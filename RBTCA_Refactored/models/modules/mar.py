import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent_attention import AgentAttentionBlock
from timm.models.layers import to_2tuple

class TextFeatureMapper(nn.Module):
    """
    Maps text embeddings to a spatially structured representation.
    Transforms [B, 25, 768] into [B, 64, 14, 14] to align with image patches.
    """
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.dim_reduce = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.len_map = nn.Linear(25, (224//patch_size)**2) # Map sequence length to patch grid (14*14=196)

    def forward(self, text_emb):
        # text_emb: [B, 25, 768]
        x = self.dim_reduce(text_emb) # [B, 25, 64]
        x = x.transpose(1, 2)         # [B, 64, 25]
        x = self.len_map(x)           # [B, 64, 196]
        x = x.reshape(-1, 64, 14, 14)
        return x

class ROIGenerator(nn.Module):
    """
    Extracts the Region of Interest (RI) using multi-scale Agent Attention.
    """
    def __init__(self, in_channels, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.attn_region_proj = nn.Linear(64, 1) # Project text features to 1 channel mask
        self.attn_mask_activation = nn.Sigmoid()
        
        # Agent Attention Blocks for multi-scale refinement
        self.attnx = AgentAttentionBlock(dim=in_channels, window_size=to_2tuple(1), num_heads=1, agent_num=1)
        self.attnx2 = AgentAttentionBlock(dim=in_channels, window_size=to_2tuple(1), num_heads=1, agent_num=4)
        self.attnx3 = AgentAttentionBlock(dim=in_channels, window_size=to_2tuple(1), num_heads=1, agent_num=16)

        self._make_windows_projections(in_channels)

    def _make_windows_projections(self, in_channels):
        def make_proj():
            return nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=self.patch_size, stride=self.patch_size),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, in_channels, kernel_size=1, stride=1)
            )
        self.x_windows_projection = make_proj()
        self.x_attn_windows_projection = make_proj()
        self.x_windows_projection2 = make_proj()
        self.x_windows_projection3 = make_proj()
        self.x_windows_activation = nn.Sigmoid()

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x, text_features):
        # x: Image [B, 1, 224, 224]
        # text_features: [B, 64, 14, 14]
        
        B, C, H, W = x.shape
        
        # 1. Generate coarse mask from text
        # Flatten text features for projection: [B, 64, 196] -> [B, 196, 64]
        tf_flat = text_features.reshape(B, 64, -1).transpose(1, 2)
        sel_attn_output = self.attn_region_proj(tf_flat) # [B, 196, 1]
        sel_attn_output = sel_attn_output.transpose(1, 2).reshape(B, 1, 14, 14)
        sel_attn_output = F.interpolate(sel_attn_output, size=(H, W), mode='bilinear', align_corners=True)
        sel_attn_mask = self.attn_mask_activation(sel_attn_output)
        
        # 2. Refine mask using Agent Attention (Multi-scale)
        # Scale 1
        x_windows = self.window_partition(x.permute(0,2,3,1).detach(), self.patch_size)
        x_windows = self.x_windows_activation(self.x_windows_projection(x_windows.permute(0,3,1,2)).permute(0,2,3,1))
        x_windows = x_windows.view(-1, 1, C)
        
        x_attn_windows = self.window_partition(sel_attn_mask.detach().expand(-1, C, -1, -1).permute(0,2,3,1), self.patch_size)
        x_attn_windows = self.x_windows_activation(self.x_attn_windows_projection(x_attn_windows.permute(0,3,1,2)).permute(0,2,3,1))
        x_attn_windows = x_attn_windows.view(-1, 1, C)
        
        attn_windows = self.attnx(x_attn_windows, attn=x_windows)
        attn_windows = attn_windows.view(-1, 1, 1, C)
        
        sel_attn_mask_lowreso = self.window_reverse(attn_windows, 1, H//self.patch_size, W//self.patch_size).permute(0,3,1,2)
        sel_attn_mask_lowreso = sel_attn_mask_lowreso.mean(dim=1, keepdim=True)

        # Scale 2
        x_windows = self.window_partition(x.permute(0,2,3,1).detach(), self.patch_size*2)
        x_windows = self.x_windows_activation(self.x_windows_projection2(x_windows.permute(0,3,1,2)).permute(0,2,3,1))
        x_windows = x_windows.view(-1, 4, C) 
        
        x_attn_windows = self.window_partition(self.x_windows_activation(sel_attn_mask_lowreso.expand(-1, C, -1, -1).permute(0,2,3,1)), 2)
        x_attn_windows = x_attn_windows.view(-1, 4, C)
        
        attn_windows = self.attnx2(x_attn_windows, attn=x_windows)
        attn_windows = attn_windows.view(-1, 2, 2, C)
        sel_attn_mask_lowreso = self.window_reverse(attn_windows, 2, H//self.patch_size, W//self.patch_size).permute(0,3,1,2)
        sel_attn_mask_lowreso = sel_attn_mask_lowreso.mean(dim=1, keepdim=True)

        # Scale 3
        x_windows = self.window_partition(x.permute(0,2,3,1).detach(), self.patch_size*7)
        x_windows = self.x_windows_activation(self.x_windows_projection3(x_windows.permute(0,3,1,2)).permute(0,2,3,1))
        x_windows = x_windows.view(-1, 49, C)
        
        x_attn_windows = self.window_partition(self.x_windows_activation(sel_attn_mask_lowreso.expand(-1, C, -1, -1).permute(0,2,3,1)), 7)
        x_attn_windows = x_attn_windows.view(-1, 49, C)
        
        attn_windows = self.attnx3(x_attn_windows, attn=x_windows)
        attn_windows = attn_windows.view(-1, 7, 7, C)
        sel_attn_mask_lowreso = self.window_reverse(attn_windows, 7, H//self.patch_size, W//self.patch_size).permute(0,3,1,2)
        sel_attn_mask_lowreso = sel_attn_mask_lowreso.mean(dim=1, keepdim=True)

        return sel_attn_output, sel_attn_mask_lowreso

class ModalityAwareRepresentation(nn.Module):
    """
    Integrates Image, Text Map, and ROI into a 3-channel MAR.
    Implementing Equation 4: I_MAR = I + CT + RI
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x, sel_attn_mask, text_features):
        # x: [B, 1, H, W]
        # sel_attn_mask: [B, 1, H, W] (upsampled refined ROI)
        # text_features: [B, 64, 14, 14]
        
        # Project text features to 1 channel and upsample to form CT
        ct = torch.mean(text_features, dim=1, keepdim=True)
        ct = F.interpolate(ct, size=x.shape[2:], mode='bilinear', align_corners=True)
        ct = torch.sigmoid(ct)
        
        # Result: [B, 3, H, W]
        return torch.cat([x, ct, sel_attn_mask], dim=1)