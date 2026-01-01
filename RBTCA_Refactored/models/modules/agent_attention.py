import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

class AgentAttentionBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., agent_num=1, if_dwc=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.agent_num = agent_num
        self.agent = nn.Parameter(torch.randn(1, agent_num, dim))
        
        self.if_dwc = if_dwc
        if self.if_dwc:
            self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, attn=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if attn is None:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            kv = self.kv(attn).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            
        agent = self.agent.expand(B, -1, -1).reshape(B, self.agent_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn_agent = (q @ agent.transpose(-2, -1)) * self.scale
        attn_agent = attn_agent.softmax(dim=-1)
        attn_agent = self.attn_drop(attn_agent)
        
        x_agent = (attn_agent @ agent)
        
        attn = (agent @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v)
        x = (attn_agent @ x)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        
        if self.if_dwc:
            x = x + self.dwc(x.permute(0, 2, 1).reshape(B, C, int(N**0.5), int(N**0.5))).flatten(2).transpose(1, 2)
            
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
