import torch
import torch.nn as nn
from .modules.mar import TextFeatureMapper, ROIGenerator, ModalityAwareRepresentation
from .modules.tca import TextConsistentAugmentation
from .segmentor import UNetSegmentor, AttnUNetSegmentor, ResUNetSegmentor
from .text_encoder import BERTEmbedder

class RBTCA_Model(nn.Module):
    """
    RBTCA Model: Region-Based Text-Consistent Augmentation.
    Pipeline: Text Encoding -> ROI Generation -> MAR Integration -> TCA Augmentation -> Segmentation.
    """
    def __init__(self, config, n_channels=1, n_classes=1, img_size=224, vis=False):
        super().__init__()
        # Extracts features from input text
        self.text_encoder = BERTEmbedder(n_embed=768, n_layer=32, max_seq_len=25)
        
        # Maps text features to spatial dimensions
        self.text_mapper = TextFeatureMapper(patch_size=16)
        
        # Generates Region of Interest mask using Agent Attention
        self.roi_generator = ROIGenerator(in_channels=n_channels, patch_size=16)
        
        # Combines Image and ROI into Modality-Aware Representation
        self.mar_integrator = ModalityAwareRepresentation()
        
        # Applies consistent augmentation to Image and ROI
        self.tca_augmentor = TextConsistentAugmentation()
        
        # Select Segmentor based on config
        segmentor_name = getattr(config, 'segmentor_name', 'UNet')
        if segmentor_name == 'UNet':
            self.segmentor = UNetSegmentor(in_channels=3, n_classes=n_classes, base_channel=config.base_channel)
        elif segmentor_name == 'AttnUNet':
            self.segmentor = AttnUNetSegmentor(in_channels=3, n_classes=n_classes, base_channel=config.base_channel)
        elif segmentor_name == 'ResUNet':
            self.segmentor = ResUNetSegmentor(in_channels=3, n_classes=n_classes, base_channel=config.base_channel)
        else:
            raise ValueError(f"Unknown segmentor: {segmentor_name}")

    def forward(self, x, text, train_mask=None):
        """
        Args:
            x: Input image tensor [B, 1, H, W] (Grayscale)
            text: Input text description
            train_mask: Ground truth mask (used for augmentation during training)
        """
        # 1. Text Encoding
        text_emb = self.text_encoder(text)
        
        # 2. Text -> Spatial Prompt (CT)
        spatial_text = self.text_mapper(text_emb)
        
        # 3. ROI Generation (RI)
        # sel_attn_output: Coarse logits [B, 1, 224, 224] (for Aux Loss)
        # sel_attn_mask: Refined low-res features [B, 1, 14, 14]
        sel_attn_output, sel_attn_mask = self.roi_generator(x, spatial_text)
        
        # Generate spatial mask for integration (Upsample refined mask)
        final_attn_mask = torch.nn.functional.interpolate(sel_attn_mask, size=x.shape[2:], mode='bilinear', align_corners=True)
        final_attn_mask = torch.sigmoid(final_attn_mask)
        
        # 4. Modality Aware Representation (MAR)
        # Integrate Image + Text(CT) + ROI(RI) into 3-channel input
        mar = self.mar_integrator(x, final_attn_mask, spatial_text)
        
        # 5. Text-Consistent Augmentation (TCA)
        if self.training and train_mask is not None:
            mar_aug, train_mask_aug = self.tca_augmentor(mar, train_mask.float())
            mar = mar_aug
            train_mask = train_mask_aug
        
        # 6. Segmentation
        logits = self.segmentor(mar)
        
        return logits, train_mask, sel_attn_output