import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
from torch.distributions.beta import Beta
import numpy as np
import random

class TextConsistentAugmentation:
    """
    Applies synchronized augmentations to the Modality-Aware Representation (MAR).
    Ensures geometric transformations and CutMix are applied consistently to both image and ROI channels.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            self.BatchRandomHorizontalFlip(),
            self.BatchRandomVerticalFlip(),
            self.BatchRandomRotation(20),
        ])
        self.mix_methods = [
            self.BatchCutMix(alpha=1.0),
        ]

    def __call__(self, imgs, masks, if_mix=True):
        # imgs: MAR [B, 4, H, W] (Image + Mask/ROI)
        # masks: Ground Truth Label [B, H, W]
        
        assert imgs.shape[0] == masks.shape[0]
        assert imgs.shape[2:] == masks.shape[2:]

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
            
        combined = torch.cat([imgs, masks], dim=1)
        
        transformed = self.transform(combined)
        
        transformed_imgs = transformed[:, :imgs.shape[1], :, :]
        transformed_masks = transformed[:, imgs.shape[1]:, :, :]

        if if_mix:
            mix_method = random.choice(self.mix_methods)
            mix_imgs, mix_masks = mix_method(transformed_imgs, transformed_masks)
            return mix_imgs, mix_masks
        else:
            return transformed_imgs, transformed_masks

    class BatchRandomHorizontalFlip:
        def __call__(self, imgs):
            if torch.rand(1) < 0.5:
                return VF.hflip(imgs)
            return imgs

    class BatchRandomVerticalFlip:
        def __call__(self, imgs):
            if torch.rand(1) < 0.5:
                return VF.vflip(imgs)
            return imgs

    class BatchRandomRotation:
        def __init__(self, degrees):
            self.degrees = degrees
        def __call__(self, imgs):
            angle = transforms.RandomRotation.get_params([-self.degrees, self.degrees])
            return VF.rotate(imgs, angle)

    class BatchCutMix:
        def __init__(self, alpha=1.0):
            self.alpha = alpha
            self.beta_dist = Beta(self.alpha, self.alpha)

        def __call__(self, imgs, masks):
            lam = self.beta_dist.sample().item()
            bs, c, h, w = imgs.size()
            index = torch.randperm(bs).to(imgs.device)
            cut_rat = np.sqrt(1.0 - lam)
            cut_h = int(h * cut_rat)
            cut_w = int(w * cut_rat)
            y = np.random.randint(0, h - cut_h)
            x = np.random.randint(0, w - cut_w)
            imgs[:, :, y:y+cut_h, x:x+cut_w] = imgs[index, :, y:y+cut_h, x:x+cut_w]
            masks[:, :, y:y+cut_h, x:x+cut_w] = masks[index, :, y:y+cut_h, x:x+cut_w]
            return imgs, masks