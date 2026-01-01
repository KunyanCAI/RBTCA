import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import os

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        
        # image is already grayscale from __getitem__
        # image = Image.fromarray(image.astype(np.uint8)) # Removed redundant step
        
        # Convert to tensor first to get [0, 1] range
        image = F.to_tensor(image)
        label = F.to_tensor(label)
        
        # Normalize image (grayscale)
        image = F.normalize(image, mean=[0.5], std=[0.5])
        
        return {'image': image, 'label': label, 'text': text}

class ImageToImage2DWithCache(Dataset):
    """
    Reads image and masks for QaTa-Covid19.
    Supports PNG format for images and masks. Images are loaded as grayscale.
    """
    def __init__(self, dataset_path, task_name, rowtext, joint_transform=None,
                 one_hot_mask=False, image_size=224):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.task_name = task_name
        self.rowtext = rowtext
        
        if self.task_name == "QaTa-Covid19":
            self.input_path = os.path.join(dataset_path, 'img')
            self.output_path = os.path.join(dataset_path, 'labelcol')
            self.images_list = [f for f in os.listdir(self.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.one_hot_mask = one_hot_mask
        else:
             raise ValueError(f"Task name {task_name} not supported in ImageToImage2DWithCache")

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if self.task_name == "QaTa-Covid19":
            image_filename = self.images_list[idx]
            mask_filename = image_filename 
            
            # Load Image as Grayscale
            image = Image.open(os.path.join(self.input_path, image_filename)).convert('L')
            # Load Mask
            mask = Image.open(os.path.join(self.output_path, mask_filename)).convert('L')
            
            # Resize
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            
            image = np.array(image)
            mask = np.array(mask)
            
            # Ensure mask is binary 0/1
            mask[mask <= 0] = 0
            mask[mask > 0] = 1
            
            # Get text from rowtext using filename
            text = self.rowtext.get(mask_filename, "No description available")

            sample = {'image': image, 'label': mask, 'text': text}

            if self.joint_transform:
                sample = self.joint_transform(sample)

            return sample, image_filename
