# -*- coding: utf-8 -*-
import argparse
import time

def get_CTranS_config():
    parser = argparse.ArgumentParser()
    
    # Model configuration
    parser.add_argument('--n_channels', type=int, default=3, help='number of channels in input image')
    parser.add_argument('--n_labels', type=int, default=1, help='number of classes in segmentation task')
    parser.add_argument('--transformer_depth', type=int, default=24, help='depth of transformer')
    parser.add_argument('--KV_size', type=int, default=960, help='size of Key/Value in attention')
    parser.add_argument('--channel_size', type=int, default=2048, help='channel size in transformer')
    parser.add_argument('--expand_ratio', type=int, default=4, help='MLP expansion ratio')
    parser.add_argument('--base_channel', type=int, default=64, help='base channel of U-Net')
    parser.add_argument('--segmentor_name', type=str, default='UNet', help='Name of the segmentor: UNet, AttnUNet, ResUNet')
    
    config = parser.parse_args([])
    return config

## PARAMETERS OF THE MODEL
save_model_path = './output/'
model_name = 'RBTCA_Model'
segmentor_name = 'UNet' 
n_channels = 1 # Grayscale images for QaTa-Covid19
n_labels = 1
epochs = 100
learning_rate = 3e-4
batch_size = 24
img_size = 224
print_freq = 1
save_path = './output/'
task_name = 'QaTa-Covid19' # Default task
train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
optimizer = 'Adam'
cosineLR = True # Use cosine learning rate scheduling
seed = 42
gpu_ids = "0" # GPU IDs to use, e.g., "0" or "0,1"
train_tf = None
val_tf = None