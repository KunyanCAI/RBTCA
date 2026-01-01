import torch.optim
from .data.loader import ImageToImage2DWithCache
from torch.utils.data import DataLoader
from .models.rbtca_model import RBTCA_Model
import torch
import torch.nn as nn
from .utils.metrics import read_text
import os
import random
import numpy as np
import logging
import torch.backends.cudnn as cudnn
from .configs import config
from .trainer import train_one_epoch, validate
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(config, logger, checkpoint_path=None):
    if config.task_name == "QaTa-Covid19":
        train_text = read_text(config.train_dataset + 'Train_Val_text.xlsx')
        val_text = read_text(config.val_dataset + 'Train_Val_text.xlsx')
        train_dataset = ImageToImage2DWithCache(config.train_dataset, config.task_name, train_text, config.train_tf, image_size=config.img_size)
        val_dataset = ImageToImage2DWithCache(config.val_dataset, config.task_name, val_text, config.val_tf, image_size=config.img_size)
    else:
        raise ValueError("Only QaTa-Covid19 is supported in this version.")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    lr = config.learning_rate
    logger.info(train_dataset.rowtext)

    if config.model_name == 'RBTCA_Model':
        config_vit = config.get_CTranS_config()
        model = RBTCA_Model(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    criterion = nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for single channel output

    if torch.cuda.device_count() > 1:
        print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    
    if config.cosineLR:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    else:
        lr_scheduler = None
        
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_pred']
        print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, start_epoch))
    else:
        start_epoch = 0
        best_pred = 0.0

    # Training loop
    for epoch in range(start_epoch + 1, config.epochs + 1):
        loss_avg, acc_avg = train_one_epoch(train_loader, model, criterion, optimizer, epoch, lr_scheduler)
        val_loss, val_acc = validate(val_loader, model, criterion)
        
        logger.info('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
            epoch, config.epochs, loss_avg, acc_avg, val_loss, val_acc))

        # Save best model based on Validation Accuracy
        if val_acc > best_pred:
            best_pred = val_acc
            save_path = config.save_path + '/best_model-' + config.model_name + '.pth.tar'
            
            # Save the underlying model state_dict (unwrap DataParallel if present)
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'best_pred': best_pred,
            }, save_path)
            print("Saved best model at {} with Val Acc: {:.4f}".format(save_path, best_pred))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    train(config, logger)