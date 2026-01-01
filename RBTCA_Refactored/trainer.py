import torch
import torch.nn as nn
from .configs import config
from .utils.metrics import AverageMeter, accuracy
import numpy as np
import time

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, lr_scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()
    end = time.time()
    
    aux_criterion = nn.BCEWithLogitsLoss() # For ROI auxiliary loss

    for i, (sampled_batch, names) in enumerate(train_loader):
        images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
        images = images.cuda()
        masks = masks.cuda()
        
        # Forward pass
        output = model(images, text, masks)
        
        # Unpack outputs
        seg_logits = output[0]       # Prediction from augmented image
        train_mask_aug = output[1]   # Augmented ground truth mask
        roi_logits = output[2]       # Coarse ROI logits (for Aux loss)
        
        # Segmentation Loss (L_seg)
        # Critical: Must use augmented mask for loss calculation
        seg_loss = criterion(seg_logits, train_mask_aug)
        
        # Auxiliary Loss (L_aux)
        if roi_logits is not None:
            # Interpolate ROI logits to match mask size if different
            # roi_logits is already 224x224 from rbtca_model but good to be safe or if config changes
            if roi_logits.shape[2:] != masks.shape[2:]:
                roi_logits = nn.functional.interpolate(roi_logits, size=masks.shape[2:], mode='bilinear', align_corners=True)
            
            # Aux loss compares Text-Derived ROI against ORIGINAL mask (to learn alignment)
            aux_loss = aux_criterion(roi_logits, masks)
            out_loss = seg_loss + 0.1 * aux_loss # Weight factor lambda = 0.1
        else:
            out_loss = seg_loss

        optimizer.zero_grad()
        out_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Measure accuracy and record loss
        train_acc = accuracy(seg_logits, masks)[0]
        losses.update(out_loss.item(), images.size(0))
        acc.update(train_acc.item(), images.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' 
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, acc=acc))

    return losses.avg, acc.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()
    end = time.time()
    
    with torch.no_grad():
        for i, (sampled_batch, names) in enumerate(val_loader):
            images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            images = images.cuda()
            masks = masks.cuda()

            # Forward pass (no mask augmentation for validation)
            output = model(images, text, train_mask=None)
            
            seg_logits = output[0]
            # output[1] is None when train_mask=None
            
            # Validation Loss
            val_loss = criterion(seg_logits, masks)

            # Measure accuracy and record loss
            val_acc = accuracy(seg_logits, masks)[0]
            losses.update(val_loss.item(), images.size(0))
            acc.update(val_acc.item(), images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc))

    print(' * Acc {acc.avg:.3f}'.format(acc=acc))
    return losses.avg, acc.avg
