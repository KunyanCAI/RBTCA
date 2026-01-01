import torch.optim
import torch
import torch.nn as nn
from .data.loader import ValGenerator, ImageToImage2DWithCache
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
from .configs import config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from .models.rbtca_model import RBTCA_Model
from .utils.metrics import jaccard_score, read_text
import numpy as np
import cv2
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    
    cv2.imwrite(save_path, predict_save * 255)
    
    return dice_pred, iou_pred


def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, test_data, test_label, dice_pred, dice_ens, model_type):
    model.eval()

    # Pass train_mask=None during inference as we don't want augmentation
    output = model(test_data.cuda(), text, train_mask=None)[0]
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs,
                                                  save_path=vis_save_path + '_predict' + model_type + '.jpg')
    return dice_pred_tmp, iou_tmp


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    test_session = "YOUR_SESSION_NAME"

    if config.task_name == "QaTa-Covid19":
        test_num = 2113
        model_type = config.model_name
        model_path = "./output/" +"/QaTa-Covid19/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
    else:
        raise ValueError("Only QaTa-Covid19 is supported in this version.")

    save_path = "./output/"  + config.task_name + '_vis/' + model_type + '/' + test_session + '/'
    vis_path = "./output/"  + config.task_name + '/visualize_test/'+ test_session +'/' 
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type == 'RBTCA_Model':
        config_vit = config.get_CTranS_config()
        model = RBTCA_Model(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    
    # Load state dict BEFORE wrapping in DataParallel
    # This ensures we can load "clean" checkpoints into the base model
    state_dict = checkpoint['state_dict']
    
    # Check for 'module.' prefix in state_dict (legacy DataParallel checkpoints)
    # and strip it if the current model is not DataParallel yet (which it isn't here)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    print('Model loaded !')

    if torch.cuda.device_count() > 1:
       print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
       model = nn.DataParallel(model)
       
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + 'Test_text.xlsx')
   
    test_dataset = ImageToImage2DWithCache(config.test_dataset, config.task_name, test_text, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    dice_pred_list = []
    iou_pred_list = []

    with tqdm(total=len(test_dataset), desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label, test_text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path + str(names) + "_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, test_text, None, lab,
                                                           vis_path + str(names),test_data,test_label,
                                                           dice_pred=dice_pred, dice_ens=dice_ens,
                                                           model_type=model_type)
            dice_pred_list.append(dice_pred_t)
            iou_pred_list.append(iou_pred_t)
            torch.cuda.empty_cache()
            pbar.update()
            
    dice_pred_array = np.array(dice_pred_list)
    iou_pred_array = np.array(iou_pred_list)

    dice_mean = np.mean(dice_pred_array)
    iou_mean = np.mean(iou_pred_array)

    dice_std = np.std(dice_pred_array)
    iou_std = np.std(iou_pred_array)
    
    print("Dice Mean:", dice_mean)
    print("Dice std:", dice_std)
    print("IoU Mean:", iou_mean)
    print("IoU std:", iou_std)