import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import dice
from torch import nn
from tqdm import tqdm

from utils_data.dice_score import multiclass_dice_coeff, dice_coeff


def eval_BCE(net, dataloader, device):
    # 验证模式
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        # mask_true = mask_true.to(device=device, dtype=torch.long)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict_ddl the mask
            mask_pred = net(image)
            mask_pred = mask_pred.squeeze()
            criteria = nn.CrossEntropyLoss()
            # criteria = nn.BCEWithLogitsLoss()
            dice_score += criteria(mask_pred, mask_true)

            # save_output = mask_pred.data.cpu().numpy()
            # pred_image = (save_output > 0) * 255

    net.train()
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


def evaluate(net, dataloader, device,beishu = 10000):
    net.eval()
    num_val_batches = len(dataloader)
    MSE_score = 0
    loss_function = nn.MSELoss()
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # image, mask_true = batch['image'], batch['mask']
        # image = batch['image']
        # mask_true = batch['mask']
        image = batch[0]
        image = image.unsqueeze(1)
        mask_true = batch[1]
        mask_true = mask_true.squeeze()
         # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32) *beishu

        mask_true = mask_true.to(device=device, dtype=torch.float32)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict_ddl the mask
            mask_pred = net(image)

            # dice_score += torch.nn.MSELoss(mask_pred, mask_pred)

            # # 保存输出
            # save_output = mask_pred.data.cpu().numpy()
            # np.savetxt(f'data/cf/process/output.csv', save_output[0], fmt="%f", delimiter=",")
            # # 保存标签
            #
            # save_target = mask_true.data.cpu().numpy()
            # np.savetxt(f'data/cf/process/target.csv', save_target[0], fmt="%f", delimiter=",")
            MSE_score += loss_function(mask_pred, mask_true)
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return MSE_score
    return MSE_score / num_val_batches


def evaluate_lung(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    MSE_score = 0
    loss_function = nn.MSELoss()
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # image, mask_true = batch['image'], batch['mask']
        # image = batch['image']
        # mask_true = batch['mask']
        image = batch[0]
        image = image.unsqueeze(1)
        mask_true = batch[1]
        mask_true = mask_true.squeeze()
         # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32) *10000

        mask_true = mask_true.to(device=device, dtype=torch.float32)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict_ddl the mask
            mask_pred = net(image)

            # dice_score += torch.nn.MSELoss(mask_pred, mask_pred)

            # # 保存输出
            # save_output = mask_pred.data.cpu().numpy()
            # np.savetxt(f'data/cf/process/output.csv', save_output[0], fmt="%f", delimiter=",")
            # # 保存标签
            #
            # save_target = mask_true.data.cpu().numpy()
            # np.savetxt(f'data/cf/process/target.csv', save_target[0], fmt="%f", delimiter=",")
            MSE_score += loss_function(mask_pred, mask_true)
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return MSE_score
    return MSE_score / num_val_batches


def evaluate2(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    MSE_score = 0
    loss_function = nn.MSELoss()
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # image, mask_true = batch['image'], batch['mask']
        image = batch['image']
        mask_true = batch['mask']
        # image = batch[0]
        # image = image.unsqueeze(1)
        # mask_true = batch[1]
        mask_true = mask_true.squeeze()
         # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32) *10000

        mask_true = mask_true.to(device=device, dtype=torch.float32)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict_ddl the mask
            mask_pred = net(image)

            # dice_score += torch.nn.MSELoss(mask_pred, mask_pred)

            # # 保存输出
            # save_output = mask_pred.data.cpu().numpy()
            # np.savetxt(f'data/cf/process/output.csv', save_output[0], fmt="%f", delimiter=",")
            # # 保存标签
            #
            # save_target = mask_true.data.cpu().numpy()
            # np.savetxt(f'data/cf/process/target.csv', save_target[0], fmt="%f", delimiter=",")
            MSE_score += loss_function(mask_pred, mask_true)
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return MSE_score
    return MSE_score / num_val_batches


