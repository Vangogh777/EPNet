import argparse
import logging
import os.path
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from PIL.Image import Image
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm
# from models import eca_resnet_FC_pre_activation
from models import eca_resnet_FC
from models import EP
from utils_data.data_loading import BasicDataset, CarvanaDataset
from evaluate import evaluate
from matplotlib import pyplot as plt
from utils_data.data_load import myDataSet


def output_to_image(masks_pred):
    probs = F.softmax(masks_pred, dim=1)[0]
    probs = probs.cpu().squeeze()

    mask = F.one_hot(probs.argmax(dim=0), 3).permute(2, 0, 1).numpy()
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def train_net(net,
              device,
              dir_img,
              dir_mask,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              ):
    # 1. Create dataset
    # 创建数据集
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)


    # 重写读数据集
    dataset = myDataSet(
        data_dir=dir_img,
        label_dir=dir_mask
    )

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=12, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # 记录最好系数
    best_dice = 5
    # (Initialize logging)
    # experiment = wandb.init(project='UNet_for_NTarget', resume='allow', anonymous='must')

    # experiment = wandb.init(project='ResNet101', resume='allow', entity="vangogh")
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.SGD(net.parameters(),learning_rate,momentum=0.9,weight_decay=1e-8)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)  # goal: maximize Dice score
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    global_step = 0
    add = 0
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # images = batch['image']
                # true_masks = batch['mask']
                images = batch[0]
                images = images.unsqueeze(1)
                true_masks = batch[1]
                true_masks = true_masks.squeeze()

                images = images.to(device=device, dtype=torch.float32) * 10000
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    # 网络输出
                    masks_pred = net(images)


                    # 计算损失
                    loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
        val_score = evaluate(net, val_loader, device)
        scheduler.step(val_score)
        logging.info('Validation MSE score: {}'.format(val_score))
        logging.info('learning rate:{}'.format(optimizer.param_groups[0]['lr']))

        # experiment.log({
        #     'learning rate': optimizer.param_groups[0]['lr'],
        #     'validation MSE': val_score,
        #     'step': global_step,
        #     'epoch': epoch,
        #     'best_MSE':best_dice
        # })

        # 保存模型
        # Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        # torch.save(net, str(dir_checkpoint / 'checkpoint.pth'))
        # logging.info(f'Checkpoint {epoch + 1} saved!')

        if val_score < best_dice:
            best_dice = val_score
            add = 0
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net, str(dir_checkpoint / 'Best_dice.pth'))
            logging.info('Best_dice:{} is saved!'.format(best_dice))
        else:
            add +=1
            if add==11:
                break




def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=30.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def main(dir_img,dir_mask):
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    # net = Resnet.resnet34()
    # net = eca_resnet_FC.eca_resnet101(num_classes=8982)
    net = EP.eca_resnet101(num_classes=4632)
    # net = Resnet_1316.resnet101()
    # net = torch.load(str(dir_checkpoint / 'Best_dice.pth'))
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  dir_mask=dir_mask,
                  dir_img=dir_img,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)


if __name__ == '__main__':
    project_name = 'cornwater'
    db = ["20db", "30db", "40db", "50db", ""]
    BV = ["BV_10_20db/", "BV_10_30db/", "BV_10_40db/", "BV_10_50db/", "BV_10/"]
    for i in range(0, len(BV)):
        print(BV[i])
        dir_img = Path('G:/Corn/' + project_name + '/'+BV[i])
        dir_mask = Path('G:/Corn/' + project_name + '/DDL_10/')
        model_path = './savemodel/' + project_name + '/'+db[i]+'正常数据/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        dir_checkpoint = Path(model_path)
        main(dir_img,dir_mask)