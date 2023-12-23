import argparse
import logging
import os

import numpy as np
import pandas
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils_data.data_loading import BasicDataset



def predict_img(net,
                full_img,
                device,
                file_name,
                scale_factor=1,
                out_threshold=0.5,

                ):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        res = output.data.cpu().numpy()
        # np.savetxt("./predict_ddl/UV/2target/ddl/test.csv", res, delimiter=',', fmt="%f")
    return res

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')

    parser.add_argument('--model', '-m', default='.\\save_model\\3Wcircle.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')

    # parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
    #                     help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    # parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')

    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def get_output_filename(filename):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return filename or list(map(_generate_name, filename))


def mask_to_image(mask: np.ndarray):
    # print("mask.ndim:", mask.ndim)
    # print("mask:", mask)
    temp = mask * 255
    # print(temp)
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def run_ddl(data_path, file_name, name,save_path, model_path,beishu = 10000):
    args = get_args()

    # file_name = f'data/BVT/15_37_42_23.csv'
    file_name = data_path  + file_name

    save_name = save_path + name +'_ddl.csv'
    # if not os.path.exists(save_name):#检查目录是否存在
    #     os.makedirs(save_name)  # 如果不存在则创建目录
    # 输入要测试的文件名和输出文件的位置

    in_files = file_name
    out_files = get_output_filename(save_name)

    # out_files = save_name
    # print(in_files, out_files)

    # net = UNet(n_channels=1, n_classes=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    # 模型地址
    # model_name = "./model/3Wcircle_lr0.001_10.pth"
    model_name = model_path
    # print(model_name)
    net = torch.load(model_name)
    net.to(device=device)

    img = pandas.read_csv(file_name, header=None) * beishu
    my_array = np.array(img)
    my_tensor = torch.tensor(my_array)

    ddl = predict_img(
        net=net,
        full_img=my_tensor,
        scale_factor=args.scale,
        out_threshold=args.mask_threshold,
        device=device,
        file_name=name,
    )
    return ddl


if __name__ == '__main__':
    args = get_args()

    # in_files = args.input
    # out_files = get_output_filenames(args)
    # file_name = in_files

    # file_name, save_name = ".\\predict_ddl\\yuan\\1.csv", ".\\predict_ddl\\Test\\24_1.png"
    file_name = f'data/BV/100_30_0_29.csv'

    name = "1_44_57_22"
    save_name = 'predict_ddl/17_58_-51_-51_47.png'
    # if not os.path.exists(save_name):#检查目录是否存在
    #     os.makedirs(save_name)  # 如果不存在则创建目录
    # 输入要测试的文件名和输出文件的位置

    in_files = file_name
    out_files = get_output_filename(save_name)

    # out_files = save_name
    # print(in_files, out_files)

    # net = UNet(n_channels=1, n_classes=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    # 模型地址
    model_name = "model/2target/ddl_model/BestRMSE_cicle_checkpoint_50epochs_0.001lr_05-31-19-27.pth"
    # print(model_name)
    net = torch.load(model_name)
    net.to(device=device)
    img = pandas.read_csv(file_name, header=None) * 1000
    my_array = np.array(img)
    my_tensor = torch.tensor(my_array)

    predict_img(
        net=net,
        full_img=my_tensor,
        scale_factor=args.scale,
        out_threshold=args.mask_threshold,
        device=device,
        file_name=name,
    )
