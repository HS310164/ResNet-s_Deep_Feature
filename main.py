# coding: UTF-8
import resnet
from opt import parse_opts
from torch import nn
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import glob
import os
from PIL import Image
import subprocess
import cv2
import HandDetection as hd
from tqdm import tqdm


def main():
    # モデル定義
    model = resnet.resnet152(pretrained=True)
    if torch.cuda.is_available():  # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'
    print('device is {0}'.format(device))
    # print(model.weight.type)
    model.to(device)
    # 絶対パスに変換
    opt = parse_opts()
    outpath = os.path.abspath(opt.output)
    apath = os.path.abspath(opt.input)
    video_names = sorted(glob.glob(os.path.join(apath, '*')))

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    for vpath in video_names:
        vname = os.path.splitext(os.path.basename(vpath))[0]

        subprocess.call('mkdir tmp', shell=True)
        subprocess.call('ffmpeg -loglevel warning -i {} tmp/image_%05d.jpg'.format(vpath), shell=True)
        images = sorted(glob.glob('tmp/*.jpg'))
        if opt.only_hand:
            print('convert to masked images')
            for im in tqdm(images):
                frame = cv2.imread(im)
                maskedframe, _ = hd.detect(frame)
                cv2.imwrite(im,maskedframe)
            print('complete convert images')

        print('extract {}\'s DeepFeatrue'.format(vname))

        outputs = input_image(images, model)

        # ファイルに保存
        if not os.path.exists(outpath):
            subprocess.call('mkdir {}'.format(outpath), shell=True)

        savename = os.path.join(outpath, vname + '.npy')
        np.save(savename, outputs)
        subprocess.call('rm -rf tmp', shell=True)


def input_image(im_paths, model):
    # 画像のリサイズ，テンソル化
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
    dataset = TransDataset(im_paths, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    outputs = []
    for idx, input in enumerate(data_loader):
        if torch.cuda.is_available():
            input = input.to('cuda')
        output = model(input)
        output = output[0]
        outputs.append(output.to('cpu').data.numpy())
    outputs = np.array(outputs)
    print(outputs.shape)
    return outputs


class TransDataset(data.Dataset):
    def __init__(self, im_paths, transform=None):
        self.im_paths = im_paths
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])
        if self.transform:
            im = self.transform(im)
        return im


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


if __name__ == '__main__':
    main()

'''
ImageNetでpretrainしたResNetからのディープ特徴抽出
python main.py --input inputs_dir/ --output outputs_dir/
input_dirには動画が格納されたフォルダを指定してください
'''
