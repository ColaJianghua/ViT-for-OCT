from sklearn.model_selection import GroupKFold

import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.ndimage.interpolation import zoom

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import sys
import time
import random

class CFG:
    data = 256 #512
    debug=False
    apex=False
    print_freq=100
    num_workers=4
    img_size=256 # appropriate input size for encoder 
    scheduler='CosineAnnealingWarmRestarts' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epoch=5 # Change epochs
    criterion= 'Lovasz' #'DiceBCELoss' # ['DiceLoss', 'Hausdorff', 'Lovasz']
    base_model='Unet' # ['Unet']
    encoder = 'vit' # ['attention','efficientnet-b5'] or other encoders from smp
    lr=1e-4
    min_lr=1e-6
    batch_size=4
    weight_decay=1e-6
    gradient_accumulation_steps=1
    seed=2021
    n_fold=5
    trn_fold= 0 #[0, 1, 2, 3, 4]
    train=True
    inference=False
    optimizer = 'Adam'
    T_0=10
    # N=5 
    # M=9
    T_max=10
    #factor=0.2
    #patience=4
    #eps=1e-6
    smoothing=1
    in_channels=3
    vit_blocks=12 #[8, 12]
    vit_linear=1024 #1024
    classes=1
    MODEL_NAME = 'R50-ViT-B_16'

main_dir = 'data'
train_dir = 'data/train-1'
masks_dir = 'data/masks-1'

def get_transform(mode='base'):
    if mode == 'base':
        base_transform = A.Compose([
            A.Resize(CFG.img_size, CFG.img_size, p=1.0),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.4, 
                             border_mode=cv2.BORDER_REFLECT),
            A.OneOf([
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p=.1),
                A.PiecewiseAffine(p=0.4),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(10,15,10),
                A.CLAHE(clip_limit=3),
                A.RandomBrightnessContrast(),            
            ], p=0.4),
            ToTensorV2()
        ], p=1.0)
        return base_transform
    
    elif mode == 'rand':
        rand_transform = A.Compose([
                RandAugment(CFG.N, CFG.M),
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.Resize(CFG.img_size, CFG.img_size, p=1.0),
                A.Normalize(),
                ToTensorV2()
            ])
        return rand_transform
    
    elif mode == 'strong':
        strong_transform = A.Compose([
                A.Transpose(p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.OneOf([
                        A.RandomGamma(),
                        A.GaussNoise()           
                    ], p=0.5),
                A.OneOf([
                        A.OpticalDistortion(p=0.4),
                        A.GridDistortion(p=0.2),
                        A.PiecewiseAffine(p=0.4),
                    ], p=0.5),
                A.OneOf([
                        A.HueSaturationValue(10,15,10),
                        A.CLAHE(clip_limit=4),
                        A.RandomBrightnessContrast(),            
                    ], p=0.5),

                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                A.Resize(CFG.img_size, CFG.img_size, p=1.0),
                ToTensorV2()
            ])
        return strong_transform
    
    elif mode == 'weak':
        weak_transform = A.Compose([
                A.Resize(CFG.img_size, CFG.img_size, p=0.5),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.4, 
                                 border_mode=cv2.BORDER_REFLECT),
                ToTensorV2()
            ], p=1.0)
        return weak_transform
    
    elif mode == 'valid':
        val_transform = A.Compose([
                A.Resize(CFG.img_size, CFG.img_size, p=1.0),
                ToTensorV2()
            ], p=1.0)
        return val_transform
    
    else:
        print('Unknown mode.')

mean = np.array([0.65459856,0.48386562,0.69428385])
std = np.array([0.15167958,0.23584107,0.13146145])

class HuBMAPDataset(Dataset):
    def __init__(self, main_dir, df, train=True, transform=None):
        """
        :param main_dir: 主目录路径，包含图像文件夹和掩码文件夹
        :param df: DataFrame，包含数据集的相关信息（id等）
        :param train: 是否用于训练集
        :param transform: 数据增强的变换
        """
        # 从传入的 DataFrame 获取图像 ID
        self.ids = df['id'].values
        print(f"Image IDs from DataFrame: {self.ids}")  # 调试信息
        
        # 获取所有图像的文件名，确保图像 ID 对应 CSV 文件中的 id
        self.fnames = [fname for fname in os.listdir(main_dir) if fname in self.ids]
        print(f"Found {len(self.fnames)} files in {main_dir}.")  # 调试信息
        print(f"Files in directory: {os.listdir(main_dir)}")  # 调试信息
        
        # 图像和标签的主目录
        self.main_dir = main_dir
        self.df = df
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if idx >= len(self.fnames):
            print(f"Index {idx} out of range for dataset with {len(self.fnames)} files.")  # 调试信息
            raise IndexError("Index out of range")
        
        # 获取当前文件名
        fname = self.fnames[idx]
        print(f"Processing file: {fname}")  # 调试信息
        
        # 读取图像
        img_path = os.path.join(self.main_dir, fname)
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")  # 调试信息
            return None, None
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if img is None:
            print(f"Failed to read image: {img_path}")  # 调试信息
            return None, None

        # 获取对应的标签（根据文件名查找DataFrame中匹配的行）
        img_labels = self.df[self.df['id'] == fname]
        
        # 创建一个空的掩码图像，和原图像大小相同
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        # 遍历每个边界框，绘制掩码
        for _, row in img_labels.iterrows():
            # 获取边界框坐标
            x, y, w, h = row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']
            
            # 绘制矩形区域，将区域内的像素值设置为1
            mask[y:y+h, x:x+w] = 1
        
        # 应用数据增强（如果有）
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        
        # 转换图像和标签格式
        img = torch.tensor(img).float() / 255  # 归一化到 [0, 1]
        img = img.permute(0, 1, 2).type(torch.FloatTensor)  # 转换为 (C, H, W) 格式
        
        mask = torch.tensor(mask).type(torch.FloatTensor)  # 保持掩码为 (H, W) 格式

        return img, mask

def vis_aug_data(dataset, length=10):
    plt.figure(figsize=(15,10))
    N = length // 2
    for i in range(length):
        try:
            image, mask = dataset[i]
            if image is None or mask is None:
                continue
            image = image.permute(1, 2, 0).numpy()  # 转换为 (H, W, C) 格式
            plt.subplot(3,4,2*i+1)
            plt.imshow(image)
            plt.axis('off')
            plt.subplot(3,4,2*i+2)
            plt.imshow(mask.numpy(), cmap='gray')
            plt.axis('off')
        except IndexError as e:
            print(e)

# 加载 CSV 文件
train_df = pd.read_csv('data/train-1.csv')
print(train_df.head())  # 调试信息

# 创建数据集
train_dataset = HuBMAPDataset('data/train-1', train_df, train=True, transform=get_transform('base'))

# 可视化数据增强后的图像和掩码
vis_aug_data(train_dataset, 6)

directory_list = os.listdir('data/train-1')
directory_list = [fnames.split('_')[0] for fnames in directory_list]
dir_df = pd.DataFrame(directory_list, columns=['id'])
dir_df

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


# 配置 VisionTransformer
config_vit = CONFIGS_ViT_seg[CFG.MODEL_NAME]
config_vit.n_classes = 1
config_vit.n_skip = 3
config_vit.pretrained_path = './R50+ViT-B_16.npz'
config_vit.transformer.dropout_rate = 0.2
config_vit.transformer.mlp_dim = 768
config_vit.transformer.num_heads = 4
config_vit.transformer.num_layers = 8

model = ViT_seg(config_vit, img_size=CFG.img_size, num_classes=config_vit.n_classes)

# 打印模型配置
print(config_vit)

class ViTHuBMAP(nn.Module):
    def __init__(self, configs=config_vit):
        super(ViTHuBMAP, self).__init__()
        
        self.model = ViT_seg(configs, img_size=CFG.img_size, num_classes=CFG.classes)
        self.model.load_from(weights=np.load(configs.pretrained_path))

    
    def forward(self, x):
        img_segs = self.model(x)
        
        return img_segs


sys.path.append('SegLossOdyssey-master')
from losses_pytorch.hausdorff import HausdorffDTLoss
from losses_pytorch.lovasz_loss import LovaszSoftmax
from losses_pytorch.focal_loss import FocalLoss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=CFG.smoothing):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice
    
    
    
class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=CFG.smoothing):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).mean()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.mean() + targets.mean() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE.mean()
    
    
class Hausdorff_loss(nn.Module):
    def __init__(self):
        super(Hausdorff_loss, self).__init__()
        
    def forward(self, inputs, targets):
        return HausdorffDTLoss()(inputs, targets)
    
class FocalDLoss(nn.Module):
    def __init__(self):
        super(FocalDLoss, self).__init__()
        
    def forward(self, inputs, targets):
        return FocalLoss()(inputs, targets)
    
    
class Lovasz_loss(nn.Module):
    def __init__(self):
        super(Lovasz_loss, self).__init__()
        
    def forward(self, inputs, targets):
        return LovaszSoftmax()(inputs, targets)
    
if CFG.criterion == 'DiceBCELoss':
    criterion = DiceBCELoss()
elif CFG.criterion == 'DiceLoss':
    criterion = DiceLoss()
elif CFG.criterion == 'FocalLoss':
    criterion = FocalDLoss()
elif CFG.criterion == 'Hausdorff':
    criterion = Hausdorff_loss()
elif CFG.criterion == 'Lovasz':
    criterion = Lovasz_loss()

def HuBMAPLoss(images, targets, model, device, loss_func=criterion):
    model.to(device)
    images = images.to(device)
    targets = targets.to(device)
    outputs = model(images)
    loss_func = loss_func
    loss = loss_func(outputs, targets)
    return loss, outputs

def train_one_epoch(epoch, model, device, optimizer, scheduler, trainloader):
    model.train()
    t = time.time()
    total_loss = 0
    
    for step, (images, targets) in enumerate(trainloader):
        loss, outputs = HuBMAPLoss(images, targets, model, device)
        loss.backward()
        if ((step+1)%4==0 or (step+1)==len(trainloader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss = loss.detach().item()
        total_loss += loss
        
        if ((step+1)%10==0 or (step+1)==len(trainloader)):
            print(
                    f'epoch {epoch} train step {step+1}/{len(trainloader)}, ' + \
                    f'loss: {total_loss/len(trainloader):.4f}, ' + \
                    f'time: {(time.time() - t):.4f}', end= '\r' if (step + 1) != len(trainloader) else '\n'
                )

            
        
def valid_one_epoch(epoch, model, device, optimizer, scheduler, validloader):
    model.eval()
    t = time.time()
    total_loss = 0
    
    for step, (images, targets) in enumerate(validloader):
        loss, outputs = HuBMAPLoss(images, targets, model, device)
        loss = loss.detach().item()
        total_loss += loss
        
        if ((step+1)%4==0 or (step+1)==len(validloader)):
            scheduler.step(total_loss/len(validloader))
        
        if ((step+1)%10==0 or (step+1)==len(validloader)):
            print(
                    f'**epoch {epoch} trainz step {step+1}/{len(validloader)}, ' + \
                    f'loss: {total_loss/len(validloader):.4f}, ' + \
                    f'time: {(time.time() - t):.4f}', end= '\r' if (step + 1) != len(validloader) else '\n'
                )

FOLDS = CFG.n_fold
gkf = GroupKFold(FOLDS)
dir_df['Folds'] = 0
for fold, (tr_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
    dir_df.loc[val_idx, 'Folds'] = fold
    
dir_df

def prepare_train_valid_dataloader(df, fold):
    train_ids = df[~df.Folds.isin(fold)]
    val_ids = df[df.Folds.isin(fold)]
    
    train_ds = HuBMAPDataset(main_dir, train_ids, train=True, transform=get_transform('base'))
    val_ds = HuBMAPDataset(main_dir, val_ids, train=True, transform=get_transform('valid'))
    
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, pin_memory=True, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, pin_memory=True, shuffle=False, num_workers=CFG.num_workers)
    
    return train_loader, val_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViTHuBMAP().to(device)
optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)

# scheduler setting
if CFG.scheduler == 'CosineAnnealingWarmRestarts':
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
elif CFG.scheduler == 'ReduceLROnPlateau':
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
elif CFG.scheduler == 'CosineAnnealingLR':
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)

