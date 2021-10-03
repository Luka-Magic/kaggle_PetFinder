# Python Libraries
from albumentations import Compose, Resize, RandomResizedCrop, CenterCrop, HorizontalFlip, VerticalFlip, ShiftScaleRotate, Cutout, RandomGridShuffle, Normalize, PadIfNeeded
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
import timm
from albumentations.pytorch import ToTensorV2
import cv2
import os
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import hydra
from omegaconf import DictConfig
import warnings


def load_data(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))

    train_df['file_path'] = train_df['Id'].apply(
        lambda x: os.path.join(data_path, f'train/{x}.jpg'))
    test_df['file_path'] = test_df['Id'].apply(
        lambda x: os.path.join(data_path, f'test/{x}.jpg'))
    return train_df, test_df


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    warnings.simplefilter('ignore')


class pf_dataset(Dataset):
    def __init__(self, df, transforms=None, output_label=True):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['file_path']

        img_bgr = cv2.imread(img_path)
        img_rgb = img_bgr[:, :, ::-1]

        if self.transforms:
            img = self.transforms(image=img_rgb)['image']

        else:
            img = img_rgb.transpose(2, 0, 1) / 256.
            img = torch.from_numpy(img).float()

        if self.output_label:
            target = self.df.iloc[index]['Pawpularity']
            return img, target

        return img


def get_train_transforms(cfg):
    return Compose([
        PadIfNeeded(min_height=cfg.img_size*3, min_width=cfg.img_size*3, border_mode=3),
        Resize(cfg.img_size, cfg.img_size),
        #         HorizontalFlip(p=0.5),
        #         VerticalFlip(p=0.5),
        #         ShiftScaleRotate(shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1), rotate_limit=(-5, 5)),
        #         # RandomGridShuffle(grid=(3, 3)),
        #         Cutout(num_holes=8, max_h_size=30, max_w_size=30),
        Normalize(max_pixel_value=255., p=1.0),
        ToTensorV2()
    ])


def get_valid_transforms(cfg):
    return Compose([
        PadIfNeeded(min_height=cfg.img_size*3,
                    min_width=cfg.img_size*3, border_mode=3),
        Resize(cfg.img_size, cfg.img_size),
        Normalize(max_pixel_value=255., p=1.0),
        ToTensorV2()
    ])


def get_test_transforms(cfg):
    return Compose([
        PadIfNeeded(min_height=cfg.img_size*3,
                    min_width=cfg.img_size*3, border_mode=3),
        Resize(cfg.img_size, cfg.img_size),
        Normalize(max_pixel_value=255., p=1.0),
        ToTensorV2()
    ])


def get_tta_transforms(cfg):
    return Compose([
        RandomResizedCrop(cfg.img_size, cfg.img_size,
                          scale=(1., 1.), ratio=(1., 1.)),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=(-0.1, 0.1),
                         scale_limit=(-0.1, 0.1), rotate_limit=(-5, 5)),
        ToTensorV2()
    ])


class pf_model(nn.Module):
    def __init__(self, model_arch, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, in_chans=3)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x


def prepare_dataloader(cfg, train_df, train_index, valid_index):

    train_ = train_df.loc[train_index, :].reset_index(drop=True)
    valid_ = train_df.loc[valid_index, :].reset_index(drop=True)

    train_ds = pf_dataset(train_, transforms=get_train_transforms(cfg))
    valid_ds = pf_dataset(valid_, transforms=get_valid_transforms(cfg))
    valid_tta_ds = pf_dataset(valid_, transforms=get_tta_transforms(cfg))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    valid_tta_loader = DataLoader(
        valid_tta_ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    return train_loader, valid_loader, valid_tta_loader


def train_one_epoch(epoch, model, loss_fn, optimizer, data_loader, device, scheduler, scaler):

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    model.train()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    preds_all = []
    labels_all = []

    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()
        labels = labels.to(device).float().view(-1, 1)

        with autocast():
            preds = model(imgs)
            loss = loss_fn(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        preds_all += [preds.detach().cpu().numpy()]
        labels_all += [labels.detach().cpu().numpy()]

        preds_temp = np.concatenate(preds_all)
        labels_temp = np.concatenate(labels_all)

        score = mean_squared_error(labels_temp, preds_temp) ** 0.5

        description = f'epoch: {epoch}, loss: {loss:.4f}, score: {score:.4f}'
        pbar.set_description(description)
    if scheduler:
        scheduler.step()

    lr = get_lr(optimizer)

    preds_epoch = np.concatenate(preds_all)
    labels_epoch = np.concatenate(labels_all)

    score_epoch = mean_squared_error(labels_epoch, preds_epoch) ** 0.5

    return score_epoch, loss.detach().cpu().numpy(), lr


def valid_one_epoch(epoch, model, loss_fn, data_loader, device):

    model.eval()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    preds_all = []
    labels_all = []

    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()
        labels = labels.to(device).float().view(-1, 1)
        with autocast():
            preds = model(imgs)

        loss = loss_fn(preds, labels)

        preds_all += [preds.detach().cpu().numpy()]
        labels_all += [labels.detach().cpu().numpy()]

        preds_temp = np.concatenate(preds_all)
        labels_temp = np.concatenate(labels_all)

        score = mean_squared_error(labels_temp, preds_temp) ** 0.5

        description = f'epoch: {epoch}, loss: {loss:.4f}, score: {score:.4f}'
        pbar.set_description(description)

    preds_epoch = np.concatenate(preds_all)
    labels_epoch = np.concatenate(labels_all)

    score_epoch = mean_squared_error(labels_epoch, preds_epoch) ** 0.5

    return score_epoch, loss.detach().cpu().numpy()


@hydra.main(config_path='.', config_name='config')
def main(cfg: DictConfig):
    wandb.login()
    seed_everything(cfg.seed)

    train_df, _ = load_data(cfg.data_path)

    folds = KFold(n_splits=cfg.fold_num, shuffle=True, random_state=cfg.seed).split(
        X=np.arange(train_df.shape[0]), y=train_df.Pawpularity.values)

    for fold, (train_index, valid_index) in enumerate(folds):
        if fold != cfg.use_fold:
            continue

        wandb.init(project='kaggle_PF_pre', entity='luka-magic',
                   name='exp_' + str(cfg.nb_num).zfill(4))

        train_loader, valid_loader, _ = prepare_dataloader(
            cfg, train_df, train_index, valid_index)

        device = torch.device(cfg.device)

        model = pf_model(cfg.model_arch, pretrained=True).to(device)

        scaler = GradScaler()

        if cfg.optimizer == 'AdamW':
            optim = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

        if cfg.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim, total_steps=cfg.epoch, max_lr=cfg.lr)

        loss_fn = nn.MSELoss()

        for epoch in tqdm(range(cfg.epoch), total=cfg.epoch):
            train_score_epoch, train_loss_epoch, lr = train_one_epoch(
                epoch, model, loss_fn, optim, train_loader, device, scheduler, scaler)
            print(f'TRAIN | epoch: {epoch}, score: {train_score_epoch}')

            with torch.no_grad():
                valid_score_epoch, valid_loss_epoch = valid_one_epoch(
                    epoch, model, loss_fn, valid_loader, device)
                print(f'VALID | epoch: {epoch}, score: {valid_score_epoch}')

            wandb.log({'train_rmse': train_score_epoch, 'train_loss': train_loss_epoch,
                       'valid_rmse': valid_score_epoch, 'valid_loss': valid_loss_epoch,
                       'epoch': epoch, 'lr': lr})


if __name__ == '__main__':
    main()
