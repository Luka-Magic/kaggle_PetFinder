# Python Libraries
import albumentations
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


def get_transforms(cfg, phase):
    if phase == 'train':
        aug = cfg.train_aug
    elif phase == 'valid':
        aug = cfg.valid_aug
    elif phase == 'tta':
        aug = cfg.tta_aug

    augs = [getattr(albumentations, name)(**kwargs)
            for name, kwargs in aug.items()]
    augs.append(ToTensorV2(p=1.))
    return albumentations.Compose(augs)


class pf_model(nn.Module):
    def __init__(self, model_arch, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained, in_chans=3)

        if model_arch == 'vit_large_patch32_384' or model_arch == 'swin_base_patch4_window12_384_in22k':
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, 1)
        elif model_arch == 'tf_efficientnet_b0':
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(self.n_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x


def prepare_dataloader(cfg, train_df, train_index, valid_index):

    train_ = train_df.loc[train_index, :].reset_index(drop=True)
    valid_ = train_df.loc[valid_index, :].reset_index(drop=True)

    train_ds = pf_dataset(train_, transforms=get_transforms(cfg, 'train'))
    valid_ds = pf_dataset(valid_, transforms=get_transforms(cfg, 'valid'))
    valid_tta_ds = pf_dataset(valid_, transforms=get_transforms(cfg, 'tta'))

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


def train_one_epoch(cfg, epoch, model, loss_fn, optimizer, data_loader, device, scheduler, scaler):

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

        if cfg.loss == 'BCEWithLogitsLoss':
            labels /= 100

        with autocast():
            preds = model(imgs)
            loss = loss_fn(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if cfg.loss == 'BCEWithLogitsLoss':
            preds_all += [np.clip(torch.sigmoid(preds).detach().cpu().numpy() * 100, 0, 100)]
            labels_all += [labels.detach().cpu().numpy() * 100]
        elif cfg.loss == 'MSELoss':
            preds_all += [np.clip(preds.detach().cpu().numpy(), 0, 100)]
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


def valid_one_epoch(cfg, epoch, model, loss_fn, data_loader, device):

    model.eval()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    preds_all = []
    labels_all = []

    for step, (imgs, labels) in pbar:
        imgs = imgs.to(device).float()
        labels = labels.to(device).float().view(-1, 1)

        if cfg.loss == 'BCEWithLogitsLoss':
            labels /= 100

        with autocast():
            preds = model(imgs)

        loss = loss_fn(preds, labels)

        if cfg.loss == 'BCEWithLogitsLoss':
            preds = np.clip(torch.sigmoid(
                preds).detach().cpu().numpy() * 100, 0, 100)
            labels = labels.detach().cpu().numpy() * 100
        elif cfg.loss == 'MSELoss':
            preds = np.clip(preds.detach().cpu().numpy(), 0, 100)
            labels = labels.detach().cpu().numpy()

        preds_all += [preds]
        labels_all += [labels]

        preds_temp = np.concatenate(preds_all)
        labels_temp = np.concatenate(labels_all)

        score = mean_squared_error(labels_temp, preds_temp) ** 0.5

        description = f'epoch: {epoch}, loss: {loss:.4f}, score: {score:.4f}'
        pbar.set_description(description)

    preds_epoch = np.concatenate(preds_all)
    labels_epoch = np.concatenate(labels_all)

    score_epoch = mean_squared_error(labels_epoch, preds_epoch) ** 0.5

    return score_epoch, loss.detach().cpu().numpy()


@hydra.main(config_path='config', config_name='default_config')
def main(cfg: DictConfig):
    wandb.login()
    seed_everything(cfg.seed)

    train_df, _ = load_data(cfg.data_path)

    train_cfg = cfg.train
    data_cfg = cfg.data

    if train_cfg.fold == 'KFold':
        folds = KFold(n_splits=cfg.fold_num, shuffle=True, random_state=cfg.seed).split(
            X=np.arange(train_df.shape[0]), y=train_df.Pawpularity.values)
    elif train_cfg.fold == 'StratifiedKFold':
        folds = StratifiedKFold(n_splits=cfg.fold_num, shuffle=True, random_state=cfg.seed).split(
            X=np.arange(train_df.shape[0]), y=train_df.Pawpularity.values)

    for fold, (train_index, valid_index) in enumerate(folds):
        if fold not in cfg.use_fold:
            continue

        if len(cfg.use_fold) == 1:
            wandb.init(project=cfg.wandb_project, entity='luka-magic',
                       name=os.getcwd().split('/')[-4], config=cfg)
        else:
            wandb.init(project=cfg.wandb_project, entity='luka-magic',
                       name=os.getcwd().split('/')[-4] + f'_{fold}', config=cfg)

        valid_rmse = {}

        train_loader, valid_loader, _ = prepare_dataloader(
            data_cfg, train_df, train_index, valid_index)

        device = torch.device(cfg.device)

        model = pf_model(cfg.model_arch, pretrained=True).to(device)

        scaler = GradScaler()

        if train_cfg.optimizer == 'AdamW':
            optim = torch.optim.AdamW(
                model.parameters(), lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2), weight_decay=train_cfg.weight_decay)

        if train_cfg.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim, total_steps=train_cfg.epoch, max_lr=train_cfg.lr, pct_start=train_cfg.pct_start, div_factor=train_cfg.div_factor, final_div_factor=train_cfg.final_div_factor)

        if train_cfg.loss == 'MSELoss':
            loss_fn = nn.MSELoss()
        elif train_cfg.loss == 'BCEWithLogitsLoss':
            loss_fn = nn.BCEWithLogitsLoss()

        for epoch in tqdm(range(train_cfg.epoch), total=train_cfg.epoch):
            # Train Start

            train_start_time = time.time()

            train_score_epoch, train_loss_epoch, lr = train_one_epoch(
                train_cfg, epoch, model, loss_fn, optim, train_loader, device, scheduler, scaler)

            train_finish_time = time.time()

            print(
                f'TRAIN {epoch}, score: {train_score_epoch:.4f}, time: {train_finish_time-train_start_time:.4f}')

            # Valid Start

            valid_start_time = time.time()

            with torch.no_grad():
                valid_score_epoch, valid_loss_epoch = valid_one_epoch(
                    train_cfg, epoch, model, loss_fn, valid_loader, device)

            valid_finish_time = time.time()

            valid_rmse[epoch] = valid_score_epoch

            print(
                f'VALID {epoch}, score: {valid_score_epoch}, time: {valid_finish_time-valid_start_time:.4f}')

            wandb.log({'train_rmse': train_score_epoch, 'train_loss': train_loss_epoch,
                       'valid_rmse': valid_score_epoch, 'valid_loss': valid_loss_epoch,
                       'epoch': epoch, 'lr': lr})

        # print Score
        valid_rmse_sorted = sorted(valid_rmse.items(), key=lambda x: x[1])
        print('-'*30)
        for i, (epoch, rmse) in enumerate(valid_rmse_sorted):
            print(f'No.{i+1} epoch{epoch}: {rmse:.5f}')
        print('-'*30)

        del model, optim, scheduler, loss_fn, valid_rmse, valid_rmse_sorted, train_loader, valid_loader, _


if __name__ == '__main__':
    main()
