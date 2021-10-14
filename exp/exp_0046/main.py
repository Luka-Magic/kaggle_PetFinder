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
import gc
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
    def __init__(self, cfg, df, phase, transforms=None, output_label=True):
        super().__init__()
        self.cfg = cfg
        self.df = df
        self.phase = phase
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['file_path']

        img_bgr = cv2.imread(img_path)
        img_rgb = img_bgr[:, :, ::-1]

        h, w, _ = img_rgb.shape
        h_pad, w_pad = max((w - h)//2, 0), max((h - w)//2, 0)
        if self.phase == 'valid':
            h_dis, w_dis = max((h - w)//2, 0), max((w - h)//2, 0)
            img_rgb = img_rgb[h_dis:h-h_dis, w_dis:w-w_dis, :]

        if self.cfg.padding == 'BORDER_WRAP':
            img_rgb = cv2.copyMakeBorder(
                img_rgb, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_WRAP)
        elif self.cfg.padding == 'BORDER_CONSTANT':
            img_rgb = cv2.copyMakeBorder(
                img_rgb, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT)

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


def rand_bbox(size, l):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1.0 - l)
    cut_W = np.int(W * cut_rat)
    cut_H = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_W // 2, 0, W)
    bbx2 = np.clip(cx + cut_W // 2, 0, W)
    bby1 = np.clip(cy - cut_H // 2, 0, H)
    bby2 = np.clip(cy + cut_H // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    new_data = data.clone()

    l = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    new_data = new_data * l + data[indices, :, :, :] * (1 - l)
    targets = (target, shuffled_target, l)
    return new_data, targets


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    new_data = data.clone()

    l = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), l)
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    l = 1 - (((bbx2 - bbx1) * (bby2 - bby1)) /
             (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, l)

    return new_data, targets


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

    train_ds = pf_dataset(cfg, train_, 'train',
                          transforms=get_transforms(cfg, 'train'))
    valid_ds = pf_dataset(cfg, valid_, 'valid',
                          transforms=get_transforms(cfg, 'valid'))
    valid_tta_ds = pf_dataset(
        cfg, valid_, 'valid', transforms=get_transforms(cfg, 'tta'))

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
            mix_p = np.random.rand()
            mix_list = list(range(cfg.init_nomix_epoch,
                            cfg.epoch-cfg.last_nomix_epoch))
            if (mix_p < cfg.mix_p) and (epoch in mix_list):
                imgs, labels = mixup(imgs, labels, 1.)
                preds = model(imgs)
                loss = loss_fn(
                    preds, labels[0]) * labels[2] + loss_fn(preds, labels[1]) * (1. - labels[2])
            else:
                preds = model(imgs)
                loss = loss_fn(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if cfg.mix_p == 0:
            if cfg.loss == 'BCEWithLogitsLoss':
                preds_all += [np.clip(torch.sigmoid(
                    preds).detach().cpu().numpy() * 100, 0, 100)]
                labels_all += [labels.detach().cpu().numpy() * 100]
            elif cfg.loss == 'MSELoss':
                preds_all += [np.clip(preds.detach().cpu().numpy(), 0, 100)]
                labels_all += [labels.detach().cpu().numpy()]

            preds_temp = np.concatenate(preds_all)
            labels_temp = np.concatenate(labels_all)

            score = mean_squared_error(labels_temp, preds_temp) ** 0.5

            description = f'epoch: {epoch}, loss: {loss:.4f}, score: {score:.4f}'
            pbar.set_description(description)
        else:
            description = f'epoch: {epoch}, mixup'
            pbar.set_description(description)
    lr = get_lr(optimizer)
    if scheduler:
        scheduler.step()
    if cfg.mix_p == 0:
        preds_epoch = np.concatenate(preds_all)
        labels_epoch = np.concatenate(labels_all)

        score_epoch = mean_squared_error(labels_epoch, preds_epoch) ** 0.5

        return score_epoch, loss.detach().cpu().numpy(), lr
    else:
        return lr


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


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    wandb.login()
    seed_everything(cfg.seed)

    train_df, _ = load_data(cfg.data_path)

    if cfg.fold == 'KFold':
        folds = KFold(n_splits=cfg.fold_num, shuffle=True, random_state=cfg.seed).split(
            X=np.arange(train_df.shape[0]), y=train_df.Pawpularity.values)
    elif cfg.fold == 'StratifiedKFold':
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
            cfg, train_df, train_index, valid_index)

        device = torch.device(cfg.device)

        model = pf_model(cfg.model_arch, pretrained=True).to(device)

        scaler = GradScaler()

        if cfg.optimizer == 'AdamW':
            optim = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

        if cfg.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim, total_steps=cfg.epoch, max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
        elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min)

        if cfg.loss == 'MSELoss':
            loss_fn = nn.MSELoss()
        elif cfg.loss == 'BCEWithLogitsLoss':
            loss_fn = nn.BCEWithLogitsLoss()

        for epoch in tqdm(range(cfg.epoch), total=cfg.epoch):
            # Train Start

            if cfg.mix_p == 0:
                train_start_time = time.time()
                train_score_epoch, train_loss_epoch, lr = train_one_epoch(
                    cfg, epoch, model, loss_fn, optim, train_loader, device, scheduler, scaler)
                train_finish_time = time.time()
                print(
                    f'TRAIN {epoch}, score: {train_score_epoch:.4f}, time: {train_finish_time-train_start_time:.4f}')
            else:
                train_start_time = time.time()
                lr = train_one_epoch(
                    cfg, epoch, model, loss_fn, optim, train_loader, device, scheduler, scaler)
                train_finish_time = time.time()
                print(
                    f'TRAIN {epoch}, mixup, time: {train_finish_time-train_start_time:.4f}')

            # Valid Start

            valid_start_time = time.time()

            with torch.no_grad():
                valid_score_epoch, valid_loss_epoch = valid_one_epoch(
                    cfg, epoch, model, loss_fn, valid_loader, device)

            valid_finish_time = time.time()

            valid_rmse[epoch] = valid_score_epoch

            print(
                f'VALID {epoch}, score: {valid_score_epoch}, time: {valid_finish_time-valid_start_time:.4f}')
            if cfg.mix_p == 0:
                wandb.log({'train_rmse': train_score_epoch, 'train_loss': train_loss_epoch,
                           'valid_rmse': valid_score_epoch, 'valid_loss': valid_loss_epoch,
                           'epoch': epoch, 'lr': lr})
            else:
                wandb.log({'valid_rmse': valid_score_epoch, 'valid_loss': valid_loss_epoch,
                           'epoch': epoch, 'lr': lr})

            model_name = os.path.join(
                '/'.join(os.getcwd().split('/')[:-2]), f"{cfg.model_arch}_fold_{fold}_{epoch}.pth")
            torch.save(model.state_dict(), model_name)

        # print Score
        valid_rmse_sorted = sorted(valid_rmse.items(), key=lambda x: x[1])
        print('-'*30)
        for i, (epoch, rmse) in enumerate(valid_rmse_sorted):
            print(f'No.{i+1} epoch{epoch}: {rmse:.5f}')
        print('-'*30)

        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
