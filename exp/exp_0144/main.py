# Python Libraries
from utils.loss import FOCALLoss, RMSELoss
from utils.mixaug import mixup, cutmix
from utils.make_columns import make_columns, len_columns
from utils.augmix import RandomAugMix
from utils.averagemeter import AverageMeter
import warnings
from omegaconf import DictConfig
import hydra
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import time
import gc
import os
import cv2
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
import albumentations


def load_data(cfg):
    data_path = cfg.data_path
    train_df = pd.read_csv(os.path.join(data_path, f'{cfg.train_csv}.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))

    train_df['file_path'] = train_df['Id'].apply(
        lambda x: os.path.join(data_path, f'train/{x}.jpg'))
    test_df['file_path'] = test_df['Id'].apply(
        lambda x: os.path.join(data_path, f'test/{x}.jpg'))

    train_df['kfold'] = -1

    if cfg.fold == 'KFold':
        folds = KFold(n_splits=cfg.fold_num, shuffle=True, random_state=cfg.seed).split(
            X=np.arange(train_df.shape[0]), y=train_df.Pawpularity.values)
    elif cfg.fold == 'StratifiedKFold':
        folds = StratifiedKFold(n_splits=cfg.fold_num, shuffle=True, random_state=cfg.seed).split(
            X=np.arange(train_df.shape[0]), y=train_df.Pawpularity.values)
    for fold, (train_index, valid_index) in enumerate(folds):
        train_df.loc[valid_index, 'kfold'] = fold

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
        self.dense_columns = make_columns(cfg.dense_columns)

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

        dense = torch.from_numpy(
            self.df.loc[index, self.dense_columns].values.astype('float'))

        if self.output_label:
            target = self.df.iloc[index]['Pawpularity']
            return img, dense, target

        return img, dense


def get_transforms(cfg, phase):
    if phase == 'train':
        aug = cfg.train_aug
    elif phase == 'valid':
        aug = cfg.valid_aug
    elif phase == 'tta':
        aug = cfg.tta_aug

    augs = [getattr(albumentations, name)(**kwargs) if name != 'RandomAugMix' else RandomAugMix(**kwargs)
            for name, kwargs in aug.items()]
    augs.append(ToTensorV2(p=1.))
    return albumentations.Compose(augs)


class pf_model(nn.Module):
    def __init__(self, cfg, out_dim=10, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            cfg.model_arch, pretrained=pretrained, in_chans=3)

        if re.search(r'vit*', cfg.model_arch) or re.search(r'swin*', cfg.model_arch):
            self.n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        elif re.search(r'tf*', cfg.model_arch):
            self.n_features = self.model.classifier.in_features
            self.model.head = nn.Identity()

        self.branch1 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 1)
        )
        self.branch10 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 10)
        )
        self.branch20 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 20)
        )
        self.branch100 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256, 100)
        )

    def forward(self, input, dense):
        features = self.model(input)
        x1 = self.branch1(features)
        x10 = self.branch10(features)
        x20 = self.branch20(features)
        x100 = self.branch100(features)
        return [x1, x10, x20, x100]


# class GradeLabelBCEWithLogits(nn.Module):
#     def __init__(self, class_num: int):
#         super().__init__()
#         self.class_num = class_num
#         self.interval = 100 // self.class_num

#     def forward(self, preds, target):
#         bs = target.shape[0]
#         dif = torch.Tensor(
#             [i for i in range(0, 100, self.interval)]).repeat(bs, 1).to('cuda:0').float()
#         target = torch.t(target.repeat(self.class_num, 1))
#         labels = torch.clamp(
#             (target - dif) / self.interval, 0., 1.)
#         bcewithlogits = F.binary_cross_entropy_with_logits
#         return bcewithlogits(preds, labels)


class GradeLabelBCEWithLogits(nn.Module):
    def __init__(self, cfg, reg_criterion):
        super().__init__()
        self.cls = cfg.cls
        self.loss = cfg.loss
        self.cls_weights = cfg.cls_weights
        self.reg_criterion = reg_criterion

    def forward(self, preds, target):
        bs = target.shape[0]
        losses = []
        for cls_i, (cls, weight) in enumerate(zip(self.cls, self.cls_weights)):
            if cls == 1:
                target_reg = target.float().view(-1, 1)
                if self.loss == 'BCEWithLogitsLoss' or self.loss == 'FOCALLoss':
                    target_reg /= 100
                losses.append(self.reg_criterion(
                    preds[cls_i], target_reg) * weight)
            else:
                interval = 100 // cls
                dif = torch.Tensor(
                    [i for i in range(0, 100, interval)]).repeat(bs, 1).to('cuda:0').float()
                target_rep = torch.t(target.repeat(cls, 1))
                labels = torch.clamp((target_rep - dif) / interval, 0., 1.)
                bcewithlogits = F.binary_cross_entropy_with_logits
                losses.append(bcewithlogits(preds[cls_i], labels) * weight)
        return sum(losses)


def prepare_dataloader(cfg, train_df, valid_df):
    train_ds = pf_dataset(cfg, train_df, 'train',
                          transforms=get_transforms(cfg, 'train'))
    valid_ds = pf_dataset(cfg, valid_df, 'valid',
                          transforms=get_transforms(cfg, 'valid'))
    valid_tta_ds = pf_dataset(
        cfg, valid_df, 'valid', transforms=get_transforms(cfg, 'tta'))

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

def get_preds(cfg, preds):
    '''
        output: numpy ()
    '''



def valid_function(cfg, epoch, model, loss_fn, data_loader, device):

    model.eval()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    preds_all = []
    labels_all = []

    losses = AverageMeter()

    for step, (imgs, dense, labels) in pbar:
        imgs = imgs.to(device).float()
        dense = dense.to(device).float()
        labels = labels.to(device).long()

        with autocast():
            preds = model(imgs, dense)
            loss = loss_fn(preds, labels)
        losses.update(loss.item(), cfg.valid_bs)

        if cfg.loss == 'BCEWithLogitsLoss' or cfg.loss == 'FOCALLoss':
            preds = np.clip(torch.sigmoid(
                preds[0]).detach().cpu().numpy() * 100, 1, 100)
        elif cfg.loss == 'MSELoss' or cfg.loss == 'RMSELoss':
            preds = np.clip(preds[0].detach().cpu().numpy(), 1, 100)
        print(preds.shape)
        # preds = get_preds(cfg, preds)

        labels = labels.detach().cpu().numpy()

        preds_all += [preds]
        labels_all += [labels]
        # preds_temp = np.sum(np.concatenate(preds_all), axis=1)
        preds_temp = np.concatenate(preds_all)
        labels_temp = np.concatenate(labels_all)

        score = mean_squared_error(labels_temp, preds_temp) ** 0.5

        description = f'epoch: {epoch}, loss: {loss:.4f}, score: {score:.4f}'
        pbar.set_description(description)

    # preds_epoch = np.sum(np.concatenate(preds_all), axis=1)
    preds_epoch = np.concatenate(preds_all)
    labels_epoch = np.concatenate(labels_all)

    score_epoch = mean_squared_error(labels_epoch, preds_epoch) ** 0.5

    return score_epoch, losses.avg


def train_valid_one_epoch(cfg, epoch, model, loss_fn, optimizer, train_loader, valid_loader, device, scheduler, scaler, best_score, model_name):
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    lr = get_lr(optimizer)

    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    preds_all = []
    labels_all = []

    losses = AverageMeter()

    for step, (imgs, dense, labels) in pbar:
        imgs = imgs.to(device).float()
        dense = dense.to(device).float()
        labels = labels.to(device).long()

        with autocast():
            mix_p = np.random.rand()
            mix_list = list(range(cfg.init_nomix_epoch,
                            cfg.epoch-cfg.last_nomix_epoch))
            if (mix_p < cfg.mix_p) and (epoch in mix_list):
                imgs, labels = mixup(imgs, labels, 1.)
                preds = model(imgs, dense)
                loss = loss_fn(
                    preds, labels[0]) * labels[2] + loss_fn(preds, labels[1]) * (1. - labels[2])
            else:
                preds = model(imgs, dense)
                loss = loss_fn(preds, labels)
        losses.update(loss.item(), cfg.train_bs)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if cfg.mix_p == 0:
            if cfg.loss == 'BCEWithLogitsLoss' or cfg.loss == 'FOCALLoss':
                preds_all += [np.clip(torch.sigmoid(
                    preds[0]).detach().cpu().numpy() * 100, 1, 100)]
                labels_all += [labels.detach().cpu().numpy() * 100]
            elif cfg.loss == 'MSELoss' or cfg.loss == 'RMSELoss':
                preds_all += [np.clip(preds[0].detach().cpu().numpy(), 1, 100)]
                labels_all += [labels.detach().cpu().numpy()]

            # preds_temp = np.sum(np.concatenate(preds_all), axis=1)
            preds_temp = np.concatenate(preds_all)
            labels_temp = np.concatenate(labels_all)
            train_score = mean_squared_error(labels_temp, preds_temp) ** 0.5

            description = f'epoch: {epoch}, loss: {loss:.4f}, score: {train_score:.4f}'
            pbar.set_description(description)

        else:
            description = f'epoch: {epoch}, mixup'
            pbar.set_description(description)

        if (step + 1) % cfg.save_step == 0 or (step + 1) == len(train_loader):
            with torch.no_grad():
                valid_score, valid_losses = valid_function(cfg, epoch, model, loss_fn,
                                                           valid_loader, device)
            model.train()

            if (step + 1) % cfg.save_step == 0:
                if cfg.mix_p == 0:
                    wandb.log({'train_rmse': train_score, 'valid_rmse': valid_score, 'train_loss': losses.avg,
                               'valid_loss': valid_losses, 'step_sum': epoch*len(train_loader) + step})
                else:
                    wandb.log({'valid_rmse': valid_score, 'train_loss': losses.avg,
                               'valid_loss': valid_losses, 'step_sum': epoch*len(train_loader) + step})

            if (step + 1) == len(train_loader):
                if cfg.mix_p == 0:
                    wandb.log({'train_rmse': train_score, 'valid_rmse': valid_score, 'train_loss': losses.avg,
                               'valid_loss': valid_losses, 'epoch': epoch, 'step_sum': epoch*len(train_loader) + step, 'lr': lr})
                else:
                    wandb.log({'valid_rmse': valid_score, 'train_loss': losses.avg,
                               'valid_loss': valid_losses, 'epoch': epoch, 'step_sum': epoch*len(train_loader) + step, 'lr': lr})

            if cfg.save:
                if best_score['score'] > valid_score:
                    torch.save(model.state_dict(), model_name)

                    best_score['score'] = valid_score
                    best_score['epoch'] = epoch
                    best_score['step'] = step
                    print(
                        f"Best score update! valid rmse: {best_score['score']}, epoch: {best_score['epoch']}, step: {best_score['step']}")
                else:
                    print(
                        f"No update. valid rmse: {valid_score}, epoch: {epoch}, step: {step}")
    if scheduler:
        scheduler.step()
    if cfg.mix_p == 0:
        # preds_epoch = np.sum(np.concatenate(preds_all), axis=1)
        preds_epoch = np.concatenate(preds_all)
        labels_epoch = np.concatenate(labels_all)
        train_score = mean_squared_error(labels_epoch, preds_epoch) ** 0.5
        print(f'TRAIN: {train_score}, VALID: {valid_score}')
    else:
        print(f'VALID: {valid_score}')


def preprocess(cfg, train_fold_df, valid_fold_df):
    if 'basic' in cfg.dense_columns or 'all' in cfg.dense_columns:
        basic_columns = ['height', 'width', 'size', 'sqrtsize', 'aspect']
        scale = MinMaxScaler()
        train_fold_df[basic_columns] = pd.DataFrame(scale.fit_transform(
            train_fold_df[basic_columns]), columns=basic_columns)
        valid_fold_df[basic_columns] = pd.DataFrame(scale.transform(
            valid_fold_df[basic_columns]), columns=basic_columns)
    return train_fold_df, valid_fold_df


def result_output(cfg, fold, valid_fold_df, model_name, save_path, device):
    model = pf_model(cfg, pretrained=False)
    model.load_state_dict(torch.load(model_name))
    features_model = model.to(device)

    ds = pf_dataset(cfg, valid_fold_df, 'valid',
                    transforms=get_transforms(cfg, 'valid'))

    data_loader = DataLoader(
        ds,
        batch_size=cfg.valid_bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    result_df = valid_fold_df.copy()

    features_model.eval()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    preds_list = []
    for step, (imgs, dense, _) in pbar:
        imgs = imgs.to(device).float()
        dense = dense.to(device).float()
        with autocast():
            with torch.no_grad():
                preds = features_model(imgs, dense)
        preds_list += [torch.sigmoid(preds).detach().cpu().numpy()]

    preds_class_all = np.concatenate(preds_list)
    preds_all = np.sum(preds_class_all, axis=1)

    result_df = pd.concat([result_df, pd.DataFrame(
        preds_class_all, columns=[f'pred_{i}' for i in range(100)])], axis=1)
    result_df['preds'] = preds_all
    return result_df


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    wandb.login()
    seed_everything(cfg.seed)

    if cfg.save:
        save_path = os.path.join(
            '/'.join(os.getcwd().split('/')[:-6]), f"outputs/{os.getcwd().split('/')[-4]}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    train_df, test_df = load_data(cfg)
    if cfg.save:
        train_df.to_csv(os.path.join(save_path, 'train.csv'))
        test_df.to_csv(os.path.join(save_path, 'test.csv'))
    save_flag = False

    for fold in range(cfg.fold_num):
        if fold not in cfg.use_fold:
            continue

        if cfg.save:
            model_name = os.path.join(
                save_path, f"{cfg.model_arch}_fold_{fold}.pth")

        if len(cfg.use_fold) == 1:
            wandb.init(project=cfg.wandb_project, entity='luka-magic',
                       name=os.getcwd().split('/')[-4], config=cfg)
        else:
            wandb.init(project=cfg.wandb_project, entity='luka-magic',
                       name=os.getcwd().split('/')[-4] + f'_{fold}', config=cfg)

        train_fold_df = train_df[train_df['kfold']
                                 != fold].reset_index(drop=True)
        valid_fold_df = train_df[train_df['kfold']
                                 == fold].reset_index(drop=True)

        train_fold_df, valid_fold_df = preprocess(
            cfg, train_fold_df, valid_fold_df)

        train_loader, valid_loader, _ = prepare_dataloader(
            cfg, train_fold_df, valid_fold_df)

        device = torch.device(cfg.device)

        model = pf_model(cfg, out_dim=10, pretrained=True).to(device)

        scaler = GradScaler()

        if cfg.optimizer == 'AdamW':
            optim = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'RAdam':
            optim = torch.optim.RAdam(
                model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

        if cfg.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim, total_steps=cfg.epoch, max_lr=cfg.lr, pct_start=cfg.pct_start, div_factor=cfg.div_factor, final_div_factor=cfg.final_div_factor)
        elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min)

        if cfg.loss == 'MSELoss':
            criterion = nn.MSELoss()
        elif cfg.loss == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif cfg.loss == 'RMSELoss':
            criterion = RMSELoss()
        elif cfg.loss == 'FOCALLoss':
            criterion = FOCALLoss(gamma=cfg.gamma)
        loss_fn = GradeLabelBCEWithLogits(cfg, criterion)

        best_score = {'score': 100, 'epoch': 0, 'step': 0}

        for epoch in tqdm(range(cfg.epoch), total=cfg.epoch):
            # Train Start
            train_valid_one_epoch(cfg, epoch, model, loss_fn, optim, train_loader,
                                  valid_loader, device, scheduler, scaler, best_score, model_name)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if cfg.save and cfg.result_output:
            if save_flag == False:
                results_df = result_output(cfg, fold, valid_fold_df,
                                           model_name, save_path, device)
                save_flag = True
            else:
                results_df = pd.concat([results_df, result_output(cfg, fold, valid_fold_df,
                                                                  model_name, save_path, device)], axis=0)
    if save_flag:
        results_df.to_csv(os.path.join(save_path, 'result.csv'), index=False)


if __name__ == '__main__':
    main()
