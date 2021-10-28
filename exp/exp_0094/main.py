# Python Libraries
from utils.loss import FOCALLoss, RMSELoss
from utils.mixaug import mixup, cutmix
from utils.make_columns import make_columns, len_columns
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
    # elif cfg.fold == 'StratifiedGroupKFold':
    #     folds = StratifiedGroupKFold(n_splits=cfg.fold_num, shuffle=True, random_state=cfg.seed).split(
    #         X=np.arange(train_df.shape[0]), y=train_df.Pawpularity.values, groups=train_df.cluster.values)
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

    augs = [getattr(albumentations, name)(**kwargs)
            for name, kwargs in aug.items()]
    augs.append(ToTensorV2(p=1.))
    return albumentations.Compose(augs)


class pf_model(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            cfg.model_arch, pretrained=pretrained, in_chans=3)

        if re.search(r'vit*', cfg.model_arch) or re.search(r'swin*', cfg.model_arch):
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, cfg.features_num)
        elif re.search(r'tf*', cfg.model_arch):
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(
                self.n_features, cfg.features_num)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(cfg.features_num +
                             len_columns(cfg.dense_columns), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, input, dense):
        features = self.model(input)
        features = self.dropout(features)
        x = torch.cat([features, dense], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x, features


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


def train_one_epoch(cfg, epoch, model, loss_fn, optimizer, data_loader, device, scheduler, scaler):

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    model.train()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    preds_all = []
    labels_all = []

    for step, (imgs, dense, labels) in pbar:
        imgs = imgs.to(device).float()
        dense = dense.to(device).float()
        labels = labels.to(device).float().view(-1, 1)

        if cfg.loss == 'BCEWithLogitsLoss' or cfg.loss == 'FOCALLoss':
            labels /= 100

        with autocast():
            mix_p = np.random.rand()
            mix_list = list(range(cfg.init_nomix_epoch,
                            cfg.epoch-cfg.last_nomix_epoch))
            if (mix_p < cfg.mix_p) and (epoch in mix_list):
                imgs, labels = mixup(imgs, labels, 1.)
                preds, _ = model(imgs, dense)
                loss = loss_fn(
                    preds, labels[0]) * labels[2] + loss_fn(preds, labels[1]) * (1. - labels[2])
            else:
                preds, _ = model(imgs, dense)
                loss = loss_fn(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if cfg.mix_p == 0:
            if cfg.loss == 'BCEWithLogitsLoss' or cfg.loss == 'FOCALLoss':
                preds_all += [np.clip(torch.sigmoid(
                    preds).detach().cpu().numpy() * 100, 1, 100)]
                labels_all += [labels.detach().cpu().numpy() * 100]
            elif cfg.loss == 'MSELoss' or cfg.loss == 'RMSELoss':
                preds_all += [np.clip(preds.detach().cpu().numpy(), 1, 100)]
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

    for step, (imgs, dense, labels) in pbar:
        imgs = imgs.to(device).float()
        dense = dense.to(device).float()
        labels = labels.to(device).float().view(-1, 1)

        if cfg.loss == 'BCEWithLogitsLoss' or cfg.loss == 'FOCALLoss':
            labels /= 100

        with autocast():
            preds, _ = model(imgs, dense)

        loss = loss_fn(preds, labels)

        if cfg.loss == 'BCEWithLogitsLoss' or cfg.loss == 'FOCALLoss':
            preds = np.clip(torch.sigmoid(
                preds).detach().cpu().numpy() * 100, 1, 100)
            labels = labels.detach().cpu().numpy() * 100
        elif cfg.loss == 'MSELoss' or cfg.loss == 'RMSELoss':
            preds = np.clip(preds.detach().cpu().numpy(), 1, 100)
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

    features_list = np.array([])
    preds_list = np.array([])
    for step, (imgs, dense, _) in pbar:
        imgs = imgs.to(device).float()
        dense = dense.to(device).float()
        with autocast():
            with torch.no_grad():
                preds, features = features_model(imgs, dense)
        if step == 0:
            features_list = features.detach().cpu().numpy()
            if cfg.loss == 'BCEWithLogitsLoss':
                preds_list = np.clip(torch.sigmoid(
                    preds).detach().cpu().numpy() * 100, 1, 100)
            elif cfg.loss == 'MSELoss':
                preds_list = np.clip(preds.detach().cpu().numpy(), 1, 100)
        else:
            features_list = np.concatenate(
                [features_list, features.detach().cpu().numpy()], axis=0)
            if cfg.loss == 'BCEWithLogitsLoss':
                preds_list = np.concatenate([preds_list, np.clip(torch.sigmoid(
                    preds).detach().cpu().numpy() * 100, 1, 100)], axis=0)
            elif cfg.loss == 'MSELoss':
                preds_list = np.concatenate([preds_list, np.clip(
                    preds.detach().cpu().numpy(), 1, 100)], axis=0)

    result_df = pd.concat([result_df, pd.DataFrame(features_list, columns=[
                          f'feature_{i}' for i in range(cfg.features_num)]), pd.DataFrame(preds_list, columns=['preds'])], axis=1)
    return result_df


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    wandb.login()
    seed_everything(cfg.seed)

    train_df, test_df = load_data(cfg)
    # if cfg.save:
    # save_path = os.path.join(
    #     '/'.join(os.getcwd().split('/')[:-6]), f"outputs/{os.getcwd().split('/')[-4]}")
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    #     train_df.to_csv(os.path.join(save_path, 'train.csv'))
    #     test_df.to_csv(os.path.join(save_path, 'test.csv'))
    # save_flag = False

    for fold in range(cfg.fold_num):
        if fold not in cfg.use_fold:
            continue
        sweep_config = {
            'name': 'exp_0094',
            'method': 'bayes',
            'metric': {
                'name': 'best_valid_rmse',
                'goal': 'minimize'
            },
            'parameters': {
                'lr': {
                    'min': 5e-6,
                    'max': 1e-3
                }
            }
        }
        sweep_id = wandb.sweep(sweep_config, project=cfg.wandb_project,
                               entity='luka-magic')

        def train():
            global cfg
            run = wandb.init(config=cfg)
            cfg = wandb.config
            train_fold_df = train_df[train_df['kfold']
                                     != fold].reset_index(drop=True)
            valid_fold_df = train_df[train_df['kfold']
                                     == fold].reset_index(drop=True)

            train_fold_df, valid_fold_df = preprocess(
                cfg, train_fold_df, valid_fold_df)

            valid_rmse = {}

            train_loader, valid_loader, _ = prepare_dataloader(
                cfg, train_fold_df, valid_fold_df)

            device = torch.device(cfg.device)

            model = pf_model(cfg, pretrained=True).to(device)

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
                loss_fn = nn.MSELoss()
            elif cfg.loss == 'BCEWithLogitsLoss':
                loss_fn = nn.BCEWithLogitsLoss()
            elif cfg.loss == 'RMSELoss':
                loss_fn = RMSELoss()
            elif cfg.loss == 'FOCALLoss':
                loss_fn = FOCALLoss(gamma=cfg.gamma)

            best_score = {'score': 100, 'epoch': 0}

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
                # if cfg.mix_p == 0:
                #     wandb.log({'train_rmse': train_score_epoch, 'train_loss': train_loss_epoch,
                #             'valid_rmse': valid_score_epoch, 'valid_loss': valid_loss_epoch,
                #             'epoch': epoch, 'lr': lr})
                # else:
                #     wandb.log({'valid_rmse': valid_score_epoch, 'valid_loss': valid_loss_epoch,
                #             'epoch': epoch, 'lr': lr})

                # if cfg.save:
                #     model_name = os.path.join(
                #         save_path, f"{cfg.model_arch}_fold_{fold}.pth")
                #     if best_score['score'] > valid_score_epoch:
                #         torch.save(model.state_dict(), model_name)

                #         best_score['score'] = valid_score_epoch
                #         best_score['epoch'] = epoch
                #         print(
                #             f"Best score update! valid rmse: {best_score['score']}, epoch: {best_score['epoch']}")
                #     else:
                #         print(
                #             f"No update. best valid rmse: {best_score['score']}, epoch: {best_score['epoch']}")

            # print Score

            valid_rmse_sorted = sorted(valid_rmse.items(), key=lambda x: x[1])
            print('='*40)
            print(f'Fold {fold}')
            for i, (epoch, rmse) in enumerate(valid_rmse_sorted):
                if i == 0:
                    best_score_epoch = epoch
                    best_valid_rmse = rmse
                print(f'No.{i+1}: {rmse:.5f} (epoch{epoch})')
            print('='*40)

            del model
            gc.collect()
            torch.cuda.empty_cache()

    #         if cfg.save and cfg.result_output:
    #             if save_flag == False:
    #                 results_df = result_output(cfg, fold, valid_fold_df,
    #                                         model_name, save_path, device)
    #                 save_flag = True
    #             else:
    #                 results_df = pd.concat([results_df, result_output(cfg, fold, valid_fold_df,
    #                                                                 model_name, save_path, device)], axis=0)
            wandb.log({'best_valid_rmse': best_valid_rmse,
                      'best_score_epoch': best_score_epoch})

        wandb.agent(sweep_id, train, count=10)
        wandb.finish()
    # if save_flag:
    #     results_df.to_csv(os.path.join(save_path, 'result.csv'), index=False)


if __name__ == '__main__':
    main()
