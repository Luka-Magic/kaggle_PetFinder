import os
import wandb
import hydra
from omegaconf import DictConfig
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
# from cuml.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

wandb.login()


def load_data(exps):
    compe_path = '/'.join(os.getcwd().split('/')[:-6])
    train_df = pd.read_csv(os.path.join(compe_path, 'data/train.csv'))
    train_preds_df = train_df[['Id', 'Pawpularity']]

    for exp in exps:
        exp_name = f'exp_{str(exp).zfill(4)}'
        path = os.path.join(compe_path, 'outputs', exp_name, 'result.csv')
        df = pd.read_csv(path)

        merge_df = df[['Id', 'preds']].rename(
            columns={'preds': exp_name})
        train_preds_df = pd.merge(train_preds_df, merge_df, on='Id')
    train_preds_df.drop('Id', axis=1, inplace=True)
    X = train_preds_df.drop('Pawpularity', axis=1)
    y = train_preds_df.Pawpularity
    return X, y


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# @hydra.main(config_path='config', config_name='config')
# def wandb_init(cfg: DictConfig):
#     sweep_cfg = dict(cfg.sweep_cfg)
#     sweep_cfg['name'] = os.getcwd().split('/')[-4]
#     count = cfg.count
#     sweep_id = wandb.sweep(
#         sweep_cfg, project='kaggle_PF_sweep', entity='luka-magic')
#     return sweep_id, count


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    # print(cfg)
    sweep_cfg = dict(cfg.sweep_cfg)
    # print(sweep_cfg)
    # sweep_cfg['name'] = os.getcwd().split('/')[-4]
    count = cfg.count
    # print(type(sweep_cfg))
    # print(sweep_cfg)
    sweep_configuration = {
        # "name": os.getcwd().split('/')[-4],
        "metric": {"name": "valid_rmse", "goal": "minimize"},
        "method": "bayes",
        "parameters": {
            "C": {
                "values": [2, 3]
            }
        }
    }
    # print(sweep_configuration)
    # print(sweep_cfg)

    sweep_id = wandb.sweep(sweep_configuration, project='kaggle_PF_sweep')

    # print('OK')

    # def train():
    #     default_cfg = {
    #         'C': 2,
    #         'epsilon': 2,
    #         'gamma': 1
    #     }
    #     wandb.init(config=default_cfg)
    #     wandb_cfg = wandb.config
    #     X, y = load_data(cfg.exps)
    #     clf = SVR(
    #         C=wandb_cfg.C,
    #         epsilon=wandb_cfg.epsilon,
    #         gamma=wandb_cfg.gamma
    #     )
    #     cv = StratifiedKFold(n_splits=cfg.fold_num,
    #                          shuffle=True, random_state=cfg.seed)

    #     score = cross_val_score(clf, X, y, scoring=make_scorer(rmse), cv=cv)

    #     wandb.log({'valid_rmse': score})
    # wandb.agend(sweep_id, train, count=count)

# @hydra.main(config_path='config', config_name='config')
# def train(cfg: DictConfig):
#     default_cfg = {
#         'C': 2,
#         'epsilon': 2,
#         'gamma': 1
#     }
#     wandb.init(config=default_cfg)
#     X, y = load_data(cfg.exps)
#     wandb_cfg = wandb.config

#     clf = SVR(
#         C=wandb_cfg.C,
#         epsilon=wandb_cfg.epsilon,
#         gamma=wandb_cfg.gamma
#     )

#     cv = StratifiedKFold(n_splits=cfg.fold_num,
#                          shuffle=True, random_state=cfg.seed)

#     score = cross_val_score(clf, X, y, scoring=make_scorer(rmse), cv=cv)

#     wandb.log({'valid_rmse': score})


if __name__ == '__main__':
    main()
