import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io.formats.format import DataFrameFormatter
import cv2
from tqdm.notebook import tqdm
import os
import math
import sys
from PIL import Image
import imagehash
import torch
import networkx as nx
from pathlib import Path
import re


def hash_grouping(df, threshold):
    funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]
    hashes = []
    for path in tqdm(df.file_path, total=df.shape[0]):
        image = cv2.imread(path)
        image = Image.fromarray(image)
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))
    hashes = torch.Tensor(np.array(hashes).astype(int))
    sims = np.array([(hashes[i] == hashes).sum(
        dim=1).cpu().numpy()/256 for i in range(hashes.shape[0])])
    duplicates = np.where(sims > threshold)

    g1 = nx.Graph()
    for i, j in tqdm(zip(*duplicates)):
        g1.add_edge(i, j)

    duplicates_groups = list(list(x) for x in nx.connected_components(g1))

    df_dict = {
        "Id": list(),
        "group": list(),
        "index_in_group": list(),
    }

    for group_idx, group in enumerate(duplicates_groups):
        for indx, indx_path in enumerate(group):
            p = Path(df.file_path[indx_path])
            img_id = p.stem.split('_')[0]
            assert len(img_id) == 32

            df_dict["Id"].append(img_id)
            df_dict["group"].append(group_idx)
            df_dict["index_in_group"].append(indx)

    df = pd.merge(df, pd.DataFrame(df_dict), on='Id')
    return df


# def remove_noisy(df, threshold):
#     gap = np.abs(df["Pawpularity"] - df["preds"])
#     df_removed = df[gap > threshold].reset_index(drop=True)
#     df_keep = df[gap <= threshold].reset_index(drop=True)
#     return df_keep, df_removed

def distribution_label(df):
    def f(x):
        for i in range(10):
            if (i+1)*10 >= x > i * 10:
                return i
    df['y'] = df.Pawpularity.apply(f)
    return df


def preprocess(data_path, phase):
    '''
    data_path: path of data
    phase: train/test/result
    '''
    df = pd.read_csv(os.path.join(data_path, f'{phase}.csv'))

    # df['file_path'] = df['Id'].apply(
    #     lambda x: f'/content/drive/MyDrive/kaggle_PetFinder/data/train/{x}.jpg')

    # hash_threshold = 0.9
    # print(f'hash threshold = {hash_threshold}')
    # df = hash_grouping(df, threshold=hash_threshold)
    # print(f"duplicate: {df['index_in_group'].sum()}")

    # remove_threshold = 25
    # print(f'remove threshold = {remove_threshold}')
    # df_keep, df_remove = remove_noisy(
    #     hash_group_df, threshold=remove_threshold)
    # print(f'keep: {df_keep.shape[0]}, remove: {df_remove.shape[0]}')

    df = distribution_label(df)

    # drop_columns = [column for column in df.columns if re.search(
    #     'feature*', column) or column == 'preds']
    # df.drop(columns=drop_columns, inplace=True)

    return df


def load_to_csv(num):
    # この関数はここでしか使わない
    data_path = '/content/drive/MyDrive/kaggle_PetFinder/data'
    processed_df = preprocess(
        data_path, 'train_4')
    processed_df.to_csv(os.path.join(
        data_path, f'train_{num}.csv'), index=False)


if __name__ == '__main__':
    args = sys.argv
    load_to_csv(args[1])
