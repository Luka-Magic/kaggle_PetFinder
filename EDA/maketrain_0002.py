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


def img_hash(series):
    funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]
    image = Image.open(series.file)
    return np.array([f(image).hash for f in funcs]).reshape(256).astype(np.uint8)


def make_hash_df(file_path_ser: pd.Series):
    '''
    file_path => hash_df
    '''
    hash_df = file_path_ser.apply(lambda x: img_hash(x))
    hash_df.columns = [f'hash_{i}' for i in range(hash_df.shape[1])]
    return hash_df


# def delete_sim_img(hash_df, theshold=0.8):
#     train_hashes = hash_df.values
#     sims = np.array([(train_hashes[i] == train_hashes).sum(axis=1)
#                     for i in range(train_hashes.shape[0])]) / 256
#     indices1, indices2 = np.where(sims > theshold)
#     indices1, indices2 = indices1[indices1 <
#                                   indices2], indices2[indices1 < indices2]
#     hash_df['sims'] = 0
#     hash_df.loc[indices1, 'sims'] = 1
#     return hash_df


def preprocess(data_path, phase):
    '''
    data_path: path of data
    phase: train/test
    '''
    df = pd.read_csv(os.path.join(data_path, f'{phase}.csv'))

    df['file_path'] = df['Id'].apply(
        lambda x: os.path.join(data_path, f'{phase}/{x}.jpg'))

    h_list = []
    w_list = []
    size_list = []
    sqrtsize_list = []
    aspect_list = []

    hash_df = make_hash_df(df['file_path'])

    add_df = pd.DataFrame()

    for i in tqdm(range(df.shape[0]), total=df.shape[0]):
        # 画像の読み込み
        img_path = df.loc[i, 'file_path']
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 基本情報
        h, w, _ = img_rgb.shape
        size = h * w
        sqrtsize = int(math.sqrt(size))
        aspect = w / h

        h_list.append(h)
        w_list.append(w)
        size_list.append(size)
        sqrtsize_list.append(sqrtsize)
        aspect_list.append(aspect)

    add_df['height'] = h_list
    add_df['width'] = w_list
    add_df['size'] = size_list
    add_df['sqrtsize'] = sqrtsize_list
    add_df['aspect'] = aspect_list

    df = pd.concat([df, add_df, hash_df], axis=1)
    df = df.drop('file_path', axis=1)
    return df


def load_to_csv(num):
    # この関数はここでしか使わない
    data_path = '/content/drive/MyDrive/kaggle_PetFinder/data'
    processed_df = preprocess(
        data_path, 'train')
    processed_df.to_csv(os.path.join(data_path, f'train_{num}.csv'))


if __name__ == '__main__':
    args = sys.argv
    load_to_csv(args[0])
