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


def delete_sim_img(hash_df, theshold=0.8):
    train_hashes = hash_df.values
    sims = np.array([(train_hashes[i] == train_hashes).sum(axis=1)
                    for i in range(train_hashes.shape[0])]) / 256
    indices1, indices2 = np.where(sims > theshold)
    indices1, indices2 = indices1[indices1 <
                                  indices2], indices2[indices1 < indices2]
    hash_df['sims'] = 0
    hash_df.loc[indices1, 'sims'] = 1
    return hash_df


def preprocess(data_path, phase):
    '''
    data_path: path of data
    phase: train/test
    '''
    df = pd.read_csv(os.path.join(data_path, f'{phase}.csv'))

    df['file_path'] = df['Id'].apply(
        lambda x: os.path.join(data_path, f'{phase}/{x}.jpg'))

    hash_df = make_hash_df(df['file_path'])

    hash_df = delete_sim_img(hash_df)

    return hash_df


def load_to_csv(num):
    # この関数はここでしか使わない
    data_path = '/content/drive/MyDrive/kaggle_PetFinder/data'
    processed_df = preprocess(
        data_path, 'train')
    processed_df.to_csv(os.path.join(data_path, f'train_{num}.csv'))


if __name__ == '__main__':
    args = sys.argv
    load_to_csv(args[0])
