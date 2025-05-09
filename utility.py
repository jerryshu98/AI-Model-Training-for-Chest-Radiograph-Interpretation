import sys
import h5py
import io
import tensorflow as tf
import os
import itertools
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from typing import List
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import subprocess
import timm
import pickle
from tqdm import tqdm, trange

# GPU Setup
# --------------------------------------------------
def setup_gpu():
    """Setup GPU for TensorFlow and manage memory growth"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"Physical GPUs: {gpus}")
        except RuntimeError as e:
            print(e)

# Data Processing Functions
# --------------------------------------------------

def str_to_array(s: str) -> np.ndarray:
    """ Convert string representation of array and shape back to NumPy array """
    data_str, shape_str = s.split(';')
    data = np.array([float(i) for i in data_str.split(',')])
    shape = tuple(map(int, shape_str.strip('()').split(',')))
    return data.reshape(shape)


def preprocess_image(jpg_bytes: bytes) -> np.ndarray:
    """ Preprocess image by resizing and converting to NumPy array """
    image = tf.io.decode_jpeg(jpg_bytes)
    image = tf.image.resize(image, [224, 224])
    return image.numpy()


def get_data_dict_part(df_part: pd.DataFrame) -> List[dict]:
    """ Generate data dictionary for a dataframe part """
    data_dict = []
    for i in tqdm(range(len(df_part)), desc="Processing part"):
        row = df_part.iloc[i]
        data_dict.append({
            'img': row['image'],
            'paths': row['image'],
            'study_id': row['study_id'],
            'dicom': row['dicom'],
        })
    return data_dict


def get_data_dict(df: pd.DataFrame, num_cores: int = 2) -> List[dict]:
    """ Process data using multiprocessing """
    parts = np.array_split(df, num_cores)
    func = partial(get_data_dict_part)
    with ProcessPoolExecutor(num_cores) as executor:
        data_dicts = executor.map(func, parts)
    return list(itertools.chain(*data_dicts))


def split_data(df: pd.DataFrame, group_column: str, n_splits: int) -> tuple:
    """ Split data based on group column for cross-validation """
    frac = 1 / n_splits
    val_idx = df[group_column].drop_duplicates().sample(frac=frac)
    df_temp = df.set_index(group_column)
    df_val = df_temp.loc[val_idx].reset_index()
    df_train = df_temp.drop(index=val_idx).reset_index()
    return df_train, df_val


def load_data_from_pickle(file_path: str) -> pd.DataFrame:
    """ Load data from pickle file """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    if 'study' in data and 'image' in data and 'dicom' in data:
        return pd.DataFrame({'study_id': data['study'], 'image': data['image'], 'dicom': data['dicom']})
    else:
        raise ValueError("Missing keys in pickle file")