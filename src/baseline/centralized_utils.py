"""
CS535 - Leveraging Federated Learning for Multi-Label Thoracic Pathology Classification
"""
import sys
import os
sys.path.append(os.path.abspath('..'))
from train import task

import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def get_hospitals_df(data_paths, filenames):
    """
    Concatenate all hospital data for centralized learning.
    """
    df_list = []
    for data_path in data_paths:
        train_files = glob.glob(os.path.join(data_path, filenames))
        df_train = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
        df_list.append(df_train)

    return pd.concat(df_list)

def show_xray(image, label):
    """
    Plot an X-Ray.
    """
    plt.imshow(image)
    indices = [i for i, num in enumerate(label) if num == 1]
    class_name = [task.DISEASE_LABELS[i] for i in indices]
    if not class_name:
        class_name = ['No Finding']
    plt.title(class_name)
    plt.axis('off')