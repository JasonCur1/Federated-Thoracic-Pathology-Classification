"""
CS535 - Leveraging Federated Learning for Multi-Label Thoracic Pathology Classification
"""
import sys
import os
sys.path.append(os.path.abspath('..'))
from train import task

from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, average_precision_score, RocCurveDisplay
import matplotlib.pyplot as plt
from torchvision import models
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import glob
import os


class DenseNet121(torch.nn.Module):
    """
    DenseNet121 model with `n_classes` outputs.
    """
    def __init__(self, n_classes, device='cpu'):
        super().__init__()
        self.device = device
        self.densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.densenet121.classifier = torch.nn.Linear(self.densenet121.classifier.in_features, n_classes)
        self.to(self.device)
    def forward(self, X):
        return self.densenet121(X)

def calculate_pos_weights(y):
    """
    Calculate positive weights for imbalanced multi-label classification.
    """
    total_samples = y.shape[0]
    positive_counts = y.sum(dim=0)
    negative_counts = total_samples - positive_counts

    positive_weights = negative_counts / positive_counts

    return np.minimum(positive_weights, 50)

def calculate_weights(y):
    """
    Calculate positive and negative weights for imbalanced multi-label classification.
    """
    total_samples = y.shape[0]

    positive_counts = y.sum(dim=0)
    negative_counts = total_samples - positive_counts

    pos_weights = (positive_counts + negative_counts) / positive_counts
    neg_weights = (positive_counts + negative_counts) / negative_counts

    return pos_weights.float(), neg_weights.float()

def get_hospitals_df(data_paths, filenames):
    """
    Concatenate all hospital data for centralized learning.
    """
    df_list = []
    for data_path in data_paths:
        files = glob.glob(os.path.join(data_path, filenames))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df_list.append(df)

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

def plot_loss(loss_train):
    """
    Plot loss curve of train set.
    """
    plt.figure(figsize=(5,3))
    plt.plot(range(1, len(loss_train)+1), loss_train, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()