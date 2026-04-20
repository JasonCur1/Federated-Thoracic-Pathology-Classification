"""
CS535 - Leveraging Federated Learning for Multi-Label Thoracic Pathology Classification
"""
import sys
import os
sys.path.append(os.path.abspath('..'))
from train import task

from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, average_precision_score
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

def train(model, train_loader, device, epochs, pos_weight=None, learning_rate=0.001):
    """
    Training loop that outputs loss, training time, and peak memory usage.
    """
    if pos_weight is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_train_loss = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    training_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0.0
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
        
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        epoch_time = time.time() - epoch_start
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1} | average loss: {avg_train_loss:.4f} | time: {epoch_time:.2f}s')
        total_train_loss.append(avg_train_loss)
    training_time = time.time() - training_start

    if torch.cuda.is_available():
        peak_memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    else:
        peak_memory_usage = None

    return total_train_loss, training_time, peak_memory_usage

def evaluate(model, test_loader, device, threshold=0.5):
    """
    Evaluates model using Jaccard, F1, Recall, Precision, ROC-AUC, Per-class ROC-AUC.
    """
    actual = []
    pred = []
    probs = []
    
    model.eval()
    pbar = tqdm(test_loader, desc=f"Evaluating")
    for (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)

            probs = torch.sigmoid(logits).cpu().numpy()

            y_pred = (probs > threshold).int().cpu().numpy()
            y_actual = labels.cpu().numpy()
    
            actual.append(y_actual)
            pred.append(y_pred)
            probs.append(probs)
    actual = np.vstack(actual)
    pred = np.vstack(pred)
    probs = np.vstack(probs)
            
    jaccard = jaccard_score(actual, pred, average='samples', zero_division=0)
    per_class_accuracy = (pred == actual).mean(axis=0) * 100
    per_class_accuracy_dict = {
        disease: acc for disease, acc in zip(task.DISEASE_LABELS, per_class_accuracy)
    }
    f1 = f1_score(actual, pred, average='macro', zero_division=0)
    per_class_f1 = f1_score(actual, pred, average='None', zero_division=0)
    per_class_f1_dict = {
        disease: f1 for disease, f1 in zip(task.DISEASE_LABELS, per_class_f1)
    }
    recall = recall_score(actual, pred, average='macro', zero_division=0)
    precision = precision_score(actual, pred, average='macro', zero_division=0)
    fpr, tpr, thresholds = roc_curve(actual, probs)
    roc_auc = roc_auc_score(actual, probs, average='macro')
    per_class_auc = roc_auc_score(actual, probs, average=None)
    per_class_auc_dict = {
        disease: auc for disease, auc in zip(task.DISEASE_LABELS, per_class_auc)
    }
    aucpr = average_precision_score(actual, probs, average='macro', zero_division=0)
    per_class_aucpr = average_precision_score(actual, probs, average=None, zero_division=0)
    per_class_aucpr_dict = {
        disease: aucpr for disease, aucpr in zip(task.DISEASE_LABELS, per_class_aucpr)
    }
    
    results = {
        'Jaccard Similarity': jaccard,
        'per_class_acc': per_class_accuracy_dict,
        'F1_score': f1, 
        'per_class_f1': per_class_f1_dict,
        'Recall': recall,
        'Precision': precision,
        'roc_curve': [fpr, tpr, thresholds],
        'roc_auc': roc_auc,
        'per_class_auc': per_class_auc_dict,
        'aucpr_curve': aucpr,
        'per_class_aucpr': per_class_aucpr_dict
    }
    return results

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