import sys
import os
sys.path.append(os.path.abspath('..'))

import io
import glob
from PIL import Image
from train import task
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


IDX_TO_LABEL = {idx: label for idx, label in enumerate(task.DISEASE_LABELS)}

class DenseNet121(torch.nn.Module):
    """
    DenseNet121 model with `n_classes` outputs.
    """
    def __init__(self, n_classes, device='cpu'):
        super().__init__()
        self.device = device
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, n_classes)
        self.to(self.device)

    def forward(self, X):
        return self.densenet(X)

def get_train_data():
    hospitals = ['hospital_a', 'hospital_b', 'hospital_c']
    hospital_data = {}
    
    patterns = {
        'train': 'train-*.parquet',
    }
    
    for hospital in hospitals:
        data_path = os.path.join('../../data/', hospital)
        hospital_data[hospital] = {}
        
        for split, pattern in patterns.items():
            files = glob.glob(os.path.join(data_path, pattern))
            hospital_data[hospital][split] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return hospital_data

def get_label_sample(df, label):
    """
    Get random sample (image, label) from `df` with class `label`.
    """
    df_single_label = df[df['label'].apply(lambda x: len(x) == 1)]
    df_filtered = df_single_label[df_single_label['label'].apply(lambda x: x[0] == label)]
    
    random_sample = df_filtered.sample(1).iloc[0]
    image_bytes = random_sample['image']
    image = Image.open(io.BytesIO(image_bytes['bytes'])).convert('RGB')
    label = random_sample['label']
    return image, label

def get_multi_hot_label(label):
    """
    Return Multi-Hot Encoding of `label`.  
    """
    LABEL_TO_IDX = {label: idx for idx, label in enumerate(task.DISEASE_LABELS)}
    multi_hot_label = torch.zeros(len(task.DISEASE_LABELS), dtype=torch.float32)
    for disease in label:
        if disease in LABEL_TO_IDX:
            multi_hot_label[LABEL_TO_IDX[disease]] = 1.0
    return multi_hot_label

def compute_heatmap(model, target_layer, image_pre):
    # Identify Target Layer
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward Pass
    logits, pred = forward_pass(model, image_pre)
    # Backward Pass
    backward_pass(model, logits, pred)
    # Heatmap
    weights = torch.mean(gradients[0], dim=[2, 3], keepdim=True)

    heatmap = torch.sum(weights * activations[0], dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    heatmap /= np.max(heatmap)

    return heatmap, pred

def forward_pass(model, image_pre):
    model.eval()
    logits = model(image_pre.unsqueeze(0))
    pred = torch.argmax(logits, dim=1).item()
    return logits, pred

def backward_pass(model, logits, pred):
    model.zero_grad()
    logits[0, pred].backward()
    return

def plot_grad_cam_grid(images, heatmaps, true_classes, pred_classes):
    n = len(images)
    cols = 5
    rows = 3
    fig, ax = plt.subplots(rows, cols, figsize=(14, 9))
    ax = ax.flatten()

    for i in range(n):
        raw_image = np.array(images[i]).astype(np.uint8)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        
        heatmap = cv2.resize(heatmaps[i], (raw_image.shape[1], raw_image.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        superimposed_img = cv2.addWeighted(raw_image, 0.6, heatmap, 0.4, 0)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        ax[i].imshow(superimposed_img)
        ax[i].set_title(f'T: {true_classes[i][0]} | P: {pred_classes[i]}')
    for i in range(len(ax)):
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()
    