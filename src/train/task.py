import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    recall_score, 
    precision_score, 
    jaccard_score,
    precision_recall_curve
)
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm

DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(DISEASE_LABELS)}

class XrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_bytes = row['image']
        image = Image.open(io.BytesIO(image_bytes['bytes'])).convert('RGB')

        label = torch.zeros(len(DISEASE_LABELS), dtype=torch.float32) # New multi hot encoded list of labels
        for disease in row['label']:
            if disease in LABEL_TO_IDX:
                label[LABEL_TO_IDX[disease]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label
        
def load_data(df_train, df_eval, df_test, batch_size=32):
    # DenseNet requires 224x224 input and Imagenet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # These mean and std values are standard for RGB. Idk why man
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainLoader = DataLoader(XrayDataset(df_train, transform=transform), batch_size=batch_size, shuffle=True)
    evalLoader = DataLoader(XrayDataset(df_eval, transform=transform), batch_size=batch_size, shuffle=False)
    testLoader = DataLoader(XrayDataset(df_test, transform=transform), batch_size=batch_size, shuffle=False)

    return trainLoader, evalLoader, testLoader
    
def load_model(num_diseases=len(DISEASE_LABELS)):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features

    # Apparently BCEWithLogitsLoss is better than sigmoid. It combines Sigmoid and BCELoss
    model.classifier = nn.Linear(num_features, num_diseases) # Last layer size = num diseases
    return model

def compute_pos_weights(df_train, cap=50.0):
    """
    Calculates positive class weights for BCEWithLogitsLoss.
    pos_weight = (Total Negatives) / (Total Positives)
    """
    pos_counts = np.zeros(len(DISEASE_LABELS))
    total_samples = len(df_train)
    
    # Count occurrences of each disease in the training set
    for labels in df_train['label']:
        for disease in labels:
            if disease in LABEL_TO_IDX:
                pos_counts[LABEL_TO_IDX[disease]] += 1
                
    # Prevent division by zero if a client is missing a disease entirely
    pos_counts = np.maximum(pos_counts, 1)
    
    neg_counts = total_samples - pos_counts
    weights = neg_counts / pos_counts
    weights = np.clip(weights, a_min=0, a_max=cap)
    
    print(f"DEBUG: Calculated Local Positive Weights: {np.round(weights, 2)}")
    return torch.tensor(weights, dtype=torch.float32)

def train(model, train_loader, epochs, device, pos_weight=None):
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()

    print(f"DEBUG: Starting training on {device}...")
    total_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in enumerate(pbar):
            if i == 0: print("DEBUG: Received first batch from Loader...")
            
            images, labels = images.to(device), labels.to(device)
            if i == 0: print(f"DEBUG: Moved first batch to {device}...")
            
            optimizer.zero_grad()
            outputs = model(images)

            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
        avg_loss = running_loss / len(train_loader)
        total_loss.append(avg_loss)
    return total_loss

# Helper to run inference without calculating metrics yet
def get_predictions(model, loader, device):
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.eval()

    all_outputs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad(): 
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    y_pred_probs = np.vstack(all_outputs)
    y_true = np.vstack(all_labels)
    
    return y_true, y_pred_probs, avg_loss

# Find best F1 threshold for each disease
def find_optimal_thresholds(y_true, y_pred_probs):
    thresholds = []
    for i in range(y_true.shape[1]):
        precision, recall, thresh = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
        
        # Calculate F1 score for all threshold points
        numerator = 2 * recall * precision
        denominator = recall + precision
        f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))
        
        best_idx = np.argmax(f1_scores)
        best_thresh = thresh[best_idx] if best_idx < len(thresh) else 0.5
        thresholds.append(best_thresh)
        
    return np.array(thresholds)

def test(model, eval_loader, test_loader, device):
    # run inference on eval set to find the optimal threshold for each disease
    y_true_eval, y_pred_probs_eval, _ = get_predictions(model, eval_loader, device)
    optimal_thresholds = find_optimal_thresholds(y_true_eval, y_pred_probs_eval)
    
    # run inference on the actual unseen test set
    y_true_test, y_pred_probs_test, test_loss = get_predictions(model, test_loader, device)
    
    # apply the tuned thresholds
    y_pred_binary = (y_pred_probs_test > optimal_thresholds).astype(int)

    metrics = {
        "global_auc": float(roc_auc_score(y_true_test, y_pred_probs_test, average='macro')),
        "global_auprc": float(average_precision_score(y_true_test, y_pred_probs_test, average='macro')),
        "global_f1": float(f1_score(y_true_test, y_pred_binary, average='macro', zero_division=0)),
        "global_recall": float(recall_score(y_true_test, y_pred_binary, average='macro', zero_division=0)),
        "global_precision": float(precision_score(y_true_test, y_pred_binary, average='macro', zero_division=0)),
        "global_jaccard": float(jaccard_score(y_true_test, y_pred_binary, average='macro', zero_division=0))
    }

    per_disease_auc = roc_auc_score(y_true_test, y_pred_probs_test, average=None)
    for i, disease in enumerate(DISEASE_LABELS):
        metrics[f"auc_{disease}"] = float(per_disease_auc[i])

    return test_loss, metrics