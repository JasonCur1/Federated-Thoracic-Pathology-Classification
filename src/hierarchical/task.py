import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
import pandas as pd
from PIL import Image
import io
import torch.distributed as dist
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

        label = torch.zeros(len(DISEASE_LABELS), dtype=torch.float32)
        for disease in row['label']:
            if disease in LABEL_TO_IDX:
                label[LABEL_TO_IDX[disease]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label
        
def load_data(df_train, df_eval, batch_size=32, is_distributed=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = XrayDataset(df_train, transform=transform)
    eval_dataset = XrayDataset(df_eval, transform=transform)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
        trainLoader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        evalLoader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler, num_workers=4, pin_memory=True)
    else:
        trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        evalLoader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainLoader, evalLoader
    
def load_model(num_diseases=len(DISEASE_LABELS)):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_diseases)
    return model

def train(model, train_loader, epochs, device, is_distributed=False):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    total_loss = []
    for epoch in range(epochs):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
            
        running_loss = 0.0
        # Only show tqdm on rank 0 to avoid console spam
        is_main_process = (not is_distributed) or (dist.get_rank() == 0)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not is_main_process)
        
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        
        # Average the loss across all GPUs
        if is_distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (avg_loss_tensor / dist.get_world_size()).item()

        total_loss.append(avg_loss)
    return total_loss

def test(model, test_loader, device, is_distributed=False):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    all_outputs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    
    if is_distributed:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (avg_loss_tensor / dist.get_world_size()).item()

    y_pred = np.vstack(all_outputs)
    y_true = np.vstack(all_labels)

    global_auc = roc_auc_score(y_true, y_pred, average='macro')
    per_disease_auc = roc_auc_score(y_true, y_pred, average=None)

    metrics = {"global_auc": global_auc}
    for i, disease in enumerate(DISEASE_LABELS):
        metrics[f"auc_{disease}"] = per_disease_auc[i]

    return avg_loss, metrics