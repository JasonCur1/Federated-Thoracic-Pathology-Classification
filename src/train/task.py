import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
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
        
def load_data(df_train, df_eval, batch_size=32):
    # DenseNet requires 224x224 input and Imagenet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # These mean and std values are standard for RGB. Idk why man
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = XrayDataset(df_train, transform=transform)
    eval_dataset = XrayDataset(df_eval, transform=transform)

    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    evalLoader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return trainLoader, evalLoader
    
def load_model(num_diseases=len(DISEASE_LABELS)):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features

    # Apparently BCEWithLogitsLoss is better than sigmoid. It combines Sigmoid and BCELoss
    model.classifier = nn.Linear(num_features, num_diseases) # Last layer size = num diseases
    return model

def train(model, train_loader, epochs, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()

    print(f"DEBUG: Starting training on {device}...")
    total_loss = 0.0
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in enumerate(pbar):
            if i == 0: print("DEBUG: Received first batch from Loader...")
            
            images, labels = images.to(device), labels.to(device)
            if i == 0: print(f"DEBUG: Moved first batch to {device}...")
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss=loss.item()

    return total_loss / len(train_loader)

def test(model, test_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.eval()
    loss = 0.0

    with torch.no_grad(): # disable weight updates. Don't need to during eval
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()

            # TODO: Calculate ROC-AUC here

    loss /= len(test_loader.dataset)
    return loss, {"accuracy": 0.0} # TODO: Placeholder for other metrics




