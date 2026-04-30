import sys
import baseline.config as config
sys.modules["config"] = config

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from baseline.config import Config, CFG
from baseline.model import ChestXrayClassifier
from baseline.datamodule import ChestXrayDataModule
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, RocCurveDisplay


def load_data(cfg):
    dm = ChestXrayDataModule(cfg=cfg)
    dm.prepare_data()
    dm.setup()
    return dm

def load_model(datamodule, checkpt_path):
    torch.serialization.add_safe_globals([Config])
    
    model = ChestXrayClassifier.load_from_checkpoint(
        checkpt_path,
        pos_weight=datamodule.pos_weight,
        cfg=CFG,
        weights_only=False
    )
    return model

def extract_logits_and_labels(model, dataloader, device=None):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting logits"):
            images = images.to(device)

            logits = model(images)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

def get_model_outputs(data_path, checkpoint_path, device):
    print('Loading datamodule...')
    cfg = Config(data_root_path=data_path)
    dm = load_data(cfg)
    print('Datamodule loaded.')

    print('\nLoading Model...')
    model = load_model(dm, checkpoint_path)
    model = model.to(device)
    print('Model loaded.')

    print('\nCreating member/non-members...')
    member_loader = dm.train_dataloader()
    val_ds = dm.val_dataloader().dataset
    test_ds = dm.test_dataloader().dataset
    non_member_ds = ConcatDataset([val_ds, test_ds])
    non_member_loader = DataLoader(non_member_ds, batch_size=dm.cfg.batch_size, shuffle=False)
    
    print('Calculating model outputs...')
    member_logits, member_y = extract_logits_and_labels(model, member_loader, device)
    non_logits, non_y = extract_logits_and_labels(model, non_member_loader, device)
    
    return member_logits, member_y, non_logits, non_y

def create_attack_dataset(member_logits, member_y, non_logits, non_y, score_fn=None):
    if score_fn is None:
        X = np.concatenate([member_logits, non_logits])
        y = np.concatenate([np.ones(len(member_y)), np.zeros(len(non_y))])
        return X, y
    
    member_scores = score_fn(member_logits, member_y).detach().cpu().numpy()
    non_scores = score_fn(non_logits, non_y).detach().cpu().numpy()

    X = np.concatenate([member_scores, non_scores]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(member_y)), np.zeros(len(non_y))])
    return X, y

def build_mia_attack_datasets(member_logits, member_y, non_logits, non_y):
    score_fns = {
        'Logits': None,
        'Loss': loss_score,
        'Confidence': confidence_score,
        'Entropy': entropy_score
    }
    
    datasets = {}
    
    for name, score_fn in score_fns.items():
        X, y = create_attack_dataset(member_logits, member_y, non_logits, non_y, score_fn=score_fn)
        datasets[name] = (X, y)
    
    X_all = np.concatenate(
        [datasets['Logits'][0],
         datasets['Loss'][0],
         datasets['Confidence'][0],
         datasets['Entropy'][0]],
        axis=1
    )
    
    y_all = datasets['Logits'][1] 
    datasets['All'] = (X_all, y_all)
    return datasets

# low loss signals it probably saw the sample before (training sample)
def loss_score(logits, y):
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(logits, y)
    return loss.mean(dim=1)

# high confidence measures how sure the model is about its predictions, so high signals its a train sample
def confidence_score(logits, y=None):
    probs = torch.sigmoid(logits)
    return probs.mean(dim=1) # or probs.max(dim=1)

# low entropy ~= less uncertainty (train sample likely), while high entropy ~= more uncertainty/flatter distribution
def entropy_score(logits, y=None, epsilon=1e-12):
    p = torch.sigmoid(logits)
    entropy = -(p * torch.log(p + epsilon) +
            (1 - p) * torch.log(1 - p + epsilon))
    return entropy.mean(dim=1)

def tpr_at_low_fpr(fpr, tpr, target_fpr):
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx]
    
def evaluation_metrics(actual, prob):
    fpr, tpr, thresholds = roc_curve(actual, prob)
    auc = roc_auc_score(actual, prob)

    j = tpr - fpr
    best_idx = np.argmax(j)
    best_threshold = thresholds[best_idx]
    
    pred = (prob >= 0.5).astype(int) # use 0.5 for attack model (since output = probs)
    accuracy = balanced_accuracy_score(actual, pred)

    target_fpr = 0.1 # low FPR for metric: TPR at low FPR
    tpr_10 = tpr_at_low_fpr(fpr, tpr, target_fpr)
    results = {
        'auc': auc,
        'balanced_accuracy': accuracy, # at 0.5
        'tpr_at_0.1_fpr': tpr_10,
        'best_threshold': best_threshold,
        'best_tpr': tpr[best_idx],
        'best_fpr': fpr[best_idx],
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }
    return results

def train_attack_model(X_train, y_train, model, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    clf = model(**model_kwargs)
    clf.fit(X_train, y_train)
    
    return clf

def evaluate_attack_model(clf, X_test, y_test):
    y_prob = clf.predict_proba(X_test)[:, 1]
    return evaluation_metrics(y_test, y_prob)

def plot_attack_roc(results_by_type, model_name):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (name, res) in zip(axes, results_by_type.items()):

        RocCurveDisplay(
            fpr=res['shadow']['fpr'],
            tpr=res['shadow']['tpr'],
            roc_auc=res['shadow']['auc'],
            name='Shadow'
        ).plot(ax=ax)

        RocCurveDisplay(
            fpr=res['victim']['fpr'],
            tpr=res['victim']['tpr'],
            roc_auc=res['victim']['auc'],
            name='Victim'
        ).plot(ax=ax)

        ax.plot([0, 1], [0, 1], '--', color='black')
        ax.set_title(name)
    fig.suptitle(model_name, fontsize=16)

    plt.tight_layout()
    plt.show()

def threshold_mia(member_logits, member_y, non_logits, non_y, score_fn, flip_sign=False):
    member_scores = score_fn(member_logits, member_y).detach().cpu().numpy()
    non_scores = score_fn(non_logits, non_y).detach().cpu().numpy()
    
    scores = np.concatenate([member_scores, non_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_scores))])

    if flip_sign: # loss and entropy
        scores = -scores
    # ROC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    # Youden’s J statistic (optimal threshold)
    # tnr = 1 - fpr
    j = tpr - fpr # sensitivity (tpr) + specificity (tnr) - 1 = tpr - fpr
    best_idx = np.argmax(j)
    best_threshold = thresholds[best_idx]

    preds = (scores >= best_threshold).astype(int)
    accuracy = balanced_accuracy_score(labels, preds)

    target_fpr = 0.1 # low FPR for metric: TPR at low FPR
    tpr_10 = tpr_at_low_fpr(fpr, tpr, target_fpr)
    results = {
        'auc': auc,
        'balanced_accuracy': accuracy, # Youden optimal here
        'tpr_at_0.1_fpr': tpr_10,
        'best_threshold': best_threshold, 
        'best_tpr': tpr[best_idx],
        'best_fpr': fpr[best_idx],
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }
    return results

def plot_mia_roc(results):
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, result in results.items():
        RocCurveDisplay(fpr=result['fpr'], tpr=result['tpr'], roc_auc=result['auc'], name=name).plot(ax=ax)

    ax.plot([0, 1], [0, 1], linestyle='--', color='black')
    ax.set_title('MIA ROC Curves')
    plt.show()
