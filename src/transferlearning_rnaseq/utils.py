
import torch
from copy import deepcopy
import logging
import numpy as np
import os
import pdb
import sys
from sklearn.model_selection import train_test_split
import polars as pl
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from explore.mlp.dataset import CustomDataset
from torch.utils.data import DataLoader
from typing import List, Union
from sklearn.metrics import classification_report
import torch.nn as nn

from mlp.model import MLP, MultiTaskMLP, GNN
from GNN.dataset import PPIDataset

def train(model, optimizer, criterion, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X.to(device))
        loss = criterion(outputs, batch_y.to(device).view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batchx, batchy in val_loader:
            batchx, batchy = batchx.to(device), batchy.to(device)
            outputs = model(batchx)
            loss = criterion(outputs, batchy.view(-1))
            val_loss += loss.item()
            predicted = torch.argmax(outputs, axis=1)
            total += batchy.size(0)
            correct += (predicted == batchy.view(-1)).sum().item()
    accuracy = correct / total
    logging.info(f"Val Accuracy: {accuracy:.4f}")
    return total_loss / len(train_loader), val_loss / len(val_loader)


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, axis=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y.view(-1)).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return {"test_accuracy": accuracy}



def train_bin(model, optimizer, criterion, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batchx, batchy in val_loader:
            batchx, batchy = batchx.to(device), batchy.to(device)
            outputs = model(batchx)
            loss = criterion(outputs, batchy.unsqueeze(1))
            val_loss += loss.item()
            predicted = (outputs > 0).float()
            total += batchy.size(0)
            correct +=( predicted == batchy.unsqueeze(1)).sum().item()
    accuracy = correct / total
    logging.info(f"Val Accuracy: {accuracy:.4f}")
    len_val_loader = len(val_loader)
    if len(val_loader)==0:
        len_val_loader = 1
    return total_loss / len(train_loader), val_loss / len_val_loader


def test_bin(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_score = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predicted = (outputs > 0).float()
            total += batch_y.size(0)
            #correct += (predicted == batch_y).sum().item()
            correct+= (predicted == batch_y.unsqueeze(1)).sum().item()
            y_true.extend(batch_y.cpu().numpy())
            y_score.extend(predicted.cpu().numpy())
    accuracy = correct / total
    print(total, correct)
    auc = roc_auc_score(y_true, y_score)
    return  {"test_accuracy": accuracy,"Roc_auc": auc, "f1_score": classification_report(y_true, y_score, zero_division=0)}



def train_multitask(model, optimizer, criterions, train_loader, val_loader, device, config):
    model.train()
    total_loss = 0
    for batch_X, batch_y_dict in train_loader:
        batch_X = batch_X.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = 0
        for task_name, criterion in criterions.items():
            # Aller chercher dynamiquement la colonne cible depuis la config
            label_col = config[task_name]["params"]["target_column"]
            y = batch_y_dict[label_col].to(device)
            out = outputs[task_name]
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                y = y.float().unsqueeze(1)
            loss += criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y_dict in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            batch_loss = 0
            for task_name, criterion in criterions.items():
                label_col = config[task_name]["params"]["target_column"]
                y = batch_y_dict[label_col].to(device)
                out = outputs[task_name]
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    y = y.float().unsqueeze(1)
                batch_loss += criterion(out, y)
            val_loss += batch_loss.item()
    return total_loss / len(train_loader), val_loss / max(1, len(val_loader))

def test_multitask(model, test_loader, device, config):
    model.eval()
    results = {task: {'correct': 0, 'total': 0} for task in model.models.keys()}
    y_true_dict = {task: [] for task in model.models.keys()}
    y_score_dict = {task: [] for task in model.models.keys()}  # Pour AUC binaire

    with torch.no_grad():
        for batch_X, batch_y_dict in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            for task_name in outputs:
                label_col = config[task_name]["params"]["target_column"]
                out = outputs[task_name]
                y = batch_y_dict[label_col].to(device)
                # Binaire
                if out.shape[-1] == 1:
                    out = out.squeeze(-1)
                    predicted = (out > 0).float()
                    correct = (predicted == y.float()).sum().item()
                    y_true_dict[task_name].extend(y.cpu().numpy())
                    y_score_dict[task_name].extend(out.cpu().numpy())  # logits pour AUC
                # Multi-classes
                else:
                    predicted = torch.argmax(out, axis=1)
                    correct = (predicted == y).sum().item()
                results[task_name]['correct'] += correct
                results[task_name]['total'] += y.size(0)

    # Calcul des métriques
    metrics = {}
    for task in results:
        accuracy = results[task]['correct'] / results[task]['total']
        metrics[task] = {"accuracy": accuracy}
        # AUC uniquement pour les binaires
        if config[task]["params_model"]["output_size"] == 1:
            try:
                auc = roc_auc_score(y_true_dict[task], y_score_dict[task])
            except ValueError:
                auc = float('nan')
            metrics[task]["roc_auc"] = auc

    print("Test metrics:", metrics)
    return metrics


def train_gnn_bin(model, optimizer, criterion, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)  # shape: [batch_size, 1]
        loss = criterion(outputs, batch.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch.y.view(-1, 1))
            val_loss += loss.item()
            predicted = (outputs > 0).float()
            correct += (predicted == batch.y.view(-1, 1)).sum().item()
            total += batch.y.size(0)
    accuracy = correct / total
    print(f"Val Accuracy: {accuracy:.4f}")
    len_val_loader = len(val_loader) if len(val_loader) > 0 else 1
    return total_loss / len(train_loader), val_loss / len_val_loader

def test_gnn_bin(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_score = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            predicted = (outputs > 0).float()
            correct += (predicted == batch.y.view(-1, 1)).sum().item()
            total += batch.y.size(0)
            y_true.extend(batch.y.cpu().numpy())
            y_score.extend(outputs.cpu().numpy().flatten())
    accuracy = correct / total
    auc = roc_auc_score(y_true, y_score)
    f1 = classification_report(y_true, (np.array(y_score) > 0).astype(float), zero_division=0)
    return {"test_accuracy": accuracy, "roc_auc": auc, "f1_score": f1}

def train_gnn_multiclass_orig(model, optimizer, criterion, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)  # shape: [batch_size, num_classes]
        loss = criterion(outputs, batch.y)  # batch.y: [batch_size], entiers
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    total = 0
    predicted = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)  # [batch_size, num_classes]
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                y = batch.y.float().unsqueeze(1)
                loss = criterion(outputs, batch.y)
            else: loss = criterion(outputs, batch.y)
            
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            #all_preds.extend(preds.cpu().numpy())
            #all_targets.extend(batch.y.cpu().numpy())
            predicted = torch.argmax(outputs, axis=1)
            total += batch.y.size(0)
            correct += (predicted == batch.y.view(-1)).sum().item()
    accuracy = correct / total
            
    print(f"Val Accuracy: {accuracy:.4f}")
    len_val_loader = len(val_loader) if len(val_loader) > 0 else 1
    return total_loss / len(train_loader), val_loss / len_val_loader

def train_gnn_multiclass(model, optimizer, criterions, train_loader, val_loader, device, config):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)  # shape: [batch_size, num_classes]
        loss = 0
        for task_name, criterion in criterions.items():
            label_col = config[task_name]["params"]["target_column"]
            #print(f'---{task_name}---')
            #print(outputs[task_name].shape)
            #print(batch.y[label_col].shape)
            if isinstance(criterion, nn.BCEWithLogitsLoss): loss += criterion(outputs[task_name], batch.y[label_col].view(-1,1))
            else: loss += criterion(outputs[task_name], batch.y[label_col])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    total = 0
    predicted = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)  # [batch_size, num_classes]
            batch_loss = 0
            for task_name, criterion in criterions.items():
                label_col = config[task_name]["params"]["target_column"]
                if isinstance(criterion, nn.BCEWithLogitsLoss): batch_loss += criterion(outputs[task_name], batch.y[label_col].view(-1, 1))
                else: batch_loss += criterion(outputs[task_name], batch.y[label_col])
            val_loss += batch_loss.item()
            #total += batch.y.size(0)
    len_val_loader = len(val_loader) if len(val_loader) > 0 else 1
    return total_loss / len(train_loader), val_loss / len_val_loader


def test_gnn_multiclass(model, test_loader, device, config):
    model.eval()
    total = 0
    predicted = 0
    correct = 0
    results = {task: {'correct':0, 'total':0} for task in model.models.keys()}
    y_true_dict = {task: [] for task in model.models.keys()}
    y_score_dict = {task: [] for task in model.models.keys()}
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            for task_name in outputs:
                label_col = config[task_name]["params"]["target_column"]
                out = outputs[task_name]
                if out.shape[-1] == 1:
                    y = batch.y[label_col].view(-1,1)
                    predicted = (out > 0).float()
                    correct = (predicted == y).sum().item()
                    y_true_dict[task_name].extend(y.cpu().numpy())
                    y_score_dict[task_name].extend(out.cpu().numpy())
                else: 
                    y = batch.y[label_col]
                    predicted = torch.argmax(out, axis=1)
                    correct = (predicted == y).sum().item()
                results[task_name]['correct'] += correct
                results[task_name]['total'] += batch.y[label_col].size(0)

    metrics = {}
    for task in results:
        accuracy = results[task]['correct'] / results[task]['total']
        metrics[task] = {'accuracy': accuracy}
        if config[task]["params_model"]["output_size"] == 1:
            try: 
                auc = roc_auc_score(y_true_dict[task], y_score_dict[task])
            except ValueError:
                auc = float('nan')
            metrics[task]["roc_auc"] = auc
    return metrics

def test_gnn_bin(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_score = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            predicted = (outputs > 0).float()
            correct += (predicted == batch.y.view(-1, 1)).sum().item()
            total += batch.y.size(0)
            y_true.extend(batch.y.cpu().numpy())
            y_score.extend(outputs.cpu().numpy().flatten())
    accuracy = correct / total
    auc = roc_auc_score(y_true, y_score)
    f1 = classification_report(y_true, (np.array(y_score) > 0).astype(float), zero_division=0)
    return {"test_accuracy": accuracy, "roc_auc": auc, "f1_score": f1}

def train_gnn_multiclass_bin(model, optimizer, criterion, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)  # shape: [batch_size, num_classes]
        loss = criterion(outputs, batch.y)  # batch.y: [batch_size], entiers
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    total = 0
    predicted = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)  # [batch_size, num_classes]
            loss = criterion(outputs, batch.y)
            
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            #all_preds.extend(preds.cpu().numpy())
            #all_targets.extend(batch.y.cpu().numpy())
            predicted = torch.argmax(outputs, axis=1)
            total += batch.y.size(0)
            correct += (predicted == batch.y.view(-1)).sum().item()
    accuracy = correct / total
            
    print(f"Val Accuracy: {accuracy:.4f}")
    len_val_loader = len(val_loader) if len(val_loader) > 0 else 1
    return total_loss / len(train_loader), val_loss / len_val_loader


def test_gnn_multiclass_bin(model, test_loader, device):
    model.eval()
    total = 0
    predicted = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            preds = outputs.argmax(dim=1)
            predicted = torch.argmax(outputs, axis=1)
            total += batch.y.size(0)
            correct += (predicted == batch.y.view(-1)).sum().item()
    accuracy = correct / total
    #report = classification_report(all_targets, all_preds, zero_division=0)
    return {"test_accuracy": accuracy}#, "classification_report": report}


def freeze_layer(model, k):
    logging.info(f"Starting to freeze layers up to layer {k}")
    
    for i, layer in enumerate(model.feature_extractor.children()):
        if i < k*4:
            logging.debug(f"Freezing layer {i}")
            for param in layer.parameters():
                param.requires_grad = False
            logging.debug(f"Layer {i} frozen successfully")
        else:
            logging.debug(f"Layer {i} left unfrozen")
    
    logging.info(f"Finished freezing layers. Layers 0 to {k-1} are now frozen.")


def load_model(model_path,output_size=1,GNN=False):
    if GNN:
        model = GNN(input_size=1,output_size=output_size)
        model.load_state_dict(torch.load(model_path))
        return model
    if isinstance(output_size, dict):
        print("dict")
        model = MultiTaskMLP(19887, output_size)
    else:
        model = MLP(19887, output_size=output_size)
    model.load_state_dict(torch.load(model_path))
    return model


def reset_weights(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def baseline_model(model):
    baseline = deepcopy(model)
    reset_weights(baseline)
    return baseline


def get_layer_info(layer):
    if  hasattr(layer, 'out_features'):
        return layer.out_features
    elif hasattr(layer, 'num_features'):
        return layer.num_features
    elif hasattr(layer, 'num_channels'):
        return layer.num_channels
    else:
        return None


def features_extractor(model, x):
    model.eval()
    return model.feature_extractor(x)


def split_data(df, test_size=0.2, random_state=None, stratify_col=None):
    if stratify_col:
        stratify_vals = df[stratify_col]
        print(stratify_vals)
    else:
        stratify_vals = None
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify = stratify_vals)
    train, val = train_test_split(train_val, test_size=test_size, random_state=random_state, stratify=None if stratify_vals is None else train_val[stratify_col])
    return train, val, test


def create_dataloader(data: pl.DataFrame, exclude_columns, label_column: Union[str, List[str]], dtype, batch_size: int=32, shuffle: bool = True):
    drop_last = True
    if (data.shape[0]% batch_size)>2:
        drop_last = False
    dataset = CustomDataset(data=data, exclude_columns=exclude_columns, label_column=label_column, dtype=dtype)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,  drop_last=drop_last, pin_memory=True)
    return dataloader

from torch_geometric.loader import DataLoader

def create_dataloader_GNN(
    data: pl.DataFrame,
    ppi,
    protein_to_gene_mapping,
    exclude_columns,
    label_column,
    dtype,
    subdir, 
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True
):
    """
    Crée un DataLoader PyTorch Geometric pour un dataset de graphes (PPIDataset).
    """
    drop_last = False if (data.shape[0] % batch_size) > 2 else True
    print(dtype)
    dataset = PPIDataset(
        root=os.path.join("data/ppi/", subdir),
        result=data,
        ppi=ppi,
        protein_to_gene_mapping=protein_to_gene_mapping,
        exclude_columns=exclude_columns,
        label_column=label_column,
        dtype=dtype
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader


def extract_features_labels(model, data_loader, device=None):
    model.eval()
    model.to(device)
    features_list = []
    labels_list = []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            features = model.feature_extractor(batch_X)
            features_list.append(features.cpu().numpy())
            labels_list.append(batch_y.cpu().numpy().astype(int))
    return np.concatenate(features_list), np.concatenate(labels_list)
"""
def extract_features_labels_gnn(model, data_loader, device=None):
    model.eval()
    model.to(device)
    features_list = []
    labels_list = []
    modules = model.feature_extractor
    with torch.no_grad():
        for batch in data_loader:
            x, edge_index = batch.x, batch.edge_index
            print(x)
            print(edge_index)
            x, edge_index = x.to(device), edge_index.to(device)
            for i in range(0, len(modules), 4):
                conv = modules[i]
                bn = modules[i+1]
                relu = modules[i+2]
                dropout = modules[i+3]
                features = conv(x, edge_index)
                features = bn(features)
                features = relu(features)
                features = dropout(features)
            features_list.append(features.cpu().numpy())
            labels_list.append(batch.y.cpu().numpy().astype(int))
    return np.concatenate(features_list), np.concatenate(labels_list)
"""
def extract_features_labels_gnn(model, data_loader, device=None):
    model.eval()
    model.to(device)
    features_list = []
    labels_list = []
    modules = model.feature_extractor
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            features = model.get_features(batch)
            features_list.append(features.cpu().numpy())
            labels_list.append(batch.y.cpu().numpy().astype(int))
    return np.concatenate(features_list), np.concatenate(labels_list)

#out = x
#for module in modules:
    #out = module(out)
