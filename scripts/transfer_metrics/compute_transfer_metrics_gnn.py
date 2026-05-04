import click
import torch
import logging
import os
import sys
import numpy as np
import polars as pl
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.loader import DataLoader
import shutil
# --- Configuration du projet ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from explore.transfert.config import config
import explore.transfert.utils as tu
from explore.transfert.Early_stopper import EarlyStopper
from explore.mlp.model import GNN  
from explore.GNN.dataset import PPIDataset
from pytorch_lightning.loggers import CSVLogger
from explore.transfert.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@click.command()
@click.option("--source", required=True, help="Chemin du modèle source (.pt)")
@click.option("--target", required=True, help="Nom du modèle cible")
@click.option("--freeze/--no-freeze", default=True, help="Geler les couches du modèle source")
@click.option("--cancer", default="", help="finetune the model on this cancer ")
@click.option("--n_version", default=None, type=int, help="number of version for logs")

def main(source, target, freeze, cancer, n_version):
    freeze_bool = "freeze" if freeze else ""
    cancer_str = "without"+cancer if cancer else cancer
    #name_csv = f"GNN_{source}_to_{target}_{cancer_str}{freeze_bool}"
    name_csv = f"transfer_metrics_{source}_to_{target}_{cancer_str}{freeze_bool}"
    csv_logger = CSVLogger(save_dir="logs", name=name_csv, version=n_version)
    
    source_dict = config[source]
    source_dict["params_model"]["model_path"] = f"models/GNN_{source}{cancer_str}.pt"
    print(f"Path pretrained model{source_dict['params_model']['model_path']}")
    target_dict = config[target]
    exclude_columns = target_dict["params"]["exclude_columns"]
    target_column = target_dict["params"]["target_column"]

    click.echo("\n" + "="*40)
    click.echo(f"Modèle '{target}' configuré avec succès")
    click.echo(f"• Freeze: {freeze}")
    click.echo(f"• Cancer: {cancer}")
    

    # --- Chargement des données ---
    data = pl.read_parquet(target_dict["params"]["data_file"])
    ppi = pl.read_parquet("data/ppi.parquet")
    #ppi = pl.read_parquet("data/ppi_random_distrib.parquet")
    protein_to_gene_mapping = pl.read_parquet("data/gene_prot.parquet")

    if cancer:
        data = data.filter((pl.col("subtype").str.contains(rf"{cancer}")))

    train, val, test = tu.split_data(data, stratify_col=target_column)
    max_train_size = len(train)
    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 8,
        "num_epochs": 50,
        "momentum": 0.9,
        "optimizer": "SGD"
    }
    csv_logger.log_hyperparams(hyperparams)
    global_step = 0

    # Configuration des tailles d'entraînement
    
    if max_train_size > 1000:
        #train_sizes = np.append(
        #    np.linspace(50, 800, num=10, dtype=int),
        #    np.linspace(800, max_train_size, num=10, dtype=int)
        #)
        train_sizes = np.linspace(50, 800, num=10, dtype=int)
    else:
        train_sizes = np.unique(np.linspace(8, max_train_size, num=10, dtype=int))


    dtype = torch.float32 if target_dict["params_model"]["output_size"] == 1 else torch.long
    shutil.rmtree("data/ppi/train/")
    val_loader = tu.create_dataloader_GNN(val, ppi,
        protein_to_gene_mapping, exclude_columns, target_column, dtype, "train/val")
    test_loader = tu.create_dataloader_GNN(test, ppi,
        protein_to_gene_mapping, exclude_columns, target_column, dtype, "train/test")
    ## freeze ou non
    k = 0
    if freeze :
        k = 3
    for train_size in tqdm(train_sizes):
        logging.info(f"\nEntraînement avec {train_size} échantillons")
        train_sub = train.slice(0, train_size)
        train_loader = tu.create_dataloader_GNN(train_sub, ppi,
            protein_to_gene_mapping, exclude_columns, target_column, dtype, "train/train")
        logging.info(f"Tailles loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        print(len(train_loader), len(val_loader), len(test_loader))

        original_model = GNN(
            input_size=1,
            output_size=source_dict["params_model"]["output_size"]
        ).to(device)
        logging.info(original_model)
        original_model.load_state_dict(torch.load(source_dict["params_model"]["model_path"]))

        #original_model = tu.load_model(f"models/GNN_{source}.pt",source_dict["params_model"]["output_size"])
        #logging.info("Computing metrics using the original model's feature extractor")
        features, labels = tu.extract_features_labels_gnn(original_model, test_loader, device)
        #features
        print(features.shape)
        original_metrics = compute_metrics(features, labels)
        logging.info(f"Original model metrics: {original_metrics}")
        csv_logger.log_metrics({"model": "test", **original_metrics, "train_samples": len(train_sub)})


        global_step += 1
        csv_logger.save()

    csv_logger.save()

if __name__ == "__main__":
    main()
