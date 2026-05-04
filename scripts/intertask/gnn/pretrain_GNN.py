import click
import torch
import logging
import os
import sys
import numpy as np
import polars as pl
from tqdm import tqdm
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# --- Configuration du projet ---
from transferlearning_rnaseq.config import config
import transferlearning_rnaseq.utils as tu
from transferlearning_rnaseq.Early_stopper import EarlyStopper
from transferlearning_rnaseq.mlp.model import GNN  # à adapter selon ton modèle
from transferlearning_rnaseq.gnn.dataset import PPIDataset
from pytorch_lightning.loggers import CSVLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@click.command()
@click.option("--modele", required=True, help="Nom du modèle à entraîner (doit exister dans config)")
@click.option("--train_size_mode", 
              type=click.Choice(['linspace', 'full'], case_sensitive=False),
              default='linspace',
              help="Mode de sélection des tailles d'entraînement")
@click.option("--cancer", default="", help="Entraîner sans ce cancer (filtrage)")
@click.option("--n_version", default=None, type=int, help="number of version for logs")
def main(modele,  train_size_mode, cancer, n_version):

    # --- Chargement de la config ---
    try:
        cfg = config[modele]
    except KeyError:
        logging.error(f"Configuration introuvable pour le modèle '{modele}'")
        return
    
    cancer_str = "without"+cancer if cancer else cancer
    target_column = cfg["params"]["target_column"]
    exclude_columns = cfg["params"]["exclude_columns"]
    csv_logger = CSVLogger(
        save_dir="logs",
        name=f"train_GNN_{modele}_{train_size_mode}{cancer_str}_training",
        version=n_version)

    
    # --- Chargement des données ---
    data = pl.read_parquet(cfg["params"]["data_file"])
    ppi = pl.read_parquet("data/ppi.parquet")
    protein_to_gene_mapping = pl.read_parquet("data/gene_prot.parquet")

    if cancer:
        data = data.filter(~(pl.col("subtype").str.contains(rf"{cancer}")))
    
    train, val, test = tu.split_data(data, stratify_col=target_column)
    print(train, val, test)
    max_train_size = len(train)
    # Hyperparamètres
    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 50,
        "momentum": 0.9,
        "optimizer": "SGD",
        "train_size_mode": train_size_mode
    }
    csv_logger.log_hyperparams(hyperparams)
    global_step = 0

    # Configuration des tailles d'entraînement
    if train_size_mode == 'linspace':
        if max_train_size > 1000:
            train_sizes = np.append(
                np.linspace(50, 800, num=10, dtype=int),
                np.linspace(800, max_train_size, num=10, dtype=int)
            )
        else:
            train_sizes = np.unique(np.linspace(1, max_train_size, num=10, dtype=int))
    else:  # 'full'
        train_sizes = [len(train)]  # Utilise la taille maximale
    print(train_sizes)
    dtype = torch.float32  if cfg["params_model"]["output_size"]==1 else torch.long

    print(f"dtype:{dtype}")
    shutil.rmtree("data/ppi/pretrain/")
    val_loader = tu.create_dataloader_GNN(val, ppi,
        protein_to_gene_mapping, exclude_columns, target_column, dtype, "pretrain/val")
    test_loader = tu.create_dataloader_GNN(test, ppi,
        protein_to_gene_mapping, exclude_columns, target_column, dtype, "pretrain/test")
    # Boucle d'entraînement
    for train_size in tqdm(train_sizes):
        logging.info(f"\nEntraînement avec {train_size} échantillons")
        train_sub = train.slice(0, train_size)
        train_loader = tu.create_dataloader_GNN(train_sub,ppi,
    protein_to_gene_mapping, exclude_columns, target_column, dtype, "pretrain/train")
        print(len(train_loader), len(val_loader), len(test_loader))
    
        # --- Initialisation du modèle ---
        model = GNN(
            input_size= 1,
            output_size=cfg["params_model"]["output_size"]
        ).to(device)
        logging.info(model)
    
        criterion = cfg["loss_function"]
        optimizer = torch.optim.SGD(model.parameters(),
            lr=hyperparams["learning_rate"],
            momentum=hyperparams["momentum"]
        )
        early_stopper = EarlyStopper(patience=5, min_delta=0.001)
        train_func = tu.train_gnn_bin if cfg["params_model"]["output_size"] == 1 else tu.train_gnn_multiclass
        test_func = tu.test_gnn_bin if cfg["params_model"]["output_size"] == 1 else tu.test_gnn_multiclass
        print(train_func)
        # --- Entraînement ---
        for epoch in tqdm(range(hyperparams ["num_epochs"])):
            model.train()
            train_loss, val_loss = train_func(model, optimizer, criterion, train_loader, val_loader, device)
            
            logging.info(f"Epoch {epoch+1}/{hyperparams['num_epochs']} - Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            csv_logger.log_metrics({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_size": train_size
            })

            if early_stopper(model, val_loss):
                logging.info("Arrêt anticipé déclenché")
                break

        # Évaluation finale
        result = test_func(model, test_loader, device)
        logging.info(f"Accuracy finale : {result}")
        csv_logger.log_metrics({
                    "epoch": epoch+1,"train_samples": len(train_sub), **result}, step=global_step)
        global_step += 1
        csv_logger.save()
        
    csv_logger.save()
    # Sauvegarde du modèle
    #model_path = f"models/GNN_{modele}{cancer_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    model_path = f"models/GNN_{modele}{cancer_str}.pt"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Modèle sauvegardé sous {model_path}")
    #from numba import cuda
    #cuda.select_device(0)
    #cuda.close()

if __name__ == "__main__":
    main()
