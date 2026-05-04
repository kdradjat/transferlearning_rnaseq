import click
import torch
from pytorch_lightning.loggers import CSVLogger
import logging
import os
import sys
import numpy as np
import polars as pl
from datetime import datetime

from transferlearning_rnaseq.config import config
import transferlearning_rnaseq.utils as tu
from transferlearning_rnaseq.Early_stopper import EarlyStopper
from transferlearning_rnaseq.mlp.model import MultiTaskMLP

# Configuration initiale
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@click.command()
@click.option("--modeles", required=True, help="Liste de modèles séparés par des virgules (ex: cancer_model,tissue_model,stage,aa5)")
@click.option("--train_size_mode", 
              type=click.Choice(['linspace', 'full'], case_sensitive=False),
              default='linspace',
              help="Mode de sélection des tailles d'entraînement")
def main(modeles, train_size_mode):
    """Script complet d'entraînement multi-modèles avec logging structuré"""
    model_keys = sorted([m.strip() for m in modeles.split(",")])
    subset_dict = {k: config[k] for k in model_keys}

    # Sélection dynamique du fichier de données
    has_stage = "stage_model" in model_keys
    has_aa5 = "aa5_model" in model_keys
    if has_stage and has_aa5:
        data_file = "data/stage_aa5.parquet"
    elif has_stage:
        data_file = "data/stage.parquet"
    elif has_aa5:
        data_file = "data/aa5.parquet"
    else:
        data_file = "data/result.parquet"
    logging.info(f"Utilisation du fichier de données : {data_file}")

    # Chargement des données
    try:
        data = pl.read_parquet(data_file)
        print(data.shape)
    except Exception as e:
        logging.error(f"Erreur de chargement des données : {str(e)}")
        return

    # Construction dynamique des exclude, target, dtype
    exclude = set()
    target = []
    dtype = {}
    for key in subset_dict:
        params = subset_dict[key]["params"]
        exclude.update(params.get("exclude_columns", []))
        target_col = params.get("target_column", "target")
        target.append(target_col)
        if subset_dict[key]["params_model"]["output_size"] == 1:
            dtype[target_col] = torch.float32
        else:
            dtype[target_col] = torch.long
    exclude = list(exclude)

    # Split data
    train, val, test = tu.split_data(data)

    # Création des dataloaders
    train_loader = tu.create_dataloader(train, exclude_columns=exclude, label_column=target, dtype=dtype)
    val_loader = tu.create_dataloader(val, exclude_columns=exclude, label_column=target, dtype=dtype)
    test_loader = tu.create_dataloader(test, exclude_columns=exclude, label_column=target, dtype=dtype)

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

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_logger = CSVLogger(
        save_dir="logs",
        name=f"train_{'_'.join(model_keys)}_{train_size_mode}_training",
        version=None)
    csv_logger.log_hyperparams(hyperparams)

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

    # Boucle d'entraînement
    for train_size in train_sizes:
        logging.info(f"\nEntraînement avec {train_size} échantillons")
        train_sub = train.slice(0, train_size)
        train_loader = tu.create_dataloader(train_sub, exclude_columns=exclude, label_column=target, dtype=dtype)

        # Initialisation du modèle multitâche
        input_size = subset_dict[model_keys[0]]["params_model"]["input_size"]
        model = MultiTaskMLP(input_size, subset_dict).to(device)
        print(model)
        criterions = {k: subset_dict[k]["loss_function"] for k in model_keys}
        optimizer = torch.optim.SGD(model.parameters(),
            lr=hyperparams["learning_rate"],
            momentum=hyperparams["momentum"]
        )
        early_stopper = EarlyStopper(patience=5, min_delta=0.001)
        train_func = tu.train_multitask
        test_func = tu.test_multitask

        # Boucle d'epochs
        for epoch in range(hyperparams["num_epochs"]):
            train_loss, val_loss = train_func(model, optimizer, criterions, train_loader, val_loader, device, subset_dict)
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
        result = test_func(model, test_loader, device, subset_dict)
        logging.info(f"Accuracy finale : {result}")
        csv_logger.log_metrics({
            "epoch": epoch+1, "train_samples": len(train_sub), **result
        })
        csv_logger.save()

    csv_logger.save()
    # Sauvegarde du modèle
    model_path = f"models/{'_'.join(model_keys)}_{train_size_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    model_path = f"models/{'_'.join(model_keys)}_{train_size_mode}.pt"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Modèle sauvegardé sous {model_path}")

    # Libération GPU (optionnel)
    #from numba import cuda
    #cuda.select_device(0)
    #cuda.close()

if __name__ == "__main__":
    main()
