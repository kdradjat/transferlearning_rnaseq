
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
from transferlearning_rnaseq.mlp.model import MLP
# Configuration initiale

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@click.command()
@click.option("--modele", required=True, help="Nom du modèle à entraîner (doit exister dans config)")
@click.option("--train_size_mode", 
              type=click.Choice(['linspace', 'full'], case_sensitive=False),
              default='linspace',
              help="Mode de sélection des tailles d'entraînement")
@click.option("--cancer", default="", help="train the model with it ")
@click.option("--n_version", default=None, type=int, help="number of version for logs")
def main(modele, train_size_mode, cancer, n_version):
    """Script complet d'entraînement pour un modèle de classification"""
    
    # Validation de la configuration
    try:
        model_config = config[modele]
        print(model_config)
    except KeyError:
        logging.error(f"Configuration introuvable pour le modèle '{modele}'")
        return

    # Initialisation du logger
    csv_logger = CSVLogger(
        save_dir="logs",
        name=f"train_{modele}_{train_size_mode}{cancer}_training",
        version=n_version)
    
    # Chargement des données
    try:
        data = pl.read_parquet(model_config["params"]["data_file"])
    except Exception as e:
        logging.error(f"Erreur de chargement des données : {str(e)}")
        return
    if cancer:
        data =  data.filter((pl.col("subtype").str.contains(rf"^*{cancer}*")))
    print(data)
    # Préparation des datasets
    exclude_columns = model_config["params"].get("exclude_columns", [])
    target_column = model_config["params"].get("target_column", "target")
    train, val, test = tu.split_data(data, stratify_col=target_column)

    # Création des dataloaders
    dtype = torch.float32  if model_config["params_model"]["output_size"]==1 else torch.long
    print(f"dtype{dtype}")
    val_loader = tu.create_dataloader(val, exclude_columns, target_column, dtype)
    test_loader = tu.create_dataloader(test, exclude_columns, target_column, dtype)

    max_train_size = len(train)
    # Hyperparamètres
    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 8,
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
            train_sizes = np.unique(np.linspace(8, max_train_size, num=10, dtype=int))
    else:  # 'full'
        train_sizes = [len(train)]  # Utilise la taille maximale
    print(train_sizes)
    # Boucle d'entraînement
    for train_size in train_sizes:
        logging.info(f"\nEntraînement avec {train_size} échantillons")
        train_sub = train.slice(0, train_size)
        train_loader = tu.create_dataloader(train_sub, exclude_columns, target_column, dtype)

        # Initialisation du modèle
        model = MLP(
            input_size=model_config["params_model"]["input_size"],
            output_size=model_config["params_model"]["output_size"]
        ).to(device)
        print(model)
        criterion = model_config["loss_function"]
        optimizer = torch.optim.SGD(model.parameters(),
            lr=hyperparams["learning_rate"],
            momentum=hyperparams["momentum"]
        )

        early_stopper = EarlyStopper(patience=5, min_delta=0.001)
        train_func = tu.train_bin if model_config["params_model"]["output_size"] == 1 else tu.train
        test_func = tu.test_bin if model_config["params_model"]["output_size"] == 1 else tu.test

        # Boucle d'epochs
        for epoch in range(hyperparams["num_epochs"]):
            train_loss, val_loss = train_func(model, optimizer, criterion, train_loader, val_loader, device)
            
            logging.info(f"Epoch {epoch+1}/{hyperparams['num_epochs']} - Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            csv_logger.log_metrics({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_size": train_size
            }, step=global_step)
            global_step += 1

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
    model_path = f"models/{modele}{cancer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    #torch.save(model.state_dict(), model_path)
    logging.info(f"Modèle sauvegardé sous {model_path}")
    #from numba import cuda
    #cuda.select_device(0)
    #cuda.close()


if __name__ == "__main__":
    main()
