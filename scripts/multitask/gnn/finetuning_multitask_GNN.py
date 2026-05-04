import click
import copy as copy
import torch
import logging
import os
import sys
import numpy as np
import polars as pl
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
import shutil
# --- Configuration du projet ---
from transferlearning_rnaseq.config import config
import transferlearning_rnaseq.utils as tu
from transferlearning_rnaseq.Early_stopper import EarlyStopper
from transferlearning_rnaseq.mlp.model import GNN, MultiTaskGNN
from transferlearning_rnaseq.gnn.dataset import PPIDataset
from pytorch_lightning.loggers import CSVLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@click.command()
@click.option("--source", required=True, help="Liste de modèles séparés par des virgules (ex: cancer_model,tissue_model,stage,aa5)")
@click.option("--target", required=True, help="Nom du modèle cible")
@click.option("--freeze/--no-freeze", default=True, help="Geler les couches du modèle source")
@click.option("--cancer", default="", help="finetune the model on this cancer ")
@click.option("--n_version", default=None, type=int, help="number of version for logs")

def main(source, target, freeze, cancer, n_version):
    freeze_bool = "freeze" if freeze else ""
    cancer_str = "without"+cancer if cancer else cancer
    sources = sorted([s.strip() for s in source.split(",")]) if "," in source else [source]
    subset_dict = {k: config[k] for k in sources}
    name_csv = f"{'_'.join(sources)}_to_{target}_{cancer_str}{freeze_bool}"
    csv_logger = CSVLogger(save_dir="logs", name=name_csv, version=n_version)
    
    if len(sources)==1:
        source_dict = copy.deepcopy(config[source])
        source_dict["params_model"]["model_path"] = f"models/GNN_{source}{cancer_str}.pt"
    else:
        source_dict = copy.deepcopy(config[sources[0]])
        source_dict["params_model"]["output_size"] = list()
        source_dict[ 'loss_function']= list()
        source_dict["params_model"]["model_path"] = f"models/GNN_{'_'.join(sources)}_full.pt"
        for s in sources : 
            source_dict["params_model"]["output_size"].append(config[s]["params_model"]["output_size"])
            source_dict[ 'loss_function'].append(config[s]["loss_function"])
    print(source_dict)
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
    protein_to_gene_mapping = pl.read_parquet("data/gene_prot.parquet")

    if cancer:
        data = data.filter(pl.col("subtype").str.contains(rf"^*{cancer}*"))

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
        train_sizes = np.append(
            np.linspace(50, 800, num=10, dtype=int),
            np.linspace(800, max_train_size, num=10, dtype=int)
        )
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

        #original_model = tu.load_model(f"models/GNN_{source}.pt",source_dict["params_model"]["output_size"])
        #logging.info("Computing metrics using the original model's feature extractor")
        #features, labels = tu.extract_features_labels(original_model, test_loader, device)
        #print(features.shape)
        #original_metrics = compute_metrics(features, labels)
        #logging.info(f"Original model metrics: {original_metrics}")
        #csv_logger.log_metrics({"model": "test", **original_metrics, "train_samples": len(train_sub)})


        
        # --- Initialisation du modèle ---
        pretrained_model = MultiTaskGNN(1, subset_dict).to(device)
        pretrained_model.load_state_dict(torch.load(source_dict["params_model"]["model_path"]))
        
        model = GNN(
            input_size=1,
            output_size=target_dict["params_model"]["output_size"]
        ).to(device)
        logging.info(model)
        model.feature_extractor = pretrained_model.feature_extractor
        model.classifier= nn.Linear(model.num_nodes * model.hidden_size, target_dict["params_model"]["output_size"])
       #print(model)
        tu.freeze_layer(model, k)  
        for i, layer in enumerate(model.feature_extractor.children()):
                logging.debug(f"Freezing layer {i}")
                for param in layer.parameters():
                    print(param.requires_grad)
                logging.debug(f"Layer {i} frozen successfully")
        print(model)
        criterion = target_dict["loss_function"]
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hyperparams["learning_rate"],
            momentum=hyperparams["momentum"]
        )
        early_stopper = EarlyStopper(patience=5, min_delta=0.001)
        train_func = tu.train_gnn_bin if target_dict["params_model"]["output_size"] == 1 else tu.train_gnn_multiclass
        test_func = tu.test_gnn_bin if target_dict["params_model"]["output_size"] == 1 else tu.test_gnn_multiclass
        model.to(device)
        # --- Entraînement ---
        for epoch in tqdm(range(hyperparams["num_epochs"])):
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
            "epoch": epoch+1,
            "train_samples": len(train_sub), **result})
        global_step += 1
        csv_logger.save()

    csv_logger.save()
    # Sauvegarde du modèle
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #model_path = f"models/GNN_{source}/{source.split('/')[-1].replace('.pt','')}_to_{target}_{cancer}frozen{k}_{timestamp}.pt"
    #torch.save(model.state_dict(), model_path)
    #name_csv = f"GNN_{source}_to_{target}_{cancer_str}{freeze_bool}"
    logging.info(f"Modèle sauvegardé sous {name_csv}")
    #from numba import cuda
    #cuda.select_device(0)
    #cuda.close()

if __name__ == "__main__":
    main()
