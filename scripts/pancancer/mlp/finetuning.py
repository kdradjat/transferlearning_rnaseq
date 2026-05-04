import click
import torch
from pytorch_lightning.loggers import CSVLogger
import logging
import os
import sys
import numpy as np

from transferlearning_rnaseq.config import config
import transferlearning_rnaseq.utils as tu
from transferlearning_rnaseq.Early_stopper import EarlyStopper
from transferlearning_rnaseq.metrics import compute_metrics
import polars as pl
from datetime import datetime
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@click.command()
@click.option("--source", required=True, help="Chemin du modèle source (.pt)")
@click.option("--target", required=True, help="Nom du modèle cible")
@click.option("--freeze/--no-freeze", default=True, help="Geler les couches du modèle source")
@click.option("--cancer", default="", help="finetune the model on this cancer ")
@click.option("--n_version", default=None, type=int, help="number of version for logs")

def main(source, target, freeze, cancer, n_version):
    """Script de transfert learning pour modèles de classification de cancer"""
    # init + load  data 
    freeze_bool = "freeze" if freeze else ""
    cancer_str = "without"+cancer if cancer else cancer
    name_csv = f"{source}_to_{target}_{cancer_str}{freeze_bool}"
    
    csv_logger = CSVLogger(save_dir="logs", name=name_csv, version=n_version)
    
    source_dict = config[source]
    source_dict["params_model"]["model_path"] = f"models/{source}{cancer_str}.pt"
    print(f"Path pretrained model{source_dict['params_model']['model_path']}")
    target_dict = config[target]
    exclude_columns = target_dict["params"]["exclude_columns"]
    target_column = target_dict["params"]["target_column"]

    click.echo("\n" + "="*40)
    click.echo(f"Modèle '{target}' configuré avec succès")
    click.echo(f"• Freeze: {freeze}")
    click.echo(f"• Cancer: {cancer}")
    
    ## preparation des données
    dtype = torch.float32  if target_dict["params_model"]["output_size"]==1 else torch.long
    data = pl.read_parquet(target_dict["params"]["data_file"])
    if cancer :
        data = data.filter(pl.col("subtype").str.contains(rf"^*{cancer}*"))
    print(data.shape)
    train, val, test = tu.split_data(data, stratify_col=target_column)
    val_loader, test_loader = tu.create_dataloader(val, exclude_columns, target_column, dtype),tu.create_dataloader(test, exclude_columns, target_column, dtype)
    original_model = tu.load_model(source_dict["params_model"]["model_path"],source_dict["params_model"]["output_size"])
    print(len(val), len(train), len(test))
    ## freeze ou non
    k = 0
    if freeze :
        k = 4
    
    # hyper param
    max_train_size = len(train)
    hyperparams = {
            "learning_rate": 1e-3,
            "batch_size": 8,
            "num_epochs": 50,
            "momentum": 0.9,
            "optimizer": "SGD",
            "activation_function": "ReLU"
        }
    csv_logger.log_hyperparams(hyperparams)
    global_step = 0
    
    # Train
    if max_train_size > 1000:
        train_sizes = np.append(
                np.linspace(50, 800, num=10, dtype=int),
                np.linspace(800, max_train_size, num=10, dtype=int)
        )
    else:
        train_sizes = np.unique(np.linspace(8, max_train_size, num=10, dtype=int))

    for train_size in train_sizes:
        print(f"Training with {train_size} samples")
        csv_logger.log_metrics({"train_size": train_size})
        train_sub = train.slice(0, train_size)
        train_loader = tu.create_dataloader(train_sub, exclude_columns, target_column, dtype)
        ### LOG INFO
        print(f"Nombre d'échantillons dans l'ensemble d'entraînement : {len(train_sub)}")
        print(f"Nombre d'échantillons dans l'ensemble de validation : {len(val)}")
        print(f"Ensemble de test : {len(test)}")

        # compute metrics on the test loader + log
        original_model = tu.load_model(source_dict["params_model"]["model_path"],source_dict["params_model"]["output_size"])
        logging.info("Computing metrics using the original model's feature extractor")
        features, labels = tu.extract_features_labels(original_model, test_loader, device)
        print(features.shape)
        original_metrics = compute_metrics(features, labels)
        logging.info(f"Original model metrics: {original_metrics}")
        csv_logger.log_metrics({"model": "test", **original_metrics, "train_samples": len(train_sub)})
        
        # compute metrics on the train loader + log
        #print(original_model, len(train_sub))
        #logging.info("Computing metrics using the original model's feature extractor")
        #features, labels = tu.extract_features_labels(original_model, train_loader, device)
        #print(features.shape)
        #original_metrics = compute_metrics(features, labels)
        #logging.info(f"Original model metrics: {original_metrics}")
        #csv_logger.log_metrics({"model": "train", **original_metrics, "train_samples": len(train_sub)})
        
        
        for exp in range(1):
            # modele
            original_model = tu.load_model(source_dict["params_model"]["model_path"],source_dict["params_model"]["output_size"])
            new_model = ModifiedModel(original_model,  target_dict["params_model"]["output_size"]).to(device)
            tu.freeze_layer(new_model, k)  
            # fonction loss, optim, train/test fonction
            criterion = target_dict["loss_function"]
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, new_model.parameters()), lr=hyperparams["learning_rate"], momentum=hyperparams["momentum"])
            train_function = tu.train_bin if target_dict["params_model"]["output_size"]==1 else tu.train
            test_function = tu.test_bin if target_dict["params_model"]["output_size"]==1 else tu.test
            # Freeze layers
            early_stopper = EarlyStopper(
                patience=5, 
                min_delta=0.001,
                restore_best_weights=True
            )
            for i, layer in enumerate(new_model.feature_extractor.children()):
                logging.debug(f"Freezing layer {i}")
                for param in layer.parameters():
                    print(param.requires_grad)
                logging.debug(f"Layer {i} frozen successfully")

            num_epochs = 50
            # training
            for epoch in range(num_epochs):
                train_loss, val_loss = train_function(new_model, optimizer, criterion, train_loader, val_loader,  device)
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
                csv_logger.log_metrics({
                    "model": "modified",
                    "frozen_layers": k,
                    "experiment": exp,
                    "epoch": epoch+1,
                    "train_samples": len(train_sub),
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
                if early_stopper(new_model, val_loss):
                    print(f"Early stopping à l'epoch {epoch}")
                    break

            result = test_function(new_model, test_loader, device)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {result}")
            csv_logger.log_metrics({"model": "modified", "frozen_layers": 0,
                    "experiment": exp,
                    "epoch": epoch+1,"train_samples": len(train_sub), **result}, step=global_step)
            csv_logger.save()
            global_step += 1
            
    # save model        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #name_new_model = f"models/{source}/{source.split('/')[-1].replace('.pt','')}_to_{target}_{cancer}frozen{k}_{timestamp}.pt"
    #torch.save(new_model.state_dict(), name_new_model)
    csv_logger.save()
    #from numba import cuda
    #cuda.select_device(0)
    #cuda.close()
    return new_model
if __name__ == "__main__":
    main()
