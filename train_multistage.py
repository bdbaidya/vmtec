import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from models.mstcn_wrapper import TeCNO, MSTCNDataset
from models.mstcn import MultiStageModel
from torch import nn
import numpy as np
import yaml
import wandb

def load_config():
    parser = argparse.ArgumentParser(description="Train MSTCN model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser = TeCNO.add_module_specific_args(parser)
    args = parser.parse_args()
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    from types import SimpleNamespace
    config.update({k: v for k, v in vars(args).items() if v is not None and k != 'config'})
    args = SimpleNamespace(**config)
    print(f"Using batch_size: {args.batch_size}")
    return args

def main():
    wandb.login(key="3cd0fd46806f5dbb7e666990676fb3d1c75e0447")
    args = load_config()

        # Define label map for surgical phases
    label_map = {
        'Preparation': 0,
        'CalotTriangleDissection': 1,
        'ClippingCutting': 2,
        'GallbladderDissection': 3,
        'GallbladderPackaging': 4,
        'CleaningCoagulation': 5,
        'GallbladderRetraction': 6
    }

        # Initialize datasets
    dataset = {
        "train": MSTCNDataset(folder_path=args.train_dir, label_map=label_map),
        "val": MSTCNDataset(folder_path=args.val_dir, label_map=label_map),
        "test": MSTCNDataset(folder_path=args.test_dir, label_map=label_map)
    }

        # Check dataset directories
    for split in ["train", "val", "test"]:
        if not os.path.exists(getattr(args, f"{split}_dir")):
            raise FileNotFoundError(f"{split}_dir {getattr(args, f'{split}_dir')} does not exist")
        if not dataset[split].file_list:
            raise ValueError(f"No .npy files found in {split}_dir {getattr(args, f'{split}_dir')}")

        # Calculate class weights for training (inverse class frequency)
    class_counts = np.zeros(len(label_map))
    for file_name in dataset["train"].file_list:
        label_name = file_name.split('_')[-1].replace('.npy', '')
        label_idx = label_map[label_name]
        class_counts[label_idx] += 1
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * len(label_map)
    dataset_weights = {"train": weights}

        # Inspect file names, labels, and shapes for all splits (limit to 10 files per split)
    for split in ["train", "val", "test"]:
        print(f"\nInspecting {split} files and their labels (showing up to 10):")
        for i, file_name in enumerate(dataset[split].file_list):
            if i >= 10:
                print("... (truncated)")
                break
            label_name = file_name.split('_')[-1].replace('.npy', '')
            label_idx = label_map.get(label_name, -1)
            print(f"File: {file_name} --> Label: {label_name} (Index: {label_idx})")
            file_path = os.path.join(getattr(args, f"{split}_dir"), file_name)
            try:
                data = np.load(file_path)
                print(f"    Shape: {data.shape}")
            except Exception as e:
                print(f"    Failed to load {file_name}: {e}")

        # Initialize MultiStageModel
    model = MultiStageModel(args)  # Pass hparams directly

        # Convert SimpleNamespace to dict for save_hyperparameters
    hparams_dict = vars(args)
    tecno = TeCNO(hparams=hparams_dict, model=model, dataset=dataset, weights=dataset_weights)

        # Ensure directories exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Set up logger and checkpoint callback
    logger = TensorBoardLogger(save_dir=args.log_dir, name="mstcn")
    wandb_logger = WandbLogger(project="VMAE-MSTCN_Final_training", log_model="all", config=hparams_dict)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
        every_n_epochs=10
    )

        # Handle devices for backward compatibility
    devices = getattr(args, 'devices', getattr(args, 'gpus', 1))

        # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=devices,
        accelerator="gpu",
        logger=logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=2
    )

        # Train the model
    trainer.fit(tecno)
    wandb.finish()

if __name__ == "__main__":
    main()