import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.mstcn_wrapper import TeCNO, MSTCNDataset
from torch import nn
import numpy as np
import yaml


def load_config():
    parser = argparse.ArgumentParser(description="Train MSTCN model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    from types import SimpleNamespace
    args = SimpleNamespace(**config)
    return args

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                                    padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = torch.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        out = self.dropout(out)
        return torch.relu(out + self.residual(x))

class MSTCN(nn.Module):
    def __init__(self, num_stages=2, num_layers=10, num_f_maps=64, feature_dim=768, num_classes=7):
        super(MSTCN, self).__init__()
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_f_maps = num_f_maps
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.stages = nn.ModuleList()
        for stage in range(num_stages):
            layers = []
            layers.append(nn.Conv1d(feature_dim if stage == 0 else num_classes, 
                                  num_f_maps, kernel_size=1))
            layers.append(nn.BatchNorm1d(num_f_maps))
            layers.append(nn.ReLU())
            for i in range(num_layers):
                dilation = 2 ** i
                layers.append(DilatedResidualLayer(dilation, num_f_maps, num_f_maps))
            layers.append(nn.Conv1d(num_f_maps, num_classes, kernel_size=1))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x):
        outputs = []
        input_features = x
        for stage in self.stages:
            out = stage(input_features)
            outputs.append(out)
            input_features = out
        return torch.stack(outputs)

def main():
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

    # Initialize model
    model = MSTCN(
        num_stages=args.mstcn_stages,
        num_layers=10,
        num_f_maps=64,
        feature_dim=768,  # Match VideoMAE feature dimension
        num_classes=7
    )
    
    # Convert SimpleNamespace to dict for save_hyperparameters
    hparams_dict = vars(args)
    tecno = TeCNO(hparams=hparams_dict, model=model, dataset=dataset, weights=dataset_weights)

    # Ensure directories exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set up logger and checkpoint callback
    logger = TensorBoardLogger(save_dir=args.log_dir, name="mstcn")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True
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

    # Test the model
    trainer.test(tecno)

if __name__ == "__main__":
    main()