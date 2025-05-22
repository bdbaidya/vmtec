import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningModule
from torchmetrics.classification import Precision, Recall, F1Score
from utils.metric_helper import AccuracyStages
from torch import nn
import numpy as np
import logging


logging.basicConfig(level=logging.DEBUG)


class MSTCNDataset(Dataset):
    def __init__(self, folder_path, label_map):
        self.folder_path = folder_path
        self.label_map = label_map
        self.class_labels = list(label_map.keys())
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        self.file_list.sort()
        # Log class distribution
        class_counts = {}
        for f in self.file_list:
            label = f.split('_')[-1].replace('.npy', '')
            class_counts[label] = class_counts.get(label, 0) + 1
        #logging.info(f"Dataset {folder_path}: Class counts: {class_counts}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        label_name = file_name.split('_')[-1].replace('.npy', '')
        label_idx = self.label_map[label_name]
        feature_path = os.path.join(self.folder_path, file_name)
        features = np.load(feature_path)
        mean = np.mean(features)
        std = np.std(features)
        #logging.debug(f"File {file_name}: {features.shape[0]} frames, Mean {mean:.4f}, Std {std:.4f}")
        if features.shape != (16, 768):
            raise ValueError(f"Unexpected feature shape {features.shape} for {file_name}")
        features = torch.tensor(features, dtype=torch.float)
        seq_len = features.shape[0]
        label_tensor = torch.full((seq_len,), label_idx, dtype=torch.long)
        return features.transpose(0, 1), label_tensor

def format_classification_input(y_pred, y_true):
    y_pred = torch.as_tensor(y_pred)
    y_true = torch.as_tensor(y_true)
    #logging.debug(f"format_classification_input: y_pred shape {y_pred.shape}, y_true shape {y_true.shape}")
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    #logging.debug(f"format_classification_input: y_pred after argmax {y_pred.shape}, y_true {y_true.shape}")
    return y_pred, y_true

class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset, weights):
        super(TeCNO, self).__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = self.hparams.batch_size
        self.dataset = dataset
        self.model = model
        self.weights_train = np.asarray(weights["train"])
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.weights_train).float().to(self.device))
        self.init_metrics()
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
    
    def init_metrics(self):
        self.train_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.val_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.max_acc_last_stage = {"epoch": 0, "acc": 0}
        self.max_acc_global = {"epoch": 0, "acc": 0, "stage": 0, "last_stage_max_acc_is_global": False}
        self.train_precision = Precision(num_classes=7, average='macro', task='multiclass')
        self.train_recall = Recall(num_classes=7, average='macro', task='multiclass')
        self.train_f1 = F1Score(num_classes=7, average='macro', task='multiclass')
        self.val_precision = Precision(num_classes=7, average='macro', task='multiclass')
        self.val_recall = Recall(num_classes=7, average='macro', task='multiclass')
        self.val_f1 = F1Score(num_classes=7, average='macro', task='multiclass')
        self.test_precision = Precision(num_classes=7, average='macro', task='multiclass')
        self.test_recall = Recall(num_classes=7, average='macro', task='multiclass')
        self.test_f1 = F1Score(num_classes=7, average='macro', task='multiclass')
    
    def custom_dice_score(self, y_pred, y_true, num_classes=7, epsilon=1e-6):
        dice_scores = torch.zeros(num_classes, device=y_pred.device)
        for c in range(num_classes):
            pred_c = (y_pred == c).float()
            true_c = (y_true == c).float()
            intersection = (pred_c * true_c).sum()
            union = pred_c.sum() + true_c.sum()
            dice_scores[c] = (2 * intersection + epsilon) / (union + epsilon)
        return dice_scores.mean()
    
    def forward(self, x):
        y_classes = self.model.forward(x)
        y_classes = torch.softmax(y_classes, dim=2)
        #logging.debug(f"forward: y_classes shape {y_classes.shape}")
        return y_classes
    
    def loss_function(self, y_classes, labels):
        stages = y_classes.shape[0]
        clc_loss = 0
        for j in range(stages):
            p_classes = y_classes[j]
            #logging.debug(f"loss_function: y_classes[{j}] shape {p_classes.shape}, labels shape {labels.shape}")
            ce_loss = self.ce_loss(p_classes, labels)
            clc_loss += ce_loss
        clc_loss = clc_loss / (stages * 1.0)
        return clc_loss
    
    def get_class_acc(self, y_true, y_classes):
        y_true = y_true.squeeze()
        y_classes = y_classes.squeeze()
        y_classes = torch.argmax(y_classes, dim=0)
        acc_classes = torch.sum(y_true == y_classes).float() / (y_true.shape[0] * 1.0)
        return acc_classes
    
    def get_class_acc_each_layer(self, y_true, y_classes):
        y_true = y_true.squeeze()
        accs_classes = []
        for i in range(y_classes.shape[0]):
            acc_classes = self.get_class_acc(y_true, y_classes[i, 0])
            accs_classes.append(acc_classes)
        return accs_classes
    
    def calc_metrics(self, y_pred, y_true, step="val"):
        y_max_pred, y_true = format_classification_input(y_pred[-1], y_true)
        # Log per-class accuracy
        per_class_acc = []
        for c in range(7):
            mask = (y_true == c)
            if mask.sum() > 0:
                acc = (y_max_pred[mask] == y_true[mask]).float().mean()
                per_class_acc.append(acc)
            else:
                per_class_acc.append(torch.tensor(float('nan'), device=y_max_pred.device))
        #logging.info(f"{step} per-class accuracy: {per_class_acc}")
        
        if step == "train":
            precision = self.train_precision(y_max_pred, y_true)
            recall = self.train_recall(y_max_pred, y_true)
            f1 = self.train_f1(y_max_pred, y_true)
            dice = self.custom_dice_score(y_max_pred, y_true)
        elif step == "val":
            precision = self.val_precision(y_max_pred, y_true)
            recall = self.val_recall(y_max_pred, y_true)
            f1 = self.val_f1(y_max_pred, y_true)
            dice = self.custom_dice_score(y_max_pred, y_true)
        else:
            precision = self.test_precision(y_max_pred, y_true)
            recall = self.test_recall(y_max_pred, y_true)
            f1 = self.test_f1(y_max_pred, y_true)
            dice = self.custom_dice_score(y_max_pred, y_true)
        return precision, recall, f1, dice
    
    def log_metrics(self, outputs, step="val"):
        precision_list = [o["precision"] for o in outputs]
        recall_list = [o["recall"] for o in outputs]
        f1_list = [o["f1"] for o in outputs]
        dice_list = [o["dice"] for o in outputs]
        # Check for NaNs
        for i, (p, r, f, d) in enumerate(zip(precision_list, recall_list, f1_list, dice_list)):
            if torch.any(torch.isnan(p)) or torch.any(torch.isnan(r)) or torch.any(torch.isnan(f)) or torch.any(torch.isnan(d)):
                logging.warning(f"NaN detected in {step} metrics for batch {i}")
        avg_precision = torch.stack([p for p in precision_list if not torch.isnan(p).any()]).mean()
        avg_recall = torch.stack([r for r in recall_list if not torch.isnan(r).any()]).mean()
        avg_f1 = torch.stack([f for f in f1_list if not torch.isnan(f).any()]).mean()
        avg_dice = torch.stack([d for d in dice_list if not torch.isnan(d).any()]).mean()
        self.log(f"{step}_avg_precision", avg_precision, on_epoch=True, on_step=False)
        self.log(f"{step}_avg_recall", avg_recall, on_epoch=True, on_step=False)
        self.log(f"{step}_avg_f1", avg_f1, on_epoch=True, on_step=False)
        self.log(f"{step}_avg_dice", avg_dice, on_epoch=True, on_step=False)
    
    def training_step(self, batch, batch_idx):
        try:
            stem, y_true = batch
            class_counts = torch.bincount(y_true.view(-1), minlength=7)
            #logging.debug(f"Epoch {self.current_epoch}, Batch {batch_idx}, Class counts: {class_counts.tolist()}")
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                #logging.debug(f"Epoch {self.current_epoch}, Batch {batch_idx}, GPU memory: {mem:.2f} GB")
            y_pred = self.forward(stem)
            loss = self.loss_function(y_pred, y_true)
            #logging.debug(f"Loss: {loss.item():.4f}")
            if not torch.isfinite(loss):
                #logging.error("Non-finite loss detected")
                raise ValueError("Non-finite loss")
            self.log("loss", loss, on_epoch=True, on_step=False, prog_bar=True)
            precision, recall, f1, dice = self.calc_metrics(y_pred, y_true, step="train")
            acc_stages = self.train_acc_stages(y_pred, y_true)
            acc_stages_dict = {f"train_S{s+1}_acc": acc_stages[s] for s in range(len(acc_stages))}
            acc_stages_dict["train_acc"] = acc_stages_dict.pop(f"train_S{len(acc_stages)}_acc")
            self.log_dict(acc_stages_dict, on_epoch=True, on_step=False)
            self.train_step_outputs.append({"loss": loss, "precision": precision, "recall": recall, "f1": f1, "dice": dice})
            torch.cuda.empty_cache()
            return {"loss": loss, "precision": precision, "recall": recall, "f1": f1, "dice": dice}
        except Exception as e:
            logging.error(f"Training step failed: {str(e)}")
            raise
    
    def on_train_epoch_end(self):
        self.log_metrics(self.train_step_outputs, step="train")
        self.train_step_outputs.clear()
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"End of training epoch {self.current_epoch}, GPU memory before clear: {mem:.2f} GB")
            torch.cuda.empty_cache()
            mem = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"End of training epoch {self.current_epoch}, GPU memory after clear: {mem:.2f} GB")
    
    def validation_step(self, batch, batch_idx):
        try:
            stem, y_true = batch
            class_counts = torch.bincount(y_true.view(-1), minlength=7)
            #logging.debug(f"Validation Batch {batch_idx}, Class counts: {class_counts.tolist()}")
            y_pred = self.forward(stem)
            val_loss = self.loss_function(y_pred, y_true)
            self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False)
            precision, recall, f1, dice = self.calc_metrics(y_pred, y_true, step="val")
            self.val_acc_stages(y_pred, y_true)
            acc_stages = self.val_acc_stages.compute()
            metric_dict = {f"val_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
            metric_dict["val_acc"] = metric_dict.pop(f"val_S{len(acc_stages)}_acc")
            self.log_dict(metric_dict, on_epoch=True, on_step=False)
            metric_dict["precision"] = precision
            metric_dict["recall"] = recall
            metric_dict["f1"] = f1
            metric_dict["dice"] = dice
            self.val_step_outputs.append(metric_dict)
            torch.cuda.empty_cache()
            return metric_dict
        except Exception as e:
            logging.error(f"Validation step failed: {str(e)}")
            raise
    
    def on_validation_epoch_end(self):
        val_acc_stage_last_epoch = torch.stack([o["val_acc"] for o in self.val_step_outputs]).mean()
        if val_acc_stage_last_epoch > self.max_acc_last_stage["acc"]:
            self.max_acc_last_stage["acc"] = val_acc_stage_last_epoch
            self.max_acc_last_stage["epoch"] = self.current_epoch
        self.log("val: max acc last Stage", self.max_acc_last_stage["acc"], on_epoch=True)
        self.log_metrics(self.val_step_outputs, step="val")
        self.val_step_outputs.clear()
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"End of validation epoch {self.current_epoch}, GPU memory before clear: {mem:.2f} GB")
            torch.cuda.empty_cache()
            mem = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"End of validation epoch {self.current_epoch}, GPU memory after clear: {mem:.2f} GB")
    
    def test_step(self, batch, batch_idx):
        try:
            stem, y_true = batch
            y_pred = self.forward(stem)
            test_loss = self.loss_function(y_pred, y_true)
            self.log("test_loss", test_loss, on_epoch=True, prog_bar=True, on_step=False)
            precision, recall, f1, dice = self.calc_metrics(y_pred, y_true, step="test")
            self.val_acc_stages(y_pred, y_true)
            acc_stages = self.val_acc_stages.compute()
            metric_dict = {f"test_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
            metric_dict["test_acc"] = metric_dict.pop(f"test_S{len(acc_stages)}_acc")
            self.log_dict(metric_dict, on_epoch=True, on_step=False)
            metric_dict["precision"] = precision
            metric_dict["recall"] = recall
            metric_dict["f1"] = f1
            metric_dict["dice"] = dice
            self.test_step_outputs.append(metric_dict)
            torch.cuda.empty_cache()
            return metric_dict
        except Exception as e:
            logging.error(f"Test step failed: {str(e)}")
            raise
    
    def on_test_epoch_end(self):
        test_acc = torch.stack([o["test_acc"] for o in self.test_step_outputs]).mean()
        self.log("test_acc", test_acc, on_epoch=True)
        self.log_metrics(self.test_step_outputs, step="test")
        self.test_step_outputs.clear()
        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0)
        return optimizer
    
    def __dataloader(self, split=None):
        dataset = self.dataset[split]
        should_shuffle = False  # Disable shuffling for all splits due to sequential data
        train_sampler = None
        if self.trainer.strategy == "ddp":
            train_sampler = DistributedSampler(dataset)
            should_shuffle = False
        logging.info(f"split: {split} - shuffle: {should_shuffle}, sampler: {train_sampler}")
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader
    
    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        #logging.info(f"Training data loader called - size: {len(dataloader.dataset)}, sampler: {type(dataloader.sampler).__name__ if dataloader.sampler else 'None'}")
        return dataloader
    
    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        #logging.info(f"Validation data loader called - size: {len(dataloader.dataset)}, sampler: {type(dataloader.sampler).__name__ if dataloader.sampler else 'None'}")
        return dataloader
    
    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        #logging.info(f"Test data loader called - size: {len(dataloader.dataset)}, sampler: {type(dataloader.sampler).__name__ if dataloader.sampler else 'None'}")
        return dataloader
    
    @staticmethod
    def add_module_specific_args(parser):
        regressiontcn = parser.add_argument_group(title='regression tcn specific args options')
        regressiontcn.add_argument("--learning_rate", default=0.001, type=float)
        regressiontcn.add_argument("--optimizer_name", default="adam", type=str)
        regressiontcn.add_argument("--batch_size", default=4, type=int)
        return parser