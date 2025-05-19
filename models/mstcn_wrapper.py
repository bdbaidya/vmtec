import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningModule
from torchmetrics.classification import Precision, Recall
from utils.metric_helper import AccuracyStages
from torch import nn
import numpy as np
import logging



class MSTCNDataset(Dataset):
    def __init__(self, folder_path, label_map):
        """
        folder_path: str, path to train/val/test folder with .npy files
        label_map: dict {phase_name (str) : class_index (int)}
        """
        self.folder_path = folder_path
        self.label_map = label_map
        self.class_labels = list(label_map.keys())  # For logging per-class metrics
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        self.file_list.sort()  # Optional but good for reproducibility

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        # File format: "index_videoName_labelName.npy"
        label_name = file_name.split('_')[-1].replace('.npy', '')
        label_idx = self.label_map[label_name]

        feature_path = os.path.join(self.folder_path, file_name)
        features = np.load(feature_path)  # Shape: [seq_len, 768], e.g., [16, 768]

        # Verify expected shape
        if features.shape != (16, 768):
            raise ValueError(f"Unexpected feature shape {features.shape} for {file_name}")

        features = torch.tensor(features, dtype=torch.float)  # Shape: [16, 768]
        seq_len = features.shape[0]  # seq_len=16
        label_tensor = torch.full((seq_len,), label_idx, dtype=torch.long)  # Shape: [16]

        return features.transpose(0, 1), label_tensor  # Shape: [768, 16], [16]

def format_classification_input(y_pred, y_true, threshold=0.5):
    """
    Format predictions and labels for classification metrics.
    Args:
        y_pred: Tensor of shape [batch, num_classes, seq_len], output probabilities
        y_true: Tensor of shape [batch, seq_len], true class indices
        threshold: Float, threshold for binary classification (not used here since multiclass)
    Returns:
        y_pred: Tensor of shape [batch * seq_len], predicted class indices
        y_true: Tensor of shape [batch * seq_len], true class indices
    """
    y_pred = torch.as_tensor(y_pred)
    y_true = torch.as_tensor(y_true)
    
    y_pred = torch.argmax(y_pred, dim=1)  # Get class indices
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    return y_pred, y_true

class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset, weights):
        super(TeCNO, self).__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = self.hparams.batch_size
        self.dataset = dataset
        self.model = model
        self.weights_train = np.asarray(weights["train"])
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.weights_train).float())
        self.init_metrics()
        # Initialize lists to store step outputs
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

    def init_metrics(self):
        self.train_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.val_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.max_acc_last_stage = {"epoch": 0, "acc": 0}
        self.max_acc_global = {"epoch": 0, "acc": 0, "stage": 0, "last_stage_max_acc_is_global": False}

        self.precision_metric = Precision(num_classes=7, average='none', task='multiclass')
        self.recall_metric = Recall(num_classes=7, average='none', task='multiclass')

    def forward(self, x):
        # Input x: [batch_size, 768, seq_len], e.g., [batch_size, 768, 16]
        y_classes = self.model.forward(x)  # Shape: [num_stages, batch_size, num_classes, seq_len]
        y_classes = torch.softmax(y_classes, dim=2)
        return y_classes

    def loss_function(self, y_classes, labels):
        stages = y_classes.shape[0]
        clc_loss = 0
        for j in range(stages):
            p_classes = y_classes[j].squeeze().transpose(1, 0)  # [batch_size, num_classes, seq_len]
            ce_loss = self.ce_loss(p_classes, labels.squeeze())  # labels: [batch_size, seq_len]
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

    def calc_precision_and_recall(self, y_pred, y_true, step="val"):
        y_max_pred, y_true = format_classification_input(y_pred[-1], y_true, threshold=0.5)
        precision = self.precision_metric(y_max_pred, y_true)
        recall = self.recall_metric(y_max_pred, y_true)
        return precision, recall

    def log_precision_and_recall(self, precision, recall, step):
        for n, p in enumerate(precision):
            if not p.isnan():
                self.log(f"{step}_precision_{self.dataset['train'].class_labels[n]}", p, on_step=True, on_epoch=True)
        for n, p in enumerate(recall):
            if not p.isnan():
                self.log(f"{step}_recall_{self.dataset['train'].class_labels[n]}", p, on_step=True, on_epoch=True)

    def log_average_precision_recall(self, outputs, step="val"):
        precision_list = [o["precision"] for o in outputs]
        recall_list = [o["recall"] for o in outputs]
        x = torch.stack(precision_list)
        y = torch.stack(recall_list)
        phase_avg_precision = [torch.mean(x[~x[:, n].isnan(), n]) for n in range(x.shape[1])]
        phase_avg_recall = [torch.mean(y[~y[:, n].isnan(), n]) for n in range(y.shape[1])]
        phase_avg_precision = torch.stack(phase_avg_precision)
        phase_avg_recall = torch.stack(phase_avg_recall)
        phase_avg_precision_over_video = phase_avg_precision[~phase_avg_precision.isnan()].mean()
        phase_avg_recall_over_video = phase_avg_recall[~phase_avg_recall.isnan()].mean()
        self.log(f"{step}_avg_precision", phase_avg_precision_over_video, on_epoch=True, on_step=False)
        self.log(f"{step}_avg_recall", phase_avg_recall_over_video, on_epoch=True, on_step=False)

    def training_step(self, batch, batch_idx):
        stem, y_true = batch  # stem: [batch_size, 768, 16], y_true: [batch_size, 16]
        y_pred = self.forward(stem)
        loss = self.loss_function(y_pred, y_true)
        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="train")
        self.log_precision_and_recall(precision, recall, step="train")
        acc_stages = self.train_acc_stages(y_pred, y_true)
        acc_stages_dict = {f"train_S{s+1}_acc": acc_stages[s] for s in range(len(acc_stages))}
        acc_stages_dict["train_acc"] = acc_stages_dict.pop(f"train_S{len(acc_stages)}_acc")
        self.log_dict(acc_stages_dict, on_epoch=True, on_step=False)
        self.train_step_outputs.append({"loss": loss, "precision": precision, "recall": recall})
        return {"loss": loss, "precision": precision, "recall": recall}

    def on_train_epoch_end(self):
        self.log_average_precision_recall(self.train_step_outputs, step="train")
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        stem, y_true = batch
        y_pred = self.forward(stem)
        val_loss = self.loss_function(y_pred, y_true)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False)
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="val")
        self.log_precision_and_recall(precision, recall, step="val")
        self.val_acc_stages(y_pred, y_true)
        acc_stages = self.val_acc_stages.compute()
        metric_dict = {f"val_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
        metric_dict["val_acc"] = metric_dict.pop(f"val_S{len(acc_stages)}_acc")
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        self.val_step_outputs.append(metric_dict)
        return metric_dict

    def on_validation_epoch_end(self):
        val_acc_stage_last_epoch = torch.stack([o["val_acc"] for o in self.val_step_outputs]).mean()
        if val_acc_stage_last_epoch > self.max_acc_last_stage["acc"]:
            self.max_acc_last_stage["acc"] = val_acc_stage_last_epoch
            self.max_acc_last_stage["epoch"] = self.current_epoch
        self.log("val: max acc last Stage", self.max_acc_last_stage["acc"], on_epoch=True)
        self.log_average_precision_recall(self.val_step_outputs, step="val")
        self.val_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        stem, y_true = batch
        y_pred = self.forward(stem)
        test_loss = self.loss_function(y_pred, y_true)
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True, on_step=False)
        precision, recall = self.calc_precision_and_recall(y_pred, y_true, step="test")
        self.log_precision_and_recall(precision, recall, step="test")
        self.val_acc_stages(y_pred, y_true)
        acc_stages = self.val_acc_stages.compute()
        metric_dict = {f"test_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))}
        metric_dict["test_acc"] = metric_dict.pop(f"test_S{len(acc_stages)}_acc")
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        self.test_step_outputs.append(metric_dict)
        return metric_dict

    def on_test_epoch_end(self):
        test_acc = torch.stack([o["test_acc"] for o in self.test_step_outputs]).mean()
        self.log("test_acc", test_acc, on_epoch=True)
        self.log_average_precision_recall(self.test_step_outputs, step="test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

    def __dataloader(self, split=None):
        dataset = self.dataset[split]
        should_shuffle = False
        if split == "train":
            should_shuffle = True
        train_sampler = None
        if self.trainer.strategy == "ddp":
            train_sampler = DistributedSampler(dataset)
            should_shuffle = False
        print(f"split: {split} - shuffle: {should_shuffle}")
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
        logging.info("training data loader called - size: {}".format(len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(len(dataloader.dataset)))
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called - size: {}".format(len(dataloader.dataset)))
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):
        regressiontcn = parser.add_argument_group(title='regression tcn specific args options')
        regressiontcn.add_argument("--learning_rate", default=0.001, type=float)
        regressiontcn.add_argument("--optimizer_name", default="adam", type=str)
        regressiontcn.add_argument("--batch_size", default=1, type=int)
        return parser