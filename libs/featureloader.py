import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


phase2label = {
    'Preparation': 0,
    'CalotTriangleDissection': 1,
    'ClippingCutting': 2,
    'GallbladderDissection': 3,
    'GallbladderPackaging': 4,
    'CleaningCoagulation': 5,
    'GallbladderRetraction': 6
}


class Cholec80FeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_paths = sorted([
            os.path.join(feature_dir, f)
            for f in os.listdir(feature_dir) if f.endswith('.npy')
        ])
        self.labels = [phase2label[f.split("_")[-1].replace(".npy", "")] for f in self.feature_paths]

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feature = np.load(self.feature_paths[idx])
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
