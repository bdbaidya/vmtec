#Train MSTCN

import os, torch, yaml
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.mstcn import MSTCN

cfg = yaml.safe_load("./config.yml")
device = torch.device(cfg["device"])

class FeatureDataset(Dataset):
    def __init__(self, feature_dir, label_map):
        self.paths = sorted([os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".pkl")])
        #self.labels = [label_map[os.path.basename(p).split("_")[2].split(".")[0]] for p in self.paths]
        self.labels = [label_map[os.path.basename(p).split("_")[2].replace(".pkl", "")] for p in self.paths]


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            features = pickle.load(f)  # [T, D]
        label = torch.full((features.shape[0],), self.labels[idx])  # [T]
        return features, label

label_map = {
    "Preparation": 0, 
    "CalotTriangleDissection": 1, 
    "ClippingCutting": 2, 
    "GallbladderDissection": 3, 
    "GallbladderPackaging": 4, 
    "CleaningCoagulation": 5, 
    "GallbladderRetraction": 6
}

dataset = FeatureDataset(cfg["feature_save_dir"], label_map)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = MSTCN(input_dim=768, num_classes=7).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(cfg["epochs"]):
    total_loss = 0
    for features, labels in tqdm(loader):
        features, labels = features.to(device), labels.to(device)
        output = model(features)
        loss = criterion(output.view(-1, 7), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss={total_loss:.4f}")

torch.save(model.state_dict(), cfg["mstcn_model_path"])
