#MSTCN Training
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from libs.model_loader import feature_dir, mstcn_epoch, mstcn_checkpoint
from libs.featureloader import phase2label, Cholec80FeatureDataset
from models.mstcn import MSTCN
import os


# Setup
dataset = Cholec80FeatureDataset(feature_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Group features per video
from collections import defaultdict
video_dict = defaultdict(lambda: {'features': [], 'labels': []})
for f in os.listdir(feature_dir):
    parts = f.split("_")
    key = parts[1]  # videoNo
    video_dict[key]['features'].append(np.load(os.path.join(feature_dir, f)))
    video_dict[key]['labels'].append(phase2label[parts[-1].replace(".npy", "")])

# Convert to tensors
video_data = []
for vid, content in video_dict.items():
    feat_stack = np.stack(content['features'])  # [T, 768] or possibly [T, 1, 768]
    if feat_stack.ndim == 3:
        feat_stack = feat_stack.squeeze(1)  # Remove singleton dim if necessary
    features = torch.tensor(feat_stack, dtype=torch.float32).unsqueeze(0)  # [1, T, 768]
    labels = torch.tensor(content['labels'], dtype=torch.long).unsqueeze(0)  # [1, T]
    video_data.append((features, labels))

# Initialize model
model = MSTCN(input_dim=768, num_classes=7)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.train()

# Training loop
for epoch in range(mstcn_epoch):
    for features, labels in video_data:
        optimizer.zero_grad()
        print(f"Feature shape: {features.shape}")
        outputs = model(features)  # [num_stages, B, T, num_classes]
        loss = 0
        for out in outputs:
            loss += F.cross_entropy(out.squeeze(0), labels.squeeze(0))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

torch.save(model.state_dict(), mstcn_checkpoint)
print("MSTCN Checkpoint Saved.")