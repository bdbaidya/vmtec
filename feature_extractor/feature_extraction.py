#Feature Extraction
import torch
from libs.model_loader import model
from utils.config_loader import config
from libs.dataloader import Cholec80Dataset
from torch.utils.data import DataLoader
import numpy as np
import os

data_path = config["processed_clip_dir"]
feature_saving_path = config["feature_save_dir"]
dataset = Cholec80Dataset(data_path)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

os.makedirs(feature_saving_path, exist_ok=True)

with torch.no_grad():
    for clips, video_name in loader:
        clips = clips.squeeze(0)  # [num_clips, C, T, H, W]
        video_name = video_name[0]
        features = []

        for clip in clips:
            clip = clip.unsqueeze(0).cuda()  # [1, C, T, H, W]
            feat = model.module.forward_features(clip) # [1, 768]
            features.append(feat.squeeze(0).cpu().numpy())

        features = np.stack(features)  # [num_clips, 768]
        np.save(os.path.join(feature_saving_path, video_name.replace(".mp4", ".npy")), features)
        print(f"Saved features for {video_name}")