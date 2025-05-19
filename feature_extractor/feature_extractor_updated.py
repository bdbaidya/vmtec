import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, parent_dir) 

import torch
from libs.model_loader import model
from utils.config_loader import config
from libs.dataloader import Cholec80Dataset
from torch.utils.data import DataLoader
import numpy as np

def extract_videomae_features_from_folder(input_folder, output_folder, seq_len=16, frame_step=1):
    """
    Extract VideoMAE features from a single folder of video clips and save all features into one output folder.
    Args:
        input_folder: Path to folder containing video clips
        output_folder: Path to save extracted feature .npy files
        seq_len: Number of clips per sequence (16 for 16 frames at 1 FPS)
        frame_step: Step between clips (1 for max coverage)
    """
    os.makedirs(output_folder, exist_ok=True)

    dataset = Cholec80Dataset(input_folder)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    label_map = {
        'Preparation': 0,
        'CalotTriangleDissection': 1,
        'ClippingCutting': 2,
        'GallbladderDissection': 3,
        'GallbladderPackaging': 4,
        'CleaningCoagulation': 5,
        'GallbladderRetraction': 6
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            clips = batch[0].squeeze(0)  # [num_clips, C, T, H, W]
            video_name = batch[1][0]
            num_clips = clips.shape[0]
            
            label_name = video_name.split('_')[-1].replace('.mp4', '') if '_' in video_name else None
            if label_name not in label_map:
                print(f"Warning: Unknown label {label_name} for {video_name}")
                continue
            label_idx = label_map[label_name]
            video_base = video_name.replace(f"_{label_name}.mp4", "")

            for start_idx in range(0, num_clips, seq_len * frame_step):
                end_idx = min(start_idx + seq_len * frame_step, num_clips)
                seq_indices = list(range(start_idx, end_idx, frame_step))
                seq_clips = clips[seq_indices]

                features = []
                current_len = seq_clips.shape[0]
                for clip in seq_clips:
                    clip = clip.unsqueeze(0).cuda()
                    feat = model.module.forward_features(clip)
                    features.append(feat.squeeze(0).cpu().numpy())

                if current_len < seq_len:
                    last_feat = features[-1] if features else np.zeros(768)
                    features.extend([last_feat] * (seq_len - current_len))

                features = np.stack(features)  # [seq_len, 768]

                npy_name = f"{video_base}_{label_name}.npy"
                np.save(os.path.join(output_folder, npy_name), features)
                print(f"Saved features: {npy_name} | shape: {features.shape}")


if __name__ == "__main__":
    input_folder = config["processed_clip_dir"]      # Update in config or hardcode path
    output_folder = config["feature_save_dir"]    # All features saved here

    extract_videomae_features_from_folder(input_folder, output_folder, seq_len=16, frame_step=1)