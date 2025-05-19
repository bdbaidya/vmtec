import torch
from libs.model_loader import model
from utils.config_loader import config
from libs.dataloader import Cholec80Dataset
from torch.utils.data import DataLoader
import numpy as np
import os

def extract_videomae_features(split='train', seq_len=16, frame_step=1):
    """
    Extract VideoMAE features for Cholec80 dataset and save as .npy files with shape [seq_len, 768].
    Labels are extracted from video file names.
    Args:
        split: Dataset split ('train', 'val', 'test')
        seq_len: Number of clips per sequence (16 for 16 frames at 1 FPS)
        frame_step: Step size between clips (1 for all clips, 12 to minimize 4-frame overlap)
    """
    # Update paths for the split
    data_path = os.path.join(config["processed_clip_dir"], split)
    feature_saving_path = os.path.join(config["feature_save_dir"], split)
    os.makedirs(feature_saving_path, exist_ok=True)

    # Initialize dataset and loader
    dataset = Cholec80Dataset(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    # Label map for surgical phases
    label_map = {
        'Preparation': 0,
        'CalotTriangleDissection': 1,
        'ClippingCutting': 2,
        'GallbladderDissection': 3,
        'GallbladderPackaging': 4,
        'CleaningCoagulation': 5,
        'GallbladderRetraction': 6
    }
    label_map_reverse = {v: k for k, v in label_map.items()}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            clips = batch[0].squeeze(0)  # [num_clips, C, T, H, W]
            video_name = batch[1][0]  # str
            num_clips = clips.shape[0]

            # Extract label from video name (e.g., 'video05_CalotTriangleDissection.mp4')
            label_name = video_name.split('_')[-1].replace('.mp4', '') if '_' in video_name else None
            if label_name not in label_map:
                print(f"Warning: Unknown label {label_name} for {video_name}")
                continue
            label_idx = label_map[label_name]
            label_name = label_name
            video_name_base = video_name.replace(f"_{label_name}.mp4", "")

            # Process clips in sequences of seq_len
            for start_idx in range(0, num_clips, seq_len * frame_step):
                end_idx = min(start_idx + seq_len * frame_step, num_clips)
                seq_indices = list(range(start_idx, end_idx, frame_step))
                seq_clips = clips[seq_indices]  # [current_len, C, T, H, W]

                # Since all clips come from the same video, they share the same label
                seq_labels = [label_idx] * len(seq_indices)

                # Extract features
                features = []
                current_len = seq_clips.shape[0]
                for clip in seq_clips:
                    clip = clip.unsqueeze(0).cuda()  # [1, C, T, H, W]
                    feat = model.module.forward_features(clip)  # [1, 768]
                    features.append(feat.squeeze(0).cpu().numpy())

                # Pad if sequence is shorter than seq_len
                if current_len < seq_len:
                    last_feat = features[-1] if features else np.zeros(768)
                    features.extend([last_feat] * (seq_len - current_len))

                features = np.stack(features)  # [seq_len, 768]

                # Save features
                #npy_name = f"{start_idx//frame_step}_{video_name_base}_{label_name}.npy"
                npy_name = f"{video_name}_{label_name}.npy"
                np.save(os.path.join(feature_saving_path, npy_name), features)
                print(f"Saved features for {npy_name} with shape {features.shape}")

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")
        extract_videomae_features(split=split, seq_len=16, frame_step=1)