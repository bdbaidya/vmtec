import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mstcn_wrapper import TeCNO, MSTCNDataset
from models.mstcn import MultiStageModel

label_map = {
    'Preparation': 0,
    'CalotTriangleDissection': 1,
    'ClippingCutting': 2,
    'GallbladderDissection': 3,
    'GallbladderPackaging': 4,
    'CleaningCoagulation': 5,
    'GallbladderRetraction': 6
}

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    from types import SimpleNamespace
    return SimpleNamespace(**config)

def load_model(config):
    hparams_dict = vars(config)
    dummy_dataset = {"train": None, "val": None, "test": None}
    dummy_weights = {"train": torch.ones(len(label_map))}

    model_core = MultiStageModel(config)
    tecno = TeCNO(hparams=hparams_dict, model=model_core, dataset=dummy_dataset, weights=dummy_weights)

    checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_file)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    tecno.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict)
    tecno.eval()
    return tecno

def group_features(test_dir):
    video_dict = defaultdict(lambda: {'features': [], 'labels': []})
    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".npy"):
            continue
        parts = f.split("_")
        if len(parts) < 3:
            continue
        video_id = parts[1]
        label_name = parts[-1].replace(".npy", "")
        label_idx = label_map.get(label_name, -1)
        if label_idx == -1:
            continue
        path = os.path.join(test_dir, f)
        try:
            feat = np.load(path)
            if feat.ndim == 3:
                feat = feat.squeeze(1)
            video_dict[video_id]['features'].append(feat)
            video_dict[video_id]['labels'].append(label_idx)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return video_dict

def plot_one_video(model, video_dict, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for video_id, content in video_dict.items():
            features_np = np.stack(content['features'])  # [T, D]
            labels_np = np.array(content['labels'])      # [T]

            features = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, D]
            labels = torch.tensor(labels_np, dtype=torch.long)  # [T]

            outputs = model.model(features)  # [num_stages, 1, T, num_classes]
            pred = outputs[-1].squeeze(0).argmax(dim=1).cpu().numpy()

            plt.figure(figsize=(12, 4))
            plt.plot(labels_np, label="Ground Truth", color='green')
            plt.plot(pred, label="Prediction", color='red', linestyle='--')
            plt.title(f"Prediction vs Ground Truth for Video {video_id}")
            plt.xlabel("Time (clip index)")
            plt.ylabel("Phase Label (0–6)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            break  # Only show one video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(config)
    video_dict = group_features(config.test_dir)
    device = 'cuda' if torch.cuda.is_available() and config.accelerator == 'gpu' else 'cpu'
    plot_one_video(model, video_dict, device=device)

if __name__ == "__main__":
    main()
