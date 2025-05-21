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
    print(video_dict)
    return video_dict

def plot_one_video(model, video_dict, device='cpu'):
    video_name = list(video_dict.keys())[0]
    data = video_dict[video_name]

    # Get features and permute to [1, dim, T]
    features = torch.tensor(data['features'][0], dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)  # [1, dim, T]
    gt = np.array(data['labels'])  # [T]

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model.model(features)  # [num_stages, 1, T, num_classes]
        pred_logits = outputs[-1].squeeze(0)  # [T, num_classes]
        pred_classes = pred_logits.argmax(dim=1).cpu().numpy()  # [T]

    # Plotting
    plt.figure(figsize=(15, 4))
    plt.plot(pred_classes, 'r--', label='Prediction')
    plt.plot(gt, 'g-', label='Ground Truth')
    plt.title(f'Prediction vs Ground Truth for Video {video_name}')
    plt.xlabel('Time (clip index)')
    plt.ylabel('Phase Label (0–6)')
    plt.ylim(-1, 7)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(config)

    video_dict = group_features(config.test_dir)
    device = 'cuda' if torch.cuda.is_available() and config.accelerator == 'gpu' else 'cpu'
    
    model.to(device)  # ✅ move model to the correct device

    plot_one_video(model, video_dict, device=device)

    
if __name__ == "__main__":
    main()
