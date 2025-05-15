#Feature extraction

import os, yaml, torch
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.vmae import VideoMAEWrapper
from libs.dataloader import Cholec80ClipDataset


cfg = yaml.safe_load("./config.yml")
device = torch.device(cfg["device"])

model = VideoMAEWrapper().to(device)
model.load_state_dict(torch.load(cfg["videomae_model_path"]))
model.eval()

dataset = Cholec80ClipDataset(cfg["preprocessed_clip_dir"])
loader = DataLoader(dataset, batch_size=1, shuffle=False)

save_dir = cfg["feature_save_dir"]
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    for i, (clip, _) in enumerate(tqdm(loader)):
        clip = clip.to(device)
        features = model.encoder(clip).squeeze(0).cpu()  # [T, D]
        video_name = os.path.basename(dataset.clip_paths[i]).replace(".mp4", ".pkl")
        with open(os.path.join(save_dir, video_name), "wb") as f:
            pickle.dump(features, f)
