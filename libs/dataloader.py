# Data Loader

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge("torch")

class Cholec80Dataset(Dataset):
    def __init__(self, video_dir, clip_len=16, img_size=224):
        self.video_dir = video_dir
        self.videos = sorted([os.path.join(video_dir, v) for v in os.listdir(video_dir) if v.endswith('.mp4')])
        self.clip_len = clip_len
        self.img_size = img_size

        self.resize = transforms.Resize((img_size, img_size))
        self.normalize = transforms.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vr = VideoReader(self.videos[idx], ctx=cpu(0))
        total_frames = len(vr)
        clips = []

        for start in range(0, total_frames - self.clip_len + 1, 1):
            clip = vr.get_batch(list(range(start, start + self.clip_len)))  # [T, H, W, C]
            clip = clip.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

            # Resize and normalize each frame
            processed = []
            for frame in clip:
                frame = self.resize(frame)
                frame = self.normalize(frame)
                processed.append(frame)

            clip_tensor = torch.stack(processed, dim=1)  # [C, T, H, W]
            clips.append(clip_tensor)

        return torch.stack(clips), os.path.basename(self.videos[idx])
