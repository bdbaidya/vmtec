from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
from decord import VideoReader, cpu
from PIL import Image

class Cholec80ClipDataset(Dataset):
    def __init__(self, clip_dir, transform=None, clip_len=16):
        self.clip_paths = sorted([
            os.path.join(clip_dir, f)
            for f in os.listdir(clip_dir)
            if f.endswith('.mp4')
        ])
        self.clip_len = clip_len
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        path = self.clip_paths[idx]
        frames = self.read_video_decord(path)
        if len(frames) < self.clip_len:
            raise ValueError(f"Video at {path} has fewer than {self.clip_len} frames.")
        frames = frames[:self.clip_len]
        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return frames_tensor, 0  # dummy label for pretraining

    def read_video_decord(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        frame_indices = list(range(min(len(vr), self.clip_len)))
        frames = [Image.fromarray(vr[i].asnumpy()) for i in frame_indices]
        return frames
