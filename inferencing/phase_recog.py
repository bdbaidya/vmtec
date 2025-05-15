import os
import cv2
import torch
import pickle
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange
from moviepy.editor import VideoFileClip
from PIL import Image
from models.vmae import VideoMAEWrapper
from models.mstcn import MSTCN

# Define label map
label_map = {
    0: "Preparation",
    1: "CalotTriangleDissection",
    2: "ClippingCutting",
    3: "GallbladderDissection",
    4: "GallbladderPackaging",
    5: "CleaningCoagulation",
    6: "GallbladderRetraction"
}

# Define transform
video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# Step 1: Load video and split into overlapping 16-frame clips at 1 FPS
def load_video_clips(video_path, clip_len=16):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    every_nth_frame = int(fps)

    frames = []
    all_clips = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % every_nth_frame == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = video_transform(frame)
            frames.append(frame)

        if len(frames) == clip_len:
            clip = torch.stack(frames, dim=0)  # [16, 3, 224, 224]
            all_clips.append(clip)
            frames.pop(0)  # sliding window

        count += 1

    cap.release()
    return all_clips  # list of [16, 3, 224, 224]

# Step 2: Load trained VideoMAE encoder
video_mae_model = VideoMAEWrapper().to(device)
checkpoint = torch.load("/kaggle/working/checkpoints/vmae/videomae.pth", map_location="cpu")
video_mae_model.load_state_dict(checkpoint)
video_mae_model.eval()

# Step 3: Extract features for all clips
def extract_features(clips, model):
    features = []
    device = next(model.parameters()).device  # get the model's device
    for clip in clips:
        clip = clip.unsqueeze(0).to(device)  # [1, 16, 3, 224, 224] on the correct device
        with torch.no_grad():
            feat = model(clip)  # [1, T, D]
        features.append(feat.squeeze(0).cpu())  # [T, D]
    return torch.cat(features, dim=0)  # [total_T, D]


# Step 4: Load trained MS-TCN model
mstcn = MSTCN(input_dim=768, num_classes=7)
mstcn.load_state_dict(torch.load("/kaggle/working/checkpoints/mstcn/mstcn.pth"))
mstcn = mstcn.to(device)
mstcn.eval()

# Step 5: Run inference
def predict_phase(features, mstcn):
    features = features.unsqueeze(0).cuda()  # [1, T, D]
    with torch.no_grad():
        out = mstcn(features)  # [1, T, C]
    preds = torch.argmax(out, dim=-1)[0]  # [T]
    return preds.cpu().numpy()  # list of int labels

# Run the full pipeline
video_path = "/kaggle/input/mstcn-cholec80-data/test_data_mstcn/00001_video01_Preparation.mp4"
clips = load_video_clips(video_path)
features = extract_features(clips, video_mae_model)  # [T, D]
preds = predict_phase(features, mstcn)               # [T]

# Step 6: Print with time info
for i, label_idx in enumerate(preds):
    print(f"Second {i}: {label_map[label_idx]}")
