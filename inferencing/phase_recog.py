import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from timm.models import create_model
#from model import MSTCN  # your MSTCN model class
import matplotlib.pyplot as plt
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Phase Label Mapping
# ------------------------
phase_map = {
    0: "Preparation",
    1: "CalotTriangleDissection",
    2: "ClippingCutting",
    3: "GallbladderDissection",
    4: "GallbladderPackaging",
    5: "CleaningCoagulation",
    6: "GallbladderRetraction"
}

# ------------------------
# Load VideoMAE model
# ------------------------
def load_videomae_model():

    model = vit_base_patch16_224()
    checkpoint = torch.load("/content/drive/My Drive/TUDresden/Research_Project/Model/checkpoints/checkpoint-799.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    return model

# ------------------------
# Load MSTCN model
# ------------------------
def load_mstcn_model():
    model = MSTCN()
    model.load_state_dict(torch.load("/content/drive/My Drive/TUDresden/Research_Project/Model/checkpoints/mstcn_cholec80.pth"))
    model.to(device)
    model.eval()
    return model

# ------------------------
# Preprocess Video Frames
# ------------------------
def preprocess_video(video_path, clip_len=16, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Break video into clips of 16 frames
    clips = []
    for i in range(0, len(frames) - clip_len + 1, clip_len):
        clip = frames[i:i + clip_len]
        clip = [transform(img).unsqueeze(1) for img in clip]  # Cx1xHxW
        clip = torch.cat(clip, dim=1)  # CxTxHxW
        clips.append(clip)

    if not clips:
        raise ValueError("Video too short to create a single 16-frame clip.")

    clips = torch.stack(clips)  # [num_clips, C, T, H, W]
    return clips

# ------------------------
# Extract Features using VideoMAE
# ------------------------
def extract_features(clips, videomae_model):
    features = []
    with torch.no_grad():
        for clip in clips:
            clip = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]
            feat = videomae_model.forward_features(clip)  # [1, 768]
            features.append(feat.squeeze(0).cpu().numpy())
    features = np.stack(features)  # [num_clips, 768]
    return features

# ------------------------
# Predict Phase using MSTCN
# ------------------------
def predict_phase(features, mstcn_model):
    features = torch.tensor(features).unsqueeze(0).float().to(device)  # [1, T, 768]
    with torch.no_grad():
        output = mstcn_model(features)  # [num_stages, 1, T, num_classes]
        preds = output[-1].squeeze(0).argmax(dim=1).cpu().numpy()  # [T]
    return preds

# ------------------------
# Plot Prediction
# ------------------------
def plot_predictions(preds):
    phase_names = [phase_map[p] for p in preds]
    unique_phases = sorted(set(preds))

    plt.figure(figsize=(12, 3))
    plt.plot(preds, label="Predicted Phase")
    plt.yticks(unique_phases, [phase_map[p] for p in unique_phases])
    plt.title("Predicted Surgical Phases")
    plt.xlabel("Clip Index")
    plt.ylabel("Phase")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------
# Main
# ------------------------
def main():
    video_path = "/content/drive/My Drive/TUDresden/Research_Project/Model/mstcn-test-data/test_data_mstcn/00370_video01_GallbladderPackaging.mp4"  # ✅ Change this
    print(f"Processing video: {video_path}")

    videomae_model = load_videomae_model()
    mstcn_model = load_mstcn_model()

    clips = preprocess_video(video_path)
    print(f"Extracted {clips.shape[0]} clips")

    features = extract_features(clips, videomae_model)
    preds = predict_phase(features, mstcn_model)

    print("Predicted Phase Indices:", preds)
    print("Predicted Phase Names:", [phase_map[p] for p in preds])

    plot_predictions(preds)

if __name__ == "__main__":
    main()
