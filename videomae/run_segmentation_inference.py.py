import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import decord
from tqdm import tqdm
from datetime import datetime
decord.bridge.set_bridge("torch")

from modeling_finetune import vit_base_patch16_224

# ---------------------------
# Configuration
# ---------------------------
num_classes = 7  # Number of Cholec80 phases
num_frames = 16
stride = 16  # Slide by 4 frames
tubelet_size = 2
image_size = 224

# Direct mapping from annotation file phase names to class indices
phase_to_index = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6
}

# ---------------------------
# Argument Parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="VideoMAE Full Video Inference for Surgical Phase Recognition")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned model checkpoint (.pth)")
    parser.add_argument("--video", type=str, required=True, help="Path to full-length video")
    parser.add_argument("--annotation", type=str, required=True, help="Path to phase annotation .txt file")
    parser.add_argument("--output_plot", type=str, default="phase_timeline.png", help="Output image filename")
    return parser.parse_args()

# ---------------------------
# Annotation Parsing
# ---------------------------
def parse_annotations(annotation_path):
    times = []
    phases = []
    with open(annotation_path, 'r') as f:
        next(f)  # Skip header row (Frame\tPhase)
        for line in f:
            try:
                time_str, phase = line.strip().split('\t')
                # Convert HH:MM:SS.sss to minutes
                dt = datetime.strptime(time_str, "%H:%M:%S.%f")
                minutes = dt.hour * 60 + dt.minute + dt.second / 60 + dt.microsecond / (60 * 1e6)
                # Map annotation phase to class index
                if phase not in phase_to_index:
                    print(f"Warning: Unknown phase '{phase}' in annotation file, skipping line.")
                    continue
                phase_idx = phase_to_index[phase]
                times.append(minutes)
                phases.append(phase_idx)
            except ValueError as e:
                print(f"Skipping invalid line: {line.strip()} (Error: {e})")
                continue
    if not times:
        raise ValueError("No valid annotations found in the file.")
    
    # Convert to arrays and sort by time
    times = np.array(times)
    phases = np.array(phases, dtype=int)
    sort_indices = np.argsort(times)
    times = times[sort_indices]
    phases = phases[sort_indices]
    
    # Group into segments
    segments = []
    start_time = times[0]
    current_phase = phases[0]
    for i in range(1, len(times)):
        if phases[i] != current_phase:
            # End of current segment
            segments.append({
                'start_time': start_time,
                'end_time': times[i],
                'phase_idx': current_phase
            })
            # Start new segment
            start_time = times[i]
            current_phase = phases[i]
    # Add final segment
    segments.append({
        'start_time': start_time,
        'end_time': times[-1],
        'phase_idx': current_phase
    })
    
    # Debugging: Print segments
    print(f"Parsed {len(segments)} ground truth segments:")
    for i, seg in enumerate(segments):
        print(f"Segment {i}: start={seg['start_time']:.2f}, end={seg['end_time']:.2f}, phase={seg['phase_idx']}")
    
    return segments

# ---------------------------
# Model Initialization
# ---------------------------
def initialize_model():
    model = vit_base_patch16_224(
        pretrained=False,
        num_classes=num_classes,
        all_frames=num_frames,
        tubelet_size=tubelet_size,
        img_size=image_size,
        use_mean_pooling=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False
    )
    return model

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(checkpoint["model"])
    except KeyError:
        model.load_state_dict(checkpoint)
    return model

# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------------------------
# Sliding Inference
# ---------------------------
def sliding_window_inference(model, video_path, device, stride=4):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    predictions = []
    
    # Calculate total number of windows for progress bar
    total_windows = (total_frames - num_frames + 1) // stride + (1 if (total_frames - num_frames) % stride else 0)
    
    # Add progress bar with tqdm
    for start in tqdm(range(0, total_frames - num_frames + 1, stride), total=total_windows, desc="Processing video"):
        indices = list(range(start, start + num_frames))
        frames = vr.get_batch(indices).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = torch.stack([transform(Image.fromarray(f.permute(1, 2, 0).numpy())) for f in frames])
        video_tensor = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, 3, T, H, W]

        with torch.no_grad():
            logits = model(video_tensor)
            pred_class = logits.argmax(dim=-1).item()
        predictions.append(pred_class)
    
    return predictions, total_frames

# ---------------------------
# Error Calculation
# ---------------------------
def calculate_error(predictions, gt_segments, stride, total_frames):
    # Create inverse mapping for display
    index_to_phase = {v: k for k, v in phase_to_index.items()}
    
    # Calculate ground truth phase durations (in minutes)
    gt_durations = {i: 0.0 for i in range(num_classes)}
    for seg in gt_segments:
        duration = seg['end_time'] - seg['start_time']
        gt_durations[seg['phase_idx']] += duration
    
    # Calculate predicted phase durations (in minutes)
    pred_durations = {i: 0.0 for i in range(num_classes)}
    for i, pred in enumerate(predictions):
        # Each prediction covers 'stride' frames (seconds)
        duration = stride / 60  # Convert to minutes
        pred_durations[pred] += duration
    
    # Calculate absolute duration errors
    duration_errors = {
        index_to_phase[i]: abs(gt_durations[i] - pred_durations[i])
        for i in range(num_classes)
    }
    
    # Calculate mean absolute error (MAE) across phases
    total_error = sum(duration_errors.values())
    mae = total_error / num_classes
    
    # Print durations for debugging
    print("Ground Truth Phase Durations (minutes):")
    for i in range(num_classes):
        print(f"  {index_to_phase[i]}: {gt_durations[i]:.2f}")
    print("Predicted Phase Durations (minutes):")
    for i in range(num_classes):
        print(f"  {index_to_phase[i]}: {pred_durations[i]:.2f}")
    
    return mae, duration_errors

# ---------------------------
# Plotting
# ---------------------------
def plot_phase_timeline(predictions, gt_segments, phase_to_index, stride, total_frames, output_path="phase_timeline.png"):
    color_map = plt.cm.get_cmap("tab20", len(phase_to_index))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Create inverse mapping for display
    index_to_phase = {v: k for k, v in phase_to_index.items()}

    # Plot ground truth (top)
    print(f"Plotting {len(gt_segments)} ground truth segments:")
    for i, seg in enumerate(gt_segments):
        start_time = seg['start_time']
        width = seg['end_time'] - seg['start_time']
        phase_idx = seg['phase_idx']
        if width <= 0:
            print(f"Skipping invalid segment {i}: start={start_time:.2f}, width={width:.2f}, phase={index_to_phase[phase_idx]}")
            continue
        print(f"Segment {i}: start={start_time:.2f}, width={width:.2f}, phase={index_to_phase[phase_idx]}")
        ax1.barh(y=0, width=width, left=start_time, height=0.5, color=color_map(phase_idx), edgecolor='black', linewidth=0.1)

    # Plot predictions (bottom)
    for i, pred in enumerate(predictions):
        start_time = (i * stride) / 60  # Convert frames to minutes
        width = stride / 60  # Segment width in minutes
        ax2.barh(y=0, width=width, left=start_time, height=0.5, color=color_map(pred), edgecolor='black', linewidth=0.1)

    # Set display limits and labels
    ax2.set_xlim(0, total_frames / 60)  # Total frames converted to minutes
    ax1.set_ylim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("Time (minutes)")
    ax1.set_title("Ground Truth Surgical Phase Timeline")
    ax2.set_title("Predicted Surgical Phase Timeline")

    # Create legend
    legend_patches = [mpatches.Patch(color=color_map(i), label=index_to_phase[i]) for i in range(num_classes)]
    ax1.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize="small")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()  # Show in console
    print(f"Saved timeline plot to {output_path}")

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    # Parse annotations
    gt_segments = parse_annotations(args.annotation)

    model = initialize_model()
    model = load_checkpoint(model, args.checkpoint)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions, total_frames = sliding_window_inference(model, args.video, device, stride=stride)

    # Calculate and print error
    mae, duration_errors = calculate_error(predictions, gt_segments, stride, total_frames)
    print(f"Mean Absolute Error (minutes): {mae:.2f}")
    print("Absolute Duration Errors (minutes):")
    for phase, error in duration_errors.items():
        print(f"  {phase}: {error:.2f}")

    plot_phase_timeline(predictions, gt_segments, phase_to_index, stride, total_frames, args.output_plot)

if __name__ == "__main__":
    main()
