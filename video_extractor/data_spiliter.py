import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import re
from tqdm import tqdm
from utils.config_loader import data_path

# Set the path to your source folder
source_folder = data_path


# Set destination folders
train_folder = os.path.join(source_folder, "train")
val_folder = os.path.join(source_folder, "val")
test_folder = os.path.join(source_folder, "test")

# Create destination folders if they don't exist
for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# Regex pattern to extract video number
pattern = re.compile(r"video(\d+)", re.IGNORECASE)

# Filter and count eligible files first
video_files = [f for f in os.listdir(source_folder) if f.endswith(".npy") and pattern.search(f)]

# Move files with a progress bar
for filename in tqdm(video_files, desc="Organizing videos", unit="file"):
    match = pattern.search(filename)
    if match:
        video_num = int(match.group(1))
        src_path = os.path.join(source_folder, filename)
        if 1 <= video_num <= 40:
            dst_folder = train_folder
        elif 41 <= video_num <= 60:
            dst_folder = val_folder
        elif 61 <= video_num <= 80:
            dst_folder = test_folder
        else:
            continue  # Ignore videos outside 1–80 range
        shutil.move(src_path, os.path.join(dst_folder, filename))