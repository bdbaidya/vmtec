import os
import shutil
import re
from libs.model_loader import data_path

# Set the path to your source folder
source_folder = data_path

# Set destination folders
train_folder = os.path.join(source_folder, "training")
val_folder = os.path.join(source_folder, "validation")
test_folder = os.path.join(source_folder, "test")

# Create destination folders if they don't exist
for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# Regex pattern to extract video number
pattern = re.compile(r"video(\d+)", re.IGNORECASE)

# Iterate through files
for filename in os.listdir(source_folder):
    if filename.endswith(".mp4"):
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
