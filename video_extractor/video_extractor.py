import os
import cv2
import glob
import json
from datetime import datetime
from collections import Counter
from tqdm import tqdm  # Import tqdm for progress bars

# Paths for checkpoint and label log
CHECKPOINT_FILE = "clip_extraction_checkpoint.json"
LABEL_FILE = "clip_labels.txt"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"last_video_index": 0, "global_clip_index": 0}

def save_checkpoint(last_video_index, global_clip_index):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_video_index": last_video_index, "global_clip_index": global_clip_index}, f)

def parse_annotation(annotation_path):
    frame_labels = {}
    with open(annotation_path, 'r') as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            time_str, label = line.strip().split('\t')
            try:
                t = datetime.strptime(time_str.split('.')[0], "%H:%M:%S")
                seconds = t.hour * 3600 + t.minute * 60 + t.second
                frame_labels[seconds] = label
            except:
                continue
    return frame_labels

def get_clip_label(start_sec, frame_labels, clip_len=16):
    labels = [frame_labels.get(start_sec + i, None) for i in range(clip_len)]
    labels = [l for l in labels if l is not None]
    return Counter(labels).most_common(1)[0][0] if labels else None

def extract_clips_as_video_1fps(video_path, annotation_path, output_dir, label_log, clip_len=16, stride=4, resize=(224, 224), global_start_index=0):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # Collect frames at 1 FPS
    one_fps_frames = []
    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_index % int(orig_fps) == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            one_fps_frames.append(frame)
        frame_index += 1
    cap.release()

    print(f"[INFO] {video_id}: Extracted {len(one_fps_frames)} 1FPS frames")

    if len(one_fps_frames) < clip_len:
        print(f"[SKIP] Not enough frames in {video_id} for even 1 clip.")
        return global_start_index

    frame_labels = parse_annotation(annotation_path)

    clip_index = global_start_index
    max_start = len(one_fps_frames) - clip_len
    i = 0
    while i <= max_start:
        clip = one_fps_frames[i:i + clip_len]
        if len(clip) < clip_len:
            break

        # Get label for this clip before saving
        label = get_clip_label(i, frame_labels, clip_len)
        if not label:
            print(f"[!] No label for clip starting at frame {i}, skipping.")
            i += stride
            continue

        filename = f"{clip_index:05d}_{video_id}_{label}.mp4"
        out_path = os.path.join(output_dir, filename)

        height, width = resize
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
        for frame in clip:
            out.write(frame)
        out.release()

        # Check if saved clip is complete
        test_cap = cv2.VideoCapture(out_path)
        saved_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        test_cap.release()

        if saved_frames != clip_len:
            print(f"[✗] Incomplete clip deleted: {filename}")
            os.remove(out_path)
        else:
            label_log.write(f"{out_path} {label}\n")
            print(f"[✓] Saved: {filename} → {label}")
            clip_index += 1

        i += stride

    return clip_index

def extract_clips_from_all_videos(video_dir, annotation_dir, output_dir, clip_len=16, stride=4, resize=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    checkpoint = load_checkpoint()
    start_video_index = checkpoint["last_video_index"]
    global_index = checkpoint["global_clip_index"]

    label_log = open(LABEL_FILE, "a")

    # Add tqdm progress bar for video processing
    with tqdm(total=len(video_paths) - start_video_index, desc="Processing videos", unit="video") as pbar:
        for i, video_path in enumerate(video_paths[start_video_index:], start=start_video_index):
            video_name = os.path.basename(video_path)
            video_id = os.path.splitext(video_name)[0]
            annotation_path = os.path.join(annotation_dir, f"{video_id}.txt")

            if not os.path.exists(annotation_path):
                print(f"[!] Missing annotation for {video_name}, skipping.")
                pbar.update(1)
                continue

            try:
                global_index = extract_clips_as_video_1fps(
                    video_path,
                    annotation_path,
                    output_dir,
                    label_log,
                    clip_len,
                    stride,
                    resize,
                    global_start_index=global_index
                )
                save_checkpoint(i + 1, global_index)
            except Exception as e:
                print(f"[✗] Error processing {video_name}: {e}")
                save_checkpoint(i, global_index)
                break

            pbar.update(1)  # Update progress bar

    label_log.close()

if __name__ == "__main__":
    extract_clips_from_all_videos(
        video_dir="/mnt/ceph/tco/TCO-Students/Homes/debashis/data/processed/videos/",
        annotation_dir="/mnt/ceph/tco/TCO-Students/Homes/debashis/data/processed/annotations/",
        output_dir="/mnt/ceph/tco/TCO-Students/Homes/debashis/data/processed/final-data/",
        clip_len=16,
        stride=4,
        resize=(224, 224)
    )
