import decord
import numpy as np
import os
import re
from tqdm import tqdm

def natural_sort_key(s):
    """Natural sorting (e.g., video2.mp4 < video10.mp4)"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]

def check_all_frames(video_path):
    try:
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        bad_frames = []
        
        for i in tqdm(range(total_frames), desc=f"Checking {os.path.basename(video_path)}", leave=False):
            try:
                frame = vr[i].asnumpy()
                if frame.size == 0 or np.all(frame == 0):
                    bad_frames.append(i)
            except Exception as e:
                bad_frames.append((i, str(e)))
        
        if not bad_frames:
            return True, f"All {total_frames} frames OK"
        else:
            return False, f"{len(bad_frames)} bad frames (first 5: {bad_frames[:5]})"
    
    except Exception as e:
        return False, f"Video load failed: {str(e)}"

def check_videos_in_range(folder_path, start_num=23, end_num=80):
    video_files = []
    for num in range(start_num, end_num + 1):
        filename = f"video{num}.mp4"
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            video_files.append(filename)
    
    if not video_files:
        print(f"No videos found between video{start_num}.mp4 and video{end_num}.mp4")
        return
    
    # Sort naturally (in case of non-sequential files)
    video_files.sort(key=natural_sort_key)
    
    print(f"Checking {len(video_files)} videos (video{start_num}.mp4 to video{end_num}.mp4):")
    print(", ".join(video_files))
    
    results = []
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        status, message = check_all_frames(video_path)
        results.append((video_file, status, message))
    
    print("\n=== Results ===")
    for video, status, msg in results:
        print(f"{'✅' if status else '❌'} {video}: {msg}")
    
    passed = sum(1 for _, s, _ in results if s)
    print(f"\nSummary: {passed}/{len(results)} passed")

if __name__ == "__main__":
    folder_path = "/mnt/ceph/tco/TCO-Students/Homes/debashis/data/processed/videos/"
    check_videos_in_range(folder_path, start_num=23, end_num=80)