import os
import numpy as np
#folder_path="/mnt/ceph/tco/TCO-Students/Homes/debashis/data/processed/test-feature/"
def check_npy_shapes(folder_path):
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    for f in npy_files:
        data = np.load(os.path.join(folder_path, f))
        print(f"File: {f}, Shape: {data.shape}")

for split in ['train', 'val', 'test']:
    folder = f'/mnt/ceph/tco/TCO-Students/Homes/debashis/data/processed/test_spilit/test-features/{split}'
    print(f"\nChecking {folder}:")
    if os.path.exists(folder):
        check_npy_shapes(folder)
    else:
        print(f"{folder} does not exist")