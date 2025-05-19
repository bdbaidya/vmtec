#!/bin/bash
#SBATCH --job-name=feature_extraction       # Job name
#SBATCH --output=/mnt/ceph/tco/TCO-Students/Homes/debashis/data/feature_extraction/feat_ext_%j.out   # Standard output log
#SBATCH --error=/mnt/ceph/tco/TCO-Students/Homes/debashis/data/feature_extraction/feat_ext_%j.err    # Standard error log
#SBATCH --gres=gpu:2                        # Request 2 GPUs
#SBATCH --nodes=1                          # Single node
#SBATCH --ntasks-per-node=1                # One task per node
#SBATCH --cpus-per-task=8                  # CPU cores per task (adjust as needed)
#SBATCH --mem=64G                          # Memory per node
#SBATCH --mail-type=END,FAIL               # Notifications on job end/fail
#SBATCH --mail-user=your.email@example.com # Your email

# Load modules if needed (uncomment & adjust)
# module load cuda/11.7
# module load python/3.8

# Activate your virtual environment if you have one
source /mnt/ceph/tco/TCO-Students/Homes/debashis/environment/conda/bin/activate

# Install any required packages if not installed (optional)
# pip install --user -r requirements.txt

# Set environment variables (optional)
export OMP_NUM_THREADS=2

python feature_extraction.py
