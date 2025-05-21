#!/bin/bash
#SBATCH --job-name=train_mstcn       # Job name
#SBATCH --output=./feat_ext_%j.out   # Standard output log
#SBATCH --error=./feat_ext_%j.err    # Standard error log
#SBATCH --gres=gpu:2                        # Request 2 GPUs
#SBATCH --nodes=1                          # Single node
#SBATCH --ntasks-per-node=1                # One task per node
#SBATCH --cpus-per-task=8                  # CPU cores per task (adjust as needed)
#SBATCH --mem=100G                          # Memory per node
#SBATCH --mail-type=END,FAIL               # Notifications on job end/fail
#SBATCH --mail-user=your.email@example.com # Your email


# Activate your virtual environment if you have one
source /mnt/ceph/tco/TCO-Students/Homes/debashis/environment/conda/bin/activate

# Install any required packages if not installed (optional)
# pip install --user -r requirements.txt

# Set environment variables (optional)
#export OMP_NUM_THREADS=2


# Run the training script with the config file
python train_multistage_memory_up.py --config ./conf/config_mstcn_clipped.yml
