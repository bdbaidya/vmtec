import yaml
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "../conf", "config.yml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Access the feature_save_dir path
vmae_checkpoint = config["videomae_model_path"]
mstcn_epoch = config["mstcn_epoch"]
mstcn_checkpoint = config["mstcn_checkpoint"]
data_path = config["main_data_source_folder"]