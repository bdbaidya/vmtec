import torch
from models.vmae import vit_base_patch16_224
import yaml

with open("../config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Access the feature_save_dir path
vmae_checkpoint = config["videomae_model_path"]
mstcn_epoch = config["mstcn_epoch"]
mstcn_checkpoint = config["mstcn_checkpoint"]
# Load model
model = vit_base_patch16_224()
checkpoint = torch.load(vmae_checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)

# Set model to evaluation and disable classification head
model.head = torch.nn.Identity()
model.eval().cuda()