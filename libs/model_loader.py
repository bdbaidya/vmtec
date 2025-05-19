import torch
from models.vmae import vit_base_patch16_224
from utils.config_loader import vmae_checkpoint
import os
# Load model
model = vit_base_patch16_224()
checkpoint = torch.load(vmae_checkpoint, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)

# Set model to evaluation and disable classification head
model.head = torch.nn.Identity()
model = torch.nn.DataParallel(model).cuda()
model.eval()