#Pretraining

import torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.vmae import VideoMAEWrapper
from libs.dataloader import Cholec80ClipDataset


cfg = yaml.safe_load("./config.yml")
device = torch.device(cfg["device"])

dataset = Cholec80ClipDataset(cfg["preprocessed_clip_dir"])
loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)

model = VideoMAEWrapper().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

model.train()

scaler = torch.cuda.amp.GradScaler()  
for epoch in range(cfg["epochs"]):
    torch.cuda.empty_cache()
    total_loss = 0
    for clips, _ in tqdm(loader):
        clips = clips.to(device)
        masked = clips.clone()
        masked[:, :, :, 64:128, 64:128] = 0  # crude spatial masking

        with torch.cuda.amp.autocast():  # NEW
            output = model(masked)
            target = model.encoder(clips).detach()
            loss = criterion(output, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()     # NEW
        scaler.step(optimizer)            # NEW
        scaler.update()                   # NEW

        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss={total_loss:.4f}")
torch.save(model.state_dict(), cfg["videomae_model_path"])
