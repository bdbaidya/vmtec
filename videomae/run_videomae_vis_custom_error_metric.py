import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
from datasets import DataAugmentationForVideoMAE
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import seaborn as sns

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]   # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script with metrics', add_help=False)
    parser.add_argument('img_path', type=str, help='input video path')
    parser.add_argument('save_path', type=str, help='save video path')
    parser.add_argument('model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth
    )
    return model

def show_images(images, title, row_idx):
    for i in range(min(6, len(images))):
        plt.subplot(3, 6, row_idx * 6 + i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        if row_idx == 0:
            plt.title(f"Frame {i+1}")
    plt.ylabel(title, rotation=0, labelpad=40, fontsize=12)

def compute_metrics(original_video, recon_video):
    """Compute MSE, PSNR, SSIM, and MAE between original and reconstructed video."""
    original = original_video.cpu().numpy()  # Shape: (1, C, T, H, W)
    reconstructed = recon_video.cpu().numpy()

    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    mae_scores = []

    for t in range(original.shape[2]):  # Iterate over frames
        orig_frame = original[0, :, t].transpose(1, 2, 0)  # HWC
        recon_frame = reconstructed[0, :, t].transpose(1, 2, 0)

        # Ensure values are in [0, 1] range
        orig_frame = np.clip(orig_frame, 0, 1)
        recon_frame = np.clip(recon_frame, 0, 1)

        # MSE
        mse = np.mean((orig_frame - recon_frame) ** 2)
        mse_scores.append(mse)

        # PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        psnr_scores.append(psnr)

        # SSIM (specify channel_axis for RGB)
        ssim_val = ssim(orig_frame, recon_frame, data_range=1.0, channel_axis=2)
        ssim_scores.append(ssim_val)

        # MAE
        mae = np.mean(np.abs(orig_frame - recon_frame))
        mae_scores.append(mae)

    return np.array(mse_scores), np.array(psnr_scores), np.array(ssim_scores), np.array(mae_scores)

def plot_metrics(mse_scores, psnr_scores, ssim_scores, mae_scores, save_path):
    """Plot metrics as time series and save to file."""
    frames = np.arange(len(mse_scores))

    # Time series plots
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(frames, mse_scores, label='MSE')
    plt.xlabel('Frame')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error Over Frames')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(frames, psnr_scores, label='PSNR', color='orange')
    plt.xlabel('Frame')
    plt.ylabel('PSNR (dB)')
    plt.title('Peak Signal-to-Noise Ratio Over Frames')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(frames, ssim_scores, label='SSIM', color='green')
    plt.xlabel('Frame')
    plt.ylabel('SSIM')
    plt.title('Structural Similarity Index Over Frames')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(frames, mae_scores, label='MAE', color='red')
    plt.xlabel('Frame')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error Over Frames')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_path}/metrics_plot.jpg")
    plt.close()

def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    with open(args.img_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    duration = len(vr)
    tmp = np.arange(0, 32, 2) + 60
    frame_id_list = tmp.tolist()

    video_data = vr.get_batch(frame_id_list).asnumpy()
    print(video_data.shape)
    img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]

    transforms = DataAugmentationForVideoMAE(args)
    img, bool_masked_pos = transforms((img, None))  # T*C,H,W
    img = img.view((args.num_frames, 3) + img.size()[-2:]).transpose(0, 1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
    bool_masked_pos = torch.from_numpy(bool_masked_pos)

    with torch.no_grad():
        img = img.unsqueeze(0)
        print(img.shape)
        bool_masked_pos = bool_masked_pos.unsqueeze(0)
        
        img = img.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        outputs = model(img, bool_masked_pos)

        # Save original video
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
        ori_img = img * std + mean  # in [0, 1]
        imgs = [ToPILImage()(ori_img[0, :, vid, :, :].cpu()) for vid, _ in enumerate(frame_id_list)]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/ori_img{id}.jpg")

        img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
        img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        img_patch[bool_masked_pos] = outputs

        # Make mask
        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        # Save reconstruction video
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
        imgs = [ToPILImage()(rec_img[0, :, vid, :, :].cpu().clamp(0, 0.996)) for vid, _ in enumerate(frame_id_list)]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/rec_img{id}.jpg")

        # Save masked video
        img_mask = rec_img * mask
        imgs = [ToPILImage()(img_mask[0, :, vid, :, :].cpu()) for vid, _ in enumerate(frame_id_list)]
        for id, im in enumerate(imgs):
            im.save(f"{args.save_path}/mask_img{id}.jpg")

        # Compute metrics
        mse_scores, psnr_scores, ssim_scores, mae_scores = compute_metrics(ori_img, rec_img)
        
        # Print average metrics
        print(f"Average MSE: {np.mean(mse_scores):.6f}")
        print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
        print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
        print(f"Average MAE: {np.mean(mae_scores):.6f}")

        # Plot and save metrics
        plot_metrics(mse_scores, psnr_scores, ssim_scores, mae_scores, args.save_path)

        # Visualize original, masked, and reconstructed frames
        plt.figure(figsize=(18, 6))
        ori_imgs = [ToPILImage()(ori_img[0, :, i, :, :].cpu()) for i in range(min(6, ori_img.shape[2]))]
        masked_imgs = [ToPILImage()(img_mask[0, :, i, :, :].cpu().clamp(0, 1)) for i in range(min(6, img_mask.shape[2]))]
        rec_imgs = [ToPILImage()(rec_img[0, :, i, :, :].cpu().clamp(0, 1)) for i in range(min(6, rec_img.shape[2]))]

        show_images(ori_imgs, 'Original', 0)
        show_images(masked_imgs, 'Masked', 1)
        show_images(rec_imgs, 'Reconstructed', 2)

        plt.tight_layout()
        plt.savefig(f"{args.save_path}/summary_plot.jpg")
        plt.close()

if __name__ == '__main__':
    opts = get_args()
    main(opts)
