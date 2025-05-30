import argparse
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np
from timm.models import create_model
from datasets import build_dataset
import modeling_finetune
import wandb
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('VideoMAE evaluation script for F1 score', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for evaluation')
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, help='Name of model to test')
    parser.add_argument('--finetune', default='', type=str, required=True, help='Path to finetuned checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str, help='Key for model state dict in checkpoint')
    parser.add_argument('--model_prefix', default='', type=str, help='Prefix for model state dict keys')
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str, help='Dataset path')
    parser.add_argument('--nb_classes', default=400, type=int, help='Number of classification types')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--num_segments', type=int, default=1, help='Number of segments')
    parser.add_argument('--sampling_rate', type=int, default=4, help='Frame sampling rate')
    parser.add_argument('--test_num_segment', type=int, default=5, help='Number of segments for testing')
    parser.add_argument('--test_num_crop', type=int, default=3, help='Number of crops for testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true', help='Use ImageNet default mean and std')
    parser.add_argument('--short_side_size', type=int, default=224, help='Short side size for video resizing')
    parser.add_argument('--tubelet_size', type=int, default=2, help='Tubelet size for video patches')
    parser.add_argument('--input_size', default=224, type=int, help='Video input size')
    parser.add_argument('--fc_drop_rate', type=float, default=0.0, help='Dropout rate for fully connected layer')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use checkpointing in model')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--use_mean_pooling', action='store_true', help='Use mean pooling')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--init_scale', default=0.001, type=float, help='Initialization scale')
    parser.add_argument('--data_set', default='Kinetics-400', choices=['Kinetics-400', 'Cholec80', 'SSV2', 'UCF101', 'HMDB51', 'image_folder'],
                        type=str, help='Dataset')
    parser.add_argument('--device', default='cuda', help='Device for testing')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of DataLoader workers')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.set_defaults(pin_mem=False)
    return parser.parse_args()


def main(args):
    print(args)

    # Set device and seed
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Initialize W&B
    try:
        os.environ["WANDB_API_KEY"] = "3cd0fd46806f5dbb7e666990676fb3d1c75e0447"
        wandb.init(project="Finetuning-VideoMAE", id="07ubhacs", resume="allow")
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {e}. Continuing without W&B logging.")
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.finish = lambda *args, **kwargs: None

    # Load test dataset
    print("Loading test dataset...")
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=sampler_test,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    print(f"Test dataset loaded with {len(dataset_test)} samples.")

    # Create model
    print("Creating model...")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        use_checkpoint=args.use_checkpoint,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    # Clean checkpoint state dict
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    from collections import OrderedDict
    new_dict = OrderedDict()
    for key in checkpoint_model.keys():
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    # Interpolate position embedding if necessary
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size)
            pos_tokens = pos_tokens.flatten(1, 3)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    print("Model and checkpoint loaded.")

    # Evaluate model and compute F1 score
    all_preds = []
    all_labels = []
    model.eval()
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(data_loader_test, desc="Evaluating"):
            videos, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            outputs = model(videos)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"F1 Score on the {len(dataset_test)} test videos: {f1:.4f}")

    # Log F1 score to W&B
    wandb.log({'F1 Score': f1})
    wandb.finish()


if __name__ == '__main__':
    args = get_args()
    main(args)