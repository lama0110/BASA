import argparse
import os
import numpy as np
from PIL import Image
import math
from safetensors.torch import save_file
import torch
import tqdm
from diffusers import AutoencoderKL


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="resolution",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training: local_rank")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    return args


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.mixed_precision == 'fp16':
        dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    ).to(device, dtype)
    
    all_info = [os.path.join(args.data_root, i) for i in sorted(os.listdir(args.data_root)) if '.jpg' in i or '.png' in i]

    os.makedirs(args.output_dir, exist_ok=True)

    work_load = math.ceil(len(all_info) / args.num_workers)
    for idx in tqdm.tqdm(range(work_load * args.local_rank, min(work_load * (args.local_rank + 1), len(all_info)), args.batch_size)):
        images = []
        paths = [os.path.join(args.data_root, item[:item.rfind('.')] + '_latent_code.safetensors') for item in all_info[idx:idx + args.batch_size]]
        for item in all_info[idx:idx + args.batch_size]:
            img = Image.open(os.path.join(args.data_root, item)).convert('RGB')
            img = img.resize((args.resolution, args.resolution))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            images.append(img)
        with torch.no_grad():
            images = torch.stack(images, dim=0)
            data = vae.encode(images.to(device, vae.dtype)).latent_dist
            means = data.mean.cpu().data
            stds = data.std.cpu().data
        for path, mean, std in zip(paths, means.unbind(), stds.unbind()):
            save_file(
                {'mean': mean, 'std': std},
                path
            )


if __name__ == '__main__':
    main(parse_args())