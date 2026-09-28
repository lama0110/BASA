import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file

from attention_processor_inference import (
    ReshapeBasaDownsampleFlexAttnProcessor,
    ReshapeBasaFlexAttnProcessor,
    init_reshape_basa_downsample_flex,
    init_reshape_basa_flex,
)


bfl_repo = "black-forest-labs/FLUX.1-dev"
device = torch.device("cuda")
dtype = torch.bfloat16

pipe = FluxPipeline.from_pretrained(bfl_repo, torch_dtype=dtype).to(device)
prompt = "A mickey mouse is eating a cake"

infer_height = 1024
infer_width = 1024

# FLUX packs the latent grid into image tokens at 1/16 of the image resolution.
token_height = infer_height // 16
token_width = infer_width // 16

num_inference_steps = 20
window_size = 32
pool_size = 4
text_length = 512

# The paper configuration uses shifted local windows plus mean-pooled global K/V memory.
down_factor = 1
ckpt_path = "./BASA/image/attn_lora_weights_32.safetensors"


def build_basa_meta(layer_index, total_layers, denoising_step):
    if down_factor == 1:
        return init_reshape_basa_flex(
            token_height,
            token_width,
            text_length=text_length,
            window_size=window_size,
            device=device,
            total_layers=total_layers,
            layer_index=layer_index,
            denoising_step=denoising_step,
            use_pooling_token=True,
            pool_size=pool_size,
            pool_text_attend=False,
        )

    return init_reshape_basa_downsample_flex(
        token_height,
        token_width,
        text_length=text_length,
        window_size=window_size,
        down_factor=down_factor,
        device=device,
        total_layers=total_layers,
        layer_index=layer_index,
        denoising_step=denoising_step,
    )


def set_initial_basa_processors(transformer, initial_denoising_step=1):
    attn_processors = {}
    total_layers = len(transformer.attn_processors)

    for layer_index, name in enumerate(transformer.attn_processors.keys()):
        basa_meta = build_basa_meta(
            layer_index=layer_index,
            total_layers=total_layers,
            denoising_step=initial_denoising_step,
        )

        if down_factor == 1:
            attn_processors[name] = ReshapeBasaFlexAttnProcessor(basa_meta=basa_meta)
        else:
            attn_processors[name] = ReshapeBasaDownsampleFlexAttnProcessor(
                down_factor=down_factor,
                basa_meta=basa_meta,
            ).to(device=device, dtype=dtype)

    transformer.set_attn_processor(attn_processors)


def update_basa_processors_for_step(transformer, denoising_step):
    """Update only partition metadata so learned processor parameters stay intact."""
    processors = transformer.attn_processors
    total_layers = len(processors)

    for layer_index, (_name, processor) in enumerate(processors.items()):
        basa_meta = build_basa_meta(
            layer_index=layer_index,
            total_layers=total_layers,
            denoising_step=denoising_step,
        )
        processor.set_basa_meta(basa_meta)


set_initial_basa_processors(pipe.transformer, initial_denoising_step=1)

# Merge LoRA updates into the original attention projections for inference.
base_state_dict = pipe.transformer.state_dict()
lora_state_dict = load_file(ckpt_path)
merged_state_dict = {k: v.clone() for k, v in base_state_dict.items()}

for k, v in lora_state_dict.items():
    if not (k.endswith(".lora_A") or k.endswith(".lora_B")):
        if k in merged_state_dict:
            merged_state_dict[k] = v.to(
                device=merged_state_dict[k].device,
                dtype=merged_state_dict[k].dtype,
            )
        else:
            merged_state_dict[k] = v.to(device=device)

rank = 128
alpha = 128
scaling = alpha / rank

lora_pairs = {}
for key, value in lora_state_dict.items():
    if key.endswith(".lora_A"):
        prefix = key[: -len(".lora_A")]
        lora_pairs.setdefault(prefix, {})["A"] = value.to(torch.float32).cpu()
    elif key.endswith(".lora_B"):
        prefix = key[: -len(".lora_B")]
        lora_pairs.setdefault(prefix, {})["B"] = value.to(torch.float32).cpu()

for key, value in base_state_dict.items():
    if not key.endswith(".weight"):
        continue
    prefix = key[: -len(".weight")]
    if prefix in lora_pairs and "A" in lora_pairs[prefix] and "B" in lora_pairs[prefix]:
        A = lora_pairs[prefix]["A"]
        B = lora_pairs[prefix]["B"]
        delta_w = torch.mm(B, A) * scaling
        merged_state_dict[key] = (value.to(torch.float32).cpu() + delta_w).to(
            device=value.device,
            dtype=value.dtype,
        )
        print("Merged:", key)

for prefix, mats in lora_pairs.items():
    weight_key = prefix + ".weight"
    if "A" in mats and "B" in mats and weight_key not in base_state_dict:
        delta_w = torch.mm(mats["B"], mats["A"]) * scaling
        merged_state_dict[weight_key] = delta_w.to(device=device)

for key in [k for k in merged_state_dict if k.endswith(".lora_A") or k.endswith(".lora_B")]:
    del merged_state_dict[key]

missing_keys, unexpected_keys = pipe.transformer.load_state_dict(merged_state_dict, strict=False)

allowed_missing = [
    ".attn.to_q.",
    ".attn.to_k.",
    ".attn.to_v.",
    ".attn.to_out.",
]
filtered_missing = [k for k in missing_keys if not any(tag in k for tag in allowed_missing)]

if filtered_missing or unexpected_keys:
    print(f"Unexpected keys: {unexpected_keys}")
    print(f"Missing keys: {filtered_missing}")


def basa_step_callback(pipe, step_index, timestep, callback_kwargs):
    # The next denoising step receives a new layer-wise shifted-window partition.
    next_denoising_step = step_index + 2
    if next_denoising_step <= num_inference_steps:
        update_basa_processors_for_step(
            pipe.transformer,
            denoising_step=next_denoising_step,
        )
    return callback_kwargs


image = pipe(
    prompt,
    height=infer_height,
    width=infer_width,
    guidance_scale=3.5,
    num_inference_steps=num_inference_steps,
    max_sequence_length=text_length,
    generator=torch.Generator("cpu").manual_seed(0),
    callback_on_step_end=basa_step_callback,
).images[0]

image.save("lora_image.png")
