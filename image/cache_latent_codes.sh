export NUM_WORKERS=4
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
torchrun --nproc_per_node=$NUM_WORKERS cache_latent_codes.py \
    --data_root="/path/to/t2i_1024" \
    --batch_size=16 \
    --num_worker=$NUM_WORKERS \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --mixed_precision='bf16' \
    --output_dir="/path/to/t2i_1024"
    
