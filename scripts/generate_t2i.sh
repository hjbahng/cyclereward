#!/bin/bash

############################################################
# Set the following variables to match your setup
############################################################

# Cycle direction
CYCLE="t2i2t"

# Text-to-image model (forward mapping)
T2I_MODELS=(
    "sd-legacy/stable-diffusion-v1-5"
    "stabilityai/stable-diffusion-xl-base-1.0"
    "stabilityai/stable-diffusion-3-medium-diffusers"
    "black-forest-labs/FLUX.1-schnell"
)

# Image-to-text model (backward mapping)
I2T_MODEL="llava-hf/llava-1.5-13b-hf"

# Dataset name 
DATASET="DCI"

# Paths 
DATA_PATH="/path/to/densely_captioned_images"
OUTPUT_PATH="/path/to/save/results"
CACHE_DIR="/path/to/download/models"
SAVE_PATH="/path/to/save/preference_dataset"

############################################################
# Generate cycle consistency preference dataset
############################################################

for T2I_MODEL in "${T2I_MODELS[@]}"; do
    python generate.py \
        --cycle $CYCLE \
        --model_name_or_path $I2T_MODEL \
        --pretrained_model_name_or_path $T2I_MODEL \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --output_path $OUTPUT_PATH \
        --cache_dir $CACHE_DIR
done

# Generate preference dataset
python make_dataset.py \
    --output_path $OUTPUT_PATH \
    --dataset $DATASET \
    --cycle $CYCLE \
    --save_path $SAVE_PATH