#!/bin/bash

############################################################
# Set the following variables to match your setup
############################################################

# Cycle direction
CYCLE="i2t2i"

# Image-to-text model (forward mapping)
I2T_MODELS=(
    "Salesforce/blip2-flan-t5-xxl"  
    "llava-hf/llava-1.5-7b-hf"           
    "llava-hf/llava-1.5-13b-hf"
    "llava-hf/llava-v1.6-mistral-7b-hf"  
    "llava-hf/llava-v1.6-34b-hf"
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"  
    "llava-hf/llava-onevision-qwen2-7b-ov-hf"    
    "OpenGVLab/InternVL2-2B"   
    "OpenGVLab/InternVL2-8B"   
    "OpenGVLab/InternVL2-26B"
    "OpenGVLab/InternVL2-40B"  
)

# Text-to-image model (backward mapping)
T2I_MODEL="stabilityai/stable-diffusion-3-medium-diffusers"

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

for I2T_MODEL in "${I2T_MODELS[@]}"; do
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