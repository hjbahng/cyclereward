# Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences
### [Paper](https://arxiv.org/abs/2506.02095) | [Project Page](https://cyclereward.github.io/) | [Dataset (I2T)](https://huggingface.co/datasets/carolineec/CyclePrefDB-I2T) | [Dataset (T2I)](https://huggingface.co/datasets/carolineec/CyclePrefDB-T2I) | [Dataset Viewer](https://cyclereward.github.io/#dataset)

[Hyojin Bahng](https://hjbahng.github.io/)\*, [Caroline Chan](https://people.csail.mit.edu/cmchan/)\*, [Fredo Durand](https://people.csail.mit.edu/fredo/), [Phillip Isola](https://web.mit.edu/phillipi/).<br>
(*Equal contribution, alphabetical order.)<br>
MIT CSAIL.

<p align="center">
    <img src="images/teaser.jpg" width="1000px">
</p>

CycleReward is a reward model trained on preferences derived from cycle consistency. Given a forward mapping $$F:X \rightarrow Y$$ and a backward mapping $$G: Y \rightarrow X$$, we define cycle consistency score as the similarity between the original input $$x$$ and its reconstruction $$G(F(x))$$. This score serves as a proxy for preference: higher cycle consistency indicates a preferred output. This provides a more scalable and cheaper signal for learning image-text alignment compared to human supervision. We construct CyclePrefDB, a preference dataset of 866K comparison pairs across image-to-text and text-to-image tasks focusing on dense captions. Trained on this dataset, CycleReward matches or surpasses models trained on human or GPT4V feedback.



## 🚀 Updates
**06/16/25:** We've now released CycleReward on Hugging Face: 

- [CycleReward-Combo](https://huggingface.co/carolineec/CycleReward-Combo)
- [CycleReward-I2T](https://huggingface.co/carolineec/CycleReward-I2T)
- [CycleReward-T2I](https://huggingface.co/carolineec/CycleReward-T2I)

## Quick Start
Install with pip:

`pip install cyclereward`

Use CycleReward to measure the alignment between an image and a caption (higher is better). 
```python
from cyclereward import cyclereward
from PIL import Image
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = cyclereward(device=device, model_type="CycleReward-Combo")

caption = "a photo of a cat"
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)
score = model.score(image, caption) 
```
We release three model variants: 
- `CycleReward-I2T` trained on image-to-text pairs
- `CycleReward-T2I` trained on text-to-image pairs
- `CycleReward-Combo` trained on both, **recommended for best results**

## Contents
- [Install](#install)
- [CyclePrefDB Dataset](#cycleprefdb-dataset)
- [Generate Preference Data](#generate-preference-data)
- [Train Reward Model](#train-reward-model)
- [Citation](#citation)

## Install
Clone this repository and install dependencies:
```bash
git clone https://github.com/hjbahng/cyclereward.git
cd cyclereward

conda create -n crwd python=3.10
conda activate crwd

pip install -r requirements.txt
```

## CyclePrefDB Dataset
CycleReward is trained on CyclePrefDB, a large-scale preference dataset based on cycle consistency. 

| Dataset | Task | Number of Pairs |
| :------ | :------ | :------ | 
| [CyclePrefDB-I2T](https://huggingface.co/datasets/carolineec/CyclePrefDB-I2T) | Image-to-text generation | 398K |
| [CyclePrefDB-T2I](https://huggingface.co/datasets/carolineec/CyclePrefDB-T2I) | Text-to-image generation | 468K |

You can load them using the Hugging Face `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("carolineec/CyclePrefDB-I2T")
```
Explore examples on our [dataset viewer](https://cyclereward.github.io/#dataset).

## Generate Preference Data
To generate your own comparison pairs using cycle consistency, you'll need:
- A set of forward models
- A backward model 
- Input images or texts

See full configuration in `scripts/generate_i2t.sh` and `scripts/generate_t2i.sh`. We detail each step below:

### 1. Prepare input data
Download the [DCI](https://github.com/facebookresearch/DCI) dataset. While we use the DCI dataset as an example, you can use **any unpaired text or image data**. When using your own data, format your input as:
```
# for image-to-text generation
image_dataset = [{'real_image': image_path}, ...]

# for text-to-image generation
text_dataset = [{'real_text': caption}, ...]
```
To use sDCI captions, follow their [instructions](https://github.com/facebookresearch/DCI?tab=readme-ov-file#clip-ready).

### 2. Generate comparison pairs and cycle consistency score
To generate cycle consistency scores using LLaVA-1.5-13B as the forward (I2T) model and Stable Diffusion 3 as the backward (T2I) model:
```
python generate.py \
    --cycle i2t2i \
    --model_name_or_path llava-hf/llava-1.5-13b-hf \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
    --dataset DCI \
    --data_path /path/to/densely_captioned_images \
    --output_path /path/to/save/results \
    --cache_dir /path/to/download/models 
```
You can repeat this for multiple forward models to construct comparison pairs.

### 3. Build preference dataset
Once generation is complete, build the preference dataset by comparing their cycle consistency scores:
```
python make_dataset.py \
    --output_path /path/to/save/results \
    --dataset DCI \
    --cycle i2t2i \
    --save_path /path/to/save/preference_dataset
```
This will produce a dataset for training the reward model.

## Train Reward Model
To train CycleReward, refer to the training scripts `scripts/train_**.sh`.

## Citation
If you find our work or any of our materials useful, please cite our paper:
```
@article{bahng2025cycle,
    title={Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences},
    author= {Bahng, Hyojin and Chan, Caroline and Durand, Fredo and Isola, Phillip},
    journal={arXiv preprint arXiv:2506.02095},
    year={2025}
}
```