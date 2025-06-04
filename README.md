# Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences
### [Paper](https://arxiv.org/abs/2506.02095) | [Project Page](https://cyclereward.github.io/) | [Dataset (I2T)](https://huggingface.co/datasets/carolineec/CyclePrefDB-I2T) | [Dataset (T2I)](https://huggingface.co/datasets/carolineec/CyclePrefDB-T2I) | [Dataset Viewer](https://cyclereward.github.io/#dataset)

[Hyojin Bahng](https://hjbahng.github.io/)\*, [Caroline Chan](https://people.csail.mit.edu/cmchan/)\*, [Fredo Durand](https://people.csail.mit.edu/fredo/), [Phillip Isola](https://web.mit.edu/phillipi/).<br>
(*Equal contribution, alphabetical order.)<br>
MIT CSAIL.

<p align="center">
    <img src="images/teaser.jpg" width="1000px">
</p>

CycleReward is a reward model trained on preferences derived from cycle consistency. Given a forward mapping $$F:X \rightarrow Y$$ and a backward mapping $$G: Y \rightarrow X$$, we define cycle consistency score as the similarity between the original input $$x$$ and its reconstruction $$G(F(x))$$. This score serves as a proxy for preference: higher cycle consistency indicates a preferred output. This provides a more scalable and cheaper signal for learning image-text alignment compared to human supervision. We construct CyclePrefDB, a large-scale preference dataset comprising 866K comparison pairs spanning image-to-text and text-to-image generation, with an emphasis on dense captions and prompts. Trained on this dataset, CycleReward matches or surpasses models trained on human or AI feedback.

## Quick Start
Run `pip install cyclereward`. The following Python code is all you need.

The basic use case is to measure the alignment between an image and a caption. **A higher score means more similar, lower means more different**. We release three model variants: `CycleReward-Combo`, `CycleReward-I2T`, `CycleReward-T2I`.

```python
from cyclereward import cyclereward
from PIL import Image
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = cyclereward(device=device, model_type="CycleReward-Combo")

caption = "a photo of a cat"
image = preprocess(Image.open("image_path")).unsqueeze(0).to(device)
score = model.score(image, caption) 
```

## CyclePrefDB Dataset
CycleReward is trained on CyclePrefDB, a large-scale preference dataset based on cycle consistency. We provide comparison pairs for both image-to-text (I2T) and text-to-image (T2I) generation, with a focus on dense captions and prompts.

| Dataset | Number of Pairs |
| ------ | ------ | 
| [CyclePrefDB-I2T](https://huggingface.co/datasets/carolineec/CyclePrefDB-I2T) | 398K |
| [CyclePrefDB-T2I](https://huggingface.co/datasets/carolineec/CyclePrefDB-T2I) | 468K |

You can use the Hugging Face [Datasets](https://huggingface.co/docs/datasets/quickstart) library to load the datasets:
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("carolineec/CyclePrefDB-I2T", split='train')
```

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