"""
Adapted from ImageReward (https://github.com/THUDM/ImageReward)
"""

import os
import torch
import torch.nn as nn
from PIL import Image

from .config import cyclereward_args
from .blip.blip_pretrain import blip_pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class CycleReward(nn.Module):
    def __init__(self, device='cpu', model_type='CycleReward-Combo', max_length=128, fix_rate=0.7):
        super().__init__()
        self.device = device
        self.model_type = model_type
        self.max_length = max_length
        
        self.blip = blip_pretrain(
            pretrained=cyclereward_args['blip_path'], 
            med_config='blip/med_config.json',  
            image_size=cyclereward_args['image_size'], 
            vit=cyclereward_args['vit']
        )
        self.preprocess = _transform(cyclereward_args['image_size'])
        self.mlp = MLP(cyclereward_args['mlp_dim'])
        
        for name, parms in self.blip.named_parameters():
            if '_proj' in name:
                parms.requires_grad_(False)
        
        # fix certain ratio of layers (setting from ImageReward)
        self.image_layer_num = 24 if cyclereward_args['vit'] == 'large' else 12
        if fix_rate > 0:
            text_fix_num = "layer.{}".format(int(12 * fix_rate))
            image_fix_num = "blocks.{}".format(int(self.image_layer_num * fix_rate))
            for name, parms in self.blip.text_encoder.named_parameters():
                parms.requires_grad_(False)
                if text_fix_num in name:
                    break
            for name, parms in self.blip.visual_encoder.named_parameters():
                parms.requires_grad_(False)
                if image_fix_num in name:
                    break

    def forward(self, batch):
        if 'Combo' in self.model_type:
            text_reward = self.text_reward(batch)
            image_reward = self.image_reward(batch)

        elif 'I2T' in self.model_type:
            text_reward = self.text_reward(batch)
            image_reward = None

        elif 'T2I' in self.model_type:
            text_reward = None
            image_reward = self.image_reward(batch)
        
        return text_reward, image_reward
    
    def text_reward(self, batch):
        images, preferred_ids, preferred_mask, rejected_ids, rejected_mask = batch["images"], batch["preferred_ids"], batch["preferred_mask"], batch["rejected_ids"], batch["rejected_mask"]
        images = images.to(self.device)
        preferred_ids = preferred_ids.to(self.device)
        preferred_mask = preferred_mask.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        rejected_mask = rejected_mask.to(self.device)

        # encode image
        image_embeds = self.blip.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        
        # encode preferred
        preferred_embeds = self.blip.text_encoder(
            preferred_ids,
            attention_mask=preferred_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ).last_hidden_state
        preferred_embeds = preferred_embeds[:,0,:].float()
        
        # encode rejected
        rejected_embeds = self.blip.text_encoder(
            rejected_ids,
            attention_mask=rejected_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ).last_hidden_state
        rejected_embeds = rejected_embeds[:,0,:].float()

        preferred_reward = self.mlp(preferred_embeds)
        rejected_reward = self.mlp(rejected_embeds)
        reward = torch.concat((preferred_reward, rejected_reward), dim=1)

        return reward

    def image_reward(self, batch):
        prompt_ids, prompt_mask, image_preferred, image_rejected = batch["prompt_ids"], batch["prompt_mask"], batch["image_preferred"], batch["image_rejected"]
        image_preferred = image_preferred.to(self.device)
        image_rejected = image_rejected.to(self.device)
        prompt_ids = prompt_ids.view(prompt_ids.shape[0], -1).to(self.device)
        prompt_mask = prompt_mask.view(prompt_mask.shape[0], -1).to(self.device)

        # encode image
        image_embeds_preferred = self.blip.visual_encoder(image_preferred)
        image_atts_preferred = torch.ones(image_embeds_preferred.size()[:-1],dtype=torch.long).to(self.device)

        image_embeds_rejected = self.blip.visual_encoder(image_rejected)
        image_atts_rejected = torch.ones(image_embeds_rejected.size()[:-1],dtype=torch.long).to(self.device)
        
        # encode preferred
        preferred_embeds = self.blip.text_encoder(
            prompt_ids,
            attention_mask=prompt_mask,
            encoder_hidden_states=image_embeds_preferred,
            encoder_attention_mask=image_atts_preferred,
            return_dict=True,
        ).last_hidden_state
        preferred_embeds = preferred_embeds[:,0,:].float()
        
        # encode rejected
        rejected_embeds = self.blip.text_encoder(
            prompt_ids,
            attention_mask=prompt_mask,
            encoder_hidden_states=image_embeds_rejected,
            encoder_attention_mask=image_atts_rejected,
            return_dict=True,
        ).last_hidden_state
        rejected_embeds = rejected_embeds[:,0,:].float()

        preferred_reward = self.mlp(preferred_embeds)
        rejected_reward = self.mlp(rejected_embeds)
        reward = torch.concat((preferred_reward, rejected_reward), dim=1)

        return reward
    
    @torch.no_grad()
    def score(self, image, prompt):
        text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        
        image_embeds = self.blip.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
        
        text_embeds = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ).last_hidden_state
        text_embeds = text_embeds[:,0,:].float() 
        
        rewards = self.mlp(text_embeds)
        return rewards
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.GELU(),

            nn.Linear(1024, 128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 16),
            nn.GELU(),
            
            nn.Linear(16, 1)
        )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.layers.apply(init_weights)
        
    def forward(self, input):
        return self.layers(input)

def download_weights(cache_dir="./checkpoints", model_type="CycleReward-Combo"):
    """Downloads the CycleReward weights."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    dst_path = os.path.join(cache_dir, f"{model_type}.pth")
    if not os.path.exists(dst_path):
        torch.hub.download_url_to_file(url=_MODELS[model_type], dst=dst_path)


def cyclereward(device="cuda", model_type="CycleReward-Combo", cache_dir="./checkpoints"):
    """Initializes the CycleReward model.
    
    Parameters
    ----------
    device : str
        The device to run the model on. 
    cache_dir : str
        The directory to cache the model in. 
    model_type : str
        The type of CycleReward model to use. 
    
    Returns
    -------
    model : CycleReward
        The CycleReward model.
    preprocess : callable
        The preprocessing function for images.
    """

    model = CycleReward(device=device)
    model.to(device)

    # download weights
    download_weights(cache_dir=cache_dir, model_type=model_type)

    # load model
    ckpt_path = os.path.join(cache_dir, f"{model_type}.pth")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.eval().requires_grad_(False)

    return model, model.preprocess

_MODELS = {
    "CycleReward-Combo": "https://github.com/hjbahng/cyclereward/releases/download/v1.0.0/CycleReward-Combo.pth",
    "CycleReward-I2T":  "https://github.com/hjbahng/cyclereward/releases/download/v1.0.0/CycleReward-I2T.pth",
    "CycleReward-T2I": "https://github.com/hjbahng/cyclereward/releases/download/v1.0.0/CycleReward-T2I.pth"
}