"""
Adapted from BLIP (https://github.com/salesforce/BLIP)
"""
import torch.distributed as dist
from torch import nn
import transformers
transformers.logging.set_verbosity_error()

from .med import BertConfig, BertModel
from .blip import create_vit, init_tokenizer, load_checkpoint


class BLIP_Pretrain(nn.Module):
    def __init__(self,                 
                 med_config = 'cycle_reward/blip/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        
        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def blip_pretrain(pretrained='', **kwargs):
    model = BLIP_Pretrain(**kwargs)
    if pretrained and get_rank() == 0:
        model, msg = load_checkpoint(model,pretrained)
    return model 
