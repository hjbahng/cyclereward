import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import itertools
from typing import Any
from dataclasses import dataclass
from transformers import BertTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from cycle_reward.config import cyclereward_args


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

DATA_KEY_MAPPING = {
    'I2T': {'input_key': 'image', 'preferred_key': 'preferred_text', 'rejected_key': 'rejected_text'},
    'T2I': {'input_key': 'prompt', 'preferred_key': 'preferred_image', 'rejected_key': 'rejected_image'}
}

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

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, i2t_data_path, t2i_data_path, split, model_type, threshold_similar, threshold_negative):
        self.model_type = model_type
        dataset = []

        if "Combo" in model_type:
            I2T_dataset = json.load(open(os.path.join(i2t_data_path, f"{split}.json"), 'r'))
            T2I_dataset = json.load(open(os.path.join(t2i_data_path, f"{split}.json"), 'r'))
            I2T_data = self.make_data(I2T_dataset, threshold_similar, threshold_negative, data_type='I2T')
            T2I_data = self.make_data(T2I_dataset, threshold_similar, threshold_negative=0.4, data_type='T2I')
            self.data = combine_datasets(I2T_data, T2I_data)
        else:
            data_path = i2t_data_path if 'I2T' in model_type else t2i_data_path
            dataset = json.load(open(os.path.join(data_path, f"{split}.json"), 'r'))
            data_type = model_type.split('-')[-1]

            self.input_key, self.preferred_key, self.rejected_key = DATA_KEY_MAPPING[data_type].values()
            self.data = self.make_data(dataset, threshold_similar, threshold_negative, data_type)

    def make_data(self, data, threshold_similar, threshold_negative, data_type='I2T'):
        input_key, preferred_key, rejected_key = DATA_KEY_MAPPING[data_type].values()
        
        pairs = []
        for item in tqdm(data, desc='making dataset'):
            input = item["input"]
            text_set = [g.strip() for g in item["generations"]]
            labels = item["scores"]
            
            for id_l in range(len(labels)):
                for id_r in range(id_l + 1, len(labels)):
                    # remove duplicates
                    if text_set[id_l].lower() == text_set[id_r].lower():
                        continue
                    
                    # remove extremely similar reconstructions
                    if abs(labels[id_l] - labels[id_r]) < threshold_similar:
                        continue

                    dict_item = {input_key: input}
                    if labels[id_l] < labels[id_r]:
                        if data_type == 'T2I':  # sbert is higher the better
                            if labels[id_r] < threshold_negative:  
                                continue
                            dict_item[preferred_key] = text_set[id_r]
                            dict_item[rejected_key] = text_set[id_l] 
                        else:                   # dreamsim is lower the better
                            if labels[id_l] > threshold_negative:  
                                continue
                            dict_item[preferred_key] = text_set[id_l]
                            dict_item[rejected_key] = text_set[id_r]     
                                 
                    elif labels[id_l] > labels[id_r]:
                        if data_type == 'I2T': 
                            if labels[id_r] < threshold_negative:  
                                continue
                            dict_item[preferred_key] = text_set[id_l]
                            dict_item[rejected_key] = text_set[id_r]
                        else:
                            if labels[id_r] > threshold_negative:
                                continue
                            dict_item[preferred_key] = text_set[id_r]
                            dict_item[rejected_key] = text_set[id_l]
                    else:
                        continue

                    # append if the item meets all conditions
                    pairs.append(dict_item)
        
        return pairs

    def __getitem__(self, index):
        item = self.data[index]
        if 'Combo' in self.model_type:
            return {
                "image": item['image'],
                "preferred_text": item['preferred_text'],
                "rejected_text": item['rejected_text'],
                "prompt": item['prompt'],
                "preferred_image": item['preferred_image'],
                "rejected_image": item['rejected_image'],
            }
        return {
            self.input_key: item[self.input_key],
            self.preferred_key: item[self.preferred_key],
            self.rejected_key: item[self.rejected_key],
        }

    def __len__(self):
        return len(self.data)

@dataclass
class DataCollator:
    def __init__(self, max_length=128, model_type='CycleReward-Combo'):
        self.model_type = model_type
        self.preprocess = _transform(cyclereward_args['image_size'])
        self.tokenizer = init_tokenizer()
        self.max_length = max_length
    
    def preprocess_image(self, image: str) -> torch.Tensor:
        return self.preprocess(Image.open(image).convert("RGB"))
    
    def _process_combo(self, features):
        images = torch.stack([self.preprocess_image(f['image']) for f in features], dim=0)
        image_preferred = torch.stack([self.preprocess_image(f['preferred_image']) for f in features], dim=0)
        image_rejected = torch.stack([self.preprocess_image(f['rejected_image']) for f in features], dim=0)
        
        prompt = [f['prompt'] for f in features]
        preferred_text = [f['preferred_text'] for f in features]
        rejected_text = [f['rejected_text'] for f in features]
        prompt = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        preferred_text = self.tokenizer(preferred_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        rejected_text = self.tokenizer(rejected_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        
        return {
            "prompt_ids": prompt.input_ids,
            "prompt_mask": prompt.attention_mask,
            "image_preferred": image_preferred,
            "image_rejected": image_rejected,
            "images": images,
            "preferred_ids": preferred_text.input_ids,
            "preferred_mask": preferred_text.attention_mask,
            "rejected_ids": rejected_text.input_ids,
            "rejected_mask": rejected_text.attention_mask,
        }
    
    def _process_t2i(self, features):
        prompt = [f['input'] for f in features]
        prompt = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        image_preferred = torch.stack([self.preprocess_image(feature['preferred_image']) for feature in features], dim=0)
        image_rejected = torch.stack([self.preprocess_image(feature['rejected_image']) for feature in features], dim=0)
        return {
            "prompt_ids": prompt.input_ids,
            "prompt_mask": prompt.attention_mask,
            "image_preferred": image_preferred,
            "image_rejected": image_rejected
        }
    
    def _process_i2t(self, features):
        images = torch.stack([self.preprocess_image(feature['input']) for feature in features], dim=0)
        preferred_text = [f['preferred_text'] for f in features]
        rejected_text = [f['rejected_text'] for f in features]
        preferred_text = self.tokenizer(preferred_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        rejected_text = self.tokenizer(rejected_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "images": images,
            "preferred_ids": preferred_text.input_ids,
            "preferred_mask": preferred_text.attention_mask,
            "rejected_ids": rejected_text.input_ids,
            "rejected_mask": rejected_text.attention_mask
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if 'Combo' in self.model_type:
            return self._process_combo(features)
        if 'T2I' in self.model_type:
            return self._process_t2i(features)
        else:
            return self._process_i2t(features)

def combine_datasets(list1, list2):
    # choose the shorter list to be repeated
    if len(list1) < len(list2):
        list1, list2 = list2, list1

    repeated_list2 = itertools.cycle(list2)
    combined_list = [{**d1, **next(repeated_list2)} for d1 in list1]

    return combined_list