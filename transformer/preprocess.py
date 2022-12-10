import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import torch
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,            
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)
import matplotlib.pyplot as plt

########
# copied this from kaggle dataset because we are not sure how GPU setup works 
# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
#os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

set_caching_enabled(True)
logging.set_verbosity_error() # ? 
torch.cuda.empty_cache()
device = torch.device("cuda")
print(device)
########
dataset = load_dataset(
    "csv", 
    data_files={
        "train": os.path.join("dataset","data_train.csv"),
        "test": os.path.join("dataset", "data_eval.csv")
    }
)

with open(os.path.join("dataset", "answer_space.txt")) as f:
    answer_space = f.read().splitlines()

dataset = dataset.map(
    lambda examples: {
        'label': [
            answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
            for ans in examples['answer']
        ]
    },
    batched=True
)

#print(dataset)
#training_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=4, shuffle=False)
#for batch_ndx, sample in enumerate(training_loader):
#    print(sample)

def showImage(id):
    image = Image.open(os.path.join("dataset", "images", dataset['test'][id]['image_id']+".png"))
    plt.imshow(image)
    plt.show()
    print("Question:\t", dataset['test'][id]['question'])
    print("Answer:\t\t", dataset['test'][id]['answer'])

@dataclass
class MultimodalCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor
    is_personal = False

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "text_features": encoded_text
            #"input_ids": encoded_text['input_ids'].squeeze(),
            #"token_type_ids": encoded_text['token_type_ids'].squeeze(),
            #"attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]):
        processed_images = self.preprocessor(
            images= [Image.open(os.path.join("personal", image_id+".png")).convert('RGB') for image_id in images] if self.is_personal
            else [Image.open(os.path.join("dataset", "images", image_id + ".png")).convert('RGB') for image_id in images],
            return_tensors="pt",
        )
        return {
            #"pixel_values": processed_images['pixel_values'].squeeze(),
            "img_features": processed_images
        }
            
    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['question']
                if isinstance(raw_batch_dict, dict) else
                [i['question'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                ([i['image_id'] for i in raw_batch_dict])
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }