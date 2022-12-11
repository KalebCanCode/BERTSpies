import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import torch
from transformers import (
    # Preprocessing
    AutoTokenizer, AutoFeatureExtractor,
    # Misc
    logging
)
import matplotlib.pyplot as plt

########
# copied this from kaggle dataset because we are not sure how GPU setup works 
set_caching_enabled(True) # cache the dataset
logging.set_verbosity_error()
torch.cuda.empty_cache() # empty cache before starting 
device = torch.device("cuda") # set gpu 
print(device)
########
# load in dataset 
dataset = load_dataset(
    "csv", 
    data_files={
        "train": os.path.join("dataset","data_train.csv"),
        "test": os.path.join("dataset", "data_eval.csv")
    }
)
# load in list of all possible answers 
with open(os.path.join("dataset", "answer_space.txt")) as f:
    answer_space = f.read().splitlines()

# map answers to their indices in the answer space list 
dataset = dataset.map(
    lambda examples: {
        'label': [
            #  Select the 1st answer if multiple answers are provided
            answer_space.index(ans.replace(" ", "").split(",")[0]) 
            for ans in examples['answer']
        ]
    },
    batched=True
)

def showImage(id):
    '''
    Function used for displaying the image during inference.
    '''
    image = Image.open(os.path.join("dataset", "images", 
    dataset['test'][id]['image_id']+".png"))
    plt.imshow(image)
    plt.show()
    print("Question:\t", dataset['test'][id]['question'])
    print("Answer:\t\t", dataset['test'][id]['answer'])


@dataclass
class MultimodalCollator:
    '''
    Custom class for collating data.  
    '''
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor
    is_personal = False # flag for whether the image should come from our 
                        # personal image dataset or not 

    def tokenize_text(self, texts: List[str]):
        '''
        Defines tokenizer for RoBERTa; returns BPE tokens. 
        '''
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
        }

    def preprocess_images(self, images: List[str]):
        '''
        Converts image to RGB and feed them to ViT feature extractor;
        then returns the extracted/normalized pixel values. 
        '''
        processed_images = self.preprocessor(
            images= [
            Image.open(
            os.path.join("personal", image_id+".png")).convert('RGB') 
            for image_id in images] if self.is_personal
            else [
            Image.open(
            os.path.join("dataset", "images", image_id + ".png")).convert('RGB') 
            for image_id in images],
            return_tensors="pt",
        )
        return {
            "img_features": processed_images 
        }
            
    def __call__(self, raw_batch_dict):
        '''
        Returns the data in our desired format. 
        ''' 
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