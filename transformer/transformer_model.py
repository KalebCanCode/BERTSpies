import torch
import torch.nn as nn
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel, Swinv2Model,         
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, num_labels = 582, hidden_dim = 512, pretrained_text = 'albert-base-v2', pretrained_image = 'microsoft/swin-tiny-patch4-window7-224'):
        super(TransformerModel, self).__init__()
        # initialize class variables 
        self.num_labels = num_labels 
        self.pretrained_text = pretrained_text 
        self.pretrained_image = pretrained_image 
        self.loss = nn.CrossEntropyLoss()
        # initialize text and image transformers 
        self.text_encoder = AutoModel.from_pretrained(self.pretrained_text)
        self.image_encoder = Swinv2Model.from_pretrained(self.pretrained_image)
        # freeze weights of pretrained models 
        #for _, param in self.text_encoder.named_parameters():
        #    param.requires_grad = False
        #for _, param in self.image_encoder.named_parameters():
        #    param.requires_grad = False 
        # fusion layer to combine output from text and image encoding outputs 
        self.fusion_input_dim = self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size # may change this later to allow pointwise multiplication 
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # classifier layer to project fusion output onto answer space 
        self.classifier = nn.Linear(hidden_dim, self.num_labels)

    def forward(self, img_features, text_features, labels): #img_features instead of pixel_values, might cause weird bug later 
        # forward pass 
        text_encoding = self.text_encoder(**text_features, return_dict = True)
        image_encoding = self.image_encoder(**img_features, return_dict = True )
        fusion_input = torch.cat([text_encoding['pooler_output'], image_encoding['pooler_output']], dim = 1) # pooler output is the CLS token 
        fusion_output = self.fusion_layer(fusion_input)
        logits = self.classifier(fusion_output)
        # calculate loss 
        loss = self.loss(logits, labels)

        return {"logits": logits, "loss": loss}
    
    



