import torch
import torch.nn as nn
from transformers import (
    # Text & Image Models
    AutoModel        
)

class TransformerModel(nn.Module):
    '''
    Pre-trained transformer model class for VQA. 
    '''
    def __init__(self, num_labels = 582, hidden_dim = 512, 
                 pretrained_text = 'roberta-base', 
                 pretrained_image = 'google/vit-base-patch16-224-in21k'):
        '''
        Initialize class variables.  
        '''
        super(TransformerModel, self).__init__()
        self.num_labels = num_labels 
        self.loss = nn.CrossEntropyLoss()
        # load in the pre-trained models 
        self.pretrained_text = pretrained_text 
        self.pretrained_image = pretrained_image  
        self.text_encoder = AutoModel.from_pretrained(self.pretrained_text)
        self.image_encoder = AutoModel.from_pretrained(self.pretrained_image)
        # initialize fusion layer 
        self.fusion_input_dim = self.text_encoder.config.hidden_size \ 
        + self.image_encoder.config.hidden_size 
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # classifier layer to project fusion output onto answer space 
        self.classifier = nn.Linear(hidden_dim, self.num_labels)

    def forward(self, img_features, text_features, labels): 
        '''
        Forward pass for the VQA model.
        '''
        # forward pass 
        # get encodings 
        text_encoding = self.text_encoder(**text_features, return_dict = True)
        image_encoding = self.image_encoder(**img_features, return_dict = True )
        # fuse by concatenating the CLS tokens 
        fusion_input = torch.cat([text_encoding['pooler_output'], 
                                    image_encoding['pooler_output']], dim = 1) 
                                    # pooler output is the CLS token 
        fusion_output = self.fusion_layer(fusion_input)
        # classify 
        logits = self.classifier(fusion_output)
        # calculate loss 
        loss = self.loss(logits, labels)

        return {"logits": logits, "loss": loss}
    
    



