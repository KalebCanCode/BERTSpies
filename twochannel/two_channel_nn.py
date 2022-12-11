import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoChanNN(nn.Module):
  '''
  Class for the two-channel vision + language model. 
  '''
  def __init__(self,image_feature_extractor, lstm_units, num_vgg_features, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.pretrained_extractor = image_feature_extractor
    self.hidden_units = lstm_units 
    # LSTM with two layers 
    self.lstm_d = nn.LSTM(512, 512, 2, batch_first=True)
    # Projection layer 
    self.lstm_fc = nn.Linear(2048, 1024)
    
    self.image_dense = nn.Sequential(
        nn.Linear(1000, 1024),
        nn.Tanh()
    )
    self.image_fc = nn.Linear(1000, 1024)
    self.question_dense = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.Tanh()
    )
    self.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Tanh(),
        nn.Linear(1024, 1000),
        nn.Dropout(0.4),
        nn.Tanh(),
        nn.Linear(1000, 582),

    )
    self.embedding = nn.Embedding(self.vocab_size, 512)
    self.tanh = nn.Tanh()

  def vision_channel(self, image):
    '''
    Uses pre-trained VGG-19 to extract the image encodings. 
    '''
    # get VGG-19 output for the images 
    with torch.no_grad():
      im_feat = self.pretrained_extractor(image)
    # project to 1024 dim
    im_feat = self.image_fc(im_feat)
    # apply l2 normalization 
    l2_norm = F.normalize(im_feat, p=2, dim=1).detach()

    return l2_norm




  def language_channel(self,questions):
    '''
    Uses two-layer LSTM to extract the word encodings. 
    '''
    embeddings = self.embedding(questions)
    embeddings = self.tanh(embeddings)

    _, (h,c) = self.lstm_d(embeddings)

    z = torch.cat((h,c), dim=2) # concatenate hidden and cell states 
    z = z.transpose(0, 1) # transpose to get batch size in the first dim 
    # reshape into two-dimensions since hidden and cell states are 3D 
    z = z.reshape(z.size()[0], 2048 ) 
    z = self.tanh(z)
    # project to 1024 dim 
    z = self.lstm_fc(z)
    return z

  
  def fuse(self, im_feat, q_feat):
    '''
    Fuses image and text encodings through pointwise multiplication. 
    '''
    return torch.mul(im_feat, q_feat) # element-wise multiplication
  

  def forward(self, data):
    '''
    Forward pass of the two-channel model. 
    '''
    image, question = data
    # image encodings 
    phi = self.vision_channel(image)
    # text encodings 
    psi = self.language_channel(question)
    # fuse
    f = self.fuse(phi, psi)
    # classify 
    output = self.classifier(f)

    return output

