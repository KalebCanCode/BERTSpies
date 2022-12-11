import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules.activation import Softmax


class TwoChanNN(nn.Module):
  def __init__(self,image_feature_extractor, lstm_units, num_vgg_features, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.pretrained_extractor = image_feature_extractor
    self.hidden_units = lstm_units 
    self.lstm_d = nn.LSTM(512, 512, 2, batch_first=True)
    self.lstm_fc = nn.Linear(2048, 1024)
     
    #nn.Linear(input_size, output_size)
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
    with torch.no_grad():
      im_feat = self.pretrained_extractor(image)
    im_feat = self.image_fc(im_feat)

    l2_norm = F.normalize(im_feat, p=2, dim=1).detach()

    return l2_norm




  def language_channel(self,questions):
    embeddings = self.embedding(questions)
    embeddings = self.tanh(embeddings)

    _, (h,c) = self.lstm_d(embeddings)

    z = torch.cat((h,c), dim=2)
    z = z.transpose(0, 1)
    z = z.reshape(z.size()[0], 2048 )

    z = self.tanh(z)
    z = self.lstm_fc(z)

    return z

  
  def fuse(self, im_feat, q_feat):
    return torch.mul(im_feat, q_feat) # element-wise multiplication
  

  def forward(self, data):
    image, question = data
    phi = self.vision_channel(image)
    psi = self.language_channel(question)
    f = self.fuse(phi, psi)

    output = self.classifier(f)
    return output

