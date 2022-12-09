import torch
import torch.nn as nn


from torch.nn.modules.activation import Softmax


class TwoChanNN(nn.Module):
  def __init__(self,image_feature_extractor, lstm_units, num_vgg_features, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.pretrained_extractor = image_feature_extractor
    self.pretrained_extractor
    self.hidden_units = lstm_units 
    #nn.LSTM(input_size, hidden_size, num_layers)
    self.lstm_a = nn.LSTM(512, 512, 1, batch_first=True)
              
    self.lstm_b = nn.LSTM(512, 512, 1, batch_first=True)

    self.lstm_c = nn.LSTM(512, 512, 5, batch_first=True)
     
    #nn.Linear(input_size, output_size)
    self.image_dense = nn.Sequential(
        #pretrained outputs num_ftrs = model_ft.fc.in_features vector
        nn.Linear(1000, 1024),# CHANGED THIS num_vgg_features, 1024)
        nn.Tanh()
    )
    self.question_dense = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.Tanh()
    )
    self.classifier = nn.Sequential(
        nn.Linear(1024, 1000),
        nn.Tanh(),
        nn.Dropout(0.5),
        #1000 or 500 because of dropout??
        nn.Linear(1000, 582),
        # do we need this tanh then softmax??
        nn.Tanh(),


    )
    self.embedding = nn.Embedding(self.vocab_size, 512)
#Motivating example  
# >>> rnn = nn.LSTM(10, 20, 2)
# >>> input = torch.randn(5, 3, 10)
# >>> h0 = torch.randn(2, 3, 20)
# >>> c0 = torch.randn(2, 3, 20)
# >>> output, (hn, cn) = rnn(input, (h0, c0))
  def language_channel(self,questions):
    embeddings = self.embedding(questions)
    #print("embedding", embeddings.size()) #(4, 25, 512)
    output_a, (final_hidden_a, final_cell_a) = self.lstm_a(embeddings) #(4, 2, 512), (1, 4, 512), (1, 4, 512)
    #print("outputa", output_a.size(), final_hidden_a.size(), final_cell_a.size())
    output_b, (final_hidden_b, final_cell_b) = self.lstm_b(nn.Tanh()(output_a), (nn.Tanh()(final_hidden_a), nn.Tanh()(final_cell_a)))
    #not sure about the dim here
    return torch.cat((final_hidden_a[0], final_cell_a[0], final_hidden_b[0], final_cell_b[0]), dim=1)
  
  def fuse(self, image_channel_features, question_features):
    im_feat = self.image_dense(image_channel_features)
    q_feat = self.question_dense(question_features)

    return torch.mul(im_feat, q_feat)
  

  def forward(self, data):
    image, question = data
    phi = self.pretrained_extractor(image)
    psi, (h, c) = self.lstm_c(question)
    #print(phi.size())
    #print(psi.size())
    f = self.fuse(phi, h[-1])
   # print(f.size())

    output = self.classifier(f)
    #print("output", output.size())
    return output

