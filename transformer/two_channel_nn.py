import torch
import torch.nn as nn


from torch.nn.modules.activation import Softmax


class TwoChanNN(nn.Module):
  def __init__(self,image_feature_extractor, lstm_units, num_vgg_features, vocab_size):
    self.vocab_size = vocab_size
    self.pretrained_extractor = image_feature_extractor
    self.hidden_units = lstm_units 
    #nn.LSTM(input_size, hidden_size, num_layers)
    self.lstm_a = nn.LSTM(512, 512, 1)
    self.lstm_b = nn.LSTM(512, 512, 1)
    #nn.Linear(input_size, output_size)
    self.image_dense = nn.Sequential(
        #pretrained outputs num_ftrs = model_ft.fc.in_features vector
        nn.Linear(num_vgg_features, 1024),
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
        nn.Linear(500, 1000),
        # do we need this tanh then softmax??
        nn.Tanh(),
        nn.Softmax()


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
    output_a, (final_hidden_a, final_cell_a) = self.lstm_a(embeddings)
    output_b, (final_hidden_b, final_cell_b) = self.lstm_b(final_hidden_a, (final_hidden_a, final_cell_a))
    #not sure about the dim here
    return torch.cat((final_hidden_a, final_cell_a, final_hidden_b, final_cell_b), dim=2)
  
  def fuse(self, image_channel_features, question_features):
    im_feat = self.image_dense(image_channel_features)
    q_feat = self.question_dense(question_features)

    return torch.mul(im_feat, q_feat)
  

  def call(self, data):
    image, question = data
    phi = self.pretrained_extractor(image)
    psi = self.language_channel(question)
    f = self.fuse(phi, psi)

    output = self.classifier(f)
    return output

