import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules.activation import Softmax


class TwoChanNN(nn.Module):
  def __init__(self,image_feature_extractor, lstm_units, num_vgg_features, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.pretrained_extractor = image_feature_extractor
    # in_features = self.pretrained_extractor.classifier[-1].in_features
    # self.pretrained_extractor.classifier = nn.Sequential(*list(self.pretrained_extractor.children())[:-1])
    self.hidden_units = lstm_units 
    #nn.LSTM(input_size, hidden_size, num_layers)
    self.lstm_a = nn.LSTM(512, 512, 1, batch_first=True)
              
    self.lstm_b = nn.LSTM(512, 512, 1, batch_first=True)

    self.lstm_c = nn.LSTM(2048, 1024, 3, batch_first=True)

    self.lstm_d = nn.LSTM(512, 512, 2, batch_first=True)
    self.lstm_fc = nn.Linear(2048, 1024)
     
    #nn.Linear(input_size, output_size)
    self.image_dense = nn.Sequential(
        #pretrained outputs num_ftrs = model_ft.fc.in_features vector
        nn.Linear(1000, 1024),# CHANGED THIS num_vgg_features, 1024)
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
        #1000 or 500 because of dropout??
        nn.Tanh(),
        nn.Linear(1000, 582),

    )
    self.embedding = nn.Embedding(self.vocab_size, 512)
    self.tanh = nn.Tanh()
#Motivating example  
# >>> rnn = nn.LSTM(10, 20, 2)
# >>> input = torch.randn(5, 3, 10)
# >>> h0 = torch.randn(2, 3, 20)
# >>> c0 = torch.randn(2, 3, 20)
# >>> output, (hn, cn) = rnn(input, (h0, c0))

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


    #print("embedding", embeddings.size()) #(4, 25, 512)
    # output_a, (final_hidden_a, final_cell_a) = self.lstm_a(embeddings) #(4, 2, 512), (1, 4, 512), (1, 4, 512)
    # #print("outputa", output_a.size(), final_hidden_a.size(), final_cell_a.size())
    # output_b, (final_hidden_b, final_cell_b) = self.lstm_b(nn.Tanh()(output_a), (nn.Tanh()(final_hidden_a), nn.Tanh()(final_cell_a)))
    # #not sure about the dim here
    # return torch.cat((final_hidden_a[0], final_cell_a[0], final_hidden_b[0], final_cell_b[0]), dim=1)
  
  def fuse(self, im_feat, q_feat):
    # im_feat = self.image_dense(image_channel_features)
    # q_feat = self.question_dense(question_features)

    return torch.mul(im_feat, q_feat)
  

  def forward(self, data):
    image, question = data
    phi = self.vision_channel(image)
    psi = self.language_channel(question)
    # psi, _ = self.lstm_c(psi)
    #print(phi.size())
    #print(psi.size())
    f = self.fuse(phi, psi)
   # print(f.size())

    output = self.classifier(f)
    #print("output", output.size())
    return output

