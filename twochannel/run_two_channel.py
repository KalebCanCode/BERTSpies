from two_channel_nn import TwoChanNN 
import argparse
import torch
import time
from preprocess import training_loader, val_loader, word2idx, dataset, convert_img, process_words, answer_space
from torchvision import models
from torchsummary import summary
from metrics import in_batch_wup_measure
import numpy as np

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs, device):
    '''
    Runs the training/eval loop. 
    '''
    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['wups'] = []
    history['val_wups'] = []

    start_time_sec = time.time()

    for epoch in range(1, epochs+1):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0
        avg_wups = []
        for batch in train_loader: 
            q_feats    = torch.stack(batch['q_tensor'], axis = 0).to(device)
            img_feats    = torch.stack(batch['image_id'], axis = 0).to(device)
            labels = torch.tensor(batch['label']).to(device)

            yhat = model((img_feats, q_feats))
            loss = loss_fn(yhat, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss         += loss.item() 

            yhat = torch.log_softmax(yhat, dim=1)
            num_train_correct  += (torch.argmax(yhat, 1) == labels).sum().item()
            num_train_examples += 32
            
            yhat_list = torch.argmax(yhat,1).tolist()
            label_list = batch['label']
            wups = in_batch_wup_measure(label_list, yhat_list)
            avg_wups.append(wups)
            
        print(num_train_correct)
        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / num_train_examples
        a = np.concatenate(avg_wups).flatten()
        avg_wups = np.mean(a)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0
        val_wups = []

        for batch in val_loader:
            q_feats    = torch.stack(batch['q_tensor'], axis = 0).to(device)
            img_feats    = torch.stack(batch['image_id'], axis = 0).to(device)
            labels = torch.tensor(batch['label']).to(device)
            with torch.no_grad(): 
                yhat = model((img_feats, q_feats))
                loss = loss_fn(yhat, labels)
                yhat = torch.log_softmax(yhat, dim=1)

            val_loss         += loss.item()
            num_val_correct  += (torch.argmax(yhat, 1) == labels).sum().item()
            num_val_examples += 32

            yhat_list = torch.argmax(yhat,1).tolist()
            label_list = batch['label']
            vwups = in_batch_wup_measure(label_list, yhat_list)
            val_wups.append(vwups)
            

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / num_val_examples
        v = np.concatenate(val_wups).flatten()
        val_wups = np.mean(v)


        if epoch % 1 ==0:
          print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, train wups:%5.2f, val wups:%5.2f, val loss: %5.2f, val acc: %5.2f' % \
                (epoch, epochs, train_loss, train_acc, avg_wups, val_wups, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['wups'].append(avg_wups)
        history['val_wups'].append(val_wups)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history, model

# importing VGG-19 
import torchvision.models as models
model_ft = models.vgg19(pretrained=True)  
# freeze parameters!  
for param in model_ft.parameters():
    param.requires_grad = False 
model_ft.to('cuda')
# build our two-channel model 
model = TwoChanNN(model_ft, 512, 4096, len(word2idx))
model.to('cuda')
# define optimizers 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss = torch.nn.CrossEntropyLoss()
# train
history, model = train(model, optimizer, loss, training_loader, val_loader, 10, 'cuda')


def model_inference(model, image_id, question):
    model.eval()
    #process image
    im = convert_img(img_id=image_id).to('cuda')
    # reshape to (batch, height, width, channel)
    im1 = torch.unsqueeze(im, dim=0)
    # same as above but for questions 
    q = process_words(word2idx, question).to("cuda")
    q1 = torch.unsqueeze(q, dim=0)

    output = model((im1,q1))

    preds = torch.argmax(output,1)

    # convert prediction to answer word/string 
    return answer_space[preds]

# inference in test set
print(model_inference(model, 'image313', 'what is below the board'))
print(model_inference(model, 'image210', 'what is behind the sofa'))
print(model_inference(model, 'image1086', 'how many sofas are there'))
print(model_inference(model, 'image1317', 'what is found below the window'))
print(model_inference(model, 'image1315', 'what is to the right of the bookshelf'))
print(model_inference(model, 'image724', 'what is to right of door'))
print(model_inference(model, 'image1160', 'what is at the bottom of the photo'))
print(model_inference(model, 'image1304', 'what is the largest object'))
print(model_inference(model, 'image629', 'what is to the left of the shelf'))
print(model_inference(model, 'image62', 'what is in front of the television monitor'))

# inference using our own pictures!
print(model_inference(model, 'personal-chair', 'what is the plastic bottle on top of'))
print(model_inference(model, 'personal-couch', 'what is on the sofa'))
print(model_inference(model, 'personal-couch', 'what is the pillow on top of'))
print(model_inference(model, 'personal-couch', 'what is in front of the sofa'))
print(model_inference(model, 'personal-couch', 'what color is the sofa'))
print(model_inference(model, 'personal-couch', 'how many shoes are there'))
print(model_inference(model, 'personal-couch', 'how many shoes are in front of the sofa'))
