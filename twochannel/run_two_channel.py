from two_channel_nn import TwoChanNN 
import argparse
import torch
import time
from preprocess import process_words, dataset
from torchvision import models
from torchsummary import summary


def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Let's Do this Two Channel Thing :D", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--extractor',           required=True,         help='Feature Extractor such as VGG16')
    parser.add_argument('--task',           required=True,              choices=['train', 'inference'],  help='Task to run')
    parser.add_argument('--feat_size', required=True,     type=int,                   help='Feature size  of extractor')
    parser.add_argument('--device',  required=True,    type=str,                   help='Device Using')
    # parser.add_argument('--data',           required=True,              help='File path to the assignment data file.')
    parser.add_argument('--epochs',         type=int,   default=10,      help='Number of epochs')
    parser.add_argument('--lstm_units',     type=int, default=512,      help='Hidden Size of lstm')
    
    
    # parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    # parser.add_argument('--batch_size',     type=int,   default=100,    help='Model\'s batch size.')
    # parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    # parser.add_argument('--window_size',    type=int,   default=20,     help='Window size of text entries.')
    # parser.add_argument('--chkpt_path',     default='',                 help='where the model checkpoint is')
    # parser.add_argument('--check_valid',    default=True,               action="store_true",  help='if training, also print validation after each epoch')
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs, device):
    summary(model)
    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    start_time_sec = time.time()

    for epoch in range(1, epochs+1):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0
        # get vocab first 

        for batch in train_loader: # this is a list of dictionaries mapping answer, image_id, . etc -> data 

            optimizer.zero_grad()
            
            # do preprocessing here
            # process_words with the vocab and the dictionary. this gives the
            # question as a tensor
            # also keep track of labels in a label tensor list 

            # stack q tensors to get a batch of question tensors
            # stack label tensors to get a batch of label tensors 
            # 

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)


        if epoch % 1 ==0:
          print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

def main(args):
    model = TwoChanNN(args.extractor, args.lstm_units, args.feat_size, 'vocab_size')
    optimizer = torch.optim.Adam
    loss = torch.nn.CrossEntropyLoss

    # maybe need this?
    training_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=4, shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=4, shuffle=False, num_workers=2)

    if args.task == 'train':
        word2idx, vocab_size, train_q, test_q  = process_words()
        history = train(model, training_loader, validation_loader, args.device)
        print(history)
    
    if args.task == 'inference':
        return None
        





    

