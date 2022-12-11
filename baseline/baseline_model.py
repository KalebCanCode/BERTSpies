import collections
import os
from datasets import load_dataset
import torch
import sys
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(1, '/Users/jean/Documents/CS1470/dl-final-project/transformer')
from metrics import batch_wup_measure
from preprocess import dataset

# load in the entire dataset, not differenting between train/test sets. 
full_dataset = load_dataset(
    "csv", 
    data_files={"data": os.path.join("dataset","data.csv")}
)
# load in the list of all possible answers.  
with open(os.path.join("dataset", "answer_space.txt")) as f:
    answer_space = f.read().splitlines()


def most_freq_ans(): 
    '''
    Goes through the entire DAQUAR dataset to return the index of the
    most frequent answer in the list of all possible answers. 
    '''
    # get a list of all the answers in the dataset 
    words = [ans for ans in full_dataset['data']['answer']]
    # map words to their frequencies in the dataset
    word2freq = collections.Counter(words)
    # return most common word and its index in the answer space 
    # (to be used as the label)
    most_freq = word2freq.most_common(1)[0][0]
    return most_freq, answer_space.index(most_freq)

def create_baseline(data):
    '''
    Creates a baseline model using scikit-learn. This model
    only predicts the answer "2" (the most frequent answer in DAQUAR),
    ignoring the question and the image. 
    '''
    most_freq, idx = most_freq_ans()
    # define dummy classifier.
    # label is the index of the answer word in answer_space, which is given
    # by most_freq_ans(). 
    dummy_clf = DummyClassifier(strategy="constant",
                                 constant = torch.tensor([idx])) 
    # fit on X, Y (X does not matter because it only needs to learn to predict 
    # Y, which is the index of "2" in the answer space.)
    dummy_clf.fit(None, torch.tensor([idx])) 
    # do prediction (will always predict 2) and get wup, accuracy, and f1 score
    preds = dummy_clf.predict(torch.rand(len(dataset[data]))) 
    labels = torch.tensor(dataset[data]['label'])
    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }

# print train and test metrics 
print("train: ", create_baseline('train'))
print("test: ", create_baseline('test'))





