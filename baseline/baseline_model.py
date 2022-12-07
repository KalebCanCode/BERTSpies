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

full_dataset = load_dataset(
    "csv", 
    data_files={"data": os.path.join("dataset","data.csv")}
)
with open(os.path.join("dataset", "answer_space.txt")) as f:
    answer_space = f.read().splitlines()


def most_freq_ans(): 
    # get a list of all the answers in the dataset 
    words = [ans for ans in full_dataset['data']['answer']]
    # map words to their frequencies in the dataset
    word2freq = collections.Counter(words)
    # return most common word and its index in the answer space (to be used as the label)
    most_freq = word2freq.most_common(1)[0][0]
    return most_freq, answer_space.index(most_freq)

#print(dataset['test']['label'])
def create_baseline():
    most_freq, idx = most_freq_ans()
    dummy_clf = DummyClassifier(strategy="constant", constant = torch.tensor([idx])) # label is the index of the answer word in answer_space..
    dummy_clf.fit(None, torch.tensor([idx])) 
    preds = dummy_clf.predict(torch.rand(len(dataset['test']))) # do prediction (will always predict 2) and get wup, accuracy, and f1 score? 
    labels = torch.tensor(dataset['test']['label'])
    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }
idk = create_baseline()
print(idk)





