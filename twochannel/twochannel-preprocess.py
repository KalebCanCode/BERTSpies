
import torch
import numpy as np
from datasets import load_dataset

dataset = load_dataset(
    "csv", 
    data_files={
        "train": os.path.join("dataset","data_train.csv"),
        "test": os.path.join("dataset", "data_eval.csv")
    }
)
def get_words(sentences):
    to_return = []
    for sentence in sentences:
        for word in sentence.split():
            to_return.append(word)
    return to_return 

def pad_sentences(sentences, max_length):
    for sentence in sentences:
        sentence += (max_length + 1 - len(sentence)) * ['<pad>'] 

def process_words():
    word2idx = {}
    vocab_size = 0
    # get all question and answers from train and test set using get_words
    train_questions, train_answers = get_words(dataset['train']['question']), get_words(dataset['train']['answer'])
    test_questions, test_answers = get_words(dataset['train']['question']), get_words(dataset['test']['answer'])
    # pad all of them (not the answers)
    train_questions, test_questions = pad_sentences(train_questions), pad_sentences(test_questions)

    for caption in train_questions:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word] # this converts the word string to the index mapping 
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1

    for caption in test_questions:
        for index, word in enumerate(caption):
            caption[index] = word2idx[word] 

    for caption in train_answers:
        for index, word in enumerate(caption):
            if word not in word2idx:
                word2idx[word] = vocab_size
            else:
                vocab_size += 1

    for caption in test_answers:
        for index, word in enumerate(caption):
            if word not in word2idx:
                word2idx[word] = vocab_size
    
    return word2idx, vocab_size, torch.tensor(train_questions), torch.tensor(test_questions)
    
    # now train and test captions should be updated to be ints, which
    # we can then cast as np arrays, then cast as tensors to be fed into 
    # the model..?
    
    



