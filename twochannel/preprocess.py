import torch
import numpy as np
import os 
from datasets import load_dataset
from PIL import Image

full_dataset = load_dataset(
    "csv", 
    data_files={
        "train": os.path.join("dataset","data_train.csv"),
        "test": os.path.join("dataset", "data_eval.csv")
    }
)
def get_words(sentences):
    to_return = []
    for sentence in sentences:
        sentence_lst = []
        for word in sentence.split():
            sentence_lst.append(word)
        to_return.append(sentence_lst)
    return to_return 

def get_sent(sentence):
    to_return = []
    for word in sentence.split():
        to_return.append(word)
    return to_return 

def pad_sentences(sentences, max_length):
    for sentence in sentences:
        sentence += (max_length - len(sentence)) * ['<pad>']
    return sentences

def pad_sentence(sentence, max_length):
    sentence += (max_length - len(sentence)) * ['<pad>']
    return sentence

def build_vocab():
    word2idx = {}
    vocab_size = 0
    # get all question and answers from train and test set using get_words
    train_questions, train_answers = get_words(full_dataset['train']['question']), get_words(full_dataset['train']['answer'])
    test_questions, test_answers = get_words(full_dataset['test']['question']), get_words(full_dataset['test']['answer'])
    # pad all of them (not the answers)
    train_questions, test_questions = pad_sentences(train_questions, 27), pad_sentences(test_questions, 27)

    for caption in train_questions:
        for index, word in enumerate(caption):
            if word not in word2idx:
                 word2idx[word] = vocab_size
                 vocab_size += 1

    for caption in test_questions:
        for index, word in enumerate(caption):
            if word not in word2idx:
                word2idx[word] = vocab_size


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
    
    return word2idx, vocab_size

    
def process_words(vocab, data): 
    '''takes a sentence in a batch and the word2idx (vocab) and converts sentence to a tensor'''
    questions = get_sent(data)
    questions = pad_sentence(questions, 27)

    for index, word in enumerate(questions):
        questions[index] = vocab[word] # this converts the word string to the index mapping 
    return torch.tensor(questions)



def convert_img(img_id):
    if img_id[0:8] == 'personal':
        image = Image.open(os.path.join("personal", img_id+".png")).convert('RGB')
    else:
        image = Image.open(os.path.join("dataset", "images", img_id+".png")).convert('RGB')
    image_numpy = np.array(image.resize((32, 32)))
    image_numpy = torch.transpose(torch.FloatTensor(image_numpy), 0, 2)
    return torch.transpose(image_numpy, 2, 1)

word2idx, vocab_size= build_vocab()

dataset = load_dataset(
    "csv", 
    data_files={
        "train": os.path.join("dataset","data_train.csv"),
        "test": os.path.join("dataset", "data_eval.csv")
    }
)

with open(os.path.join("dataset", "answer_space.txt")) as f:
    answer_space = f.read().splitlines()

dataset = dataset.map(
    lambda examples: {
        'label': [
            answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
            for ans in examples['answer']
        ]
    },
    batched=True
)

dataset = dataset.map(
    lambda examples: {
        'q_tensor': [
            process_words(word2idx, q)
            for q in examples['question']
        ]
    },
    batched=True
)

dataset = dataset.map(
    lambda examples: {
        'i_tensor': [
            convert_img(id)
            for id in examples['image_id']
        ]
    },
    batched=True
)

def collate_fn(list_items):
    label = []
    img = []
    q = []
    for d in list_items: 
        label.append(d['label'])
        img.append(torch.tensor(d['i_tensor']))
        q.append(torch.tensor(d['q_tensor']))
    return {'q_tensor': q, 'image_id': img, 'label': label}

training_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=30, shuffle=True, collate_fn = collate_fn)
val_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=30, shuffle=True, collate_fn = collate_fn)