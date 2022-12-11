from torch import nn
import torch
from transformers import Trainer, TrainingArguments, AutoFeatureExtractor, AutoTokenizer
from preprocess import dataset, MultimodalCollator, device, showImage, answer_space
from transformer_model import TransformerModel
from metrics import compute_metrics

def create_trainer(model, train_dataset, eval_dataset, data_collator):
    '''
    Creates the huggingface trainer object to run the train/eval loop.
    '''
    args = TrainingArguments(
    output_dir = '/Users/jean/Documents/CS1470/dl-final-project/transformer/transformer-checkpoints',
    seed=12345, 
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,          
    metric_for_best_model='wups',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    num_train_epochs=12,
    fp16=True, # allows for faster training of larger models and minibatch sizes 
    dataloader_num_workers=8,# speed up data transfer between cpu and gpu 
    load_best_model_at_end=True,
)
    # create huggingface trainer 
    trainer = Trainer(model = model, args = args, train_dataset = train_dataset, eval_dataset = eval_dataset, data_collator = data_collator, compute_metrics = compute_metrics)
    return trainer 
    

def train_model(): 
    '''
    Builds the data collator, tokenizer/extractor, model, and runs the
    training and eval loop. 
    '''
    # initialize tokenizer, feature extractor and pass into multimodal collator 
    text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # initialize components 
    collator = MultimodalCollator(text_tokenizer, feature_extractor)
    model = TransformerModel().to(device) # move model to gpu 
    trainer = create_trainer(model, dataset['train'], dataset['test'], collator)
    # train 
    train_metrics = trainer.train() 
    # evaluate
    eval_metrics = trainer.evaluate() 

    return model, collator, train_metrics, eval_metrics 

def model_inference(model, collator):
    '''
    Runs inference with the trained model on instances in the 
    DAQUAR testing set. 
    '''
    # take the 1858th-1868th instances in the test set 
    sample = collator(dataset['test'][1858:1868])
    # get tensors to feed into our model 
    img_feat = sample['img_features'].to(device)
    text_feat = sample['text_features'].to(device)
    labels = sample['labels'].to(device)
    # put in evaluation mode
    model.eval() 
    # forward pass
    output = model(img_features = img_feat, text_features = text_feat, labels = labels) 
    preds = output['logits'].argmax(axis = -1).cpu().numpy() 
    for i in range(1858, 1868):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        showImage(i) # display the image that goes with this question 
        print("Prediction:\t", answer_space[preds[i-1858]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")

def model_personal_inference(model, collator):
    '''
    Runs inference with the trained model on instances in the 
    testing set containing our custom images. 
    '''
    collator.is_personal = True # preprocess using the personal dataset
    sample = \
    collator({'answer': ['chair', 'pillow', 'sofa', 'shoe', 'gray', '2', '2'], 
    'image_id': ['personal-chair', 
    'personal-couch', 
    'personal-couch', 
    'personal-couch', 
    'personal-couch', 
    'personal-couch', 
    'personal-couch'], 
    'label': torch.tensor([107, 384, 453, 438, 251, 11, 11]), 
    'question': ['what is the plastic bottle on top of',
    'what is on the sofa',
    'what is the pillow on top of',
    'what is in front of the sofa',
    'what color is the sofa',
    'how many shoes are there',
    'how many shoes are in front of the sofa']})
    img_feat = sample['img_features'].to(device)
    text_feat = sample['text_features'].to(device)
    labels = sample['labels'].to(device)
    model.eval() 
    output = model(img_features = img_feat, 
                   text_features = text_feat, 
                   labels = labels)
    preds = output['logits'].argmax(axis = -1).cpu().numpy() 
    for i in range(0, 7):
        print(answer_space[preds[i]]) # display the answer 
    collator.is_personal = False # set it back to false

def load_model():
    '''
    Creates the tokenizer/extractor, collator, and model; then,
    loads the saved trained model weights so that we don't have 
    to train from scratch. 
    '''
    # initialize tokenizer, feature extractor and pass into multimodal collator 
    text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        'google/vit-base-patch16-224-in21k')
    # initialize components 
    collator = MultimodalCollator(text_tokenizer, feature_extractor)
    model = TransformerModel().to(device) # move model to gpu 
    # load weights
    state_dict = torch.load(
        "/Users/jean/Documents/CS1470/dl-final-project/transformer/"+
        "transformer-checkpoints/checkpoint-3400/pytorch_model.bin") 
    model.load_state_dict(state_dict)
    return model, collator

###========== run this block to train the model ===============
#model, collator, train_metrics, eval_metrics = train_model()
###==========#==========#==========#==========#==========#=======

# get the model with the saved weights 
model, collator = load_model()
# run inference with DAQUAR test set 
model_inference(model, collator)
# run inference with our custom images/questions. 
model_personal_inference(model, collator)