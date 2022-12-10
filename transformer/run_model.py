from torch import nn
import torch
from transformers import Trainer, TrainingArguments, AutoFeatureExtractor, AutoTokenizer
from preprocess import dataset, MultimodalCollator, device, showImage, answer_space
from transformer_model import TransformerModel
from metrics import compute_metrics

def create_trainer(model, train_dataset, eval_dataset, data_collator):
    args = TrainingArguments(
    output_dir = '/Users/jean/Documents/CS1470/dl-final-project/transformer/transformer-checkpoints',
    seed=12345, 
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,             # Save only the last 3 checkpoints at any given time while training 
    metric_for_best_model='wups',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    num_train_epochs=12,
    fp16=True,
    # warmup_ratio=0.01,
    # learning_rate=5e-4,
    # weight_decay=1e-4,
    # gradient_accumulation_steps=2,
    dataloader_num_workers=8,
    load_best_model_at_end=True,
)

    # training_args = TrainingArguments(
        
    #     seed = 1470,
    #     save_strategy = 'epoch',
    #     evaluation_strategy = 'epoch',
    #     logging_strategy = 'epoch',
    #     metric_for_best_model = 'wups',
    #     remove_unused_columns = False,
    #     num_train_epochs = 5,
    #     load_best_model_at_end = True, 
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=32,
    #     learning_rate = 5e-4,
    #     warmup_ratio=0.01,
    #     fp16 = True, # allows for faster training of larger models and minibatch sizes 
    #     dataloader_num_workers = 4 # speed up data transfer between cpu and gpu 
    #     )
    trainer = Trainer(model = model, args = args, train_dataset = train_dataset, eval_dataset = eval_dataset, data_collator = data_collator, compute_metrics = compute_metrics)
    return trainer 
    

def train_model(): 
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
    sample = collator(dataset['test'][1858:1868])
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
        showImage(i)
        print("Prediction:\t", answer_space[preds[i-1858]])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$")

def model_personal_inference(model, collator):
    collator.is_personal = True # preprocess using the personal dataset
    sample = collator({'answer': ['chair', 'pillow', 'sofa', 'shoe', 'gray', '2', '2'], 
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
    output = model(img_features = img_feat, text_features = text_feat, labels = labels)
    preds = output['logits'].argmax(axis = -1).cpu().numpy() 
    print(answer_space[preds[i] for i in range(0, 8)])
    collator.is_personal = False # set it back to false

#model, collator, train_metrics, eval_metrics = train_model()
def load_model():
    # initialize tokenizer, feature extractor and pass into multimodal collator 
    text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # initialize components 
    collator = MultimodalCollator(text_tokenizer, feature_extractor)
    model = TransformerModel().to(device) # move model to gpu 
    state_dict = torch.load("/Users/jean/Documents/CS1470/dl-final-project/transformer/transformer-checkpoints/checkpoint-3400/pytorch_model.bin")
    model.load_state_dict(state_dict)
    return model, collator

model, collator = load_model()
model_inference(model, collator)
model_personal_inference(model, collator)