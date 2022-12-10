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
    model.is_personal = True
    sample = collator({'answer': ['pillow'], 'image_id': ['personal-couch'], 'label': torch.tensor([384]), 'question': ['what is on the couch']})
    img_feat = sample['img_features'].to(device)
    text_feat = sample['text_features'].to(device)
    labels = sample['labels'].to(device)
    model.eval() 
    output = model(img_features = img_feat, text_features = text_feat, labels = labels)
    preds = output['logits'].argmax(axis = -1).cpu().numpy() 
    print(answer_space[preds[0]])
    model.is_personal = False # set it back to false

#model, collator, train_metrics, eval_metrics = train_model()
text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# initialize components 
collator = MultimodalCollator(text_tokenizer, feature_extractor)
model = TransformerModel()
checkpoint = torch.load('/Users/jean/Documents/CS1470/dl-final-project/transformer/transformer-checkpoints/runs/Dec05_22-02-26_jeans-mbp-2.devices.brown.edu/events.out.tfevents.1670295746.jeans-mbp-2.devices.brown.edu.28802.0')
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
model_inference(model, collator)
model_personal_inference(model, collator)