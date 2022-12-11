# BERT Spies With Its Little Eye... Exploring DL Models for Visual Question Answering! 
- Kaleb Newman (knewman7) and Jean Yoo (syoo28)

## Write-ups
- [Initial Proposal](https://docs.google.com/document/d/1Z4r1GkDDwVOqxmIV4XX5k4O0qltpDV-t1r4HskmrpLY/edit?usp=sharing)
- [Check-in](https://docs.google.com/document/d/10-mT3ACLgZQA4HUKl9OeobX3EFiwjnylXC3y7QeCDl0/edit?usp=sharing)
- [Final Report](https://docs.google.com/document/d/1IRxeYBmB52jgfnPeFVLG4XoflD7sOoSPXmC8kn82U40/edit?usp=sharing)

## Poster
- [Click here!](https://docs.google.com/presentation/d/1rihk-RQizOLtd67WkPpp-awQ4Q0Jr8zs/edit?usp=sharing&ouid=107485465646607954788&rtpof=true&sd=true)

## Main files/folders 
- `baseline`: contains the file that builds and runs the baseline model on the training and testing set. 
- `dataset`: contains the images, questions, and answers from the DAQUAR dataset. 
- `personal`: contains the custom images taken by us for model inference. 
- `transformer`:
    - `preprocess.py`: code for loading in the dataset in tensor form. 
    - `transformer_model.py`: contains the architecture for our transformer-based model. 
    -   `metrics.py`: code for calculating WUPs, accuracy, and f1 score. 
    - `run_model.py`: builds the transformer-based model and runs the train/eval loop, as well as model inference. 
- `twochannel`: 
    - `preprocess.py`: maps the words to their vocab indices, converts images to rgb tensors, and defines the collating function for the DataLoader. 
    - `two_channel_nn.py`: architecture for our two-channel vision + langauge model. 
    - `metrics.py`: same code as transformer, but per-batch rather than per-epoch. 
    - `run_two_channel.py`: runs the training/eval loop on the two-channel model, and carries out inference for both the DAQUAR and custom testing set. 
- `vqa.ipynb`: contains the output/results from running both models. in notebook form, because we had to run the files through Google Colab to use their GPU! 
## Brief Rundown of Metrics 
- Baseline: 0.04 accuracy, 0.07 WUPs
- Two-channel: 0.22 accuracy, 0.27 WUPs
- RoBERTa + ViT: 0.30 accuracy, 0.34 WUPs
- More detailed results in the poster + final write-up! 
## Documentations/Resources used 
- [HuggingFace Tokenizer](https://huggingface.co/transformers/v3.0.2/main_classes/tokenizer.html)
- [How to freeze parameters](https://discuss.huggingface.co/t/how-to-freeze-layers-using-trainer/4702)
- [How to use the Trainer API](https://huggingface.co/course/chapter3/3?fw=pt)
- [How to write a custom training loop in PyTorch](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [RoBERTa documentation](https://huggingface.co/docs/transformers/model_doc/roberta)
- [ViT documentation](https://huggingface.co/docs/transformers/model_doc/vit)
- [WUPs understanding](https://blog.thedigitalgroup.com/words-similarityrelatedness-using-wupalmer-algorithm#:~:text=The%20Wu%20%26%20Palmer%20calculates%20relatedness,)
- [What are Synsets?](https://www.nltk.org/howto/wordnet.html)
- [DAQUAR dataset/WUPs code](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge/)
- [How to use PyTorch's DataLoader](https://pytorch.org/docs/stable/data.html)
- [How to extract hidden and cell states?](https://discuss.pytorch.org/t/retrieving-hidden-and-cell-states-from-lstm-in-a-language-model/34989/5)
- [How to checkpoint model params?](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)
- [Using scikit-learn's Dummy Classifier](https://scikit-learn.org/0.24/modules/generated/sklearn.dummy.DummyClassifier.html)
