#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from preprocess import answer_space

def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score. Similarity threshold
    is the lowest similarity score needed to be considered similar.
    Much of this code was taken from the WUP code provided by the
    authors of the DAQUAR dataset. 
    """
    def get_semantic_field(a):
        '''
        Gets the WordNet Synset for the input word,
        i.e. nouns that are the synonyms of a. 
        '''
        semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
        return semantic_field

    if a==b:
        # if a and b are the same word, return 1 
        return 1.0

    if a==[] or b==[]:
        # no comparison possible 
        return 0


    interp_a = get_semantic_field(a) 
    interp_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        # if no synsets were found 
        return 0

    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y) # use wordnet's wup similiarity 
            if local_score > global_max: # if we found a new max WUPS 
                global_max=local_score

    if global_max < similarity_threshold:
        interp_weight = 0.1 # downweight the WUPS; considered not similar enough 
    else:
        interp_weight = 1.0 # similar 

    final_score=global_max*interp_weight # downweight if needed 
    return final_score 

def batch_wup_measure(labels, preds):
    '''
    Calculates the mean WUP measure for a batch. 
    '''
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for 
                  label, pred in zip(labels, preds)]
    return np.mean(wup_scores)


def compute_metrics(eval_tuple):
    '''
    Compute WUPS, accuracy, and f1 score for this batch. 
    '''
    logits, labels = eval_tuple
    preds = logits.argmax(axis=-1)
    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro')
    }