import torch
import numpy as np
from util import load_targets, prep_targets, get_masked_indices
from transformers import BertForMaskedLM, BertTokenizer

#TODO: create class
# setup tokenizer and model
def load_model(modelname):
    tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(modelname)
    return tokenizer, model

def test_mask_dom(sentences, indices, model, tokenizer, n_tokens=5):
    prep_targets(sentences, indices)
    top_masked_tokens, top_masked_probs = [], []
    for sent in sentences:
        # tokenize
        tokens = tokenizer.tokenize(sent)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        # get masked token indices
        mask_idx = get_masked_indices(tokens)[0]

        # predict
        predictions = model(tokens_tensor)[0]
        n_tokens = n_tokens
        
        top_tokens, top_probs = [], []
        # get mask token predictions
        probs = torch.nn.functional.softmax(predictions[0, mask_idx], dim=-1) 
        top_probs, top_indices = torch.topk(probs, n_tokens, sorted=True) # get highest probabilites
        top_tokens = tokenizer.convert_ids_to_tokens(top_indices) # get tokens with highest probability
        top_masked_tokens.append(top_tokens)
        top_masked_probs.append(top_probs.tolist())
        print(sent)
        print('most likely tokens: {}'.format(top_tokens))
        print('top probabilities: {}'.format(top_probs.tolist()))
    return top_masked_tokens, top_masked_probs

def analyze_output(tokens, probs):
    dom_ranking = [t.index('a') for t in tokens]
    dom_probs = []
    for i, p in enumerate(probs):
        rank = dom_ranking[i]
        dom_probs.append(p[rank])
    return dom_ranking, dom_probs