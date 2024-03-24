import re
import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer

# read and prepare targets from input file, return pandas dataframe
def load_targets(filename, mode, remove_dom=False):
    assert mode in ['mask_dom', 'mask_dobj']
    with open(filename, mode='r', encoding='utf-8') as f:
        ids, sentences, dobjects, indices, en_sents = [],[], [], [], []
        next(f)
        for line in f:
            cols = line.split('\t')
            ids.append(cols[0].strip())
            en_sents.append(cols[2].strip())
            sent = cols[1].strip()
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, sent)
            do = matches[0]
            dobjects.append(do)
            if mode == 'mask_dom':
                idx = find_dom_index(sent.split())
            elif mode == 'mask_dobj':
                sent_list = sent.split()
                if remove_dom:
                    dom_idx = find_dom_index(sent_list)[0]
                    sent_list.pop(dom_idx)
                    sent = ' '.join(sent_list)
                idx = find_dobj_indices(sent_list)
            sentences.append(sent)
            indices.append(idx)
    return pd.DataFrame(
            {'id': ids,
            'sentence': sentences,
            'en_sentence': en_sents,
            'mask_idx': indices,
            'dobj': dobjects
            })

# helper function: find index of dom-marker (a/ al) in target sentence
def find_dom_index(sent_list, char='['):
    idx = -1
    for i, w in enumerate(sent_list):
        if w[0] == char:
            idx = i-1
    if idx < 0:
        print('Index not found in sentence: {}'.format(sent_list))
    return [idx]

# helper function: find index of direct object in target sentence
def find_dobj_indices(sent_list, start='[', end=']', remove_dom=False):
    idx1, idx2 = 0, 0
    for i, w in enumerate(sent_list):
        if w[0] == start:
            idx1 = i
        if w[-1] == end:
            idx2 = i
    idx = list(range(idx1, idx2+1))
    if idx == []:
        print('Index not found in sentence: {}'.format(sent_list))
    return idx

# setup tokenizer and model
def load_model(modelname):
    tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(modelname)
    return tokenizer, model

# initialize file for MLM outputs, print header
def initialize_result_file(outputfile, num_mask, top_n):
    with open(outputfile, mode='w', encoding='utf-8') as f:
        f.write('id'+'\t'+'sentence')
        for i in range(num_mask):
            f.write('\t'+'masked_token'+str(i+1))
            for j in range(top_n-1):
                f.write('\t'+'predicted_token'+str(j+1))
                f.write('\t'+'probability'+str(j+1))
            f.write('\t'+'predicted_token'+str(j+2)+'\t'+'probability'+str(j+2))
        f.write('\n')

# look for specific words in MLM predictions
def search_fillers(fillers, probs, wordlist):
    fillers_found, filler_probs, filler_ranks = [], [], []
    for i, filler in enumerate(fillers):
        if filler in wordlist:
            fillers_found.append(filler)
            filler_probs.append(probs[i])
            filler_ranks.append(i)
    return fillers_found, filler_probs, filler_ranks

# get input into correct format
def prep_input(inputfile, outputfile):
    with open(inputfile, mode='r', encoding='utf-8') as f:
        with open(outputfile, mode='w', encoding='utf-8') as out_f:
            for line in f.readlines():
                cols = line.split('\t')
                sentence = cols[1]
                pattern = r'(\w+\s+\w+)\s+/\s+(\w+\s+\w+)'
                replacement = r'[\1]'
                new_sentence = re.sub(pattern, replacement, sentence)
                cols[1] = new_sentence
                out_f.write('\t'.join(cols))

# get targets into correct format
def prep_input2(inputfile, outputfile, single_token_dobj=False):
    with open(outputfile, mode='w', encoding='utf-8') as out_f:
        out_f.write('id\tes\ten\n')
        with open(inputfile, mode='r', encoding='utf-8') as f:
            next(f)
            cnt = 0
            for i, line in enumerate(f.readlines()):
                if i % 2 == 0:
                    sentence = line.strip()
                    cnt += 1
                    sent_list = sentence.split()
                    dobj1 = '[' + sent_list[5] 
                    if single_token_dobj:
                        dobj1 += ']'
                    sent_list[5] = dobj1
                    if not single_token_dobj:
                        dobj2 = sent_list[6] + ']'
                        sent_list[6] = dobj2
                    sentence = ' '.join(sent_list)
                else:
                    en_sentence = line.strip()[1:-1]
                    out_f.write('\t'.join([str(cnt), sentence, en_sentence]))
                    out_f.write('\n')

# mark direct object with [] in english translations of targets
def mark_dobj_en(file, start, end):
    sentences, en_sentences, ids = [], [], []
    with open(file, mode='r', encoding='utf-8') as f:
        next(f)
        for line in f:
            cols = line.split('\t')
            ids.append(cols[0])
            sentences.append(cols[1])
            en_sent = cols[2]
            sent_list = en_sent.strip('\n').split()
            sent_list[start] = '[' + sent_list[start]
            sent_list[end] = sent_list[end] + ']'
            en_sent = ' '.join(sent_list)
            en_sentences.append(en_sent)
    df = pd.DataFrame({'id': ids, 'es': sentences, 'en': en_sentences})
    return df