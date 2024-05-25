import re
import os
import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer


def load_targets(input_path, source, mask_type=None, remove_dom=False):
    df = pd.read_csv(input_path, sep='\t')
    idx = []
    if source is not None:
        df = df[df['source']==source]
    if mask_type == 'dom':
        idx = list(df['dom_idx'])
    else:
        sentences = []
        for i, row in df.iterrows():
            sent = row['sentence']
            dom_idx = row['dom_idx']
            sent_list = sent.split()
            if remove_dom:
                if row['dom'] == 'al':
                    sent_list[dom_idx] = '[el'
                    sent_list[dom_idx+1] = sent_list[dom_idx+1][1:]
                else:
                    sent_list.pop(dom_idx)
                    dom_idx -= 1
            sentences.append(' '.join(sent_list)) # join to string
            dobj_len = len(row['dobject'].split())
            if mask_type == 'noun':
                idx.append(dom_idx+dobj_len)
            else:
                if dobj_len > 1: # skip sentence, no separate determiner
                    idx.append(dom_idx+1)
                else:
                    idx.append(-1)
        df['sentence'] = sentences
    df['sentence'] = df['sentence'].str.replace('[', '').str.replace(']', '') # remove brackets
    if mask_type is not None:
        df['mask_idx'] = idx
        df = df[df['mask_idx'] > 0] # remove sentences that have no tokens to mask
    return df

# helper function: find index of dom-marker (a/ al) in target sentence
def find_dom_index(sent_list, char='['):
    idx = -1
    for i, w in enumerate(sent_list):
        if w in ['a', 'al'] and sent_list[i+1][0] == char:
            idx = i
            break
    if idx < 0:
        print('Index not found in sentence: {}'.format(sent_list))
    return idx

# helper function: find index of direct object in target sentence
def find_dobj_indices(sent_list, start='[', end=']'):
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

# look for specific words in MLM predictions
def search_fillers(fillers, probs, wordlist):
    fillers_found, filler_probs, filler_ranks = [], [], []
    for i, filler in enumerate(fillers):
        if filler in wordlist:
            fillers_found.append(filler)
            filler_probs.append(probs[i])
            filler_ranks.append(i)
    return fillers_found, filler_probs, filler_ranks

# get targets into correct format
def prep_input(inputfile, outputfile, idx=5):
    with open(outputfile, mode='w', encoding='utf-8') as out_f:
        out_f.write('id\tes\ten\n')
        with open(inputfile, mode='r', encoding='utf-8') as f:
            next(f)
            cnt = 0
            for line in f:
                    line_list = line.split('\t')
                    print(line_list)
                    sentence = line_list[1]
                    cnt += 1
                    sent_list = sentence.split()
                    if idx < len(sent_list):
                        sent_list[idx] = '[' + sent_list[idx] 
                    else:
                        sent_list[-1] = '[' + sent_list[-1] 
                    if idx+1 < len(sent_list):
                        dobj2 = sent_list[idx+1] + ']'
                        sent_list[idx+1] = dobj2
                    else: 
                        sent_list[-1] = sent_list[-1]+']'
                    sentence = ' '.join(sent_list)
                    id = line_list[0]
                    en_sentence = line_list[2]
                    en_sent_list = en_sentence.split()
                    if idx-1 < len(en_sent_list):
                        en_sent_list[idx-1] = '[' + en_sent_list[idx-1]
                    else:
                        en_sent_list[-1] = '[' + en_sent_list[-1]
                    if idx < len(en_sent_list):
                        en_sent_list[idx] = en_sent_list[idx] + ']'
                    else:
                        en_sent_list[-1] = en_sent_list[-1] + ']'
                    en_sentence = ' '.join(en_sent_list)
                    out_f.write('\t'.join([id, sentence, en_sentence+'\n']))

def merge_test_data(directory):
    es_sents, en_sents, dom_markers, dobjects, dom_idxs, genus, numerus, source, conditions = [], [], [], [], [], [], [], [], []
    cnt = 0 
    for file in os.listdir(directory):
        print(file)
        sou = file.split('_')[0]
        name = file.split('_')[1]
        cond = name.split('-')[0]
        print(sou)
        path = directory + file
        with open(path, mode='r', encoding='utf-8') as f:
            next(f)
            for line in f.readlines():
                cnt += 1
                line_list = line.split('\t')
                es_sent = line_list[1]
                pattern = r'\[(.*?)\]'
                matches = re.findall(pattern, es_sent)
                dobject = matches[0]
                sent_list = es_sent.split()
                dom_idx = find_dom_index(sent_list)
                dom = sent_list[dom_idx] if dom_idx > -1 else ''
                if len(dobject.split()) > 1:
                    det = dobject.split()[0]
                    det_sing = ['la', 'el', 'un', 'una']
                    det_fem = ['la', 'las', 'una'] 
                    if det in det_sing: 
                        num = 'sing'
                    else:
                        num = 'pl'
                    if det in det_fem:
                        gen = 'f'
                    else:
                        gen = 'm'
                else:
                    num = 'sing'
                    gen = 'm'
                source.append(sou)
                conditions.append(cond)
                es_sents.append(es_sent)
                dom_markers.append(dom)
                dobjects.append(dobject)
                dom_idxs.append(dom_idx)
                genus.append(gen)
                numerus.append(num)
                en_sents.append(line_list[2].strip('\n'))
    ids = list(range(1, cnt+1)) 
    animate = ['?' for x in ids]
    definite = animate
    affected = animate
    df = pd.DataFrame(
            {'id': ids,
             'source': source,
             'condition': conditions,
             'sentence': es_sents,
             'dom': dom_markers, 
             'dom_idx': dom_idxs,
             'dobjects': dobjects,
             'gender': genus, 
             'number': numerus,
             'animate': animate,
             'definite': definite,
             'affected': affected,
            'en_sentence': en_sents
            })
    return df
