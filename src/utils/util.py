import re
import os
import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer


def load_targets(input_path, source, mask_type='dom', remove_dom=False):
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
                if sent_list[dom_idx] == 'al':
                    sent_list[dom_idx] == 'el'
                else:
                    sent_list.pop(dom_idx)
                    dom_idx -= 1
            # print(f'sent_list: {sent_list}')
            sentences.append(' '.join(sent_list)) # join to string
            # print(f'dobj: {row["dobject"]}')
            article = row['dobject'].split()[0]
            if article in ['la', 'los', 'las', 'un', 'una', 'su']: # , 'el']: # filter out dobjects without article
                idx.append(dom_idx+1)
                # print(f'index: {dom_idx+1}')
            # elif sent_list[dom_idx] == 'al':
            #     idx.append(dom_idx)
            else: # skip sentence, no separate article
                print(f'SKIP SENTENCE: {sent_list}')
                idx.append(-1)
                # print(f'index: {-1}')
        df['sentence'] = sentences
    df['sentence'] = df['sentence'].str.replace('[', '').str.replace(']', '') # remove brackets
    # filter for sentences with separate article
    # print(f'df shape before filtering: {df.shape}')
    df['mask_idx'] = idx
    if mask_type == 'article':
        df = df[df['mask_idx'] > 0] # remove sentences that have no tokens to mask
    # print(f'df shape after: {df.shape}')
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

def articlemasking_postprocessing(dir):
    ordered_files = ['ms-2013-results.tsv','sa-2020-results.tsv','re-2021-results.tsv', 
                     're-2021-modified-results.tsv', 'hg-2023-results.tsv']
    dfs = []
    for file in ordered_files:
        # read in results with dom
        dom_path = os.path.join(dir, 'dom', file)
        dom_df = pd.read_csv(dom_path, sep='\t')
        # drop irrelevant and rename columns
        dom_df.drop(columns=['dom_rank', 'def_rank', 'indef_rank'], inplace=True)
        new_columns = ['id', 'condition', 'input_sentence', 'masked', 'dom_top_fillers',
                       'dom_probabilities', 'dom_dom_prob', 'dom_def_prob', 'dom_indef_prob']
        dom_df.columns = new_columns
        # modify sentences a --> (a)
        new_sentences = []
        for sent in list(dom_df['input_sentence']):
            new_sent = sent.replace(' a ', ' (a) ')
            new_sentences.append(new_sent)
        dom_df['input_sentence'] = new_sentences
        # add discrepancy column
        dom_df['dom_discrepancy'] = dom_df['dom_def_prob'] - dom_df['dom_indef_prob']
        # read in unmarked results
        unmarked_path = os.path.join(dir, 'unmarked', file)
        unmarked_df = pd.read_csv(unmarked_path, sep='\t')
        # drop irrelevant and rename columns
        unmarked_df.drop(columns=['condition', 'input_sentence', 'masked', 'dom_rank', 'def_rank', 'indef_rank'], inplace=True)
        new_columns = ['id', 'unmarked_top_fillers', 'unmarked_probabilities', 
                       'unmarked_dom_prob', 'unmarked_def_prob', 'unmarked_indef_prob']
        unmarked_df.columns = new_columns
        unmarked_df['unmarked_discrepancy'] = unmarked_df['unmarked_def_prob'] - unmarked_df['unmarked_indef_prob']
        merged_df = pd.merge(dom_df, unmarked_df, on='id', how='inner')
        dfs.append(merged_df)
    full_df = pd.concat(dfs)
    full_df.replace('nonaffected', 'non-affected', inplace=True)
    full_df.to_csv(os.path.join(dir, 'merged-results.tsv'), sep='\t')
    return full_df

def count_article_disc(df):
    df = df[df['condition'] != 'inanimate']
    df = df[df['masked'] != 'el']
    df = df[df['masked'] != 'un']
    print(df.shape)
    print('DOM')
    count_greater = (df['dom_discrepancy'] > 0.0).sum()
    # print('def > indef: ', count_greater)
    count_smaller = (df['dom_discrepancy'] < 0.0).sum()
    print('indef > def: ', count_smaller)
    print('UNMARKED')
    count_greater = (df['unmarked_discrepancy'] > 0.0).sum()
    print('def > indef: ', count_greater)
    count_smaller = (df['unmarked_discrepancy'] < 0.0).sum()
    print('indef > def: ', count_smaller)
    return count_greater, count_smaller

def count_dom_rank(dir):
    ordered_files = ['ms-2013-results.tsv','sa-2020-results.tsv','re-2021-results.tsv', 
                     're-2021-modified-results.tsv', 'hg-2023-results.tsv']
    dfs = []
    for file in ordered_files:
        # read in results with dom
        dom_path = os.path.join(dir, file)
        df = pd.read_csv(dom_path, sep='\t')
        dfs.append(df)
    full_df = pd.concat(dfs)
    print(full_df.shape)
    count_dom = (full_df['dom_rank'] == 0).sum()
    count_other = (full_df['dom_rank'] > 0).sum()
    return count_dom, count_other
    

if __name__ == '__main__':
    # df = articlemasking_postprocessing('results/fill-mask/article-masking/BETO')
    # count_article_disc(df)
    dir = 'results/fill-mask/dom-masking/BETO/'
    count_dom, count_other = count_dom_rank(dir)
    print(f'BETO dom count: {count_dom}, other count: {count_other}')

    dir = 'results/fill-mask/dom-masking/mBERT/'
    count_dom, count_other = count_dom_rank(dir)
    print(f'mBERT dom count: {count_dom}, other count: {count_other}')