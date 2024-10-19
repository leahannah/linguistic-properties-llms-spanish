import re
import os
import torch
import pandas as pd
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer


def load_targets(input_path, source=None, mask_type='dom', remove_dom=False):
    """ load targets from csv file
    :param input_path: path to csv file containing test sentences
    :param source: optional, filters for given source dataset (default None)
    :param mask_type: word to be masked, either 'dom' or 'article' (default 'dom')
    :param remove_dom: if True, dom gets removed from sentences (default False)
    :return: pandas dataframe containing test sentences, translations, dom indices, and some additional info
    """
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
            sentences.append(' '.join(sent_list)) # join to string
            article = row['dobject'].split()[0]
            if article in ['la', 'los', 'las', 'un', 'una', 'su']: 
                idx.append(dom_idx+1)
            else: # skip sentence, no separate article
                print(f'SKIP SENTENCE: {sent_list}')
                idx.append(-1)
        df['sentence'] = sentences
    df['sentence'] = df['sentence'].str.replace('[', '').str.replace(']', '') # remove brackets
    # filter for sentences with separate article
    df['mask_idx'] = idx
    if mask_type == 'article':
        df = df[df['mask_idx'] > 0] # remove sentences that have no tokens to mask
    return df

# setup tokenizer and model
def load_model(modelname):
    """ load tokenizer and model from huggingface transformers library
    :param modelname: name of the model
    :return: loaded tokenizer and model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(modelname).to(device)
    return tokenizer, model

def find_dom_index(sent_list, char='['):
    """ helper function: find index of dom-marker (a/ al) in test sentence based on DO tagged with []
    :param sent_list: list of words in sentence
    :param char: character indicating the start of the DO (default '[')
    :return: index of DOM preposition
    """
    idx = -1
    for i, w in enumerate(sent_list):
        if w in ['a', 'al'] and sent_list[i+1][0] == char:
            idx = i
            break
    if idx < 0:
        print('Index not found in sentence: {}'.format(sent_list))
    return idx

def find_dobj_indices(sent_list, start='[', end=']'):
    """ helper function: find index of direct object in test sentence
    :param sent_list: list of words in sentence
    :param start: character indicating the start of the DO (default '[')
    :param end: character indicating the end of the DO (default ']')
    :return: list of direct object indices
    """
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

def preprocess_input(inputfile, outputfile, idx=5):
    """ preprocess test data, add [] to tag direct objects (manual postprocessing applied)
    :param inputfile: path to file containing ids, test sentences, translations
    :param outputfile: output path
    :param idx: index where direct object is expected to start
    """
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
    """ merge test data from different sources into single file
    :param directory: path to directory containing test data files
    :return: pandas df containing merged data
    """
    es_sents, en_sents, dom_markers, dobjects, dom_idxs, genus, numerus, source, conditions = [], [], [], [], [], [], [], [], []
    cnt = 0 
    # iterate over data files
    for file in os.listdir(directory):
        print(file)
        # get source and condition
        sou = file.split('_')[0]
        name = file.split('_')[1]
        cond = name.split('-')[0]
        print(sou)
        path = directory + file
        with open(path, mode='r', encoding='utf-8') as f:
            next(f)
            # iterate over lines
            for line in f.readlines():
                # extract information
                cnt += 1
                line_list = line.split('\t')
                es_sent = line_list[1]
                pattern = r'\[(.*?)\]'
                matches = re.findall(pattern, es_sent)
                dobject = matches[0]
                sent_list = es_sent.split()
                dom_idx = find_dom_index(sent_list)
                dom = sent_list[dom_idx] if dom_idx > -1 else ''
                # add info on gender and number of DO
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
                # add info to lists
                source.append(sou)
                conditions.append(cond)
                es_sents.append(es_sent)
                dom_markers.append(dom)
                dobjects.append(dobject)
                dom_idxs.append(dom_idx)
                genus.append(gen)
                numerus.append(num)
                en_sents.append(line_list[2].strip('\n'))
    ids = list(range(1, cnt+1))  # add ids
    # animate = ['?' for x in ids]
    # definite = animate
    # affected = animate
    # create dataframe
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
            #  'animate': animate,
            #  'definite': definite,
            #  'affected': affected,
            'en_sentence': en_sents
            })
    return df

def reorder_test_data(filename):
    """ reorder test data so that it aligns with order presented in thesis, save to file
    :param filename: path to merged test data file
    :return: pandas df containing reordered data
    """
    # read data and correct minor things
    df = pd.read_csv(filename, sep='\t')
    df.replace('nonaffected', 'non-affected', inplace=True)
    df.replace('animate-animal', 'animal', inplace=True)
    df.replace('animate-human', 'human', inplace=True)

    # define order
    sources = ['ms-2013','sa-2020','re-2021', 're-2021-modified', 'hg-2023']
    conditions = ['animate', 'inanimate', 'human', 'animal', 'definite', 'indefinite', 'affected', 'non-affected']

    # sort df based on categorical column
    df['source'] = pd.Categorical(df['source'], categories=sources, ordered=True)
    df['condition'] = pd.Categorical(df['condition'], categories=conditions, ordered=True)
    df_sorted = df.sort_values(['source', 'condition'])
    
    # add ids based on order
    ids = list(range(1, df.shape[0]+1))
    df_sorted['id'] = ids

    # print(df_sorted)
    df_sorted.to_csv(f'{filename[:-4]}-sorted.tsv', index=False, sep='\t')
    return df_sorted

def articlemasking_postprocessing(dir):
    """ postprocess results from article masking experiments, create single table with all predictions
    :param dir: directory containing marked and unmarked folder
    :return: pandas df with merged predictions for a model
    """
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
    full_df.to_csv(os.path.join(dir, 'merged-results.tsv'), sep='\t', index=False)
    return full_df

def count_article_disc(df, remove_masc=True):
    """ count the number of test sentences where the model prefers a definite, and indefinite, article
    :param df: dataframe containing article-masking results
    :param remove_masc: if True, sentences with masc. sing. DOs are removed (default True)
    :return: counts of definite preferred, indefinite preferred for DOM and unmarked sentences
    """
    df = df[df['condition'] != 'inanimate']
    if remove_masc:
        df = df[df['masked'] != 'el']
        df = df[df['masked'] != 'un']
    print(df.shape)
    print('DOM')
    dom_greater = (df['dom_discrepancy'] > 0.0).sum()
    dom_smaller = (df['dom_discrepancy'] < 0.0).sum()
    print('indef > def: ', dom_smaller.sum())
    print('UNMARKED')
    unmarked_greater = (df['unmarked_discrepancy'] > 0.0).sum()
    unmarked_smaller = (df['unmarked_discrepancy'] < 0.0).sum()
    print('indef > def: ', unmarked_smaller.sum())
    return dom_greater, dom_smaller, unmarked_greater, unmarked_smaller

def count_dom_rank(dir):
    """ count how often DOM is the top prediction in dom-masking
    :param dir: directory containing dom-masking results for model
    :return: count where DOM is top prediction, count where other word is top prediction
    """
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
    
def merge_stats_dom(results_path):
    """ merge statistics of dom-masking results for both models
    :param results_path: path to results
    :return: pandas df containing merged statistics
    """
    dfs = []
    for model in ['BETO', 'mBERT']:
        path = os.path.join(results_path, model, 'statistics.tsv')
        df = pd.read_csv(path, sep='\t')
        df.drop(columns=['experiment', 'model', 'filler_type'], inplace=True)
        df.rename(columns={'mean': f'{model}-mean', 'std': f'{model}-std', 
                           'median': f'{model}-median', 'rank': f'{model}-mean-rank'}, inplace=True)
        dfs.append(df)
    merged_df = pd.merge(dfs[0], dfs[1], on=['source', 'condition'], how='inner')
    # merged_df = merged_df.drop(columns=['experiment_x', 'experiment_y'])
    print(merged_df.head())
    print(merged_df.columns)
    print(merged_df.shape)
    merged_df.to_csv(os.path.join(results_path, 'merged_stats.tsv'), sep='\t', index=False)
    return merged_df

def merge_stats_sentscore(results_path):
    """ merge statistics of sentence-score results for both models
    :param results_path: path to results
    :return: pandas df containing merged statistics
    """
    dfs = []
    for model in ['BETO', 'mBERT']:
        path = os.path.join(results_path, model, 'statistics.tsv')
        df = pd.read_csv(path, sep='\t')
        df.drop(columns=['model'], inplace=True)
        df.rename(columns={'mean_score_dom': f'{model}-mean_score_dom', 
                           'mean_score_unmarked': f'{model}-mean_score_unmarked', 
                           'mean_disc': f'{model}-mean_disc', 
                           'std_disc': f'{model}-std_disc'}, inplace=True)
        dfs.append(df)
    merged_df = pd.merge(dfs[0], dfs[1], on=['source', 'condition'], how='inner')
    print(merged_df.head())
    print(merged_df.columns)
    print(merged_df.shape)
    merged_df.to_csv(os.path.join(results_path, 'merged_stats.tsv'), sep='\t', index=False)
    return merged_df

def merge_articlemasking_stats(results_file, remove_masc=True):
    """ calculate statistics of article-masking results for one model
    :param results_path: path to results file
    :param remove_masc: if True, sentences with masc. sing. DOs are removed (default True)
    :return: pandas df containing statistics
    """
    df = pd.read_csv(results_file, sep='\t')
    # remove undesired test sentences
    df = df[df['condition'] != 'inanimate']
    if remove_masc:
        df = df[df['masked'] != 'el']
        df = df[df['masked'] != 'un']
    print(df.head())
    print(df.columns)
    print(df.shape)
    df.drop(columns=['dom_dom_prob', 'unmarked_dom_prob'])
    dom_columns = ['dom_def_prob', 'dom_indef_prob', 'dom_discrepancy']
    unmarked_columns = ['unmarked_def_prob', 'unmarked_indef_prob', 'unmarked_discrepancy']
    # calculate statistics
    means1, stds1, medians1 = [], [], []
    for col in dom_columns:
        stat_list = list(df[col])
        means1.append(round(np.mean(stat_list), 4))
        stds1.append(round(np.std(stat_list), 4))
        medians1.append(round(np.median(stat_list), 4))
    means2, stds2, medians2 = [], [], []
    for col in unmarked_columns:
        stat_list = list(df[col])
        means2.append(round(np.mean(stat_list), 4))
        stds2.append(round(np.std(stat_list), 4))
        medians2.append(round(np.median(stat_list), 4))
    # create dataframe and save
    out_df = pd.DataFrame({'measurement': ['definite probability', 'indefinite probability', 'discrepancy'],
                           'dom_mean': means1, 'dom_std': stds1, 'dom_median': medians1,
                           'unmarked_mean': means2, 'unmarked_std': stds2, 'unmarked_median': medians2})
    out_path = os.path.dirname(results_file)
    out_df.to_csv(os.path.join(out_path, 'merged_stats.tsv'), sep='\t', index=False)
    return out_df