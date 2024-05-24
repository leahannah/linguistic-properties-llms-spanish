import json
import pandas as pd
import numpy as np
import os
import pathlib
from util import load_targets, load_model
from mlm_sentence import MLMSentence
# suppress warnings
pd.set_option('mode.chained_assignment', None)
np.set_printoptions(suppress=True)

# parse config file
config_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '..', 'config.json')
with open(config_path) as f:
    config = json.load(f)

# access parameters
INPUT_FILE = config['INPUT_FILE']
EXPERIMENT_TYPE = config['EXPERIMENT_TYPE']
SOURCE = [config['SOURCE']]
CONDITION = config['CONDITION']
REMOVE_DOM = config['REMOVE_DOM']
MODEL_NAME = config['MODEL_NAME']
NUM_TOP_PREDS = config['NUM_PREDS']
PRINT_MODE = config['PRINT_TO_CONSOLE']
SAVE_MODE = config['SAVE_RESULTS']
MASK_TYPE = config['MASK_TYPE']

model_mapping = {'dccuchile/bert-base-spanish-wwm-cased': 'BETO',
                 'google-bert/bert-base-multilingual-cased': 'mBERT',
                 'microsoft/mdeberta-v3-base': 'mDeBERTa'}

print(f'Start MLM experiment {EXPERIMENT_TYPE} with {INPUT_FILE}')

# load model
tokenizer, model = load_model(MODEL_NAME)

# SOURCE = ['hg-2023', 'ms-2013', 'sa-2020', 're-2021', 'self-re', 'self-sa']
for SOURCE in SOURCE:
    print(f'SOURCE: {SOURCE}')
    print()
    # load targets
    input_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '../data/', INPUT_FILE)
    if EXPERIMENT_TYPE == 'dom-masking': 
        REMOVE_DOM = False
        MASK_TYPE = 'dom'
    data = load_targets(input_path, source=SOURCE, condition=CONDITION, mask_type=MASK_TYPE, remove_dom=REMOVE_DOM)
    if PRINT_MODE:
        print(data.head())
        print(data.shape)
        print()

    # find unique conditions
    conditions = sorted(list(set(list(data['condition']))))
    statistics = []
    for cond in conditions:
        # filter data for condition
        df = data[data['condition'] == cond]

        # run experiment
        print(f'start experiment with condition {cond}')
        print()

        # initialize lists
        dom_rank, dom_prob = [], []
        if EXPERIMENT_TYPE == 'dobject-masking':
            def_rank, def_prob, indef_rank, indef_prob = [], [], [], []
        # compute masked language model probabilites for sentence
        inputs, fillers, probabilities, masked_tokens = [], [], [], []
        for index, row in df.iterrows():
            mlm_sent = MLMSentence(row['sentence'], row['mask_idx'], model, tokenizer, top_k=-1)
            mlm_sent.compute_mlm_fillers_probs() # compute mask fillers and probabilities
            input_sent = mlm_sent.get_sentence() # get input sentence as string
            inputs.append(input_sent)
            top_fillers = mlm_sent.get_top_fillers() # get top predicted fillers
            fillers.append(top_fillers)
            top_probs = mlm_sent.get_top_probabilities() # get probabilities for top fillers
            probabilities.append(np.round(top_probs, decimals=4))
            masked_tokens.append(mlm_sent.get_masked_token()) # get tokens masked
            filler, prob = mlm_sent.get_filler_prob() # get most likely filler and its probability
            rank1, prob1 = mlm_sent.get_list_prob_rank(['a', 'al']) # get rank and probability for most likely dom filler
            dom_rank.append(rank1)
            dom_prob.append(prob1)
            if EXPERIMENT_TYPE == 'dobject-masking':
                # get rank and probability for most likely definite article
                rank2, prob2 = mlm_sent.get_list_prob_rank(['el', 'la', 'las', 'los'])
                def_rank.append(rank2)
                def_prob.append(prob2)
                # get rank and probability for most likely indefinite article
                rank3, prob3 = mlm_sent.get_list_prob_rank(['un', 'una', 'unas', 'unos'])
                indef_rank.append(rank3)
                indef_prob.append(prob3)
            if PRINT_MODE:
                print(input_sent)
                print(row['en_sentence'])
                print(f'masked token: {mlm_sent.get_masked_token()}')
                print(f'top predictions: {top_fillers}')
                if EXPERIMENT_TYPE == 'dom-masking':
                    print(f'most likely filler tokens: {filler} probability {prob: .4f}')
                    print(f'dom probability: {prob1:.4f} rank: {rank1}')
                else:
                    filler, prob = mlm_sent.get_filler_prob(rank=0)
                    print(f'TOP filler token: {filler}, probability {prob:.4f}')
                    print(f'dom probability: {prob1:.4f} rank: {rank1}')
                    print(f'definite article probability: {prob2:.4f} rank: {rank2}')
                    print(f'indefinite article probability: {prob3:.4f} rank: {rank3}')
                print()

        # add columns to df
        df['input_sentence'] = inputs
        df['masked'] = masked_tokens
        df['top_fillers'] = fillers
        df['probabilities'] = probabilities

        stats = {'dom': [round(np.mean(dom_prob), 4), round(np.std(dom_prob), 4), round(np.median(dom_prob), 4), round(np.mean(dom_rank), 4)]}
        if EXPERIMENT_TYPE == 'dobject-masking':
            stats['def'] = [round(np.mean(def_prob), 4), round(np.std(def_prob), 4), round(np.median(def_prob), 4), round(np.mean(def_rank), 4)]
            stats['indef'] = [round(np.mean(indef_prob), 4), round(np.std(indef_prob), 4), round(np.median(indef_prob), 4), round(np.mean(indef_rank), 4)]
        statistics.append(stats)

        # save
        if SAVE_MODE:
            print('save results')
            print()
            # initialize output
            str_type = ''
            modelname = model_mapping[MODEL_NAME] if MODEL_NAME in model_mapping else MODEL_NAME
            if EXPERIMENT_TYPE == 'dobject-masking':
                str_type = '-unmarked-' if REMOVE_DOM else '-dom-' 
                if MASK_TYPE == 'det':
                    str_type += 'mask_det'
                if MASK_TYPE == 'noun':
                    str_type += 'mask_noun'
                output_path = os.path.join(pathlib.Path(__file__).parent.absolute(), f'../results/{EXPERIMENT_TYPE}/{str_type.strip("-")}/', modelname)
            else:
                output_path = os.path.join(pathlib.Path(__file__).parent.absolute(), f'../results/{EXPERIMENT_TYPE}/', modelname)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # save
            filename = f'{SOURCE}-{cond}{str_type}-results.tsv'
            str_input = f'{SOURCE}_{cond}'
            df_print = df[['id', 'input_sentence', 'masked', 'top_fillers', 'probabilities']]
            full_path = os.path.join(pathlib.Path(__file__).parent.absolute(), output_path, filename)
            df_print.to_csv(full_path, index=False, sep='\t')
            stats_path = os.path.join(pathlib.Path(__file__).parent.absolute(), output_path, 'statistics.tsv')
            with open(stats_path, mode='a', encoding='utf-8') as f:
                str_type = EXPERIMENT_TYPE + str_type
                for key in stats.keys():
                    f.write(f'{str_input}\t{str_type}\t{modelname}\t{key}\t{stats[key][0]}\t{stats[key][1]}\t{stats[key][2]}\t{stats[key][3]}\n')

# print
if PRINT_MODE:
    print(f'{SOURCE} {EXPERIMENT_TYPE} result statistics')
    for i, cond in enumerate(conditions):
        print(f'condition: {cond}')
        stats = statistics[i]
        for key in stats.keys():
            print(f'{key} mean probability: {stats[key][0]}, std: {stats[key][1]}, median: {stats[key][2]}, mean rank {stats[key][3]}')
        print()


print(f'Successfully completed {EXPERIMENT_TYPE} with {INPUT_FILE}')