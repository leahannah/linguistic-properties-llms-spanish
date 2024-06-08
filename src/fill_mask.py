import pandas as pd
import numpy as np
import os
import pathlib
from util import load_targets, load_model
from mlm_sentence import MLMSentence

# suppress warnings
pd.set_option('mode.chained_assignment', None)
np.set_printoptions(suppress=True)


def main(MODEL_NAME, INPUT_FILE, SOURCE, TYPE, MASK_TYPE, REMOVE_DOM, PRINT_MODE, SAVE_MODE):
    model_mapping = {'dccuchile/bert-base-spanish-wwm-cased': 'BETO',
                     'google-bert/bert-base-multilingual-cased': 'mBERT',
                     'microsoft/mdeberta-v3-base': 'mDeBERTa'}

    print(f'Start fill mask experiment {TYPE} with {INPUT_FILE}')

    # load model
    tokenizer, model = load_model(MODEL_NAME)

    # load targets
    input_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '../data/', INPUT_FILE)
    if TYPE == 'dom-masking':
        REMOVE_DOM = False
        MASK_TYPE = 'dom'
    full_data = load_targets(input_path, source=SOURCE, mask_type=MASK_TYPE, remove_dom=REMOVE_DOM)
    if PRINT_MODE:
        print(full_data.head())
        print(full_data.shape)
        print()

    # create list of sources in data if no source specified
    if SOURCE is None:
        SOURCE = list(full_data['source'].unique())
    else:
        SOURCE = [SOURCE]

    for source in SOURCE:
        print(f'SOURCE: {source}')
        print()

        # filter data for source
        data = full_data[full_data['source'] == source]

        # find unique conditions
        conditions = list(data['condition'].unique())
        statistics = []
        inputs, fillers, probabilities, masked_tokens, dom_ranks, dom_probs, condis = [], [], [], [], [], [], []
        if TYPE == 'dobject-masking':
            def_ranks, def_probs, indef_ranks, indef_probs = [], [], [], []
        for cond in conditions:
            # filter data for condition
            df = data[data['condition'] == cond]

            # run experiment
            print(f'start experiment with condition {cond}')
            print()

            # initialize lists
            dom_rank, dom_prob = [], []
            if TYPE == 'dobject-masking':
                def_rank, def_prob, indef_rank, indef_prob = [], [], [], []
            # compute masked language model probabilites for sentence
            for index, row in df.iterrows():
                mlm_sent = MLMSentence(row['sentence'], model, tokenizer, index=row['mask_idx'], top_k=-1)
                mlm_sent.compute_mlm_fillers_probs()  # compute mask fillers and probabilities
                input_sent = mlm_sent.get_sentence()  # get input sentence as string
                inputs.append(input_sent)
                top_fillers = mlm_sent.get_top_fillers()  # get top predicted fillers
                fillers.append(top_fillers)
                top_probs = mlm_sent.get_top_probabilities()  # get probabilities for top fillers
                probabilities.append(np.round(top_probs, decimals=4))
                masked_tokens.append(mlm_sent.get_masked_token())  # get tokens masked
                filler, prob = mlm_sent.get_filler_prob()  # get most likely filler and its probability
                rank1, prob1 = mlm_sent.get_list_prob_rank(
                    ['a', 'al'])  # get rank and probability for most likely dom filler
                dom_rank.append(rank1)
                dom_prob.append(prob1)
                condis.append(cond)
                if TYPE == 'dobject-masking':
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
                    if TYPE == 'dom-masking':
                        print(f'most likely filler tokens: {filler} probability {prob: .4f}')
                        print(f'dom probability: {prob1:.4f} rank: {rank1}')
                    else:
                        filler, prob = mlm_sent.get_filler_prob(rank=0)
                        print(f'TOP filler token: {filler}, probability {prob:.4f}')
                        print(f'dom probability: {prob1:.4f} rank: {rank1}')
                        print(f'definite article probability: {prob2:.4f} rank: {rank2}')
                        print(f'indefinite article probability: {prob3:.4f} rank: {rank3}')
                    print()
            dom_probs.extend(dom_prob)
            dom_ranks.extend(dom_rank)
            stats = {'dom': [round(np.mean(dom_prob), 4), round(np.std(dom_prob), 4), round(np.median(dom_prob), 4),
                             round(np.mean(dom_rank), 4)]}
            if TYPE == 'dobject-masking':
                stats['def'] = [round(np.mean(def_prob), 4), round(np.std(def_prob), 4), round(np.median(def_prob), 4),
                                round(np.mean(def_rank), 4)]
                stats['indef'] = [round(np.mean(indef_prob), 4), round(np.std(indef_prob), 4),
                                  round(np.median(indef_prob), 4), round(np.mean(indef_rank), 4)]
                def_probs.extend(def_prob)
                def_ranks.extend(def_rank)
                indef_probs.extend(indef_prob)
                indef_ranks.extend(indef_rank)
            statistics.append(stats)

        # add columns to df
        data['input_sentence'] = inputs
        data['masked'] = masked_tokens
        data['top_fillers'] = fillers
        data['probabilities'] = probabilities
        data['condition'] = condis
        data['dom_prob'] = [round(x, 4) for x in dom_probs]
        data['dom_rank'] = [round(x, 4) for x in dom_ranks]
        if TYPE == 'dobject-masking':
            data['def_prob'] = [round(x, 4) for x in def_probs]
            data['def_rank'] = [round(x, 4) for x in def_ranks]
            data['indef_prob'] = [round(x, 4) for x in indef_probs]
            data['indef_rank'] = [round(x, 4) for x in indef_ranks]

        # print
        if PRINT_MODE:
            print(data.head())
            print(data.shape)
            print()
            print(f'{source} {TYPE} result statistics')
            for i, cond in enumerate(conditions):
                print(f'condition: {cond}')
                stats = statistics[i]
                for key in stats.keys():
                    print(
                        f'{key} mean probability: {stats[key][0]}, std: {stats[key][1]}, median: {stats[key][2]}, mean rank {stats[key][3]}')
                print()

        # save
        if SAVE_MODE:
            print('save results')
            print()
            # initialize output
            str_type = ''
            modelname = model_mapping[MODEL_NAME] if MODEL_NAME in model_mapping else MODEL_NAME
            if TYPE == 'dobject-masking':
                str_type = 'unmarked' if REMOVE_DOM else 'dom'
                output_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
                                           f'../results/fill-mask/{TYPE}/{str_type}/', modelname)
            else:
                output_path = os.path.join(pathlib.Path(__file__).parent.absolute(), f'../results/fill-mask/{TYPE}/', modelname)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # save
            filename = f'{source}-results.tsv'
            if TYPE == 'dobject-masking':
                df_print = data[['id', 'condition', 'input_sentence', 'masked', 'top_fillers', 'probabilities',
                                 'dom_prob', 'dom_rank', 'def_prob', 'def_rank', 'indef_prob', 'indef_rank']]
            else:
                df_print = data[['id', 'condition', 'input_sentence', 'masked', 'top_fillers',
                                 'probabilities', 'dom_prob', 'dom_rank']]
            full_path = os.path.join(pathlib.Path(__file__).parent.absolute(), output_path, filename)
            df_print.to_csv(full_path, index=False, sep='\t')
            stats_path = os.path.join(pathlib.Path(__file__).parent.absolute(), output_path, 'statistics.tsv')
            exp_type = TYPE + '-' + str_type if TYPE == 'dobject' else TYPE
            for i, cond in enumerate(conditions):
                stats = statistics[i]
                with open(stats_path, mode='a', encoding='utf-8') as f:
                    for key in stats.keys():
                        f.write(
                            f'{source}\t{cond}\t{exp_type}\t{modelname}\t{key}\t{stats[key][0]}\t{stats[key][1]}\t{stats[key][2]}\t{stats[key][3]}\n')

    print(f'Successfully completed fill-mask {TYPE} experiment with {MODEL_NAME}')
