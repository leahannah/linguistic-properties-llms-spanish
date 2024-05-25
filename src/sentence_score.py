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


def main(MODEL_NAME, INPUT_FILE, SOURCE, PRINT_MODE, SAVE_MODE):
    model_mapping = {'dccuchile/bert-base-spanish-wwm-cased': 'BETO',
                     'google-bert/bert-base-multilingual-cased': 'mBERT',
                     'microsoft/mdeberta-v3-base': 'mDeBERTa'}

    print(f'Start sentence score experiment with {INPUT_FILE}')

    # load model
    tokenizer, model = load_model(MODEL_NAME)

    # define outputpath
    modelname = model_mapping[MODEL_NAME]
    output_path = os.path.join(pathlib.Path(__file__).parent.absolute(), f'../results/sentence-score/', modelname)

    # load targets
    input_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '../data/', INPUT_FILE)
    full_data = load_targets(input_path, source=SOURCE)
    if PRINT_MODE:
        print(full_data.head())
        print(full_data.columns)
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
        # initialize lists
        condis, inputs, statistics, scores_dom, scores_unmarked, discrepancies = [], [], [], [], [], []
        for cond in conditions:
            # filter data for condition
            df = data[data['condition'] == cond]

            # run experiment
            print(f'start experiment with condition {cond}')
            print()

            discs = []
            # compute masked language model probabilites for sentence
            for index, row in df.iterrows():
                condis.append(cond)
                sent = row['sentence']
                # get score with dom
                mlm_sent1 = MLMSentence(sent, model, tokenizer, index=-1, top_k=-1)
                input_sent1 = mlm_sent1.get_sentence()
                score = mlm_sent1.sentence_score()
                log = mlm_sent1.sentence_score(log=True)
                token_scores, ranks = mlm_sent1.sentence_score(log=True, per_token=True, return_ranks=True)
                # get score without dom
                sent_unmarked = sent.replace(' a ', ' ').replace(' al ', ' el ')
                mlm_sent2 = MLMSentence(sent_unmarked, model, tokenizer, index=-1, top_k=-1)
                input_sent2 = mlm_sent2.get_sentence()
                score_unmarked = mlm_sent2.sentence_score()
                log_unmarked = mlm_sent2.sentence_score(log=True)
                token_scores_unmarked, ranks_unmarked = mlm_sent2.sentence_score(log=True, per_token=True, return_ranks=True)
                scores_dom.append(score)
                scores_unmarked.append(score_unmarked)
                disc = log - log_unmarked
                inputs.append(sent.replace(' a ', ' (a) ').replace(' al ', ' al/el '))
                discs.append(disc)
                if PRINT_MODE:
                    print(input_sent1)
                    print(row['en_sentence'])
                    print(f'masked: {mlm_sent1.get_masked_token()}')
                    print(f'score with dom: {score :4f}')
                    print(f'log: {log}')
                    print(f'token scores: {np.round(token_scores, 4)}, ranks: {ranks}')
                    print(input_sent2)
                    print(f'score unmarked: {score_unmarked :4f}')
                    print(f'log: {log_unmarked}')
                    print(f'token scores: {np.round(token_scores_unmarked, 4)}, ranks: {ranks_unmarked}')
                    print(f'discrepancy: {disc :4f}')
                    print()
            discrepancies.extend(discs)
            stats = {'discrepancy': [round(np.mean(discs), 4), round(np.std(discs), 4), round(np.median(discs), 4),]}
            statistics.append(stats)

        # print
        if PRINT_MODE:
            print(data.head())
            print(data.shape)
            print()
            print(f'{source} result statistics')
            for i, cond in enumerate(conditions):
                print(f'condition: {cond}')
                stats = statistics[i]
                for key in stats.keys():
                    print(
                        f'{key} mean: {stats[key][0]}, std: {stats[key][1]}, median: {stats[key][2]}')
                print()

        # add columns to df
        data['input_sentence'] = inputs
        data['conditions'] = condis
        data['score_dom'] = scores_dom
        data['score_unmarked'] = scores_unmarked
        data['discrepancy'] = discrepancies
        # # save
        # if SAVE_MODE:
            # print('save results')
            # print()
            # # initialize output
            # str_type = ''
            # modelname = model_mapping[MODEL_NAME] if MODEL_NAME in model_mapping else MODEL_NAME
            # if TYPE == 'dobject-masking':
            #     str_type = '-unmarked-' if REMOVE_DOM else '-dom-'
            #     str_type += MASK_TYPE
            #     output_path = os.path.join(pathlib.Path(__file__).parent.absolute(),
            #                                f'../results/fill-mask/{TYPE}/{str_type.strip("-")}/', modelname)
            # else:
            #     output_path = os.path.join(pathlib.Path(__file__).parent.absolute(), f'../results/fill-mask/{TYPE}/',
            #                                modelname)
            # if not os.path.exists(output_path):
            #     os.makedirs(output_path)
            # # save
            # filename = f'{SOURCE}-{str_type}-results.tsv'
            # df_print = data[['id', 'input_sentence', 'masked', 'top_fillers', 'probabilities']]
            # full_path = os.path.join(pathlib.Path(__file__).parent.absolute(), output_path, filename)
            # df_print.to_csv(full_path, index=False, sep='\t')
            # stats_path = os.path.join(pathlib.Path(__file__).parent.absolute(), output_path, 'statistics.tsv')
            # for i, cond in enumerate(conditions):
            #     stats = statistics[i]
            #     with open(stats_path, mode='a', encoding='utf-8') as f:
            #         str_type = TYPE + str_type
            #         for key in stats.keys():
            #             f.write(
            #                 f'{SOURCE}-{cond}\t{str_type}\t{modelname}\t{key}\t{stats[key][0]}\t{stats[key][1]}\t{stats[key][2]}\t{stats[key][3]}\n')


    print(f'Successfully completed sentence-score experiment with {INPUT_FILE}')
