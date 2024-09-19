import pandas as pd
import numpy as np
import os
import pathlib
from utils.util import load_targets, load_model
from mlm_sentence import MLMSentence

# suppress warnings
pd.set_option('mode.chained_assignment', None)
np.set_printoptions(suppress=True)

# execute sentence score experiment for a given dataset and model
def main(MODEL_NAME, INPUT_FILE, SOURCE, PRINT_MODE, SAVE_MODE):
    model_mapping = {'dccuchile/bert-base-spanish-wwm-cased': 'BETO',
                     'google-bert/bert-base-multilingual-cased': 'mBERT'}
    modelname = model_mapping[MODEL_NAME]

    print(f'Start sentence score experiment with {INPUT_FILE} and model {modelname}')

    # load model
    tokenizer, model = load_model(MODEL_NAME)

    # load targets
    input_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '../data/', INPUT_FILE)
    full_data = load_targets(input_path, source=SOURCE)
    if PRINT_MODE:
        print(full_data.head())
        print(full_data.columns)
        print(full_data.shape)
        print()
    
    # initialize output
    if SAVE_MODE:
        # define outputpath
        output_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 
                                   f'../results/sentence-score/', modelname)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        stats_path = os.path.join(pathlib.Path(__file__).parent.absolute(), output_path, 'statistics.tsv')
        if not os.path.exists(stats_path):
            with open(stats_path, mode='w', encoding='utf-8') as f:
                f.write(f'source\tcondition\tmodel\tmean_score_dom\tmean_score_unmarked')
                f.write(f'\tmean_disc\tstd_disc\tmedian_disc\n')

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
                mlm_sent1 = MLMSentence(sent, -1, model, tokenizer)
                input_sent1 = mlm_sent1.get_sentence()
                score = mlm_sent1.sentence_score(log=False)
                prob = mlm_sent1.sentence_score()
                token_scores = mlm_sent1.sentence_score(log=False, per_token=True, return_ranks=False)
                # get score without dom
                sent_unmarked = sent.replace(' a ', ' ').replace(' al ', ' el ')
                mlm_sent2 = MLMSentence(sent_unmarked, -1, model, tokenizer)
                input_sent2 = mlm_sent2.get_sentence()
                score_unmarked = mlm_sent2.sentence_score(log=False)
                prob_unmarked = mlm_sent2.sentence_score()
                token_scores_unmarked = mlm_sent2.sentence_score(log=False, per_token=True, return_ranks=False)
                scores_dom.append(np.round(score, 4))
                scores_unmarked.append(np.round(score_unmarked, 4))
                disc = score - score_unmarked
                inputs.append(sent.replace(' a ', ' (a) ').replace(' al ', ' al/el '))
                discs.append(disc) #np.round(disc, 4))
                if PRINT_MODE:
                    print(input_sent1)
                    print(row['en_sentence'])
                    print(f'score with dom: {score :4f}')
                    print(f'token scores: {np.round(token_scores, 4)}')
                    print(input_sent2)
                    print(f'score unmarked: {score_unmarked :4f}')
                    print(f'token scores: {np.round(token_scores_unmarked, 4)}')
                    print(f'discrepancy: {disc :4f}')
                    print()
            discrepancies.extend(discs)
            stats = {'score_dom': [round(np.mean(scores_dom), 4), round(np.std(scores_dom), 4), round(np.median(scores_dom), 4)], 
                     'score_unmarked': [round(np.mean(scores_unmarked), 4), round(np.std(scores_unmarked), 4), round(np.median(scores_unmarked), 4)],
                    'discrepancy': [round(np.mean(discs), 4), round(np.std(discs), 4), round(np.median(discs), 4)]}
            statistics.append(stats)

        # add columns to df
        data['input_sentence'] = inputs
        data['condition'] = condis
        data['score_dom'] = scores_dom
        data['score_unmarked'] = scores_unmarked
        data['discrepancy'] = discrepancies

        # print
        if PRINT_MODE:
            print(data.head())
            print(data.columns)
            print(data.shape)
            print()
            print(f'{source} result statistics')
            for i, cond in enumerate(conditions):
                print(f'condition: {cond}')
                stats = statistics[i]
                for key in stats.keys():
                    print(f'key: {key}, stats: {stats[key]}')
                print()

        # save
        if SAVE_MODE:
            print('save results')
            print()
            # save
            filename = f'{source}-results.tsv'
            df_print = data[['id', 'condition', 'input_sentence', 'score_dom', 'score_unmarked', 'discrepancy']]
            full_path = os.path.join(pathlib.Path(__file__).parent.absolute(), output_path, filename)
            df_print.to_csv(full_path, index=False, sep='\t')
            stats_path = os.path.join(pathlib.Path(__file__).parent.absolute(), output_path, 'statistics.tsv')
            for i, cond in enumerate(conditions):
                stats = statistics[i]
                with open(stats_path, mode='a', encoding='utf-8') as f:
                    f.write(f"{source}\t{cond}\t{modelname}\t{stats['score_dom'][0]}")
                    f.write(f"\t{stats['score_unmarked'][0]}\t{stats['discrepancy'][0]}")
                    f.write(f"\t{stats['discrepancy'][1]}\t{stats['discrepancy'][2]}\n")

    print(f'Successfully completed sentence-score experiment with {INPUT_FILE}')
