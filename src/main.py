import json
from util import load_targets, load_model, initialize_result_file, search_fillers
from mlm_sentence import MLMSentence
import numpy as np

#TODO: plot results
# parse config file
with open('../config.json') as f:
    config = json.load(f)

# access parameters
experiment_type = config['EXPERIMENT_TYPE']
remove_dom = config['REMOVE_DOM']
input_file = config['INPUT_FILE']
modelname = config['MODEL_NAME']
num_top_preds = config['NUM_PREDS']
print_mode = config['PRINT_TO_CONSOLE']
save_mode = config['SAVE_RESULTS']
plot_mode = config['PLOT_RESULTS']

print(f'Start MLM experiment {experiment_type} with {input_file}')
print()

# load targets
input_path = f'../data/{input_file}'
mode = 'mask_dom' if experiment_type == 'dom-masking' else 'mask_dobj'
if mode == 'mask_dom': remove_dom = False
df = load_targets(input_path, mode=mode, remove_dom=remove_dom)
if print_mode:
    print(df.head())
    print()

# load model
model_mapping = {'dccuchile/bert-base-spanish-wwm-cased': 'BETO',
                 'google-bert/bert-base-multilingual-cased': 'mBERT',
                 'microsoft/mdeberta-v3-base': 'mDeBERTa'}
tokenizer, model = load_model(modelname)

# initialize output
if save_mode:
    filename = input_file.replace('-targets', '')
    str_mode = ''
    if mode == 'mask_dobj':
        str_mode = '-unmarked' if remove_dom else '-dom' 
    name = model_mapping[modelname] if modelname in model_mapping else modelname
    output_path = f'../results/{experiment_type}/{name}{str_mode}-{filename}'
    num_mask = len(df['mask_idx'].iloc[0])
    # initialize_result_file(output_path, num_mask=num_mask, top_n=num_top_preds)

# run experiment
print('start experiment')
print()
# compute masked language model probabilites for sentence
fillers, probabilities, dom_rank, dom_prob, masked_tokens = [], [], [], [], []
for index, row in df.iterrows():
    mlm_sent = MLMSentence(row, model, tokenizer)
    top_fillers = mlm_sent.get_top_fillers()
    fillers.append(top_fillers)
    top_probs = mlm_sent.get_top_probabilities()
    probabilities.append(top_probs)
    masked_tokens.append(mlm_sent.get_masked_tokens())
    if mode == 'mask_dom':
        filler, prob = mlm_sent.get_token_prob(0)
        rank, prob = mlm_sent.get_dom_prob_rank()
        dom_rank.append(rank)
        dom_prob.append(round(prob, 4))
    if print_mode:
        print(row['sentence'])
        print(row['en_sentence'])
        print(f'masked tokens: {mlm_sent.get_masked_tokens()}')
        print(f'top predictions: {top_fillers}')
        if mode == 'mask_dom':
            print(f'most likely filler token: {filler} probability {prob: .4f}')
            print(f'dom probability: {prob} rank: {rank}')
        else:
            mask_cnt = mlm_sent.get_num_masks()
            for i in range(mask_cnt):
                filler, prob = mlm_sent.get_token_prob(0, mask=i+1)
                print(f'{i+1} filler token: {filler}, probability {prob: .4f}')
            def_articles = ['la', 'el', 'las', 'los']
            indef_articles = ['una', 'un', 'unas', 'unos']
            dom_markers = ['a', 'al']
            wordlists = [def_articles, indef_articles, dom_markers]
            for w in wordlists:
                fillers_found, filler_probs, filler_ranks = search_fillers(top_fillers[0], top_probs[0], w)
                if len(fillers_found) > 0:
                    print(w)
                    print(f'found {fillers_found} at ranks {filler_ranks} with probability {filler_probs}')
        print()
    # if save_mode: 
    #     mlm_sent.save_to_file(output_path)

# add columns to df
df['masked'] = masked_tokens
df['top_fillers'] = fillers
df['probabilities'] = probabilities
if print_mode:
    print(df)
    print(df.columns)

# print results
if experiment_type == 'dom-masking':
    df['dom_rank'] = dom_rank
    df['dom_prob'] = dom_prob
    if print_mode:
        print('mean probability dom: {:.4f}'.format(np.mean(df['dom_prob'])))

if save_mode:
    df_print = df[['id', 'sentence', 'masked', 'top_fillers', 'probabilities']]
    df_print.to_csv(output_path, index=False, sep='\t')

print(f'Successfully completed {experiment_type} with {input_file}')