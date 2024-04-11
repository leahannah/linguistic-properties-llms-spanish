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
    output_path = f'../results/{experiment_type}/'
    filename = f'{name}{str_mode}-{filename}'
    num_mask = len(df['mask_idx'].iloc[0])
    # initialize_result_file(output_path, num_mask=num_mask, top_n=num_top_preds)

# run experiment
print('start experiment')
print()
# initialize lists
dom_rank, dom_prob = [], []
if mode == 'mask_dobj':
    def_rank, def_prob, indef_rank, indef_prob = [], [], [], []
# compute masked language model probabilites for sentence
fillers, probabilities, masked_tokens = [], [], []
for index, row in df.iterrows():
    mlm_sent = MLMSentence(row, model, tokenizer)
    top_fillers = mlm_sent.get_top_fillers()
    fillers.append(top_fillers)
    top_probs = mlm_sent.get_top_probabilities()
    probabilities.append(top_probs)
    masked_tokens.append(mlm_sent.get_masked_tokens())
    filler, prob = mlm_sent.get_token_prob(0)
    rank, prob = mlm_sent.get_filler_prob_rank(['a', 'al'])
    if rank > -1:
        dom_rank.append(rank)
    dom_prob.append(prob)
    if mode == 'mask_dobj':
        rank2, prob2 = mlm_sent.get_filler_prob_rank(['el', 'la', 'las', 'los'])
        def_rank.append(rank2)
        def_prob.append(prob2)
        rank3, prob3 = mlm_sent.get_filler_prob_rank(['un', 'una', 'unas', 'unos'])
        indef_rank.append(rank3)
        indef_prob.append(prob3)
    if print_mode:
        print(row['sentence'])
        print(row['en_sentence'])
        print(f'masked tokens: {mlm_sent.get_masked_tokens()}')
        print(f'top predictions: {top_fillers}')
        if mode == 'mask_dom':
            print(f'most likely filler tokens: {filler} probability {prob: .4f}')
            print(f'dom probability: {prob:.4f} rank: {rank}')
        else:
            mask_cnt = mlm_sent.get_num_masks()
            for i in range(mask_cnt):
                filler, prob = mlm_sent.get_token_prob(0, mask=i+1)
                print(f'{i+1} TOP filler token: {filler}, probability {prob:.4f}')
            print(f'dom probability: {prob:.4f} rank: {rank}')
            print(f'definite article probability: {prob2:.4f} rank: {rank2}')
            print(f'indefinite article probability: {prob3:.4f} rank: {rank3}')
    print()

# add columns to df
    
df['masked'] = masked_tokens
df['top_fillers'] = fillers
df['probabilities'] = probabilities
if print_mode:
    print(df)

stats = {'dom': [round(np.mean(dom_prob), 4), round(np.std(dom_prob), 4), round(np.mean(dom_rank), 4)]}
if mode == 'mask_dobj':
    stats['def'] = [round(np.mean(def_prob), 4), round(np.std(def_prob), 4), round(np.mean(def_rank), 4)]
    stats['indef'] = [round(np.mean(indef_prob), 4), round(np.std(indef_prob), 4), round(np.mean(indef_rank), 4)]
    if remove_dom==True: experiment_type+='-unmarked'
if print_mode:
    print(f'mean probability dom: {np.mean(dom_prob):.4f} std: {np.std(dom_prob):.4f}, mean rank: {np.mean(dom_rank)}')
    if mode == 'mask_dobj':
        print(f'mean probability definite: {np.mean(def_prob):.4f} std: {np.std(def_prob):.4f}, mean rank: {np.mean(def_rank)}')
        print(f'mean probability dom: {np.mean(indef_prob):.4f} std: {np.std(indef_prob):.4f}, mean rank: {np.mean(indef_rank)}')
# save
if save_mode:
    df_print = df[['id', 'sentence', 'masked', 'top_fillers', 'probabilities']]
    df_print.to_csv(output_path+filename, index=False, sep='\t')
    with open(output_path+'statistics.tsv', mode='a', encoding='utf-8') as f:
        for key in stats.keys():
            f.write(f'{input_file}\t{experiment_type}\t{name}\t{key}\t{stats[key][0]}\t{stats[key][1]}\t{stats[key][2]}\n')

print()
print(f'Successfully completed {experiment_type} with {input_file}')