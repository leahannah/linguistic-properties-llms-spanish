import json
from util import load_targets, load_model
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
mask_det_only = config['MASK_DET_ONLY']
mask_noun_only = config['MASK_NOUN_ONLY']
assert experiment_type in ['dom-masking', 'dobject-masking']

model_mapping = {'dccuchile/bert-base-spanish-wwm-cased': 'BETO',
                 'google-bert/bert-base-multilingual-cased': 'mBERT',
                 'microsoft/mdeberta-v3-base': 'mDeBERTa'}

print(f'Start MLM experiment {experiment_type} with {input_file}')
print()

# initialize output
if save_mode:
    filename = input_file.replace('-targets', '')
    str_experiment_type = ''
    if experiment_type == 'dobject-masking':
        str_experiment_type = '-unmarked' if remove_dom else '-dom' 
        if mask_det_only and not mask_noun_only:
            str_experiment_type += '-mask_det'
        if not mask_det_only and mask_noun_only:
            str_experiment_type += '-mask_noun'
    name = model_mapping[modelname] if modelname in model_mapping else modelname
    output_path = f'../results/{experiment_type}/'
    filename = f'{name}{str_experiment_type}-{filename}'

# run experiment
print('start experiment')
print()

# load model
tokenizer, model = load_model(modelname)

# load targets
input_path = f'../data/{input_file}'
if experiment_type == 'dom-masking': remove_dom = False
df = load_targets(input_path, mode=experiment_type, remove_dom=remove_dom, det_only=mask_det_only, noun_only=mask_noun_only)
if print_mode:
    print(df.head())
    print()

# initialize lists
dom_rank, dom_prob = [], []
if experiment_type == 'dobject-masking':
    def_rank, def_prob, indef_rank, indef_prob = [], [], [], []
# compute masked language model probabilites for sentence
inputs, fillers, probabilities, masked_tokens = [], [], [], []
for index, row in df.iterrows():
    mlm_sent = MLMSentence(row, model, tokenizer)
    input_sent = mlm_sent.get_sentence()
    inputs.append(input_sent)
    top_fillers = mlm_sent.get_top_fillers()
    fillers.append(top_fillers)
    top_probs = mlm_sent.get_top_probabilities()
    probabilities.append(top_probs)
    masked_tokens.append(mlm_sent.get_masked_tokens())
    filler, prob = mlm_sent.get_token_prob(0)
    rank1, prob1 = mlm_sent.get_filler_prob_rank(['a', 'al'])
    dom_rank.append(rank1)
    dom_prob.append(prob1)
    if experiment_type == 'dobject-masking':
        rank2, prob2 = mlm_sent.get_filler_prob_rank(['el', 'la', 'las', 'los'])
        def_rank.append(rank2)
        def_prob.append(prob2)
        rank3, prob3 = mlm_sent.get_filler_prob_rank(['un', 'una', 'unas', 'unos'])
        indef_rank.append(rank3)
        indef_prob.append(prob3)
    if print_mode:
        print(input_sent)
        print(row['en_sentence'])
        print(f'masked tokens: {mlm_sent.get_masked_tokens()}')
        print(f'top predictions: {top_fillers}')
        if experiment_type == 'dom-masking':
            print(f'most likely filler tokens: {filler} probability {prob: .4f}')
            print(f'dom probability: {prob1:.4f} rank: {rank1}')
        else:
            mask_cnt = mlm_sent.get_num_masks()
            for i in range(mask_cnt):
                filler, prob = mlm_sent.get_token_prob(0, mask=i+1)
                print(f'{i+1} TOP filler token: {filler}, probability {prob:.4f}')
            print(f'dom probability: {prob1:.4f} rank: {rank1}')
            print(f'definite article probability: {prob2:.4f} rank: {rank2}')
            print(f'indefinite article probability: {prob3:.4f} rank: {rank3}')
        print()

# add columns to df
df['input_sentence'] = inputs
df['masked'] = masked_tokens
df['top_fillers'] = fillers
df['probabilities'] = probabilities
if print_mode:
    print(df)

stats = {'dom': [round(np.mean(dom_prob), 4), round(np.std(dom_prob), 4), round(np.mean(dom_rank), 4)]}
if experiment_type == 'dobject-masking':
    stats['def'] = [round(np.mean(def_prob), 4), round(np.std(def_prob), 4), round(np.mean(def_rank), 4)]
    stats['indef'] = [round(np.mean(indef_prob), 4), round(np.std(indef_prob), 4), round(np.mean(indef_rank), 4)]
    # if remove_dom==True: experiment_type+='-unmarked'
if print_mode:
    print(f'mean probability dom: {np.mean(dom_prob):.4f} std: {np.std(dom_prob):.4f}, mean rank: {np.mean(dom_rank)}')
    if experiment_type == 'dobject-masking':
        print(f'mean probability definite: {np.mean(def_prob):.4f} std: {np.std(def_prob):.4f}, mean rank: {np.mean(def_rank)}')
        print(f'mean probability indefinite: {np.mean(indef_prob):.4f} std: {np.std(indef_prob):.4f}, mean rank: {np.mean(indef_rank)}')
# save
if save_mode:
    df_print = df[['id', 'input_sentence', 'masked', 'top_fillers', 'probabilities']]
    df_print.to_csv(output_path+filename, index=False, sep='\t')
    with open(output_path+'statistics.tsv', mode='a', encoding='utf-8') as f:
        experiment_type += str_experiment_type
        for key in stats.keys():
            f.write(f'{input_file}\t{experiment_type}\t{name}\t{key}\t{stats[key][0]}\t{stats[key][1]}\t{stats[key][2]}\n')

print()
print(f'Successfully completed {experiment_type} with {input_file}')