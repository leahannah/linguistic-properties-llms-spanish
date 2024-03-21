import json
from util import load_targets, load_model, initialize_result_file
from mlm_sentence import MLMSentence
import numpy as np

#TODO: plot results
#TODO: run with dobject masking
# parse config file
with open('../config.json') as f:
    config = json.load(f)

# access parameters
experiment_type = config['EXPERIMENT_TYPE']
input_file = config['INPUT_FILE']
modelname = config['MODEL_NAME']
num_top_preds = config['NUM_PREDS']
print_mode = config['PRINT_TO_CONSOLE']
save_mode = config['SAVE_RESULTS']
plot_mode = config['PLOT_RESULTS']

print(f'Start MLM experiment {experiment_type}')
print()

# load targets
input_path = f'../data/{input_file}'
mode = 'mask_dom' if experiment_type == 'dom-masking' else 'mask_dobj'
df = load_targets(input_path, mode=mode)
if print_mode:
    print(df.head())

# load model
model_mapping = {'dccuchile/bert-base-spanish-wwm-cased': 'BETO',
                 'google-bert/bert-base-multilingual-cased': 'mBERT',
                 'microsoft/mdeberta-v3-base': 'mDeBERTa'}
tokenizer, model = load_model(modelname)

# initialize output
if save_mode:
    name = model_mapping[modelname] if modelname in model_mapping else modelname
    output_path = f'../results/{experiment_type}/{name}-{input_file}'
    num_mask = len(df['mask_idx'].iloc[0])
    initialize_result_file(output_path, num_mask=num_mask, top_n=num_top_preds)

# test mask dom
# compute masked language model probabilites for sentence
fillers, probabilities, dom_rank, dom_prob = [], [], [], []
for index, row in df.iterrows():
    mlm_sent = MLMSentence(row, model, tokenizer)
    top_fillers = mlm_sent.get_top_fillers()
    fillers.append(top_fillers)
    top_probs = mlm_sent.get_top_probabilities()
    probabilities.append(top_probs)
    filler, prob = mlm_sent.get_token_prob(0)
    rank, prob = mlm_sent.get_dom_prob_rank()
    dom_rank.append(rank)
    dom_prob.append(round(prob, 4))
    if print_mode:
        print(row['sentence'])
        print(row['en_sentence'])
        print(f'masked tokens: {mlm_sent.get_masked_tokens()}')
        print(f'top predictions: {top_fillers}')
        print(f'most likely filler token: {filler}, probability {prob: .4f}')
        print(f'dom probability: {prob} rank: {rank}')
    if save_mode: 
        mlm_sent.save_to_file(output_path)

# add columns to df
df['top_fillers'] = fillers
df['filler_probs'] = probabilities
df['dom_rank'] = dom_rank
df['dom_prob'] = dom_prob

# print results
if experiment_type == 'dom_masking' and print_mode:
    print(df[['dom_rank', 'dom_prob', 'sentence']])
    print('mean probability dom: {:.4f}'.format( np.mean(df['dom_prob'])))

print(f'Successfully completed {experiment_type} with {input_file}')