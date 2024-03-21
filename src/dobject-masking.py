from util import load_targets, load_model, initialize_result_file
from mlm_sentence import MLMSentence
import numpy as np

# specify input and output path
# TODO: use argparse or config file
input_file = '../data/affected-targets.tsv'
output_file = '../results/dobject-masking/unmarked-affected.tsv'
top_n = 5

# load targets
df = load_targets(input_file, mode='mask_dobj', remove_dom=True)
print(df.head())
num_mask = len(df['mask_idx'].iloc[0])
initialize_result_file(output_file, num_mask=num_mask, top_n=top_n)

# load model
tokenizer, model = load_model('dccuchile/bert-base-spanish-wwm-cased')

# test mask dom
# compute masked language model probabilites for sentence
fillers, probabilities = [], []
for index, row in df.iterrows():
    print(row['sentence'])
    print(row['en_sentence'])
    mlm_sent = MLMSentence(row, model, tokenizer)
    print(f'masked tokens: {mlm_sent.get_masked_tokens()}')
    top_fillers = mlm_sent.get_top_fillers()
    fillers.append(top_fillers)
    print(f'top predictions: {top_fillers}')
    top_probs = mlm_sent.get_top_probabilities()
    probabilities.append(top_probs)
    mask_cnt = mlm_sent.get_num_masks()
    for i in range(mask_cnt):
        filler, prob = mlm_sent.get_token_prob(0, mask=i+1)
        print(f'{i+1} filler token: {filler}, probability {prob: .4f}')
    mlm_sent.save_to_file(output_file)

# add columns to df
df['top_fillers'] = fillers
df['filler_probs'] = probabilities
# print(df.head())
