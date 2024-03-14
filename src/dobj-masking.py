from util import load_targets, load_model, initialize_result_file
from mlm_sentence import MLMSentence
import numpy as np

# specify input and output path
# TODO: use argparse or config file
input_file = '../data/animacy/animate-human-targets.tsv'
output_file = '../results/animacy/beto-animate-human-targets.tsv'
top_n = 5

# load targets
df = load_targets(input_file, mode='mask_dobj')
num_mask = len(df['mask_idx'].iloc[0])
initialize_result_file(output_file, num_mask=num_mask, top_n=top_n)

# load model
tokenizer, model = load_model('dccuchile/bert-base-spanish-wwm-cased')

# test mask dom
# compute masked language model probabilites for sentence
fillers, probabilities, dom_rank, dom_prob = [], [], [], []
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
    filler, prob = mlm_sent.get_token_prob(0)
    print(f'most likely filler token: {filler}, probability {prob: .4f}')
    rank, prob = mlm_sent.get_dom_prob_rank()
    dom_rank.append(rank)
    dom_prob.append(round(prob, 4))
    print(f'dom probability: {prob} rank: {rank}')
    mlm_sent.save_to_file(output_file)

# add columns to df
df['top_fillers'] = fillers
df['filler_probs'] = probabilities
df['dom_rank'] = dom_rank
df['dom_prob'] = dom_prob

# print results
print(df[['dom_rank', 'dom_prob', 'sentence']])
print('mean probability dom: {:.4f}'.format( np.mean(df['dom_prob'])))
