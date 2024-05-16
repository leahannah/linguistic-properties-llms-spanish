import torch
from transformers import BertConfig, BertTokenizer, BertModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# from https://krishansubudhi.github.io/deeplearning/2019/09/26/BertAttention.html
# TODO: modify to make it usable
model_type = 'dccuchile/bert-base-spanish-wwm-cased'
config = BertConfig.from_pretrained(model_type)
config.output_attentions=True
model = BertModel.from_pretrained(model_type,config = config)
tokenizer = BertTokenizer.from_pretrained(model_type)

text1 = 'Ana tocó'
text2 = 'el [MASK].'
tok1 = tokenizer.tokenize(text1)
tok2 = tokenizer.tokenize(text2)

p_pos = len(tok1) # position for token
tok = tok1+tok2
print(f'POS: {p_pos}')
print(f'TOK: {tok}')

ids = torch.tensor(tokenizer.convert_tokens_to_ids(tok)).unsqueeze(0)
with torch.no_grad():
    output = model(ids)
attentions = torch.cat(output[2])
print(f'shape: {attentions.shape}')

attentions = attentions.permute(2,1,0,3)
print(f'permuted: {attentions.shape}')

layers = len(attentions[0][0])
heads = len(attentions[0])
seqlen = len(attentions)

attentions_pos = attentions[p_pos]
print(f'pos: {attentions_pos.shape}')
cols = 2
rows = int(heads/cols)

print (f'Attention weights for token {tok[p_pos]}')
mean_att = np.mean(np.array(attentions_pos), axis=0)
print(f'shape {mean_att.shape}')
fig = sns.heatmap(mean_att, vmin = 0, vmax = 0.7, xticklabels = tok)
fig.set(title='mean multihead attention', ylabel='layers')
plt.savefig('mean_att1.png')
# plt.show()

fig, axes = plt.subplots(rows, cols, figsize = (14,30))
axes = axes.flat
for i,att in enumerate(attentions_pos):
    # print(f'att shape {att.shape}')
    sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i], xticklabels = tok)
    axes[i].set_title(f'head - {i} ' )
    axes[i].set_ylabel('layers')
plt.savefig('multi_att1.png')
# plt.show()
