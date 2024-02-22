import re

def load_targets(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        sentences, dobjects, dom_idx = [], [], []
        next(f)
        for line in f:
            cols = line.split('\t')
            sent = cols[1].strip()
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, sent)
            do = matches[0]
            idx = find_dom_index(sent.split())
            sentences.append(sent)
            dobjects.append(do)
            dom_idx.append([idx])
    return sentences, dobjects, dom_idx

def find_dom_index(sent_list, char='['):
    idx = -1
    for i, w in enumerate(sent_list):
        if w[0] == char:
            idx = i-1
    if idx < 0:
        print('Index not found in sentence: {}'.format(sent_list))
    return idx

# get targets into correct format
def prep_input(filename, out_filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        with open(out_filename, mode='w', encoding='utf-8') as out_f:
            for line in f.readlines():
                cols = line.split('\t')
                sentence = cols[1]
                print(sentence)
                pattern = r'(\w+\s+\w+)\s+/\s+(\w+\s+\w+)'
                replacement = r'[\1]'
                new_sentence = re.sub(pattern, replacement, sentence)
                print(new_sentence)
                cols[1] = new_sentence
                out_f.write('\t'.join(cols))

def prep_targets(sentences, indices):
    for i, s  in enumerate(sentences):
        idx = indices[i]
        s = s.replace('[', '').replace(']', '')
        s_list = s.split()
        for masked_index in idx:
            s_list[masked_index] = '[MASK]' 
        s = ' '.join(s_list)
        s = '[CLS] ' + s + ' [SEP]'
        sentences[i] = s

def get_masked_indices(tokenized):
    masked_indices = []
    for i, token in enumerate(tokenized):
        if token == '[MASK]':
            masked_indices.append(i)
    return masked_indices