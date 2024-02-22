
from analysis import load_model, test_mask_dom, analyze_output
from util import load_targets, prep_targets
import numpy as np


# load model
tokenizer, model = load_model('dccuchile/bert-base-spanish-wwm-cased')

# test mask dom
# affected targets
file1 = 'data/affected-targets.tsv' # affected targets
# load targets
sentences1, dobjects1, indices1 = load_targets(file1)
tokens1, probs1 = test_mask_dom(sentences1, indices1, model, tokenizer)
dom_ranking1, dom_probs1 = analyze_output(tokens1, probs1)

# non-affected targets
file2 = 'data/non-affected-targets.tsv' # affected targets
# load targets
sentences2, dobjects2, indices2 = load_targets(file2)
tokens2, probs2 = test_mask_dom(sentences2, indices2, model, tokenizer)
dom_ranking2, dom_probs2 = analyze_output(tokens2, probs2)# affected targets
