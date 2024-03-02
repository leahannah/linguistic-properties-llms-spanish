from util import load_targets, load_model
from mlm_sentence import MLMSentence

# load model
tokenizer, model = load_model('dccuchile/bert-base-spanish-wwm-cased')

# test mask dom
# affected targets
file = 'data/non-affected-targets.tsv' # affected targets
# # load targets
ids, sentences, dobjects, indices = load_targets(file)
print(sentences[0])

# compute masked language model probabilites for sentence
for i, sent in enumerate(sentences):
    mlm_sent = MLMSentence(ids[i], sent, indices[i], model, tokenizer)
    # print(f'sentence: {mlm_sent.get_sentence()}')
    # print(f'masked tokens: {mlm_sent.get_masked_tokens()}')
    # print(f'top predictions: {mlm_sent.get_top_tokens()}')
    # tok, prob = mlm_sent.get_token_prob(0)
    # print(f'most likely token: {tok}, probability {prob: .4f}')
    # print(f'dom probability and rank: {mlm_sent.get_token_prob_rank("a")}')
    mlm_sent.save_to_file('results/beto-nonaffected-mask-dom.tsv')

# tokens1, probs1 = compute_masked_tokens_probs(sentences1, indices1, model, tokenizer, result_file='results/beto-affected-mask-dom.tsv')
# dom_ranking1, dom_probs1 = analyze_output_dom(tokens1, probs1)

# # non-affected targets
# file2 = 'data/non-affected-targets.tsv' # affected targets
# # load targets
# sentences2, dobjects2, indices2 = load_targets(file2)
# tokens2, probs2 = compute_masked_tokens_probs(sentences2, indices2, model, tokenizer, result_file='results/beto-nonaffected-mask-dom.tsv')
# dom_ranking2, dom_probs2 = analyze_output_dom(tokens2, probs2)# affected targets

# # print results
# print('dom ranking affected: {}'.format(dom_ranking1))
# print('dom probabilities affected: {}\n mean: {:.4f}'.format(dom_probs1, np.mean(dom_probs1)))

# print('dom ranking non-affected: {}'.format(dom_ranking2))
# print('dom probabilities non-affected: {}\n mean: {:.4f}'.format(dom_probs2, np.mean(dom_probs2)))

