import torch
import re
import numpy as np


class MLMSentence:
    def __init__(self, sentence, indices, model, tokenizer, sentence_score=False, top_n=5):
        """
        constructor to create a new MLMSentence object
        :param sentence: a sentence marked with []
        :param indices: token indices to be masked
        :param model: a loaded pretrained Spanish model suitable for MLM
        :param tokenizer: a loaded tokenizer based on the same model
        :param top_n: number of most likely fillers to predict for a [MASK]
        :param: mode='fill-mask' to perform fill mask experiment, 'sentence-score' to score a sentence
        """
        self.sentence, self.masked_tokens = self.prep_input(sentence, indices)  # prepare sentence
        self.num_masks = len(self.masked_tokens)
        self.model = model
        self.tokenizer = tokenizer
        self.top_n = top_n #TODO: auf top_n verzichten, nach bestimmten Token in allen Vorhersagen suchen, -1 for all tokens in vocab
        #TODO: conditional: run function separately
        self.top_fillers, self.top_probs = self.compute_mlm_tokens_probs()  # execute mlm for sentence

    def get_sentence(self):
        """
        Get the string sentence that is input to the model
        :return: string sentence
        """
        return self.sentence

    def get_masked_tokens(self):
        """
        Get the masked tokens
        :return: list of masked tokens
        """
        return self.masked_tokens

    def get_num_masks(self):
        """
        get the number of masked tokens
        :return: number of masked tokens as integer
        """
        return self.num_masks

    def get_top_fillers(self):
        """
        get the top_n fillers for each masked token
        :return: list of lists containing most likely fillers
        """
        return self.top_fillers

    def get_top_probabilities(self):
        """
        get the probabilities for the top_n filler tokens for each masked token
        :return: list of lists containing softmax probabilities
        """
        return self.top_probs

    def get_filler_prob(self, rank=0, mask=0):
        """
        get filler and its probability at given rank for given mask index
        :param rank: the rank of the filler, 0 returns the most likely filler (default 0)
        :param mask: mask index, 0 returns the filler for first mask (default 0)
        :return: filler and probability
        """
        if rank > self.top_n - 1 or rank < 0:
            print(f'Rank {rank} is out of range for top {self.top_n} ranking (first rank is 0)')
            return
        if mask > self.num_masks - 1:
            print(f'Mask {mask} is out of range, only {self.num_masks} tokens masked')
            return
        return self.top_fillers[mask][rank], self.top_probs[mask][rank]

    def get_token_prob_rank(self, token, mask=0):
        """
        get rank and probability for given token being predicted at given mask index
        :param token: list of tokens to look for
        :param mask: mask index, 0 checks for token in predictions for first mask (default 0)
        :return: rank and probability for token, if not found 5 and 0.0
        """
        rank, prob = 5, 0.0
        if mask > self.num_masks - 1 or self.get_num_masks() == 0:
            print(f'Mask {mask} is out of range, only {self.num_masks} tokens masked')
            return
        if token in self.top_fillers[mask]:
            rank = self.top_fillers[mask].index(token)
            prob = self.top_probs[mask][rank]
        return rank, prob

    # find instances of list in fillers (e.g. dom markers, (in)definite articles)
    def get_list_prob_rank(self, li, mask=0):
        """
        get probability of an item list of tokens being predicted at given mask index
        :param li: list of tokens to look for
        :param mask: mask index, 0 checks for token in predictions for first mask (default 0)
        :return: rank and probability for most likely from list, if none found 5 and 0.0
        """
        rank, prob = 5, 0.0
        if mask > self.num_masks or self.get_num_masks() == 0:
            print(f'Mask {mask} is out of range, only {self.num_masks} tokens masked')
            return
        for token in li:
            new_rank, new_prob = self.get_token_prob_rank(token, mask=mask)
            if new_rank < rank:
                rank = new_rank
                prob = new_prob
        return rank, prob

    def compute_mlm_tokens_probs(self):
        """
        compute most likely fillers and their softmax probabilities for all masked tokens
        :return: two lists of lists containing most likely fillers and their probability
        """
        # tokenize
        s = '[CLS]' + self.sentence + '[SEP]'
        tokens = self.tokenizer.tokenize(s)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        # predict
        predictions = self.model(tokens_tensor)[0]
        # get mask token predictions
        top_fillers, top_probs = [], []
        # get masked token indices
        mask_idx = self.get_masked_indices(tokens)
        for msk in mask_idx:
            # TODO: sorted probs for all tokens in vocab --> compute dom rank even if > 5
            # sorted_probs, sorted_idx = torch.topk(log_probs, k=probs.shape[0], sorted=True)
            # index = sorted_idx.tolist().index(indexed_masked)
            probs = torch.nn.functional.softmax(predictions[0, msk], dim=-1) # get softmwax outputs
            top_p, top_indices = torch.topk(probs, self.top_n, sorted=True)  # get highest probabilites
            top_t = self.tokenizer.convert_ids_to_tokens(top_indices)  # get tokens with highest probability
            top_probs.append(top_p.tolist())
            top_fillers.append(top_t)
        return top_fillers, top_probs

    def sentence_score(self, log=True, return_ranks=False, per_token=False):
        """
        function to compute a likelihood score for a sentence based on the softmax (log) probabilities of each token
        :param log: if True, score is based on logarithmic probability (default True)
        :param return_ranks: if True, function returns a list of ranks for each token in the probability distribution (default False)
        :param per_token: if True, function returns a list of (log) probabilites for each token (default False)S
        :return: the (log) probability score for the sentence or a list of probabilities, optional: list of ranks
        """
        probabilities, ranks = [], []
        for i in range(len(self.sentence.split())):
            s, masked = self.prep_input(self.sentence, indices=[i]) # mask token at index i
            s = '[CLS]' + s + '[SEP]' # add special tokens
            # tokenize
            tokens = self.tokenizer.tokenize(s)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokenized_masked = self.tokenizer.tokenize(masked[0])
            indexed_masked = self.tokenizer.convert_tokens_to_ids(tokenized_masked)[0]
            # predict
            predictions = self.model(tokens_tensor)[0]
            # get masked token indices
            mask_idx = self.get_masked_indices(tokens)[0]
            # get (log) probabilities
            if log:
                probs = torch.nn.functional.log_softmax(predictions[0, mask_idx], dim=-1)
            else:
                probs = torch.nn.functional.softmax(predictions[0, mask_idx], dim=-1)
            # return rank of masked token
            if return_ranks:
                top_probs, top_idx = torch.topk(probs, probs.shape[0], sorted=True)
                rank = top_idx.tolist().index(indexed_masked)
                ranks.append(rank)
            # get probability of masked token being predicted for [MASK]
            filler_prob = probs[indexed_masked]
            probabilities.append(float(filler_prob))
        # return
        if per_token: # return list of probabilities
            return (probabilities, ranks) if return_ranks else probabilities
        else: # return probability score
            if log: # sum for logarithmic probabilities
                return (np.sum(probabilities), ranks) if return_ranks else np.sum(probabilities)
            else: # product for probabilities
                return (np.prod(probabilities), ranks) if return_ranks else np.prod(probabilities)

    # helper function to add special tokens to sentence in order to use in MLM
    def prep_input(self, sent, indices):
        """
        prepare sentence by inserting [MASK] tokens and begin and end of sentence markers
        :param sentence: the sentence to be prepped
        :param indices: token indices to be masked
        :return: the prepared sentence, and the tokens replaced with [MASK]
        """
        s_list = re.findall(r'[\w?!,]+|[^\s\w]+', sent)
        masked_tokens = []
        if type(indices) is not list:
            indices = [indices]
        # iteratively mask tokens
        for masked_index in indices:
            if masked_index in range(len(s_list)):
                masked_tokens.append(s_list[masked_index])
                s_list[masked_index] = '[MASK]'
        s = ' '.join(s_list)
        return s, masked_tokens

    # helper function to get masked indices from tokenized sentence
    def get_masked_indices(self, tokens):
        """
        get indices of [MASK] tokens
        :param tokens: list of tokenized sentence
        :return: list of indices
        """
        masked_indices = []
        for i, token in enumerate(tokens):
            if token == '[MASK]':
                masked_indices.append(i)
        return masked_indices


from util import load_model

tokenizer, model = load_model('dccuchile/bert-base-spanish-wwm-cased')

sentence1 = 'Cristina saludó a la mujer.'
mlm_sent1 = MLMSentence(sentence1, indices=-1, model=model, tokenizer=tokenizer, sentence_score=True)
score1, ranks1 = mlm_sent1.sentence_score(return_ranks=True)

sentence2 = 'Cristina saludó la mujer.'
mlm_sent2 = MLMSentence(sentence2, indices=-1, model=model, tokenizer=tokenizer, sentence_score=True)
score2, ranks2 = mlm_sent2.sentence_score(return_ranks=True)

print(f'DOM score: {score1}, ranks: {ranks1}')
print(f'unmarked score: {score2}, ranks: {ranks2}')



# # basic info about sentence
# print(mlm_sent.get_sentence())
# print(mlm_sent.get_masked_tokens())
# print(mlm_sent.get_num_masks())
# print()
#
# # info about fill mask predictions
# print(mlm_sent.get_top_fillers())
# print(mlm_sent.get_top_probabilities())
# print(mlm_sent.get_filler_prob())
# print(mlm_sent.get_token_prob_rank('a'))
# print(mlm_sent.get_list_prob_rank(['a', 'al']))
