import math
import torch
import re
import numpy as np


class MLMSentence:
    def __init__(self, sentence, index, model, tokenizer):
        """
        constructor to create a new MLMSentence object
        :param sentence: a sentence
        :param index: token index to be masked
        :param model: a loaded pretrained Spanish model suitable for MLM
        :param tokenizer: a loaded tokenizer based on the same model
        :param top_k: number of most likely fillers to predict for a [MASK]
        """
        self.sentence = sentence
        self.index = index
        self.model = model
        self.tokenizer = tokenizer
        self.top_fillers, self.top_probs  = None, None  # will be calculated in function

    def set_mask_index(self, new_index):
        """
        Set the index of the token to be masked
        :return:
        """
        self.index = new_index

    def get_sentence(self):
        """
        Get the string sentence that is input to the model
        :return: string sentence
        """
        return self.sentence

    def get_masked_token(self):
        """
        Get the masked token
        :return: string masked token
        """
        return self.masked_token

    def get_top_fillers(self, num=5):
        """
        get the top_k fillers for masked token
        :param num: number of most likely fillers to return
        :return: list of lists containing most likely fillers
        """
        if self.top_fillers is None:
            print(f'No top fillers have been computed yet. Run function compute_mlm_fillers_probs first.')
        if num <= len(self.top_fillers):
            fillers = self.top_fillers[:num]
        else:
            fillers = self.top_fillers
        return fillers

    def get_top_probabilities(self, num=5):
        """
        get the probabilities for the top_k filler tokens for masked token
        :param num: number of highest probabilities to return
        :return: list of lists containing softmax probabilities
        """
        if self.top_fillers is None:
            print(f'No top probabilities have been computed yet. Run function compute_mlm_fillers_probs first.')
        if num < len(self.top_probs):
            probs = self.top_probs[:num]
        else:
            probs = self.top_probs
        return probs

    def get_filler_prob(self, rank=0):
        """
        get filler and its probability at given rank for given mask index
        :param rank: the rank of the filler, 0 returns the most likely filler (default 0)
        :return: filler and probability
        """
        if self.top_fillers is None or self.top_probs is None:
            print(f'No fillers and probabilities have been computed yet. Run function compute_mlm_fillers_probs first.')
            return
        if rank > self.vocab_size - 1 or rank < 0:
            print(f'Rank {rank} is out of range for top {self.vocab_size} ranking (first rank is 0)')
            return
        return self.top_fillers[rank], self.top_probs[rank]

    def get_token_prob_rank(self, token):
        """
        get rank and probability for given token being predicted at given mask index
        :param token: list of tokens to look for
        :return: rank and probability for token, if not found 5 and 0.0
        """
        rank, prob = self.vocab_size, 0.0
        if self.top_fillers is None or self.top_probs is None:
            print(f'No fillers and probabilities have been computed yet. Run function compute_mlm_fillers_probs first.')
            return
        if token in self.top_fillers:
            rank = self.top_fillers.index(token)
            prob = self.top_probs[rank]
        return rank, prob

    # find instances of list in fillers (e.g. dom markers, (in)definite articles)
    def get_list_prob_rank(self, li):
        """
        get probability of an item list of tokens being predicted at given mask index
        :param li: list of tokens to look for
        :return: rank and probability for most likely from list, if none found top_k and 0.0
        """
        rank, prob = self.vocab_size, 0.0
        if self.top_fillers is None or self.top_probs is None:
            print(f'No fillers and probabilities have been computed yet. Run function compute_mlm_fillers_probs first.')
            return
        for token in li:
            new_rank, new_prob = self.get_token_prob_rank(token)
            if new_rank < rank:
                rank = new_rank
                prob = new_prob
        return rank, prob

    def compute_mlm_fillers_probs(self):
        """
        compute most likely fillers and their softmax probabilities for masked token
        :return: two lists of lists containing most likely fillers and their probability
        """
        # prepare input
        s, self.masked_token = self.prep_input(self.sentence, self.index)  # prepare sentence
        s = '[CLS]' + s + '[SEP]'
        tokens = self.tokenizer.tokenize(s)
        print(f'mlm tokens: {tokens}')
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        # predict
        predictions = self.model(tokens_tensor)[0]
        # get masked token index
        mask_idx = self.get_masked_index(tokens)
        probs = torch.nn.functional.softmax(predictions[0, mask_idx], dim=-1) # get softmax outputs
        self.vocab_size = probs.shape[0]
        top_probs, top_indices = torch.topk(probs, probs.shape[0], sorted=True)  # get highest probabilites
        top_fillers = self.tokenizer.convert_ids_to_tokens(top_indices)  # get tokens with highest probability
        self.top_fillers = top_fillers
        self.top_probs = top_probs.tolist()
        return top_fillers, top_probs

    # based on https://github.com/simonepri/lm-scorer
    def sentence_score_old(self, reduce='mean', log=False, per_token=False, return_ranks=False):
        """
        function to compute a likelihood score for a sentence based on the softmax (log) probabilities of each token
        :param reduce: method to reduce token probabilities to a single score, either mean or prod (default mean)
        :param log: if True, score is based on logarithmic probability (default False)
        :param per_token: if True, function returns a list of (log) probabilites for each token (default False)
        :param return_ranks: if True and per_token True, function additionally returns a list of ranks for each token in the probability distribution (default False)
        :return: the (log) probability score for the sentence or a list of probabilities plus optional: list of ranks
        """
        probabilities, ranks = [], []
        for i in range(len(self.sentence.split())):
            s, masked = self.prep_input(self.sentence, index=i) # mask token at index i
            # print(f'sentence: {s}, masked: {masked}')
            s = '[CLS]' + s + '[SEP]' # add special tokens
            # tokenize
            tokens = self.tokenizer.tokenize(s)
            # print(f'tokens: {tokens}')
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokenized_masked = self.tokenizer.tokenize(masked)
            indexed_masked = self.tokenizer.convert_tokens_to_ids(tokenized_masked)[0]
            # predict
            predictions = self.model(tokens_tensor)[0]
            # get masked token index
            mask_idx = self.get_masked_index(tokens)
            # get (log) probabilities
            if log:
                probs = torch.nn.functional.log_softmax(predictions[0, mask_idx], dim=-1)
            else:
                probs = torch.nn.functional.softmax(predictions[0, mask_idx], dim=-1)
            # return rank of masked token
            top_probs, top_indices = torch.topk(probs, probs.shape[0], sorted=True)  # get highest probabilites
            top_fillers = self.tokenizer.convert_ids_to_tokens(top_indices)
            rank = top_fillers.index(masked)
            prob = top_probs[rank]
            if per_token and return_ranks:
                ranks.append(rank)
            # get probability of masked token being predicted for [MASK]
            probabilities.append(float(prob))
        probabilities = torch.tensor(probabilities)
        # return
        if per_token: # return list of probabilities
            probabilities = probabilities.tolist()
            return (probabilities, ranks) if return_ranks else probabilities
        else: # return probability
            if reduce == 'prod': # product
                if log: # sum for logarithmic probabilities
                    score = probabilities.sum()
                else: # product for probabilities
                    score = probabilities.prod()
            else: # mean
                if log:
                    score = probabilities.logsumexp(0) - math.log(probabilities.shape[0])
                else:
                    score = probabilities.mean()
            return float(score)
        

    def sentence_score(self, reduce='mean', log=False, per_token=False, return_ranks=False):
        """
        Function to compute a likelihood score for a sentence based on the softmax (log) probabilities of each token.
        Adjusted to handle WordPiece tokenization and words split into multiple subwords.
        
        :param reduce: method to reduce token probabilities to a single score, either mean or prod (default mean)
        :param log: if True, score is based on logarithmic probability (default False)
        :param per_token: if True, function returns a list of (log) probabilities for each token (default False)
        :param return_ranks: if True and per_token True, function additionally returns a list of ranks for each token in the probability distribution (default False)
        :return: the (log) probability score for the sentence or a list of probabilities plus optional: list of ranks
        """
        probabilities, ranks = [], []
        tokens = self.tokenizer.tokenize('[CLS] ' + self.sentence + ' [SEP]')  # tokenize the full sentence including special tokens
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        for word_idx, word in enumerate(self.sentence.split()):  # loop through words
            word_tokens = self.tokenizer.tokenize(word)  # check if the word is split into subwords
            word_probs = []  # to accumulate probabilities for subwords of the same word
            word_ranks = []  # to accumulate ranks for subwords
            
            for subword_idx, subword in enumerate(word_tokens):  # handle each subword
                s, masked = self.prep_input(self.sentence, index=word_idx)  # mask token at index of word
                s = '[CLS] ' + s + ' [SEP]'  # add special tokens

                tokens_masked = self.tokenizer.tokenize(s)  # tokenize the masked sentence
                indexed_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)
                tokens_tensor = torch.tensor([indexed_masked])

                # Get prediction scores
                predictions = self.model(tokens_tensor)[0]
                mask_idx = self.get_masked_index(tokens_masked)  # index of the masked token

                # Calculate (log) softmax probabilities
                if log:
                    probs = torch.nn.functional.log_softmax(predictions[0, mask_idx], dim=-1)
                else:
                    probs = torch.nn.functional.softmax(predictions[0, mask_idx], dim=-1)
                
                # Get the index of the current subword in the probability distribution
                subword_id = self.tokenizer.convert_tokens_to_ids([subword])[0]
                
                # Append the probability for the current subword
                word_probs.append(probs[subword_id].item())
                
                # Optionally calculate rank
                if per_token and return_ranks:
                    top_probs, top_indices = torch.topk(probs, probs.shape[0], sorted=True)
                    rank = (top_indices == subword_id).nonzero(as_tuple=True)[0].item()
                    word_ranks.append(rank)

            # Combine probabilities of subwords (average for simplicity)
            word_prob = sum(word_probs) / len(word_probs)
            probabilities.append(word_prob)
            
            if per_token and return_ranks:
                ranks.append(word_ranks)
        
        probabilities = torch.tensor(probabilities)
        
        # Return results
        if per_token:
            probabilities = probabilities.tolist()
            return (probabilities, ranks) if return_ranks else probabilities
        else:
            if reduce == 'prod':
                score = probabilities.sum() if log else probabilities.prod()
            else:
                score = probabilities.logsumexp(0) - math.log(probabilities.shape[0]) if log else probabilities.mean()
            return float(score)

    # helper function to add special tokens to sentence in order to use in MLM
    def prep_input(self, sent, index):
        """
        prepare sentence by inserting [MASK] tokens and begin and end of sentence markers
        :param sentence: the sentence to be prepped
        :param index: token index to be masked
        :return: the prepared sentence, and the tokens replaced with [MASK]
        """
        s_list = re.findall(r'[\w?!,]+|[^\s\w]+', sent)
        # mask token if index in sentence range
        masked_token = None
        if index in range(len(s_list)):
            masked_token = s_list[index]
            s_list[index] = '[MASK]'
        s = ' '.join(s_list)
        return s, masked_token

    # helper function to get masked index from tokenized sentence
    def get_masked_index(self, tokens):
        """
        get index of [MASK] tokens
        :param tokens: list of tokenized sentence
        :return: int index
        """
        masked_index = None
        for i, token in enumerate(tokens):
            if token == '[MASK]':
                masked_index = i
        return masked_index
    
# # test
# from util import load_model
# tokenizer, model = load_model('dccuchile/bert-base-spanish-wwm-cased')
# sentence = 'El ladrón secuestró a la señora.'
# mlm = MLMSentence(sentence, 3, model, tokenizer)
# print(mlm.get_sentence())
# # experiment 1
# mlm.compute_mlm_fillers_probs()
# print(mlm.get_top_fillers())
# print(mlm.get_top_probabilities())
# # experiment 2
# print(mlm.sentence_score())
# print(mlm.sentence_score(per_token=True))
# # experiment 3
# mlm.set_mask_index(4)
# mlm.compute_mlm_fillers_probs()
# print(mlm.get_top_fillers())
# print(mlm.get_top_probabilities())
