import torch

class MLMSentence:
    def __init__(self, data, model, tokenizer, top_n=5):
        self.id = data['id']
        self.sentence, self.masked_tokens = self.prep_target(data['sentence'], data['mask_idx'])
        self.num_masks = len(self.masked_tokens)
        self.model = model
        self.tokenizer = tokenizer
        self.tokens = self.tokenizer.tokenize(self.sentence)
        self.top_n = top_n
        self.top_fillers, self.top_probs = self.compute_mlm_tokens_probs()

    def get_sentence(self):
        return self.sentence
    
    def get_masked_tokens(self):
        return self.masked_tokens

    def get_top_fillers(self):
        return(self.top_fillers)
    
    def get_top_probabilities(self):
        return(self.top_probs)
    
    def get_num_masks(self):
        return(self.num_masks)
    
    def get_token_prob(self, rank, mask=1):
        if rank > self.top_n-1 or rank < 0:
            print(f'Rank {rank} is out of range for top {self.top_n} ranking (first rank is 0)')
            return
        if mask > self.num_masks:
            print(f'Mask {mask} is out of range, only {self.num_masks} tokens masked')
            return
        return self.top_fillers[mask-1][rank], self.top_probs[mask-1][rank]

    def get_token_prob_rank(self, token, mask=1):
        rank, prob = 5, 0.0
        if mask > self.num_masks:
            print(f'Mask {mask} is out of range, only {self.num_masks} tokens masked')
            return
        if token in self.top_fillers[mask-1]:
            rank = self.top_fillers[mask-1].index(token)
            prob = self.top_probs[mask-1][rank]
        return rank, prob

    # find instances of list in fillers (e.g. dom markers, (in)definite articles)
    def get_filler_prob_rank(self, list, mask=1):
        rank, prob = 5, 0.0
        if mask > self.num_masks:
            print(f'Mask {mask} is out of range, only {self.num_masks} tokens masked')
            return
        for token in list:
            if token in self.top_fillers[mask-1]:
                new_rank = self.top_fillers[mask-1].index(token)
                new_prob = self.top_probs[mask-1][new_rank]
                if new_rank < rank:
                    rank = new_rank
                    prob = new_prob
        return rank, prob

    def compute_mlm_tokens_probs(self):
        # tokenize
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(self.tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        # predict
        predictions = self.model(tokens_tensor)[0]
        # get mask token predictions
        top_fillers, top_probs = [], []
        # get masked token indices
        mask_idx = self.get_masked_indices()
        for msk in mask_idx:
            probs = torch.nn.functional.softmax(predictions[0, msk], dim=-1) 
            top_p, top_indices = torch.topk(probs, self.top_n, sorted=True) # get highest probabilites
            top_t = self.tokenizer.convert_ids_to_tokens(top_indices) # get tokens with highest probability
            top_probs.append(top_p.tolist())
            top_fillers.append(top_t)
        return top_fillers, top_probs

    # helper function to add special tokens to sentence in order to use in MLM
    def prep_target(self, sentence, indices):
        s = sentence
        s = s.replace('[', '').replace(']', '')
        s_list = s.split()
        masked_tokens = []
        if type(indices) != list:
            indices = [indices]
        # iteratively mask tokens
        for masked_index in indices:
            masked_tokens.append(s_list[masked_index])
            s_list[masked_index] = '[MASK]' 
        s = ' '.join(s_list)
        s = '[CLS] ' + s + ' [SEP]'
        return s, masked_tokens
    
    # helper function to get masked indices from tokenized sentence
    def get_masked_indices(self):
        masked_indices = []
        for i, token in enumerate(self.tokens):
            if token == '[MASK]':
                masked_indices.append(i)
        return masked_indices

