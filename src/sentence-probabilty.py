import torch
from transformers import BertTokenizer, BertForNextSentencePrediction


tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
model = BertForNextSentencePrediction.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

def next_sentence_prediction(sentence1, sentence2):
    encoded = tokenizer(sentence1, sentence2, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded)
    probs = torch.softmax(outputs[0], dim=1) # prob for true at position 0
    return probs.tolist()[0][0]

sent1 = ''
sent2 = 'Miguel vio la pelota.'

prob = next_sentence_prediction(sent1, sent2)
print(f'Next sentence probability: {prob :6f}')