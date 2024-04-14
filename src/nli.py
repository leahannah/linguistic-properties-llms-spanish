import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_nli_model(modelname):
    # setup NLI model
    model_name = modelname
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def nli_binary(premise, hypothesis, model, tokenizer, threshold=90.0):
    label_names = ['entailment', 'neutral', 'contradiction']
    print(premise)
    input1 = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output1 = model(input1['input_ids'])
    prediction1 = torch.softmax(output1['logits'][0], -1).tolist()
    prediction1 = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction1, label_names)}
    # premise without dom
    prem2 = premise.replace(' a ', ' ') 
    print(prem2)
    input2 = tokenizer(prem2, hypothesis, truncation=True, return_tensors="pt")
    output2 = model(input2['input_ids']) 
    prediction2 = torch.softmax(output2['logits'][0], -1).tolist()
    prediction2 = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction2, label_names)}
    return prediction1, prediction2

model, tokenizer = load_nli_model(modelname='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')

premises = ['Busco a una secretaria.',
            'Necesito a una mujer.',
            'Busco a una niña.',
            'Necesito a un médico.',
            'Busco a un hombre.',
            'Necesito a un artesano.']
            # 'Quiero a una mujer con ojos azules.',
            # 'Eduardo conoce a una mujer de ojos azules.',
            # 'Manuel encuentra a una mujer con ojos azules.',
            # 'Ana miró a una mujer con ojos azules.',
            # 'Estaba dibujando a una niña.']
hypothesis = ['There is a particular secretary I am looking for.',
              'There is a particular woman I need.',
              'There is a particular girl I am looking for.',
              'There is a particular doctor I need.',
              'There is a particular man I am looking for.',
              'There is a particular craftsman I need.']
            #  'I love a woman with blue eyes.',
            #   'Eduardo knows a woman with blue eyes.',
            #   'Manuel finds a woman with blue eyes.',
            #   'Ana looked at a woman with blue eyes.',
            #   'She was portraying a girl.']

for i, prem in enumerate(premises):
    pred1, pred2 = nli_binary(prem, hypothesis[i], model, tokenizer)
    print(f'prediction with dom: {pred1}')
    print(f'without dom: {pred2}')
    print()