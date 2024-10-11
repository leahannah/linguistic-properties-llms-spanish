# Examining linguistic properties of Large Language Models in Spanish

## Description
This repository contains code used to analyze how multilingual BERT (Devlin et al, 2019) and BETO (Canete et al, 2020) deal with Differential Object Marking (DOM) in Spanish. Three experiments are used, all leveraging the Masked Language Model (MLM) property of BERT. More precisely, the fill-mask experiments analyze the predictions when masking DOM and the article in a sentence, and the sentence-scoree experiment compares the probability assigned to a sentence with and without DOM. This is done on the basis of stimuli data from four linguistic studies ().

## Contents
* `data/`: Tables containing test sentences from linguistic studies.
* `plots/`: Plots generated from the results.
* `results/`: Outcomes of the three experiments.
* `src/`: Python scripts implementing the experiments. 
	* `fill_mask.py` is the implementation of the fill-mask experiments for DOM-masking and article-masking.
	* `sentence_score.py` is for the sentence score experiment. The sentence score is implemented based on the [lm-scorer library](https://github.com/simonepri/lm-scorer).
 	* `mlm_sentence.py` contains a class that receives a sentence and generates fillers and probabilities for a masked token, or assigns a single probability to a sentence. This class is used for all three experiments.
	*  `utils/` contains scripts with functions for pre- and postprocessing of data, and for plotting the results.
* `config.json`: Configuration file where experiments, input data and experimental parameters are specified.
* `main.py` main script accessing `config.json` and executing the specified experiment.
* `requirements.txt`: Python libraries required to run the experiments.

## Run experiments
### Requirements
* create venv
* install requirements

### Execution
* modify config
* run main.py

### Outputs
* results saved to results, plotted in plots 

## Structure
```
GenderBias
├── data_preparation
│   ├── create_de_gender_wordlists.py
│   ├── embeddings.py
│   └── preprocessing.py
├── embeddings
│   ├── vectors-de.tsv
│   ├── vectors-en_de.tsv
│   ├── vectors-en_es.tsv
│   ├── vectors-es.tsv
│   ├── vocab-de.tsv
│   ├── vocab-en_de.tsv
│   ├── vocab-en_es.tsv
│   ├── vocab-es.tsv
├── en-en.py
├── experiment1.py
├── experiment2.py
├── plots
│   ├── freq-attributes-de.png
│   ├── freq-attributes-en_de.png
│   ├── freq-attributes-en_es.png
│   ├── freq-attributes-es.png
│   ├── freq-occupations-en_de.png
│   ├── freq-occupations-en_es.png
│   ├── mweat-dot_occupations_en-de.png
│   ├── mweat-dot_occupations_en-es.png
│   ├── mweat_occupations_en-de.png
│   ├── mweat_occupations_en-en.png
│   ├── mweat_occupations_en-es.png
│   ├── weat_common-nouns_de.png
│   ├── weat_common-nouns_es.png
│   ├── weat_occupations_de.png
│   └── weat_occupations_es.png
├── README.md
├── requirements.txt
├── results
│   ├── attributes-freqs-en_de.tsv
│   ├── attributes-freqs-en_es.tsv
│   ├── attribute-stats.tsv
│   ├── mweat-dot_occupations_de.tsv
│   ├── mweat-dot_occupations_es.tsv
│   ├── mweat_occupations_de.tsv
│   ├── mweat_occupations_en_es.tsv
│   ├── mweat_occupations_es.tsv
│   ├── mweat-stats.tsv
│   ├── occupations-freqs-en_de.tsv
│   ├── occupations-freqs-en_es.tsv
│   ├── occupation-stats.tsv
│   ├── weat_common-nouns_de.tsv
│   ├── weat_common-nouns_es.tsv
│   ├── weat-dot_occupations_en-de.tsv
│   ├── weat-dot_occupations_en-es.tsv
│   ├── weat-dot_occupations_en.tsv
│   ├── weat_occupations_de.tsv
│   ├── weat_occupations_en-de.tsv
│   ├── weat_occupations_en-en_es.tsv
│   ├── weat_occupations_en-es.tsv
│   ├── weat_occupations_es.tsv
│   ├── weat-stats_de.tsv
│   └── weat-stats_es.tsv
├── util.py
├── weat.py
└── wordlists
    ├── de_definitional_pairs.json
    ├── de_occupation_words_with_EN_translations.txt
    ├── en_definitional_pairs.json
    ├── es_definitional_pairs.json
    ├── es_occupation_words_with_EN_translations.txt
    ├── f_nouns_de.txt
    ├── f_nouns_es.txt
    ├── m_nouns_de.txt
    ├── m_nouns_es.txt
    └── n_nouns_de.txt

```


## References
[1] Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). Spanish pre-trained BERT model and evaluation data. In Proceedings of PML4DC at ICLR 2020.
[2] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, & T. Solorio (Eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 4171–4186). Association for Computational Linguistics. [https://doi.org/10.18653/v1/N19-1423          ](https://doi.org/10.18653/v1/N19-1423           )      <br>
[3] Heredero, D. R., & García, M. G. (2023). Differential object marking in Spanish: The effect of affectedness. Caplletra. Revista Internacional de Filologia, (74), 259-285. [https://doi.org/10.7203/Caplletra.74.26043          ](https://doi.org/10.7203/Caplletra.74.26043          )           <br>
[4] Montrul, S., & Sánchez-Walker, N. (2013). Differential object marking in child and adult Spanish heritage speakers. Language Acquisition, 20(2), 109-132.<br>
[5] Reina, J. C., García, M. G., & Von Heusinger, K. (2021). Differential object marking in Cuban Spanish. Differential object marking in romance, 339.<br>
[6] Sagarra, N., Bel, A., & Sánchez, L. (2020). Animacy hierarchy effects on L2 processing of Differential Object Marking. The Acquisition of Differential Object Marking, 26, 183-206.
