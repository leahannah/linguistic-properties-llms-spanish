# Examining linguistic properties of Large Language Models in Spanish

## Description
This repository contains code used to analyze how multilingual BERT (Devlin et al, 2019) and BETO (Canete et al, 2020) deal with Differential Object Marking (DOM) in Spanish. Three experiments are used, all leveraging the Masked Language Model (MLM) property of BERT. More precisely, predicted fillers are analyzed when masking DOM and the article in a sentence, and the probability assigned to sentences with and without DOM are compared. This is done on the basis of stimuli data from four linguistic studies ().

## Content
* `embeddings`: German, Spanish and English trained vectors and vocabulary
* `wordlists`: Lists of gender-related attribute words, lists of common nouns and occupations as target words for all three languages
* (`data`): The seed corpora used to obtain embeddings and wordlists are not provided here, and need to be downloaded for reproduction (see Quickstart).

## Run experiments
### Requirements
* `experiment1.py`: Monolingual experiment where German and Spanish common nouns and occupations are scored with WEAT to predict grammatical gender. Results and plots are saved. Script can be executed directly, the two parameters at the top can be modified:
	* `lang`: Language of wordlists to used for the experiment, either *de* for German or *es* for Spanish
	* `num_nouns`: Amount of common nouns per gender to use for the experiment, maximum is 3000
* `experiment2.py`: Bilingual experiment where gender bias in occupations is compared for either English and German, or English and Spanish.  English words are scored with WEAT, German and Spanish words are scored using mWEAT.  Results and plots are saved. Script can be executed directly, the three parameters at the top can be modified:
	*   `lang`: Language of wordlists to used for the experiment, either *de* for German or *es* for Spanish
	* `useDot`: If true, dot product is used instead of cosine similarity to get similarity of two vectors (in line with code from Zhou et al, 2019)
	* `getFreqs`: If true, vocabulary frequencies for attributes and occupational targets are determined, saved and analyzed
* `en-en.py`:  Additional experiment done for completeness. Get WEAT bias scores from embeddings trained on two different English Corpora, plot and save the results.
* `weat.py` Implements bias score, effect size and test statistic for WEAT based on https://github.com/e-mckinnie/WEAT, and bias score and test statistic for mWEAT based on https://github.com/shaoxia57/Bias_in_Gendered_Languages
* `util.py`: Contains useful functions needed in the experiment scripts
### Execution

### Outputs
* `plots`: Plots of the results with different languages and measures
* `results`: Bias scores and vocabulary frequencies for the wordlists, plus statistics of those

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
Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). Spanish pre-trained BERT model and evaluation data. In Proceedings of PML4DC at ICLR 2020.<br>
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, & T. Solorio (Eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 4171–4186). Association for Computational Linguistics. [https://doi.org/10.18653/v1/N19-1423 ](https://doi.org/10.18653/v1/N19-1423  )      <br>
Heredero, D. R., & García, M. G. (2023). Differential object marking in Spanish: The effect of affectedness. Caplletra. Revista Internacional de Filologia, (74), 259-285. [https://doi.org/10.7203/Caplletra.74.26043 ](https://doi.org/10.7203/Caplletra.74.26043 )           <br>
Montrul, S., & Sánchez-Walker, N. (2013). Differential object marking in child and adult Spanish heritage speakers. Language Acquisition, 20(2), 109-132.<br>
Reina, J. C., García, M. G., & Von Heusinger, K. (2021). Differential object marking in Cuban Spanish. Differential object marking in romance, 339.<br>
Sagarra, N., Bel, A., & Sánchez, L. (2020). Animacy hierarchy effects on L2 processing of Differential Object Marking. The Acquisition of Differential Object Marking, 26, 183-206.
