# Examining linguistic properties of Large Language Models in Spanish

## Description
This repository contains code used to analyze how multilingual BERT [2] and BETO [1] deal with Differential Object Marking (DOM) in Spanish. Three experiments are used, all leveraging the Masked Language Model (MLM) property of BERT. More precisely, the fill-mask experiments analyze the predictions when masking DOM and the direct object article in a sentence, and the sentence-score experiment compares the probability assigned to a sentence with and without DOM. This is done on the basis of stimuli data from four linguistic studies [3, 4, 5, 6].

## Contents
* `data/`: Tables containing test input data from linguistic studies [3, 4, 5, 6].
* `plots/`: Plots generated from the results.
* `results/`: Tables with experimental outputs.
* `src/`: Python scripts implementing the experiments. 
	* `fill_mask.py` is the implementation of the fill-mask experiments for DOM-masking and article-masking.
	* `sentence_score.py` is for the sentence-score experiment. The sentence-score is implemented based on the [lm-scorer library](https://github.com/simonepri/lm-scorer).
 	* `mlm_sentence.py` contains a class that receives a sentence and generates fillers and probabilities for a masked token, or assigns a single probability to a sentence. This class is used for all three experiments.
	*  `utils/` contains scripts with functions for pre- and postprocessing of data, and for plotting the results.
* `config.json`: Configuration file where experiments, input data and experimental parameters are specified.
* `main.py` main script accessing `config.json` and executing the specified experiment.
* `requirements.txt`: Python libraries required to run the experiments.

## Run experiments
The code for this repository is written in Python 3.11.0 on Ubuntu 18.04.6 LTS. 

### Requirements
* Optional: create a virtual environment with `python3 -m venv .venv`
	* activate virtual environment: `source .venv/bin/activate`
* Install the required packages into the .venv using `python3 -m pip install -r requirements.txt`

### Execution
First modify the parameters in `config.json` as needed:
```
{
    "INPUT_FILE": "testdata-merged.tsv",
    "SOURCE": null,
    "PRINT_TO_CONSOLE": true,
    "SAVE_RESULTS": false,
    "OUTPATH": null,
    "MODEL_NAME": "dccuchile/bert-base-spanish-wwm-cased",
    "EXPERIMENT": "fill-mask",
    "TYPE": "dom-masking",
    "REMOVE_DOM": true
}
```
* `INPUT_FILE`: Input tsv file containing test data. Required to contain correctly formatted data, at least the columns: id, source, condition, sentence, dom_idx, dobject, en_sentence.
* `SOURCE`: Option to run experiment only using data from one specific source study, insert one of the abbreviations "ms-2013" [4], "sa-2020" [6], "re-2021", "re-2021-modified" [5], "hg-2023" [3].
* `PRINT_TO_CONSOLE`: If true, outputs will be printed to console while running experiment.
* `SAVE_RESULTS`: If true, outputs will be saved to automatically created subdirectory in  `results/` folder.
* `OUTPATH`: Option to specify a customized path where the results should be saved.
* `MODEL_NAME`: Name of huggingface transformers BERT model. Experiments are implemented for [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) and [multilingual BERT](https://huggingface.co/google-bert/bert-base-multilingual-cased), but can be executed with other Spanish BERT-like models as well.
* `EXPERIMENT`: Experiment type, either "fill-mask" or "sentence-score". Please note that the latter can take more than 30 minutes to run, depending on hardware.
* `TYPE`: If `EXPERIMENT` is "fill-mask", choose between "dom-masking" and "article-masking".
* `REMOVE_DOM`: If `TYPE` is "article-masking", input False to execute with sentences containing DOM, and True to run with sentences where DOM is removed.
<br>

To execute the specified experiment with the parameters specified in `config.json`, run `main.py`

### Outputs
* The results are saved to `results/` by default, the `plots/` are generated using the functions in `src/utils/plot.py`.

## Structure
```
linguistic-properties-llms-spanish
|
│   .gitignore
│   config.json
│   main.py
│   README.md
│   requirements.txt
|
├───data
│   │   testdata-merged.tsv
│   │
│   └───single-files
│
├───plots
│   ├───fill-mask
│   │   ├───article-masking
│   │   │
│   │   └───dom-masking
|   |
│   └───sentence-score
│
├───results
│   ├───fill-mask
│   │   ├───article-masking
│   │   │   
│   │   └───dom-masking
│   │
│   └───sentence-score
│
├───src
   │   fill_mask.py
   │   mlm_sentence.py
   │   sentence_score.py
   │
   ├───utils
         plot.py
         util.py
         __init__.py

```


## References
1. Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). Spanish pre-trained BERT model and evaluation data. In Proceedings of PML4DC at ICLR 2020.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, & T. Solorio 	(Eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short 	Papers) (pp. 4171–4186). Association for Computational Linguistics. [https://doi.org/10.18653/v1/N19-1423                                                                ](https://doi.org/10.18653/v1/N19-1423                                                                 )      <br>
3. Heredero, D. R., & García, M. G. (2023). Differential object marking in Spanish: The effect of affectedness. Caplletra. Revista Internacional de Filologia, (74), 259-285. 	
	[https://doi.org/10.7203/Caplletra.74.26043                                                            ](https://doi.org/10.7203/Caplletra.74.26043                                                            )           <br>
4. Montrul, S., & Sánchez-Walker, N. (2013). Differential object marking in child and adult Spanish heritage speakers. Language Acquisition, 20(2), 109-132.<br>
5. Reina, J. C., García, M. G., & Von Heusinger, K. (2021). Differential object marking in Cuban Spanish. Differential object marking in romance, 339.<br>
6. Sagarra, N., Bel, A., & Sánchez, L. (2020). Animacy hierarchy effects on L2 processing of Differential Object Marking. The Acquisition of Differential Object Marking, 26, 183-206.
