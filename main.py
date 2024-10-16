import json
import os
import pathlib
import time
from src import fill_mask, sentence_score
# execute experiments based on config.json file

# measure time
start_time = time.time()
print(f'Start time: {time.ctime()}')

# parse config file
config_path = 'config.json'
with open(config_path) as f:
    config = json.load(f)

# access parameters
EXPERIMENT = config['EXPERIMENT']
MODEL_NAME = config['MODEL_NAME']
INPUT_FILE = config['INPUT_FILE']
SOURCE = config['SOURCE']
PRINT_MODE = config['PRINT_TO_CONSOLE']
SAVE_MODE = config['SAVE_RESULTS']
TYPE = config['TYPE']
REMOVE_DOM = config['REMOVE_DOM']
OUTPATH = config['OUTPATH']

# run fill-mask experiment
if EXPERIMENT == 'fill-mask':
    # sanity check to check if TYPE parameter is valid
    valid_types = ['dom-masking', 'article-masking']
    if TYPE not in valid_types:
        print(f'TYPE parameter {TYPE} is not in the set of valid options.\n'
              f'Choose from: {valid_types}.')
    fill_mask.main(MODEL_NAME, INPUT_FILE, SOURCE, TYPE, REMOVE_DOM, PRINT_MODE, SAVE_MODE, OUTPATH)
# run sentence-score experiment
elif EXPERIMENT == 'sentence-score':
    sentence_score.main(MODEL_NAME, INPUT_FILE, SOURCE, PRINT_MODE, SAVE_MODE, OUTPATH)
# invalid EXPERIMENT param
else:
    valid_experiments = ['fill-mask', 'sentence-score']
    print(f'EXPERIMENT parameter {EXPERIMENT} is not in the set of valid experiments.\n'
          f'Choose from: {valid_experiments}.')

# measure time
end_time = time.time()
total_seconds = end_time-start_time
print(f'Measured time: {int(total_seconds/60)}.{int(total_seconds%60)} minutes')
print(f'End time: {time.ctime()}')