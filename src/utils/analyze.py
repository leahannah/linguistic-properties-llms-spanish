import os
import pandas as pd
import pathlib

def find_highest_lowest_preds(df, column):
    conditions = list(df['condition'].unique())
    for cond in conditions:
        print(f'CONDITION: {cond}')
        sorted_df = df[df['condition']==cond].sort_values(by=column, ascending=False)
        #sorted_df = sorted_df[['input_sentence', column]]
        print('HIGHEST')
        print(sorted_df.head(2))
        print('LOWEST')
        print(sorted_df.tail(2))
        print()

def dom_contrast_outliers(dir1, dir2, file):
    # get highest and lowest predictions
    # dom masking experiment
    df1 = pd.read_csv(os.path.join(dir1, file), sep='\t')
    print('DOM MASKING')
    find_highest_lowest_preds(df1, 'dom_prob')
    # sentence score experiment
    df2 = pd.read_csv(os.path.join(dir2, file), sep='\t')
    print(df2.columns)
    print('SENTENCE SCORE')
    print('DISCREPANCY')
    find_highest_lowest_preds(df2, 'discrepancy')
    print('DOM')
    find_highest_lowest_preds(df2, 'score_dom')
    print('UNMARKED')
    find_highest_lowest_preds(df2, 'score_unmarked')

if __name__ == '__main__':
    # Set options to display all rows and columns
    pd.set_option('display.max_rows', None)    # Show all rows
    pd.set_option('display.max_columns', None) # Show all columns
    # define directory and column name
    dir1 = 'results/fill-mask/dom-masking/BETO/'
    dir2 = 'results/sentence-score/BETO/'
    # iterate over results files and read into df
    ordered_files = ['ms-2013-results.tsv','sa-2020-results.tsv','re-2021-results.tsv', 
                     're-2021-modified-results.tsv', 'hg-2023-results.tsv']
    file = 'ms-2013-results.tsv'
    source = file[:-12]
    print(f'SOURCE: {source}')
    dom_contrast_outliers(dir1, dir2, file)