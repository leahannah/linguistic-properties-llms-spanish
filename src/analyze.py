import os
import pathlib
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_barplot(inpath, title, filename, errorbar=False):
    df = pd.read_csv(inpath, sep='\t')
    print(df.head())
    print(df.shape)
    print(df.columns)
    if 'filler_type' in df.columns:
        filler_types = df['filler_type'].unique()
    else:
        filler_types = ['sent_score']
    sns.set_context('paper', rc={'font.size':15,'axes.titlesize':15,'axes.labelsize':12})
    plt.clf()
    df['input_condition'] = df['input'] + '_' + df['condition']
    mapping = {'dom': 'DOM', 'def': 'definite article', 'indef': 'indefinite article', 'sent_score': 'sentence score'}
    for ftype in filler_types:
        if len(filler_types) > 1:
            filtered_df = df[df['filler_type'] == ftype]
        else:
            filtered_df = df
        if filtered_df['mean'].mean() > 0.0:
            ax = sns.barplot(data=filtered_df, x='input_condition', y='mean', hue='input',
                             palette=sns.color_palette('colorblind'))
            if errorbar:
                plt.errorbar(x=filtered_df['input_condition'], y=filtered_df['mean'],
                             yerr=filtered_df['std'], fmt='none', c='black', capsize=2)
            ax.get_legend().set_visible(False)
            ax.set_title(f'{title} {mapping[ftype]}')
            # ax.set_ylim(ymax=1.0)
            plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
            plt.tight_layout()
            outpath = inpath.replace('results', 'plots')
            cutoff = len(outpath.split('/')[-1]) + len(outpath.split('/')[-2]) + 1
            outpath = outpath[:-cutoff]
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            print(f'INPATH: {inpath}')
            print(f'OUTPATH: {outpath}')
            plt.savefig(f'{outpath}/{ftype}-{filename}')
            plt.show()


def create_boxplot(inpath, title):
    df = pd.read_csv(inpath, sep='\t')
    print(df.head())
    print(df.shape)
    print(df.columns)
    df['top_fillers'] = json.loads(df['top_fillers'])
    df['probabilities'] = json.loads(df['probabilities'])


filepath = os.path.join(pathlib.Path(__file__).parent.absolute(), '..', 'results/sentence-score/BETO/statistics.tsv')
create_barplot(filepath, 'BETO mean discrepancy', 'BETO-mean-discrepancy.png', errorbar=False)