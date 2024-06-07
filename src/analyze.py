import os
import pathlib
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_barplot(inpath, title, filename):
    df = pd.read_csv(inpath, sep='\t')
    print(df.head())
    print(df.shape)
    print(df.columns)
    filler_types = df['filler_type'].unique()
    sns.set_context('paper', rc={'font.size':15,'axes.titlesize':15,'axes.labelsize':12})
    plt.clf()
    df['input_condition'] = df['input'] + '_' + df['condition']
    #TODO: plot mean with std if available
    for ftype in filler_types:
        filtered_df = df[df['filler_type'] == ftype]
        if filtered_df['mean_prob'].mean() > 0.0:
            ax = sns.barplot(data=filtered_df, x='input_condition', y='mean_prob', hue='input',
                             palette=sns.color_palette('colorblind'))
            ax.get_legend().set_visible(False)
            ax.set_title(f'{ftype} {title}')
            plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
            plt.tight_layout()
            outpath = inpath.replace('results', 'plots')
            cutoff = len(outpath.split('/')[-1]) + len(outpath.split('/')[-2]) + 1
            outpath = outpath[:-cutoff]
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            print(f'INPATH: {inpath}')
            print(f'OUTPATH: {outpath}')
            plt.savefig(outpath+'/'+filename)
            plt.show()


def create_boxplot(inpath, title):
    df = pd.read_csv(inpath, sep='\t')
    print(df.head())
    print(df.shape)
    print(df.columns)
    df['top_fillers'] = json.loads(df['top_fillers'])
    df['probabilities'] = json.loads(df['probabilities'])


filepath = os.path.join(pathlib.Path(__file__).parent.absolute(), '..', 'results/fill-mask/dom-masking/mBERT/statistics.tsv')
create_barplot(filepath, 'mean probability', 'mBERT-mean-prob.png')