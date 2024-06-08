import os
import pathlib
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_barplot(inpath, modelname, measure, errorbar=False):
    df = pd.read_csv(inpath, sep='\t')
    print(df.head())
    print(df.shape)
    print(df.columns)
    if 'filler_type' in df.columns:
        filler_types = df['filler_type'].unique()
    else:
        filler_types = ['sent_score']
    df['input_condition'] = df['input'] + '_' + df['condition']
    mapping = {'dom': 'DOM', 'def': 'definite article', 'indef': 'indefinite article', 'sent_score': 'sentence score'}
    for ftype in filler_types:
        if len(filler_types) > 1:
            filtered_df = df[df['filler_type'] == ftype]
        else:
            filtered_df = df
        filtered_df.sort_values(by='input_condition', inplace=True)
        if filtered_df['mean'].mean() > 0.0:
            sns.set_context('paper', rc={'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 10,
                                         'ytick.labelsize': 12, 'legend.fontsize': 12, 'legend.title_fontsize': 12})
            fig, ax = plt.subplots(figsize=(10, 7))
            ax = sns.barplot(data=filtered_df, x='input_condition', y='mean', hue='condition')
            if errorbar:
                plt.errorbar(x=filtered_df['input_condition'], y=filtered_df['mean'],
                             yerr=filtered_df['std'], fmt='none', c='black', capsize=2)
            ax.set_title(f'{modelname} {mapping[ftype]} {measure} per dataset and condition')
            conditions = list(filtered_df['input'].unique())
            items = range(len(filtered_df['input_condition'].unique()))
            tick_pos = [x - 0.5 for x in list(items[1::2])]
            ax.set_xticks(tick_pos, labels=conditions)
            sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
            ax.set_ylim(ymax=1.0)
            ax.yaxis.grid(True)
            # add vertical lines after every two boxes (1 dataset)
            for pos in tick_pos:
                ax.axvline(pos + 1, linewidth=0.5, color='grey')
            plt.xlabel('Dataset')
            plt.ylabel(f'Mean {measure}')
            plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
            plt.tight_layout()
            outpath = inpath.replace('results', 'plots')
            cutoff = len(outpath.split('/')[-1]) + len(outpath.split('/')[-2]) + 1
            outpath = outpath[:-cutoff]
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            filename = f'{modelname}-mean_{measure}'
            plt.savefig(os.path.join(outpath, filename))
            plt.show()


def create_boxplot(dir, modelname, measure):
    dfs = []
    for file in os.listdir(dir):
        if file != 'statistics.tsv':
            df = pd.read_csv(os.path.join(dir, file), sep='\t')
            dfs.append(df)
            source = file[:-12]
            df['input'] = [source for x in range(df.shape[0])]
            df['input_condition'] = df['input'] + '_' + df['condition']
    merged_df = pd.concat(dfs)
    merged_df.sort_values(by='input_condition', inplace=True)
    print(merged_df.head())
    print(merged_df.shape)
    print(merged_df.columns)
    sns.set_context('paper', rc={'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 10,
                                 'ytick.labelsize': 12, 'legend.fontsize': 12, 'legend.title_fontsize': 12})
    fig, ax = plt.subplots(figsize=(10, 7))
    ax = sns.boxplot(data=merged_df, x='input_condition', y='dom_prob', hue='condition', ax=ax)
    ax.set_title(f'{modelname} DOM {measure} per dataset and condition')
    conditions = list(merged_df['input'].unique())
    items = range(len(merged_df['input_condition'].unique()))
    tick_pos = [x - 0.5 for x in list(items[1::2])]
    ax.set_xticks(tick_pos, labels=conditions)
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
    ax.yaxis.grid(True)
    # add vertical lines after every two boxes (1 dataset)
    for pos in tick_pos:
        ax.axvline(pos+1, linewidth=0.5, color='grey')
    plt.xlabel('Dataset')
    plt.ylabel('Probability')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
    plt.tight_layout()
    outpath = dir.replace('results', 'plots').replace(f'{modelname}/', '')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = f'{modelname}-{measure}-boxplot.png'
    plt.savefig(os.path.join(outpath, filename))
    plt.show()

filepath = os.path.join(pathlib.Path(__file__).parent.absolute(), '..', 'results/fill-mask/dom-masking/mBERT/')
create_boxplot(filepath, modelname='mBERT', measure='probability')