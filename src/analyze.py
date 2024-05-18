import os
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_barplot(inpath, name):
    df = pd.read_csv(inpath, sep='\t')
    print(df)
    print(df.shape)
    filler_types = df['filler_type'].unique()
    sns.set_context('paper', rc={'font.size':15,'axes.titlesize':15,'axes.labelsize':12})
    plt.clf()
    # plot mean with std if available
    for ftype in filler_types:
        filtered_df = df[df['filler_type'] == ftype]
        if filtered_df['mean_prob'].mean() > 0.0:
            ax = sns.barplot(data=filtered_df, x='input', y='mean_prob', hue='model', palette=sns.color_palette('colorblind'))
            ax.set_title(f'{name} {ftype}')
            plt.savefig(f'{name}-{ftype}.png')
            plt.show()

create_barplot('../results/dom-masking/statistics.tsv', 'dom-masking')
create_barplot('../results/dobject-masking/dom-mask_det/BETO/statistics.tsv', 'dobj-masking-dom')