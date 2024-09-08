import os
import pathlib
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_barplot(inpath, modelname, errorbar=False):
    measure = 'probability' if 'fill-mask' in inpath else 'discrepancy'
    add_str = ''
    if 'article-masking' in inpath:
        add_str = ' unmarked' if 'unmarked' in inpath else ' dom'
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
        filtered_df.sort_values(by='input', inplace=True)
        if filtered_df['mean'].mean() > 0.0:
            sns.set_context('paper', rc={'axes.titlesize': 18, 'axes.labelsize': 14, 'xtick.labelsize': 10,
                                         'ytick.labelsize': 12, 'legend.fontsize': 12, 'legend.title_fontsize': 12})
            fig, ax = plt.subplots(figsize=(10, 7))
            ax = sns.barplot(data=filtered_df, x='input_condition', y='mean', hue='condition', ax=ax)
            if errorbar:
                plt.errorbar(x=filtered_df['input_condition'], y=filtered_df['mean'],
                             yerr=filtered_df['std'], fmt='none', c='black', capsize=2)
            ax.set_title(f'{modelname} {mapping[ftype]} {measure}{add_str}', pad=20, loc='center')
            sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
            if measure == 'probability':
                ax.set_ylim(ymax=1.0)
            inputs = list(filtered_df['input'])
            positions = []
            stepsize = -1.5
            for inp in list(filtered_df['input'].unique()):
                stepsize += inputs.count(inp)
                if stepsize < 0:
                    stepsize = 0.5
                positions.append(stepsize)
            ax.set_xticks(positions, labels=list(filtered_df['input'].unique()))
            ax.set_axisbelow(True)
            ax.yaxis.grid(True)
            # add vertical lines after every two boxes (1 dataset)
            for pos in positions:
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
            filename = f'{modelname}-mean_{ftype}_{measure}'
            plt.savefig(os.path.join(outpath, filename))
            # plt.show()


def create_probability_boxplot(dir, modelname):
    measure = 'Probability'
    add_str = ''
    if 'article-masking' in dir:
        add_str = ' unmarked' if 'unmarked' in dir else ' dom'
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
    p_columns = [x for x in merged_df.columns if '_prob' in x]
    sns.set_context('paper', rc={'axes.titlesize': 18, 'axes.labelsize': 14, 'xtick.labelsize': 10,
                                 'ytick.labelsize': 12, 'legend.fontsize': 12, 'legend.title_fontsize': 12})
    mapping = {'dom_prob': 'DOM', 'def_prob': 'definite article', 'indef_prob': 'indefinite article'}
    # iterate over prob columns
    for col_name in p_columns:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax = sns.boxplot(data=merged_df, x='input_condition', y=col_name, hue='condition', ax=ax)
        ax.set_title(f'{modelname} {mapping[col_name]} {measure.lower()}{add_str}', pad=20, loc='center')
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(ymax=1.05)
        ax.yaxis.grid(True)
        positions = []
        stepsize = -1.5
        for inp in list(merged_df['input'].unique()):
            df_filtered = merged_df[merged_df['input'] == inp]
            steps = len(list(df_filtered['condition'].unique()))
            stepsize += steps
            if stepsize < 0:
                stepsize = 0.5
            positions.append(stepsize)
        ax.set_xticks(positions, labels=list(merged_df['input'].unique()))
        # add vertical lines after every two boxes (1 dataset)
        for pos in positions:
            ax.axvline(pos + 1, linewidth=0.5, color='grey')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        print(f'INPUTS {merged_df["input"].unique()}')
        print(f'POSITIONS {positions}')
        plt.xlabel('Dataset')
        plt.ylabel(measure)
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
        plt.tight_layout()
        outpath = dir.replace('results', 'plots').replace(f'/{modelname}', '')
        print('OUT ', outpath)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        filename = f'{modelname}-{col_name}-boxplot.png'
        plt.savefig(os.path.join(outpath, filename))
        plt.show()


def create_sentencescore_boxplots(dir, modelname):
    figure, axis = plt.subplots(3, 2, figsize=(10, 13))
    figure.subplots_adjust(hspace=0.4, wspace=0.3)
    row, col = 0, 0
    for file in sorted(os.listdir(dir)):
        if file != 'statistics.tsv':
            print(file)
            print(row, col)
            df1 = pd.read_csv(os.path.join(dir, file), sep='\t')
            print(df1.columns)
            df2 = df1.copy()
            df1.drop(columns=['score_unmarked'], inplace=True)
            df1['type'] = ['DOM' for _ in range(df1.shape[0])]
            df1.rename(columns={'score_dom': 'score'}, inplace=True)
            df2.drop(columns=['score_dom'], inplace=True)
            df2['type'] = ['unmarked' for _ in range(df2.shape[0])]
            df2.rename(columns={'score_unmarked': 'score'}, inplace=True)
            df = pd.concat([df1, df2])
            print(df.head())
            print(df.shape)
            print(df.columns)
            ax = sns.boxplot(data=df, y='score', x='condition', hue='type', ax=axis[row, col], showmeans=True)
            ax.set_title(file[:-12])
            ax.legend_.remove()
            if col == 1:
                row += 1
                col = 0
            else:
                col += 1
    plt.delaxes(axis[-1, -1])
    handles, labels = axis[0, 0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc='center', bbox_to_anchor=(0.6, 0.25), bbox_transform=figure.transFigure,
                           title='Type', fontsize='large', title_fontsize='x-large')
    outpath = dir.replace('results', 'plots').replace(f'/{modelname}', '')
    print('OUT ', outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = f'{modelname}-multi-boxplot.png'
    plt.savefig(os.path.join(outpath, filename))
    plt.show()


def create_all_barplots(dir_path):
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            print(subdir)
            print(file)
            if file == 'statistics.tsv':
                if 'BETO' in subdir:
                    modelname = 'BETO'
                else:
                    modelname = 'mBERT'
                path = os.path.join(subdir, file)
                create_barplot(path, modelname=modelname)


def create_fillmask_boxplots(dir_path):
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            print(subdir)
            print(file)
            if file == 'statistics.tsv':
                if 'BETO' in subdir:
                    modelname = 'BETO'
                else:
                    modelname = 'mBERT'
                path = subdir
                create_probability_boxplot(path, modelname=modelname)


if __name__ == '__main__':
    dir = os.path.join(pathlib.Path(__file__).parent.absolute(), '..', 'results/fill-mask/dom-masking/BETO')
    create_fillmask_boxplots(dir)
    import matplotlib
    print(matplotlib.__version__)
    import seaborn
    print(seaborn.__version__)
