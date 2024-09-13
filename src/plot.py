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
            filename = f'{modelname}-mean-{ftype}-{measure}'
            plt.savefig(os.path.join(outpath, filename))
            plt.show()


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
    # merged_df.sort_values(by='input_condition', inplace=True)
    merged_df = reorder_by_data_condition(merged_df) # get datasets into correct order
    print(merged_df.head())
    print(merged_df.shape)
    print(merged_df.columns)
    print(merged_df['condition'].unique())
    p_columns = [x for x in merged_df.columns if '_prob' in x]
    sns.set_context('paper', rc={'axes.titlesize': 18, 'axes.labelsize': 14, 'xtick.labelsize': 12,
                                 'ytick.labelsize': 12, 'legend.fontsize': 12, 'legend.title_fontsize': 12})
    mapping = {'dom_prob': 'DOM', 'def_prob': 'definite article', 'indef_prob': 'indefinite article'}
    # iterate over prob columns
    for col_name in p_columns:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax = sns.boxplot(data=merged_df, x='input_condition', y=col_name, hue='condition', ax=ax,
                         showmeans=True, meanprops={"marker": "d", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"7"})
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
        plt.xlabel('Dataset')
        plt.ylabel(measure)
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
        plt.tight_layout()
        outpath = dir.replace('results', 'plots').replace(f'/{modelname}', '')
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        filename = f'{modelname}-{col_name}-boxplot.png'
        filename = filename.replace('_', '-')
        plt.savefig(os.path.join(outpath, filename))
        plt.show()


def create_sentencescore_boxplots(dir, modelname=None):
    if not modelname:
        modelname = 'BETO' if 'BETO' in dir else 'mBERT'
    figure, axis = plt.subplots(2, 3, figsize=(10, 7))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    row, col = 0, 0
    ordered_files = ['ms-2013-results.tsv','sa-2020-results.tsv','re-2021-results.tsv', 
                     're-2021-modified-results.tsv', 'hg-2023-results.tsv']
    for file in ordered_files:
        df1 = pd.read_csv(os.path.join(dir, file), sep='\t')
        df1.replace('nonaffected', 'non-affected', inplace=True)
        df1.replace('animate-human', 'human', inplace=True)
        df1.replace('animate-animal', 'animal', inplace=True)
        df2 = df1.copy()
        df1.drop(columns=['score_unmarked'], inplace=True)
        df1['type'] = ['DOM' for _ in range(df1.shape[0])]
        df1.rename(columns={'score_dom': 'score'}, inplace=True)
        df2.drop(columns=['score_dom'], inplace=True)
        df2['type'] = ['unmarked' for _ in range(df2.shape[0])]
        df2.rename(columns={'score_unmarked': 'score'}, inplace=True)
        df = pd.concat([df1, df2])
        sns.set_context('paper', rc={'axes.titlesize': 16, 'axes.labelsize': 16, 'xtick.labelsize': 16,                                 'ytick.labelsize': 16, 'legend.fontsize': 16, 'legend.title_fontsize': 16})
        ax = sns.boxplot(data=df, y='score', x='condition', hue='type', ax=axis[row, col], 
                            showmeans=True, meanprops={"marker": "d", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"7"})
        ax.set(xlabel=None)
        ax.set_ylim(ymin=0.0, ymax=0.65)
        ax.set_title(file[:-12])
        ax.tick_params(axis='x', labelsize=14)  # Set font size for x-axis tick labels
        ax.tick_params(axis='y', labelsize=14) 
        ax.set_ylabel('Score', fontsize=14)  # Increase y-axis label font size
        ax.legend_.remove()
        if col == 2:
            row += 1
            col = 0
        else:
            col += 1
    plt.delaxes(axis[-1, -1])
    handles, labels = axis[0, 0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc='right', bbox_to_anchor=(1.1, 0.5), 
                       bbox_transform=figure.transFigure, title='Version', fontsize=14, title_fontsize='x-large')
    figure.suptitle(f'{modelname} sentence scores', fontsize=18)#, y=1.02)
    outpath = dir.replace('results', 'plots').replace(f'/{modelname}', '')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = f'{modelname}-multi-boxplot.png'
    plt.savefig(os.path.join(outpath, filename))
    plt.show()

def create_sentencescore_boxplots(dir, modelname=None):
    if not modelname:
        modelname = 'BETO' if 'BETO' in dir else 'mBERT'
    figure, axis = plt.subplots(2, 3, figsize=(10, 7))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    row, col = 0, 0
    ordered_files = ['ms-2013-results.tsv','sa-2020-results.tsv','re-2021-results.tsv', 
                     're-2021-modified-results.tsv', 'hg-2023-results.tsv']
    for file in ordered_files:
        df1 = pd.read_csv(os.path.join(dir, file), sep='\t')
        df1.replace('nonaffected', 'non-affected', inplace=True)
        df1.replace('animate-human', 'human', inplace=True)
        df1.replace('animate-animal', 'animal', inplace=True)
        df2 = df1.copy()
        df1.drop(columns=['score_unmarked'], inplace=True)
        df1['type'] = ['DOM' for _ in range(df1.shape[0])]
        df1.rename(columns={'score_dom': 'score'}, inplace=True)
        df2.drop(columns=['score_dom'], inplace=True)
        df2['type'] = ['unmarked' for _ in range(df2.shape[0])]
        df2.rename(columns={'score_unmarked': 'score'}, inplace=True)
        df = pd.concat([df1, df2])
        sns.set_context('paper', rc={'axes.titlesize': 16, 'axes.labelsize': 16, 'xtick.labelsize': 16,                                 'ytick.labelsize': 16, 'legend.fontsize': 16, 'legend.title_fontsize': 16})
        ax = sns.boxplot(data=df, y='score', x='condition', hue='type', ax=axis[row, col], 
                            showmeans=True, meanprops={"marker": "d", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":"7"})
        ax.set(xlabel=None)
        ax.set_ylim(ymin=0.0, ymax=0.65)
        ax.set_title(file[:-12])
        ax.tick_params(axis='x', labelsize=14)  # Set font size for x-axis tick labels
        ax.tick_params(axis='y', labelsize=14) 
        ax.set_ylabel('Score', fontsize=14)  # Increase y-axis label font size
        ax.legend_.remove()
        if col == 2:
            row += 1
            col = 0
        else:
            col += 1
    plt.delaxes(axis[-1, -1])
    handles, labels = axis[0, 0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc='right', bbox_to_anchor=(1.1, 0.5), 
                       bbox_transform=figure.transFigure, title='Version', fontsize=14, title_fontsize='x-large')
    figure.suptitle(f'{modelname} sentence scores', fontsize=18)#, y=1.02)
    outpath = dir.replace('results', 'plots').replace(f'/{modelname}', '')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = f'{modelname}-multi-boxplot.png'
    plt.savefig(os.path.join(outpath, filename))
    plt.show()
    
def create_discrepancy_scatterplot(dir, modelname=None):
    models = ['BETO', 'mBERT']
    figure, axis = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.2, hspace=0.2, right=1.8) 
    col = 0
    for model in models:
        path = os.path.join(dir, model)
        print('PATH ', path)
        dfs = []
        for file in os.listdir(path):
            if file != 'statistics.tsv':
                df = pd.read_csv(os.path.join(path, file), sep='\t')
                dfs.append(df)
                source = file[:-12]
                df['input'] = [source for x in range(df.shape[0])]
        merged_df = pd.concat(dfs)
        merged_df = reorder_by_condition(merged_df) # get datasets into correct order
        print(merged_df)
        print(merged_df.shape)
        print(merged_df.columns)
        print(merged_df['condition'].unique())
        sns.set_context('paper', rc={'axes.titlesize': 22, 'axes.labelsize': 14, 'xtick.labelsize': 14,
                                            'ytick.labelsize': 14, 'legend.fontsize': 14, 'legend.title_fontsize': 14})
        ax = sns.scatterplot(data=merged_df, x='index', y='discrepancy', hue='condition', 
                            ax=axis[col], s=50)
        ax.set_title(f'{model} sentence-score discrepancy', loc='center')
        ax.set_ylim(ymax=0.35, ymin=-0.25)
        ax.legend_.remove()
        ax.set_ylabel('discrepancy', fontsize=16) 
        ax.set_xlabel('index', fontsize=16) 
        ax.tick_params(axis='x', labelsize=16)  # Set font size for x-axis tick labels
        ax.tick_params(axis='y', labelsize=16)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        # ax.set_aspect(aspect=0.67)
        col += 1
    handles, labels = axis[0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc='right', bbox_to_anchor=(1.0, 0.50), 
                       bbox_transform=figure.transFigure, title='condition', fontsize=16, title_fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    outpath = dir.replace('results', 'plots').replace(f'/{modelname}', '')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = f'sentencescore-discrepancy.png'
    plt.savefig(os.path.join(outpath, filename))
    # plt.show()

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

def reorder_by_data_condition(df):
    new_df = df
    new_df = new_df.replace('nonaffected', 'non-affected')
    data_mapping = {'ms-2013': 1, 'sa-2020': 2, 're-2021': 3, 're-2021-modified': 4, 'hg-2023': 5}
    new_df['data_index'] = [data_mapping[x] for x in list(new_df['input'])]
    cond_mapping = {'animate': 1, 'inanimate': 2,
                    'animate-human': 1, 'animate-animal':2,
                    'definite': 1, 'indefinite': 2,
                    'affected': 1,  'non-affected': 2}
    new_df['cond_index'] = [cond_mapping[x] for x in list(new_df['condition'])]
    return new_df.sort_values(by=['data_index', 'cond_index'])

def reorder_by_condition(df):
    new_df = df
    new_df = new_df.replace('nonaffected', 'non-affected')
    new_df = new_df.replace('animate-human', 'human')
    new_df = new_df.replace('animate-animal', 'animal')
    cond_mapping = {'animate': 1, 'inanimate': 2,
                    'human': 3, 'animal': 4,
                    'definite': 5, 'indefinite': 6,
                    'affected': 7,  'non-affected': 8}
    new_df['cond_index'] = [cond_mapping[x] for x in list(new_df['condition'])]
    new_df.sort_values(by=['cond_index'], inplace=True)
    new_df['index'] = [x for x in range(1, new_df.shape[0]+1)]
    return new_df

if __name__ == '__main__':
    dir = os.path.join(pathlib.Path(__file__).parent.absolute(), '..', 'results/sentence-score/')
    create_discrepancy_scatterplot(dir)