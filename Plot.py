import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('bmh')
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['legend.loc'] = 'lower right'


# List of benchmarks to plot
files_to_plot = ['/Users/samlerman/Code/UnifiedML/Benchmarking/drqv2/dmc_cheetah_run.csv',
                 '/Users/samlerman/Code/UnifiedML/Benchmarking/drqv2/dmc_cup_catch.csv']


def plot(df, key='Reward', name='Curve'):
    tasks = np.sort(df.task.unique())
    cols = 2
    assert tasks.shape[0] % cols == 0
    rows = tasks.shape[0] // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    for i, task in enumerate(tasks):
        data = df[df['task'] == task]
        task = ' '.join([y.capitalize() for y in task.split('_')])
        data.columns = [' '.join([y.capitalize() for y in x.split('_')]) for x in data.columns]
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        hue_order = np.sort(data.Agent.unique())
        sns.lineplot(x='Step', y=key, data=data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax)
        ax.set_title(f'{task}')

    plt.tight_layout()
    plt.savefig(f'{name}.png')


def standardize_header(name):
    if name == 'episode_reward':
        return 'reward'
    return name


df = pd.concat([pd.read_csv(file).rename(standardize_header, axis='columns') for file in files_to_plot])

# Just standardizing
if 'step' not in df.columns:
    action_repeat = 2
    df['step'] = df['frame'] // action_repeat

plot(df)
