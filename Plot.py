import glob
from pathlib import Path

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
experiment_names = ['humanoid', '_munch_trust_ent']
path = Path.cwd()
files_to_plot = sum([glob.glob(f'{str(path)}/**/Eval*{name}.csv', recursive=True) for name in experiment_names], [])
also_files_to_plot = ['humanoid_walk']
files_to_plot += sum([glob.glob(f'{str(path)}/**/{name}.csv', recursive=True) for name in also_files_to_plot], [])

print('Plotting from', path)
for file_name in files_to_plot:
    print(file_name)


def plot(df, key='Reward', name='Curve2'):
    tasks = np.sort(df.task.unique())
    cols = int(np.floor(np.sqrt(tasks.shape[0])))
    while tasks.shape[0] % cols != 0:
        cols -= 1
    assert tasks.shape[0] % cols == 0, f'{tasks.shape[0]} tasks, {cols} columns invalid'
    rows = tasks.shape[0] // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

    for i, task in enumerate(tasks):
        data = df[df['task'] == task]
        task = ' '.join([y.capitalize() for y in task.split('_')])
        data.columns = [' '.join([y.capitalize() for y in x.split('_')]) for x in data.columns]

        data.loc[data['Agent'] == 'SPR', 'Agent'] = 'SPR-general (mine)'

        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col] if cols > 1 else axs
        hue_order = np.sort(data.Agent.unique())

        sns.lineplot(x='Step', y=key, data=data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax)
        ax.set_title(f'{task}')

    plt.tight_layout()
    plt.savefig(f'{name}.png')


def standardize_header(name):
    if name == 'episode_reward':
        return 'reward'
    return name


# df = pd.concat([pd.read_csv(file).rename(standardize_header, axis='columns') for file in files_to_plot])

# Just standardizing
# if 'step' not in df.columns and 'Step' not in df.columns:
#     action_repeat = 2
#     df['step'] = df['frame'] // action_repeat


i = 0
# TODO automatically append experiment name if same agent from different experiment already encountered
n = ['-munch_trust_ent' if '_munch_trust_ent' in file else '' for file in files_to_plot]
to_c = []
for file in files_to_plot:
    bla = pd.read_csv(file).rename(standardize_header, axis='columns')
    if 'step' not in bla.columns and 'Step' not in bla.columns:
        action_repeat = 2
        bla['step'] = bla['frame'] // action_repeat
    if 'time' not in bla.columns:
        action_repeat = 2
        bla['time'] = bla['hour']
    if 'hour' not in bla.columns:
        action_repeat = 2
        bla['hour'] = bla['time']
    bla['agent'] = bla['agent'] + f'{n[i]}'
    to_c.append(bla)
    i += 1
df = pd.concat(to_c, ignore_index=True)

plot(df)
