from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-colorblind')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

output_dir = Path('../visualization/exp')

df = []
for csv in output_dir.glob('*/*/correlations.csv'):
    try:
        df_ = pd.read_csv(csv, index_col=None)
        df_['model_name'] = csv.parents[1].stem
        df_['dataset_name'] = csv.parent.stem
        df.append(df_)
    except pd.errors.EmptyDataError:
        warnings.warn(f'{csv} is empty')

df = pd.concat(df)
df = df.sort_values(['dataset_name', 'column'])
df.to_csv(output_dir / 'gather.csv', index=None)

# plotting
df_plot = []  # filtering
drop_dataset_names = ('absci_7d', 'absci_7b', 'absci_7', 'commonvl')

for (dataset_name, column), df_ in df.groupby(['dataset_name', 'column']):
    if dataset_name in drop_dataset_names:
        continue

    if True:  # show all datasets
    # if (df_['p_value_pair'] < 0.05).any():  # show only significant datasets
        df_plot.append(df_)

df_plot = pd.concat(df_plot)

df_plot = df_plot.set_index(['dataset_name', 'column'])  # sort by src_mean_pair
df_plot['y'] = df_plot['src_mean_pair'].abs()
index = df_plot.loc[df_plot['model_name'] == 'pabt5'].sort_values('y', ascending=False).index
df_plot = df_plot.loc[index].reset_index()
df_plot['x'] = df_plot['dataset_name'] + '_' + df_plot['column']

# comparison with SOTA
model_names = ('pabt5', 'progen2-oas', 'progen2-base', 'esm1v', 'esm2_3B')
dfs_model = {k: v for k, v in df_plot.groupby('model_name')}
df_model_ref = dfs_model['pabt5']
n_datasets = len(df_model_ref)

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), gridspec_kw={'height_ratios': [1, 4.5]})
for i_model, model_name in enumerate(model_names, 1):
    shift = 0.1 * (i_model - (len(model_names) + 1) / 2)
    df_model = dfs_model[model_name]
    kwargs = {
        'x': np.arange(n_datasets) + shift,
        'label': model_name,
        'markersize': 5,
        'fmt': '.',
    }

    ax0.errorbar(
        y=[a - b for a, b in zip(df_model['src_mean_pair'].abs(), df_model_ref['src_mean_pair'].abs())],
        yerr=[(a ** 2 + b ** 2) ** 0.5 for a, b in \
              zip(df_model['src_std_pair'].tolist(), df_model_ref['src_std_pair'].tolist())],
        **kwargs
    )
    ax1.errorbar(
        y=df_model['src_mean_pair'].abs(),
        yerr=df_model['src_std_pair'],
        **kwargs
    )

ax0.plot([-1, n_datasets - 0.5], [0, 0], 'k--', linewidth=0.5)
ax0.set_xticks(np.arange(n_datasets))
ax0.tick_params(axis='x', length=0)
ax0.set_xticklabels([])
ax0.set_xlim(-0.5, n_datasets - 0.5)
ax0.set_ylim(-0.5, 0.5)
ax0.set_yticks([-0.5, 0, 0.5])
ax0.set_ylabel('Δ', rotation=90)

ax1.set_ylim(0, 1)
ax1.set_xticks(np.arange(n_datasets))
xticklabels = [x.replace('_exp', '') for x in df_model_ref['dataset_name'].tolist()]
ax1.set_xticklabels(xticklabels, rotation=35, fontsize=9, ha='right')
ax1.set_xlabel('')
ax1.set_ylabel('|Spearman Rank Correlation|', fontsize=10)
ax1.legend(loc='upper right')

plt.tight_layout(h_pad=0.1)
fig.savefig(output_dir / 'sota_comparison.png', dpi=300)
plt.close(fig)

# ablation study
dfs_ablation = {
    'pabt5': df_model_ref.rename(columns={
        'src_mean_pair': 'src_mean',
        'src_std_pair': 'src_std',
    }),
    'decoder-only': df_plot.loc[df_plot['model_name'] == 'pabt5_dec'].rename(columns={
        'src_mean_pair': 'src_mean',
        'src_std_pair': 'src_std',
    }),
    'no pretraining': df_plot.loc[df_plot['model_name'] == 't5_3b'].rename(columns={
        'src_mean_pair': 'src_mean',
        'src_std_pair': 'src_std',
    }),
    'light-to-heavy': df_plot.loc[df_plot['model_name'] == 'pabt5'].rename(columns={
        'src_mean_H': 'src_mean',
        'src_std_H': 'src_std',
    }),
    'heavy-to-light': df_plot.loc[df_plot['model_name'] == 'pabt5'].rename(columns={
        'src_mean_L': 'src_mean',
        'src_std_L': 'src_std',
    }),
}
df_model_ref = dfs_ablation['pabt5']

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), gridspec_kw={'height_ratios': [1, 4.5]})
for i_model, (model_name, df_model) in enumerate(dfs_ablation.items(), 1):
    shift = 0.1 * (i_model - (len(model_names) + 1) / 2)
    kwargs = {
        'x': np.arange(n_datasets) + shift,
        'label': model_name,
        'markersize': 5,
        'fmt': '.',
    }

    ax0.errorbar(
        y=[a - b for a, b in zip(df_model['src_mean'].abs(), df_model_ref['src_mean'].abs())],
        yerr=[(a ** 2 + b ** 2) ** 0.5 for a, b in \
              zip(df_model['src_std'].tolist(), df_model_ref['src_std'].tolist())],
        **kwargs
    )
    ax1.errorbar(
        y=df_model['src_mean'].abs(),
        yerr=df_model['src_std'],
        **kwargs
    )

ax0.plot([-1, n_datasets - 0.5], [0, 0], 'k--', linewidth=0.5)
ax0.set_xticks(np.arange(n_datasets))
ax0.tick_params(axis='x', length=0)
ax0.set_xticklabels([])
ax0.set_xlim(-0.5, n_datasets - 0.5)
ax0.set_ylim(-0.5, 0.5)
ax0.set_yticks([-0.5, 0, 0.5])
ax0.set_ylabel('Δ', rotation=90)

ax1.set_ylim(0, 1)
ax1.set_xticks(np.arange(n_datasets))
xticklabels = [x.replace('_exp', '') for x in df_model_ref['dataset_name'].tolist()]
ax1.set_xticklabels(xticklabels, rotation=35, fontsize=9, ha='right')
ax1.set_xlabel('')
ax1.set_ylabel('|Spearman Rank Correlation|', fontsize=10)
ax1.legend(loc='upper right')

plt.tight_layout(h_pad=0.1)
fig.savefig(output_dir / 'ablation_comparison.png', dpi=300)
plt.close(fig)
