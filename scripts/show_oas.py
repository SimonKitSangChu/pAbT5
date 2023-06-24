import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import itertools

from pabt5.dataset import get_oas_statistics, load_oas_dataframe, get_antibody_info

plt.style.use('seaborn-colorblind')
tqdm.pandas()

# cmd options
cli = argparse.ArgumentParser()
cli.add_argument('--output_dir', type=str, default='visualization/oas_statistics')
cli.add_argument('--format', type=str, default='png', help='output figure format')
args = cli.parse_args()


def pairwise_dataframe(df):
    code = df.stack()
    code.index = code.index.droplevel(1)
    code.name = 'code'
    code = code.to_frame()
    pair = code.join(code, rsuffix='_2')
    return pd.crosstab(pair['code'], pair['code_2'])


output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

stats = get_oas_statistics()

# dataset by species
data = stats['Species']
data_species = {
    'human': data['human'],
    'rat': data['rat_SD'],
    'mouse': data['mouse_BALB/c'] + data['mouse_C57BL/6'],
}

ax = plt.pie(
    x=data_species.values(),
    labels=data_species.keys(),
    autopct='%.0f%%',
    startangle=90,
    textprops={'fontsize': 'x-large'},
)
plt.tight_layout()
plt.savefig(output_dir / f'species.{args.format}', dpi=300)


# pairwise statistics
df_oas = load_oas_dataframe()
vdj_cols = {col: col.replace('_call_', '') for col in df_oas.columns if '_call_' in col}
df_oas = df_oas.rename(columns=vdj_cols)
vdj_cols = ['vheavy', 'dheavy', 'jheavy', 'vlight', 'jlight']

df_human = df_oas[df_oas['species'] == 'human']  # restricted to human

csv = output_dir / 'pairwise_statistics.csv'
if csv.exists():
    df_ = pd.read_csv(csv)
else:
    def _assign_germline(row):
        info1, info2 = get_antibody_info([
            ('1', row['sequenceA']),
            ('2', row['sequenceB']),
        ], assign_germline=True)
        info_h, info_l = (info1, info2) if info1['is_heavy'] else (info2, info1)
        return pd.Series([info_h['v_gene'], info_h['j_gene'], info_l['v_gene'], info_l['j_gene']])

    df_ = df_human.progress_apply(_assign_germline, axis=1)
    df_.to_csv(csv, index=None)

df_human[['vheavy', 'jheavy', 'vlight', 'jlight']] = df_

cols_h = ['vheavy', 'jheavy']
cols_l = ['vlight', 'jlight']
for col in cols_h + cols_l:
    def coarse_germline(x):
        return x.split('*')[0]

    df_human[col] = df_human[col].apply(coarse_germline)

for col1, col2 in itertools.combinations(cols_h + cols_l, 2):  # pairwise recombination
    freq_org = pd.crosstab(df_human[col1], df_human[col2])
    index = sorted(freq_org.index)
    columns = sorted(freq_org.columns)

    freq_org_ = freq_org.reindex(index, axis=0).reindex(columns, axis=1)
    freq_org_ = freq_org_.fillna(0)

    # heatmap in original families
    ax = sns.heatmap(
        freq_org_.loc[freq_org.index, freq_org.columns],
        square=True,
        annot=False,
        vmin=0,
        cmap='Blues',
        cbar=True,
    )
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_dir / f'{col1}_{col2}.png', dpi=300)
    plt.close()

    df_ = freq_org_.loc[freq_org.index, freq_org.columns]
    df_.to_csv(output_dir / f'{col1}_{col2}.csv', index=None)

