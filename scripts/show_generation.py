import argparse
import pandas as pd
from pathlib import Path

from datasets import load_from_disk
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
from tqdm import tqdm
import plotly.graph_objs as go
import seaborn as sns

from pabt5.util import spaceout
from pabt5.alignment import blastp_pair, alignment2hsp
from pabt5.dataset import get_antibody_info, get_region_identities


# helper function(s)
def func_is_heavy(sequenceA, sequenceB):
    infoA, infoB = get_antibody_info((
        ('A', sequenceA.replace(' ', '')),
        ('B', sequenceB.replace(' ', '')),
    ))
    if infoA['is_heavy'] == infoB['is_heavy']:
        return None
    return infoA['is_heavy']


def clean_species(species):
    if 'human' in species:
        return 'human'
    elif 'mouse' in species:
        return 'mouse'
    elif 'rat' in species:
        return 'rat'
    else:
        return None


def get_pairwise_identity(seq1, seq2, percent=True, **kwargs):
    alignments = blastp_pair(seq1, seq2, **kwargs)
    if not alignments:
        return 0

    hsp = alignment2hsp(alignments[0])
    if percent:
        return hsp.identities / hsp.align_length
    else:
        return hsp.identities


def _match_h(x):
    if x['pair_order'] == 'h2l':
        return not x['is_heavy_g']
    elif x['pair_order'] == 'l2h':
        return x['is_heavy_g']
    else:
        raise ValueError


def _get_chain_type(x):
    if x['pair_order'] == 'h2l':
        return x['chain_type_l']
    elif x['pair_order'] == 'l2h':
        return x['chain_type_h']
    else:
        raise ValueError


def _match_chain_type(x):
    if x['pair_order'] == 'h2l':
        return x['chain_type_l'] == x['chain_type_g']
    elif x['pair_order'] == 'l2h':
        return x['chain_type_h'] == x['chain_type_g']
    else:
        raise ValueError


def _match_species(x):
    if x['pair_order'] == 'h2l':
        return x['species_l'] == x['species_g']
    elif x['pair_order'] == 'l2h':
        return x['species_h'] == x['species_g']
    else:
        raise ValueError


def _match_family(x, family_name):
    if x['pair_order'] == 'h2l':
        return x[f'{family_name}_l'] == x[f'{family_name}_g']
    elif x['pair_order'] == 'l2h':
        return x[f'{family_name}_h'] == x[f'{family_name}_g']
    else:
        raise ValueError


def _format_ax(ax):
    ax.legend(loc='lower right')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    for v in ax.spines.values():
        v.set_visible(False)


def get_baseline(df_sub, df):
    n_uniques_sub = len(df_sub.groupby(['sequence_h', 'sequence_l']))
    n_uniques = len(df.groupby(['sequence_h', 'sequence_l']))
    return n_uniques_sub / n_uniques, n_uniques


plt.style.use('seaborn-colorblind')
colors = ('#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9')

generate_config = {
    'max_new_tokens': 130,
    'return_dict_in_generate': True,
    'do_sample': True,
    'top_p': 0.9,
    'num_return_sequences': 10,
    'temperature': 1,
}

# cmd options
cli = argparse.ArgumentParser()
cli.add_argument('--dataset_dir', type=str, default=None, help='include oas dataset in TSNE if provided')
cli.add_argument('--checkpoint_dir', type=str, help='path to model checkpoint')
cli.add_argument('--output_dir', type=str, default='visualization/generation')
cli.add_argument('--format', type=str, default='png', help='output figure format')
args = cli.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

csv = output_dir / 'generation.csv'
if csv.exists():
    df = pd.read_csv(csv)
else:
    # load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir)
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False, return_tensors='pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # load oas test set
    dataset = load_from_disk(args.dataset_dir)['test']

    df = dataset.to_pandas()
    df['A_is_heavy'] = df.apply(lambda x: func_is_heavy(x['sequenceA'], x['sequenceB']), axis='columns')
    df = df[~df['A_is_heavy'].isna()]  # 4 cases of homo chain pairing

    df['sequence_vh'] = df.apply(lambda x: x['sequenceA'] if x['A_is_heavy'] else x['sequenceB'], axis=1)
    df['sequence_vl'] = df.apply(lambda x: x['sequenceB'] if x['A_is_heavy'] else x['sequenceA'], axis=1)
    df['identifier'] = df['__index_level_0__']
    df = df[['sequence_vh', 'sequence_vl', 'identifier']]
    df = df.drop_duplicates(['sequence_vh', 'sequence_vl'])  # drop duplication due to symmetrization

    # generate sequences and classify
    data = []
    for _, sr in tqdm(df.iterrows(), total=len(df), desc='generate sequences'):
        pair_id = sr['identifier']
        sequence_h = sr['sequence_vh']
        sequence_l = sr['sequence_vl']
        info_vh, info_vl = get_antibody_info([
            ('vh', sequence_h.replace(' ', '')),
            ('vl', sequence_l.replace(' ', '')),
        ], assign_germline=True, output_regions=True)

        sequence_pairs = {
            'h2l': (sequence_h, sequence_l),
            'l2h': (sequence_l, sequence_h),
        }

        for pair_order, (sequenceA, sequenceB) in sequence_pairs.items():
            input_ids = tokenizer.encode(spaceout(sequenceA))
            labels = tokenizer.encode(spaceout(sequenceB))
            input_ids = torch.tensor(input_ids, device=device).view(1, -1)
            labels = torch.tensor(labels, device=device).view(1, -1)

            generate_outputs = model.generate(input_ids=input_ids, **generate_config)
            sequences = [tokenizer.decode(sequence_ids, skip_special_tokens=True).replace(' ', '')
                for sequence_ids in generate_outputs.sequences]

            for sequence_ids in generate_outputs.sequences:
                sequence = tokenizer.decode(sequence_ids, skip_special_tokens=True).replace(' ', '')
                info_vg = get_antibody_info([('tmp', sequence)], assign_germline=True, output_regions=True)[0]
               
                sequence_target = sequence_h.replace(' ', '') if pair_order == 'l2h' else sequence_l.replace(' ', '')
                identities = get_region_identities(info_vg, info_vh if pair_order == 'l2h' else info_vl)
                # identity = get_pairwise_identity(  # replaced by region identities
                #     sequence,
                #     sequence_target,
                #     percent=True,
                #     qcov_hsp_perc=90,
                # )

                # (chain type, species, family, identity with original sequence)
                datum = {
                    'sequence_h': sequence_h.replace(' ', ''),
                    'sequence_l': sequence_l.replace(' ', ''),
                    'pair_order': pair_order,
                    'generate_sequence': sequence,
                    **identities,
                    **{f'{k}_h': v for k, v in info_vh.items()},
                    **{f'{k}_l': v for k, v in info_vl.items()},
                    **{f'{k}_g': v for k, v in info_vg.items()},
                }
                data.append(datum)

    df = pd.DataFrame(data)
    df.to_csv(csv, index=None)

# by category evaluation
df['pair'] = df['sequence_h'] + '_' + df['sequence_l']
cols_family = ['v_gene_h', 'j_gene_h', 'v_gene_l', 'j_gene_l', 'v_gene_g', 'j_gene_g']
for col in cols_family:
    df[col] = df[col].apply(lambda x: x.split('-')[0].split('*')[0])

df['match_hl'] = df.apply(_match_h, axis='columns')
df['match_chain_type'] = df.apply(_match_chain_type, axis='columns')
df['match_species'] = df.apply(_match_species, axis='columns')
df['match_v_gene'] = df.apply(lambda x: _match_family(x, family_name='v_gene'), axis='columns')
df['match_j_gene'] = df.apply(lambda x: _match_family(x, family_name='j_gene'), axis='columns')

# FR and CDR identities and lengths (non-gap)
for region in ['FR1', 'FR2', 'FR3', 'FR4', 'CDR1', 'CDR2', 'CDR3', 'gapped_sequence']:
    df[f'{region}_t'] = df.apply(lambda x: x[f'{region}_h'] if x['pair_order'] == 'l2h' else x[f'{region}_l'], axis='columns')
    df[f'{region}_t_length'] = df[f'{region}_t'].apply(lambda x: len(x.replace('-', '')) if isinstance(x, str) else float('nan'))
    df[f'{region}_g_length'] = df[f'{region}_g'].apply(lambda x: len(x.replace('-', '')) if isinstance(x, str) else float('nan'))

cols_id = [col for col in df.columns if 'identity' in col]
cols_length = [col for col in df.columns if 'length' in col]

group = df.groupby('pair_order')
df_id = group[cols_id].mean().join(group[cols_id].std(), lsuffix='_mean', rsuffix='_std')
df_length = group[cols_length].mean().join(group[cols_length].std(), lsuffix='_mean', rsuffix='_std')

df_id.to_csv(output_dir / 'id.csv')
df_length.to_csv(output_dir / 'length.csv')

# cross-species mixing in generation
df['consistent_species'] = df.apply(lambda x: x['species_h'] == x['species_l'], axis='columns')
df_ = df[df['consistent_species']]
group = df_.groupby(['species_h', 'pair_order'])
df_mean = group['match_species'].mean()
df_mean = df_mean.reset_index()
df_mean = df_mean.pivot_table(index=['species_h'], columns=['pair_order'])
df_mean = df_mean['match_species'].rename(columns={'h2l': 'heavy-to-light', 'l2h': 'light-to-heavy'})

ax = sns.heatmap(df_mean, cmap='Blues', vmin=0, vmax=1, annot=True, cbar=True, square=True,)
ax.set_xlabel('Translation')
ax.set_ylabel('Species')

plt.tight_layout()
ax.figure.savefig(output_dir / 'species_heatmap.png', dpi=300)
plt.close()

# human specific evaluation
df_human = df[(df['species_h'] == 'human') & (df['species_l'] == 'human')]
df_ = df_human.copy()

df_h2l = df_human[df_human['pair_order'] == 'h2l']
df_l2h = df_human[df_human['pair_order'] == 'l2h']

data = [
    (
        'antibody',
        '',
        '',
        len(df_h2l) + len(df_l2h),  # generated match
        len(df_h2l) + len(df_l2h),  # generated count
        (len(df_h2l) + len(df_l2h)) // generate_config['num_return_sequences'],  # observed count
        (len(df_h2l) + len(df_l2h)) // generate_config['num_return_sequences'],  # parent count
    ),
    (
        'heavy',
        'heavy',
        'antibody',
        df_l2h['match_hl'].sum(),
        len(df_l2h),
        len(df_l2h) // generate_config['num_return_sequences'],
        (len(df_h2l) + len(df_l2h)) // generate_config['num_return_sequences'],
    ),
    (
        'light',
        'light',
        'antibody',
        df_h2l['match_hl'].sum(),
        len(df_h2l),
        len(df_h2l) // generate_config['num_return_sequences'],
        (len(df_h2l) + len(df_l2h)) // generate_config['num_return_sequences'],
    ),
    (
        'H',
        'H',
        'heavy',
        df_l2h['match_hl'].sum(),
        len(df_l2h),
        len(df_l2h) // generate_config['num_return_sequences'],
        (len(df_h2l) + len(df_l2h)) // generate_config['num_return_sequences'],
    ),
]

greek_letter_map = {
    'L': 'λ',
    'K': 'κ',
}

for chain_type, df_chain_type in df_h2l.groupby('chain_type_l'):
    data.append((
        greek_letter_map.get(chain_type, chain_type),
        greek_letter_map.get(chain_type, chain_type),
        'light',
        int(df_chain_type['match_chain_type'].sum()),
        len(df_chain_type),
        len(df_chain_type) // generate_config['num_return_sequences'],
        len(df_h2l) // generate_config['num_return_sequences'],
    ))

    for family1, df_family1 in df_chain_type.groupby('v_gene_l'):
        data.append((
            family1,
            family1,
            greek_letter_map.get(chain_type, chain_type),
            int(df_family1['match_v_gene'].sum()),
            len(df_family1),
            len(df_family1) // generate_config['num_return_sequences'],
            len(df_chain_type) // generate_config['num_return_sequences'],
        ))

        for family2, df_family2 in df_family1.groupby('j_gene_l'):
            data.append((
                family1 + ' - ' + family2,
                family2,
                family1,
                int((df_family2['match_v_gene'] & df_family2['match_j_gene']).sum()),
                len(df_family2),
                len(df_family2) // generate_config['num_return_sequences'],
                len(df_family1) // generate_config['num_return_sequences'],
            ))

for family1, df_family1 in df_l2h.groupby('v_gene_h'):
    data.append((
        family1,
        family1,
        'H',
        int(df_family1['match_v_gene'].sum()),
        len(df_family1),
        len(df_family1) // generate_config['num_return_sequences'],
        len(df_l2h) // generate_config['num_return_sequences'],
    ))

    for family2, df_family2 in df_family1.groupby('j_gene_h'):
        data.append((
            family1 + ' - ' + family2,
            family2,
            family1,
            int((df_family2['match_v_gene'] & df_family2['match_j_gene']).sum()),
            len(df_family2),
            len(df_family2) // generate_config['num_return_sequences'],
            len(df_family1) // generate_config['num_return_sequences'],
        ))

data = pd.DataFrame(
    data,
    columns=['ids', 'labels', 'parents', 'generated_match', 'generated_count', 'observed_count', 'parent_count']
)
data['accuracy'] = data['generated_match'] / data['generated_count']
data.to_csv(output_dir / 'human_evaluation.csv', index=False)

fig = go.Figure(
    data=go.Sunburst(
        ids=data['ids'],
        labels=data['labels'],
        parents=data['parents'],
        values=data['observed_count'],
        branchvalues='total',
        insidetextfont={'size': 18},
        opacity=1,
    ),
)
marker = fig.data[0].marker
marker.colors = [0] + data['accuracy'][1:].to_list()  # set root to white
marker.colorscale = 'Blues'
marker.cmin = 0
marker.cmax = 1
marker.showscale = True

marker.colorbar.title = 'Recovery rate'
marker.colorbar.titlefont = dict(size=18)
marker.colorbar.title.side = 'right'
# marker.colorbar.tickvals = [0, 0.5, 1]
# marker.colorbar.ticktext = [0, 0.5, 1]
marker.colorbar.tickfont = dict(size=16)

fig.update_layout(
    margin=dict(t=0, l=0, r=0, b=0),
)

# fig.show()
dpi = 300
fig.write_image(output_dir / 'sunbursts.png', width=3.5 * dpi, height=3 * dpi, scale=10)
