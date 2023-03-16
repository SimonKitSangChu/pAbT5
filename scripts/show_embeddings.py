import argparse
import pandas as pd
from pathlib import Path
import itertools
import warnings

from datasets import load_from_disk
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
from tqdm import tqdm

from pabt5.util import spaceout
from pabt5.dataset import get_antibody_info, load_oas_dataframe


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


def pairwise_dataframe(df):
    code = df.stack()
    code.index = code.index.droplevel(1)
    code.name = 'code'
    code = code.to_frame()
    pair = code.join(code, rsuffix='_2')
    return pd.crosstab(pair['code'], pair['code_2'])


def _format_ax(ax):
    ax.legend(loc='lower right')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    for v in ax.spines.values():
        v.set_visible(False)


def _format_heatmap_axes(fig, axes, col1, col2):
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('observed')
    axes[1].set_xlabel('generated')
    axes[1].set_yticks([])
    # fig.supylabel(col1)
    # fig.supxlabel(col2)


plt.style.use('seaborn-colorblind')
colors = ('#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9')
colors_tab20 = matplotlib.cm.get_cmap('tab20').colors

generate_config = {
    'max_new_tokens': 130,
    'return_dict_in_generate': True,
    'do_sample': True,
    'top_k': 5,
    'num_return_sequences': 8,
    'temperature': 1,
}

# cmd options
cli = argparse.ArgumentParser()
cli.add_argument('--generate', default=False, action='store_true', help='keep generate sequences for TSNE plot')
cli.add_argument('--no-generate', action='store_false', dest='generate')
cli.add_argument('--oas_dataset_dir', type=str, default=None)
cli.add_argument('--checkpoint_dir', type=str, help='path to model checkpoint')
cli.add_argument('--output_dir', type=str, default='visualization/embeddings')
cli.add_argument('--keep_oas_metadata', default=False, action='store_true',
                 help='keep oas species info from spreadsheet header and family')
cli.add_argument('--format', type=str, default='png', help='output figure format')
cli.add_argument('--drop_duplicate', default=False, action='store_true')
cli.add_argument('--no_drop_duplicate', action='store_false')
args = cli.parse_args()

if args.keep_oas_metadata and args.generate:
    warnings.warn('assign species and family info from observed antibody onto generated sequences')

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

pt = output_dir / f'embeddings.pt'
if pt.exists():
    data = torch.load(pt)
else:
    # load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir)
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False, return_tensors='pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # load oas test set
    dataset = load_from_disk(args.oas_dataset_dir)['test']

    df = dataset.to_pandas()
    df['A_is_heavy'] = df.apply(lambda x: func_is_heavy(x['sequenceA'], x['sequenceB']), axis='columns')
    df = df[~df['A_is_heavy'].isna()]  # 4 cases of homo chain pairing

    df['sequence_vh'] = df.apply(lambda x: x['sequenceA'] if x['A_is_heavy'] else x['sequenceB'], axis=1)
    df['sequence_vl'] = df.apply(lambda x: x['sequenceB'] if x['A_is_heavy'] else x['sequenceA'], axis=1)
    df['identifier'] = df['__index_level_0__']
    df = df[['sequence_vh', 'sequence_vl', 'identifier']]

    # generate embeddings
    data = {}
    for _, sr in tqdm(df.iterrows(), total=len(df), desc='generate embeddings'):
        pair_id = sr['identifier']
        sequence_h = sr['sequence_vh']
        sequence_l = sr['sequence_vl']
        info_vh, info_vl = get_antibody_info([
            ('heavy', sequence_h),
            ('light', sequence_l),
        ], assign_germline=True)

        sequence_pairs = {
            'h2l': (sequence_h, sequence_l),
            'l2h': (sequence_l, sequence_h),
        }
        data[pair_id] = {
            'sequence_h': sequence_h.replace(' ', ''),
            'sequence_l': sequence_l.replace(' ', ''),
            'info_vh': info_vh,
            'info_vl': info_vl,
        }

        for pair_order, (sequenceA, sequenceB) in sequence_pairs.items():
            input_ids = tokenizer.encode(spaceout(sequenceA))
            labels = tokenizer.encode(spaceout(sequenceB))
            input_ids = torch.tensor(input_ids, device=device).view(1, -1)
            labels = torch.tensor(labels, device=device).view(1, -1)

            with torch.no_grad():
                output = model.forward(
                    input_ids=input_ids,
                    labels=labels,
                    output_hidden_states=True,
                    return_dict=True,
                )
                decoder_hidden_state = output.decoder_hidden_states[-1].cpu().numpy()

            with torch.no_grad():
                output = model.forward(
                    input_ids=labels,
                    labels=input_ids,  # placeholder
                    output_hidden_states=True,
                    return_dict=True
                )
                encoder_hidden_state = output.encoder_last_hidden_state.cpu().numpy()

            data[pair_id][pair_order] = {'original': {
                'decoder_hidden_state': decoder_hidden_state,
                'encoder_hidden_state': encoder_hidden_state,
            }}

            if args.generate:
                generate_outputs = model.generate(input_ids=input_ids, output_hidden_states=True, **generate_config)
                sequences = [tokenizer.decode(sequence_ids, skip_special_tokens=True).replace(' ', '')
                             for sequence_ids in generate_outputs.sequences]

                encoder_hidden_states = []
                decoder_hidden_states = []

                for sequence in sequences:
                    decoder_input_ids = tokenizer.encode(spaceout(sequence))  # ensure variable-length decoder_input_ids
                    decoder_input_ids = torch.tensor(decoder_input_ids, device=model.device).view(1, -1)

                    with torch.no_grad():
                        output = model.forward(
                            input_ids=input_ids,
                            labels=decoder_input_ids,
                            output_hidden_states=True,
                            return_dict=True
                        )
                    decoder_hidden_state = output.decoder_hidden_states[-1].cpu().numpy()
                    decoder_hidden_states.append(decoder_hidden_state)

                    with torch.no_grad():
                        output = model.forward(
                            input_ids=decoder_input_ids,
                            labels=input_ids,  # dummy
                            output_hidden_states=True,
                            return_dict=True
                        )
                    encoder_hidden_state = output.encoder_hidden_states[-1].cpu().numpy()
                    encoder_hidden_states.append(encoder_hidden_state)

                info_generate = get_antibody_info([
                    (str(i), sequence) for i, sequence in enumerate(sequences)
                ], assign_germline=True)

                data[pair_id][pair_order]['generate'] = {
                    'sequences': sequences,
                    'info': info_generate,
                    'decoder_hidden_state': decoder_hidden_states,
                    'encoder_hidden_state': encoder_hidden_states,
                }

    torch.save(data, pt)

# data aggregation and TSNE projection (and human-specific)
df_oas = load_oas_dataframe()
vdj_cols = {col: col.replace('_call_', '') for col in df_oas.columns if '_call_' in col}
df_oas = df_oas.rename(columns=vdj_cols)
vdj_cols = ['vheavy', 'dheavy', 'jheavy', 'vlight', 'jlight']

csv = output_dir / 'embeddings.csv'
if csv.exists():
    df = pd.read_csv(csv)
    if not args.generate:
        df = df[~df['generate']]
else:
    df = []  # format: [[ generate, is_heavy, chain type, species, *vdj_heavy, *vdj_light, *hidden_states ]]
    data = torch.load(pt)

    for datum in tqdm(data.values(), total=len(data), desc='add metadata'):
        sequence_h = datum['sequence_h']
        sequence_l = datum['sequence_l']
        info_vh = datum['info_vh']
        info_vl = datum['info_vl']

        df_ = df_oas[(df_oas['sequenceA'] == sequence_h) & (df_oas['sequenceB'] == sequence_l)]
        if len(df_) == 0:
            raise ValueError('antibody pair not found in oas dataframe')
        elif len(df_) > 1:
            warnings.warn('duplicate pair found in oas_dataframe')

        if args.keep_oas_metadata:
            sr = df_.iloc[0]  # vlight, vheavy, dheavy, ...
            vdj = list(sr[vdj_cols])
            species_h = species_l = sr['species']
        else:
            species_h = info_vh['species']
            species_l = info_vl['species']
            vdj = (info_vh['v_gene'], None, info_vh['j_gene'], info_vl['v_gene'], info_vl['j_gene'])

        vh_hidden_state = datum['l2h']['original']['encoder_hidden_state'][0].mean(axis=0)
        vl_hidden_state = datum['h2l']['original']['encoder_hidden_state'][0].mean(axis=0)
        df.append([sequence_h, False, True, info_vh['chain_type'], info_vh['species'], *vdj, *vh_hidden_state])
        df.append([sequence_l, False, False, info_vl['chain_type'], info_vl['species'], *vdj, *vl_hidden_state])

        # generate
        if args.generate:
            for sequence_gh, sequence_gl, vh_hidden_state, vl_hidden_state, info_gh, info_gl in zip(
                    datum['l2h']['generate']['sequences'],
                    datum['h2l']['generate']['sequences'],
                    datum['l2h']['generate']['encoder_hidden_state'],
                    datum['h2l']['generate']['encoder_hidden_state'],
                    datum['l2h']['generate']['info'],
                    datum['h2l']['generate']['info'],
            ):
                species_h = info_gh['species']
                species_l = info_gl['species']
                vdj = (info_gh['v_gene'], None, info_gh['j_gene'], info_gl['v_gene'], info_gl['j_gene'])
                vh_hidden_state = vh_hidden_state[0].mean(axis=0)
                vl_hidden_state = vl_hidden_state[0].mean(axis=0)
                df.append([sequence_gh, True, True, info_gh['chain_type'], species_h, *vdj, *vh_hidden_state])
                df.append([sequence_gl, True, False, info_gl['chain_type'], species_l, *vdj, *vl_hidden_state])

    hidden_state_size = vh_hidden_state.shape[-1]
    df = pd.DataFrame(
        df,
        columns=['decoder_sequence', 'generate', 'is_heavy', 'chain_type', 'species', *vdj_cols, *range(hidden_state_size)]
    )
    df['species'] = df['species'].apply(clean_species)

    if args.drop_duplicate:
        df = df.drop_duplicates('decoder_sequence')

    tsne = TSNE(n_components=2, init='pca', n_jobs=-1, perplexity=10)
    X = df[range(hidden_state_size)].values
    X_tsne = tsne.fit_transform(X)

    df['tsne_0'] = X_tsne[:, 0]
    df['tsne_1'] = X_tsne[:, 1]

    df.to_csv(output_dir / 'embeddings.csv', index=None)

# # heavy v.s. light (generate) in OAS dataset
ax = None
if args.generate:
    df_vh_generate = df[df['is_heavy'] & df['generate']]
    df_vl_generate = df[~df['is_heavy'] & df['generate']]
    ax = df_vh_generate.plot.scatter(x='tsne_0', y='tsne_1', color=colors[0], s=0.25, alpha=0.5, ax=ax)
    ax = df_vl_generate.plot.scatter(x='tsne_0', y='tsne_1', color=colors[1], s=0.25, alpha=0.5, ax=ax)

df_vh_original = df[df['is_heavy'] & ~df['generate']]
df_vl_original = df[~df['is_heavy'] & ~df['generate']]
ax = df_vh_original.plot.scatter(x='tsne_0', y='tsne_1', color=colors[0], label='heavy chain', alpha=0.5, s=10,  ax=ax)
ax = df_vl_original.plot.scatter(x='tsne_0', y='tsne_1', color=colors[1], label='light chain', alpha=0.5, s=10, ax=ax)

_format_ax(ax)
plt.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=12)
plt.tight_layout()
figname = output_dir / f'heavy_light.{args.format}'
plt.savefig(figname, dpi=300)
plt.close()

# by species
ax = None
species = ('human', 'mouse', 'rat')

for i, species_ in enumerate(species):
    df_spc = df[df['species'] == species_]
    for is_generate, df_gen in df_spc.groupby('generate'):
        if is_generate and not args.generate:
            continue

        ax = df_gen.plot.scatter(
            x='tsne_0',
            y='tsne_1',
            color=colors[i + 2],  # different color from H/L
            label=None if is_generate else species_,
            alpha=0.5,
            s=0.25 if is_generate else 10,
            ax=ax,
        )

_format_ax(ax)
plt.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=12)
plt.tight_layout()
figname = output_dir / f'by_species.{args.format}'
plt.savefig(figname, dpi=300)
plt.close()

# color by chain type (human specific)
df_human = df[df['species'] == 'human'].copy()
ax = None
chain2color = {'H': colors[0], 'K': colors[1], 'L': colors[2]}
greek_letter_map = {
    'L': 'λ',
    'K': 'κ',
}

for (is_generate, chain_type), df_ in df_human.groupby(['generate', 'chain_type']):
    if is_generate and not args.generate:
        continue

    ax = df_.plot.scatter(
        x='tsne_0',
        y='tsne_1',
        color=chain2color[chain_type],
        s=0.25 if is_generate else 10,
        alpha=0.5,
        label=None if is_generate else greek_letter_map.get(chain_type, chain_type),
        ax=ax
    )

_format_ax(ax)
plt.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=12)
plt.tight_layout()
figname = output_dir / f'chain_type.{args.format}'
plt.savefig(figname, dpi=300)
plt.close()

# color by (VD)J family (human specific)
cols_h = ['vheavy', 'jheavy']
cols_l = ['vlight', 'jlight']
cols = cols_h + cols_l + ['tsne_0', 'tsne_1']

for col in cols_h + cols_l:  # drop subfamily information
    df_human[col] = df_human[col].apply(lambda x: x.split('-')[0].split('*')[0])

for col in cols_h + cols_l:
    is_heavy_family = 'heavy' in col
    families = sorted(df_human[col].unique())

    ax = None
    for i, family in enumerate(families):
        c = colors_tab20[i % len(colors_tab20)]

        # TSNE for the correct chain type
        idx = ~df_human['generate'] & (df_human[col] == family) & (df_human['is_heavy'] == is_heavy_family)
        df_ = df_human.loc[idx, cols].dropna()
        ax = df_.plot.scatter(x='tsne_0', y='tsne_1', s=10, alpha=0.5, color=c, label=family, ax=ax)

        # void TSNE for the incorrect chain type
        idx = ~df_human['generate'] & (df_human[col] == family) & (df_human['is_heavy'] != is_heavy_family)
        df_ = df_human.loc[idx, cols].dropna()
        ax = df_.plot.scatter(x='tsne_0', y='tsne_1', s=10, alpha=0, ax=ax)

        if args.generate:
            idx = df_human['generate'] & (df_human[col] == family) & (df_human['is_heavy'] == ('heavy' in col))
            df_ = df_human.loc[idx, cols].dropna()
            ax = df_.plot.scatter(x='tsne_0', y='tsne_1', s=0.25, alpha=0.5, color=c, label=None, ax=ax)

    _format_ax(ax)
    # bbox_to_anchor = (1.15, 0),
    plt.legend(
        loc='lower right' if is_heavy_family else 'lower left',
        bbox_to_anchor=(1.15, 0) if is_heavy_family else (-0.15, 0),
    )
    plt.tight_layout()
    figname = output_dir / f'{col}_family.{args.format}'
    plt.savefig(figname, dpi=300)
    plt.close()

# pairwise frequency in heatmap
df_org = df_human[~df_human['generate']]
freq_org = pairwise_dataframe(df_org[cols_h + cols_l])
freq_org.to_csv(output_dir / 'full_pairwise_org.csv')

for col1, col2 in itertools.combinations(cols_h + cols_l, 2):  # pairwise recombination
    freq_org = pd.crosstab(df_org[col1], df_org[col2])
    index = sorted(freq_org.index)
    columns = sorted(freq_org.columns)

    freq_org_ = freq_org.reindex(index, axis=0).reindex(columns, axis=1)
    freq_org_ /= freq_org_.sum().sum()
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
