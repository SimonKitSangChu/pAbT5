import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from pabt5.dataset import get_antibody_info


# helper function(s)
def heatmap(figname, attention, query: str = 'query', key: str = 'key', skip_eos: bool = False,
            query_cdr_ids=None, key_cdr_ids=None, overwrite: bool = False):
    if figname.exists() and not overwrite:
        return

    fig, ax = plt.subplots()
    if skip_eos:
        plt.imshow(attention[:-1, :-1], cmap='hot')
    else:
        plt.imshow(attention, cmap='hot')

    if key_cdr_ids:
        ax.set_xticks([i - skip_eos for ids in key_cdr_ids for i in ids])
    if query_cdr_ids:
        ax.set_yticks([i - skip_eos for ids in query_cdr_ids for i in ids])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel(key)
    ax.set_ylabel(query)

    plt.tight_layout(pad=0.3)
    fig.savefig(figname, dpi=300)
    plt.close(fig)


plt.style.use('seaborn-colorblind')  # define color palette

# cmd options
cli = argparse.ArgumentParser()
cli.add_argument('--evaluation_dir', type=str, help='path to evaluation/generation directory')
cli.add_argument('--plot_per_head', default=False, help='plot attention map per layer per head')
cli.add_argument('--plot_per_layer', default=False, help='plot attention map per layer')
cli.add_argument('--output_dir', default='visualization')
cli.add_argument('--format', type=str, default='png', help='output figure format')
cli.add_argument('--overwrite', default=False, action='store_true')
args = cli.parse_args()

src_dir = Path(args.evaluation_dir)
if not src_dir.exists():
    raise FileNotFoundError('evaluation directory not found')

# figure generation
pbar = tqdm(sorted(src_dir.glob('*.pt')))
for pt in pbar:
    data = torch.load(pt)

    for attention_name in ('cross_attentions', 'decoder_attentions', 'encoder_attentions'):
        output_dir = Path(f'{args.output_dir}/{attention_name}')
        output_dir.mkdir(exist_ok=True, parents=True)

        attention = data['label'][attention_name]  # shape = (n_layers, n_heads, n_query, n_key)
        attention = attention.astype(float)  # float16 to full float
        n_layers, n_heads, n_query, n_key = attention.shape

        # find CDR loop ids
        if attention_name == 'cross_attentions':
            key_sequence = data['label']['encoder_sequence'][0]
            query_sequence = data['label']['label_sequence'][0]
            assert len(query_sequence) == n_query - 1
            assert len(key_sequence) == n_key - 1
        elif attention_name == 'decoder_attentions':
            key_sequence = data['label']['label_sequence'][0]
            query_sequence = data['label']['label_sequence'][0]
            assert len(key_sequence) == len(query_sequence) == n_query - 1 == n_key - 1
        elif attention_name == 'encoder_attentions':
            key_sequence = data['label']['encoder_sequence'][0]
            query_sequence = data['label']['encoder_sequence'][0]
            assert len(key_sequence) == len(query_sequence) == n_query - 1 == n_key - 1
        else:
            raise ValueError('only support encoder_attentions, decoder_attentions and cross_attentions')

        key_info, query_info = get_antibody_info(
            [('key', key_sequence), ('query', query_sequence)],
        )

        if args.plot_per_head or args.plot_per_layer:
            output_subdir = output_dir / pt.stem
            output_subdir.mkdir(exist_ok=True, parents=True)
        else:
            output_subdir = None

        plot_kwargs = {
            'key': 'key (heavy)' if key_info['is_heavy'] else 'key (light)',
            'query': 'query (heavy)' if query_info['is_heavy'] else 'query (light)',
            'overwrite': args.overwrite,
        }

        # per layer per head
        for i_layers in range(n_layers):
            for i_heads in range(n_heads):
                pbar.set_postfix_str(f'layer {i_layers} head {i_heads}')
                if args.plot_per_head:
                    heatmap(
                        figname=output_subdir / f'layer{i_layers}_head{i_heads}.{args.format}',
                        attention=attention[i_layers, i_heads],
                        **plot_kwargs
                    )

            # per layer average across head
            if args.plot_per_layer:
                heatmap(
                    figname=output_subdir / f'layer{i_layers}.{args.format}',
                    attention=attention[i_layers].mean(axis=0),
                    **plot_kwargs
                )

        # average across layer and head
        heatmap(
            figname=output_dir / f'{pt.stem}.{args.format}',
            attention=attention.mean(axis=0).mean(axis=0),
            **plot_kwargs
        )
