import argparse
from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logomaker
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import softmax
import torch
from transformers import T5Tokenizer
from tqdm import tqdm

from pabt5.util import read_json, sequences2records
from pabt5.alignment import (
    psiblast_pssm,
    clustalw_msa,
    msa_matrix,
    AA_LETTERS,
)

# cmd options
cli = argparse.ArgumentParser()
cli.add_argument('--blastdb', type=str, help='path to blastp database')
cli.add_argument('--evaluation_dir', type=str, help='path to evaluation/generation directory')
cli.add_argument('--generate_methods', nargs='+', type=str)
cli.add_argument('--output_dir', type=str, default='visualization/alignment', help='path to output directory')
cli.add_argument('--dump_generate', default=True, action='store_true',
    help='dump figure of aligned generated sequences')
cli.add_argument('--no-dump_generate', dest='dump_generate', action='store_false')
cli.add_argument('--dump_logits', default=True, action='store_true',
    help='dump figure of logits in comparison with PSSM')
cli.add_argument('--no-dump_logits', dest='dump_logits', action='store_false')
cli.add_argument('--format', type=str, default='png', help='output figure format')
cli.add_argument('--overwrite', default=False, action='store_true')
args = cli.parse_args()

# offload cmd options
blastdb = Path(args.blastdb)

evaluation_dir = Path(args.evaluation_dir)
if not evaluation_dir.exists():
    raise FileNotFoundError('evaluation directory not found')

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# load amino acid indices from tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
vocabs = tokenizer.get_vocab()
vocabs_aa = {k: v for k, v in vocabs.items() if k[-1] in AA_LETTERS}
vocabs_aa = {k[-1]: vocabs_aa[k] for k in sorted(vocabs_aa.keys())}
idx_aa = list(vocabs_aa.values())

# figure generation
js_list = sorted(evaluation_dir.glob('*.json'))
for js in tqdm(js_list, desc='analysis: sequence alignment'):
    if 'evaluate' in js.stem:  # evaluate.json and evaluate_average.json
        continue

    datum = read_json(js)
    label_sequence = datum['label_sequence']
    hash_ = datum['hash']
    record_label = SeqRecord(
        seq=Seq(datum['label_sequence']),
        id='label',
        name='',
        description='',
    )

    # figure 1. alignment within label and generate
    if args.dump_generate:
        for generate_method in datum['generate']:
            if generate_method not in args.generate_methods:
                continue

            output_subdir = output_dir / generate_method
            output_subdir.mkdir(exist_ok=True, parents=True)
            figname = output_subdir / f'fig1_{hash_}.{args.format}'
            if figname.exists() and not args.overwrite:
                continue

            sequences = datum['generate'][generate_method]['generate_sequence']
            records = sequences2records(sequences)
            records = [record_label] + records

            alignment = clustalw_msa(records, head_id='label')
            df_label = msa_matrix(alignment, normalize=False, strategy='first', drop_gap=True)
            df_generate = msa_matrix(alignment, normalize=True, strategy='not first', drop_gap=True)

            logo = logomaker.Logo(
                df_generate, 
                color_scheme='chemistry',
                stack_order='small_on_top',
                figsize=(0.5 * df_generate.shape[0], 2),
            )
            logomaker.Logo(
                -df_label,
                color_scheme='chemistry',
                ax=logo.ax,
                flip_below=False,
                alpha=0.5,
            )

            logo.ax.set_ylim(-1, 1)
            logo.ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(figname, dpi=300)
            plt.close()

    # figure 2. label pssm v.s. logits
    #  independent of decoder_start_ids and generate_method
    figname = output_dir / f'fig2_{hash_}.{args.format}'
    if args.dump_logits:
        if figname.exists() and not args.overwrite:
            continue

        df_pssm = psiblast_pssm(record_label, db=blastdb)
        df_pssm = pd.DataFrame(softmax(df_pssm.values, axis=-1), columns=df_pssm.columns)

        logits = torch.load(evaluation_dir / f'{hash_}.pt')['label']['logits'].squeeze(0)
        assert logits.ndim == 2
        logits = logits[:-1, idx_aa]
        prob = softmax(logits, axis=-1)
        df_prob = pd.DataFrame(prob)
        df_prob.columns = AA_LETTERS

        logo = logomaker.Logo(
            df_prob,
            color_scheme='chemistry',
            stack_order='small_on_top',
            figsize=(0.5 * df_prob.shape[0], 2),
        )
        logomaker.Logo(
            -df_pssm,
            color_scheme='chemistry',
            stack_order='small_on_top',
            ax=logo.ax,
            flip_below=False,
            alpha=0.5,
        )

        logo.ax.set_ylim(-1, 1)
        logo.ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(figname, dpi=300)
        plt.close()

