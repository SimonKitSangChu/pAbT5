import argparse
from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from tokenizers import Tokenizer
from tqdm import tqdm

from pabt5.util import spaceout
from pabt5.metrics import pseudolikelihood_esm
from pabt5.model import T5DecoderForCausalLM
from pabt5.progen.modeling import ProGenForCausalLM


def evaluate_t5(model, tokenizer, sequence_h, sequence_l, return_perplexity=True):
    sequence_h = spaceout(sequence_h.replace(' ', ''))
    sequence_l = spaceout(sequence_l.replace(' ', ''))
    input_ids_h = torch.tensor([tokenizer.encode(sequence_h)], device=model.device)
    input_ids_l = torch.tensor([tokenizer.encode(sequence_l)], device=model.device)

    with torch.no_grad():
        loss_lh = model.forward(input_ids=input_ids_l, labels=input_ids_h).loss.cpu().item()
        loss_hl = model.forward(input_ids=input_ids_h, labels=input_ids_l).loss.cpu().item()
        loss = (loss_lh + loss_hl) / 2

    sr = pd.Series([loss_lh, loss_hl, loss], index=['H', 'L', 'pair'])
    if return_perplexity:
        sr = sr.apply(np.exp)

    return sr


def evaluate_progen2(model, tokenizer, sequence_h, sequence_l, return_perplexity=True, linker='GGGGSGGGGSGGGGS'):
    sequence_h = sequence_h.replace(' ', '')
    sequence_l = sequence_l.replace(' ', '')
    sequence = '1' + sequence_h + linker + sequence_l + '2'

    input_ids_h = torch.tensor([tokenizer.encode(sequence_h).ids], device=model.device)
    input_ids_l = torch.tensor([tokenizer.encode(sequence_l).ids], device=model.device)
    input_ids = torch.tensor([tokenizer.encode(sequence).ids], device=model.device)

    with torch.no_grad():
        loss_h = model.forward(input_ids=input_ids_h, labels=input_ids_h).loss.cpu().item()
        loss_l = model.forward(input_ids=input_ids_l, labels=input_ids_l).loss.cpu().item()
        loss = model.forward(input_ids=input_ids, labels=input_ids).loss.cpu().item()

    sr = pd.Series([loss_h, loss_l, loss], index=['H', 'L', 'pair'])
    if return_perplexity:
        sr = sr.apply(np.exp)

    return sr


def evaluate_esm(model, alphabet, sequence_h, sequence_l, device=None,
                 batch_size=128, return_perplexity=True):
    device = 'cpu' if device is None else device
    sequence_h = sequence_h.replace(' ', '')
    sequence_l = sequence_l.replace(' ', '')

    perplexity_h = pseudolikelihood_esm(
        model=model,
        alphabet=alphabet,
        sequence=sequence_h,
        batch_size=batch_size,
        device=device,
    )
    perplexity_l = pseudolikelihood_esm(
        model=model,
        alphabet=alphabet,
        sequence=sequence_l,
        batch_size=batch_size,
        device=device,
    )
    perplexity = (perplexity_h + perplexity_l) / 2

    sr = pd.Series([perplexity_h, perplexity_l, perplexity], index=['H', 'L', 'pair'])
    if return_perplexity:
        sr = sr.apply(np.exp)

    return sr


def to_numeric(x):
    try:
        return float(x)
    except ValueError:
        return None


def bootstrapped_src(x, y, n_samples=1000):
    x, y = np.array(x), np.array(y)

    src, p_value = [], []
    for _ in range(n_samples):
        idx = np.random.choice(len(x), len(x), replace=True)
        src_, p_value_ = spearmanr(x[idx], y[idx], nan_policy='omit')
        src.append(src_)
        p_value.append(p_value_)

    return np.array(src), np.array(p_value)


# cmd options
cli = argparse.ArgumentParser()
cli.add_argument('--input_csv', type=str, required=True)
cli.add_argument('--checkpoint_dir', type=str, required=True)
cli.add_argument('--output_dir', default='visualization/exp')
cli.add_argument('--use_cuda', default=False, action='store_true')
cli.add_argument('--n_samples', type=int, default=1000, help='number of samples for bootstrapping')
cli.add_argument('--overwrite', default=False, action='store_true')
cli.add_argument('--model_type', default='t5', choices=['t5', 't5-decoder', 'esm', 'progen2'])
cli.add_argument('--tokenizer', default='Rostlab/prot_t5_xl_uniref50')
args = cli.parse_args()

# offload cmd options
csv_in = Path(args.input_csv)

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# prepare model
csv = output_dir / f'{csv_in.stem}_out.csv'
if csv.exists():
    df = pd.read_csv(csv)
else:
    if args.model_type == 'progen2':
        with open(args.tokenizer) as f:
            tokenizer = Tokenizer.from_str(f.read())
        model = ProGenForCausalLM.from_pretrained(args.checkpoint_dir)
        func = lambda x: evaluate_progen2(model, tokenizer, x['sequence_h'], x['sequence_l'])
    elif args.model_type == 'esm':
        model, alphabet = torch.hub.load('facebookresearch/esm:main', args.checkpoint_dir)
        func = lambda x: evaluate_esm(model, alphabet, x['sequence_h'], x['sequence_l'],
                                      device=device, batch_size=128)
    elif args.model_type == 't5':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir)
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, do_lower_case=False)
        func = lambda x: evaluate_t5(model, tokenizer, x['sequence_h'], x['sequence_l'])
    elif args.model_type == 't5-decoder':
        model = T5DecoderForCausalLM.from_pretrained(args.checkpoint_dir)
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer, do_lower_case=False)
        func = lambda x: evaluate_t5(model, tokenizer, x['sequence_h'], x['sequence_l'])
    else:
        raise ValueError('model_type not recognized')

    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    model = model.to(device)

    # evaluate antibody pair(s)
    df = pd.read_csv(csv_in)
    tqdm.pandas(desc='evaluate')
    df[['H', 'L', 'pair']] = df.progress_apply(func, axis='columns')
    df.to_csv(csv, index=None)

# plotting
output_subdir = output_dir / csv_in.stem
output_subdir.mkdir(exist_ok=True, parents=True)

if args.model_type == 'progen2':
    cols_out = {
        'H': 'heavy',
        'L': 'light',
        'pair': 'heavy-GS-light',
    }
else:
    cols_out = {
        'H': 'light-to-heavy',
        'L': 'heavy-to-light',
        'pair': 'bidirectional',
    }

data = []
cols_eval = [col for col in df.columns if col not in cols_out and col not in ('sequence_l', 'sequence_h')]
csv = output_subdir / 'correlations.csv'

for col_y in tqdm(cols_eval, desc='plotting'):
    name_y = col_y.replace(' ', '').replace('/', '')
    df[col_y] = df[col_y].apply(to_numeric)
    df_ = df[list(cols_out) + [col_y]].dropna()
    if len(df_) <= 2:
        warnings.warn(f'{col_y} has too few numerical values thus skipped')
        continue

    # plot
    png = output_subdir / f'{name_y}_scatter.png'
    if not png.exists() or args.overwrite:
        fig, axes = plt.subplots(ncols=len(cols_out), figsize=(6*len(cols_out), 6), tight_layout=True)
        if len(cols_out) == 1:
            axes = [axes]

        for i, (col_x, name_x) in enumerate(cols_out.items()):
            df_.plot.scatter(x=col_x, y=col_y, ax=axes[i])
            axes[i].set_xlabel(name_x, fontsize=16)
            if i != 0:
                axes[i].set_ylabel('')
            else:
                axes[i].set_ylabel(col_y, fontsize=16)

        fig.savefig(png, dpi=300)
        plt.close(fig)

    # rank-rank plot
    png = output_subdir / f'{name_y}_scatter_rank.png'
    if not png.exists() or args.overwrite:
        fig, axes = plt.subplots(ncols=len(cols_out), figsize=(6*len(cols_out), 6), tight_layout=True)
        if len(cols_out) == 1:
            axes = [axes]

        for i, (col_x, name_x) in enumerate(cols_out.items()):
            axes[i].scatter(x=df_[col_x].rank(), y=df_[col_y].rank())
            axes[i].set_xlabel(f'{name_x} (rank)', fontsize=16)
            if i != 0:
                axes[i].set_ylabel('')
            else:
                axes[i].set_ylabel(f'{col_y} (rank)', fontsize=16)

        fig.savefig(png, dpi=300)
        plt.close(fig)

    # jointplot
    for col_x, name_x in cols_out.items():
        png = output_subdir / f'jg_{name_y}_{name_x}.png'
        if not png.exists() or args.overwrite:
            jg = sns.jointplot(data=df_, x=col_x, y=col_y, kind='scatter')
            jg.set_axis_labels(name_x, col_y)
            jg.figure.savefig(png, dpi=300)
            plt.close()

    # statistics
    if not csv.exists():
        datum = {}
        for col_x in cols_out:
            src, p_value = spearmanr(df[col_x], df[col_y], nan_policy='omit')
            datum[f'src_{col_x}'] = src
            datum[f'p_value_{col_x}'] = p_value

            src, _ = bootstrapped_src(df[col_x], df[col_y], n_samples=args.n_samples)
            datum[f'src_mean_{col_x}'] = np.mean(src)
            datum[f'src_std_{col_x}'] = np.std(src)

        datum['column'] = col_y
        data.append(datum)

if not csv.exists():
    df_sta = pd.DataFrame.from_dict(data)
    df_sta.to_csv(csv, index=None)
