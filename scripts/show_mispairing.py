import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

from pabt5.util import spaceout
from pabt5.dataset import get_antibody_info, desymmetrize_dataset
from pabt5.alignment import is_heavyA


# helper function(s)
def get_species(sequence):
    return get_antibody_info([('tmp', sequence.replace(' ', ''))])[0]['species']


def tokenize_pair(sequence_vh, sequence_vl, device):
    vh_tokens = tokenizer.encode(spaceout(sequence_vh))
    vl_tokens = tokenizer.encode(spaceout(sequence_vl))
    vh_tokens = torch.tensor([vh_tokens], device=device)
    vl_tokens = torch.tensor([vl_tokens], device=device)
    return vh_tokens, vl_tokens


def get_perplexity(model, input_ids, labels):
    with torch.no_grad():
        output = model.forward(input_ids=input_ids, labels=labels)
        return float(torch.exp(output.loss).cpu().item())


class ZeroClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict_single(self, x):
        assert len(x) == 1
        return x[0] > 0

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def predict_proba(self, X):
        probs = []
        for x in X:
            prob = float(self.predict_single(x))
            probs.append([prob, 1 - prob])
        return np.array(probs)


def get_metrics(df, col_true='hetero', n_resample=100):
    model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000)
    df_val = df[df['split'] == 'val']
    X_val = df_val['perplexity'].values.reshape(-1, 1)
    y_val = df_val['pairing'].str.startswith(col_true).astype(int).values
    model.fit(X_val, y_val)

    df_test = df[df['split'] == 'test']
    X_test = df_test['perplexity'].values.reshape(-1, 1)
    y_test = df_test['pairing'].str.contains(col_true).astype(int).values

    acccuracies = []
    aurocs = []

    for _ in range(n_resample):
        X_test_resampled, y_test_resampled = resample(X_test, y_test)
        y_pred = model.predict(X_test_resampled)
        y_pred_proba = model.predict_proba(X_test_resampled)[:, 1]
        acccuracies.append(accuracy_score(y_test_resampled, y_pred))
        aurocs.append(roc_auc_score(y_test_resampled, y_pred_proba))

    accuracies = np.array(acccuracies)
    aurocs = np.array(aurocs)

    return {
        'accuracy': np.mean(accuracies),
        'auroc': np.mean(aurocs),
        'accuracy_std': np.std(accuracies),
        'auroc_std': np.std(aurocs),
    }


sns.set_theme(palette='colorblind')

# cmd options
cli = argparse.ArgumentParser()
cli.add_argument('--output_dir', default='visualization/mispairing')
cli.add_argument('--checkpoint_dir', type=str)
cli.add_argument('--dataset_dir', type=str)
cli.add_argument('--hard_mode', action='store_true', default=False,
                 help='shuffle all encoder binding partners')
cli.add_argument('--format', default='png', help='output figure format')
args = cli.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# evaluate mispairing
csv = output_dir / 'mispairing.csv'
if csv.exists():
    df_eval = pd.read_csv(csv)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir)
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataset_path = args.dataset_dir
    dataset_dict = load_from_disk(dataset_path)

    df_eval = []
    for split in ('val', 'test'):
        dataset = desymmetrize_dataset(dataset_dict[split])
        df = dataset.to_pandas()
        df = df.rename(columns={
            '__index_level_0__': 'identifier',
            'sequenceA': 'sequence_vh',
            'sequenceB': 'sequence_vl'}
        )
        df = df[['sequence_vh', 'sequence_vl', 'identifier']]

        df['species_h'] = df['sequence_vh'].apply(get_species)
        df['species_l'] = df['sequence_vl'].apply(get_species)
        df['match_species'] = df['species_h'] == df['species_l']
        species = set(df['species_h'].tolist())

        for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
            df_spec = df[df['species_h'] != row['species_h']]  # random cross-species partner
            row_spec = df_spec.sample(frac=1).iloc[0]
            vh_tokens_cs, vl_tokens_cs = tokenize_pair(
                row_spec['sequence_vh'], row_spec['sequence_vl'], device)

            if args.hard_mode:
                row_chain = df.sample(frac=1).iloc[0]  # random homo partner
                vh_tokens_ho, vl_tokens_ho = tokenize_pair(
                    row_chain['sequence_vh'], row_chain['sequence_vl'], device)

                row_chain = df.sample(frac=1).iloc[0]  # random hetero partner
                vh_tokens_he, vl_tokens_he = tokenize_pair(
                    row_chain['sequence_vh'], row_chain['sequence_vl'], device)

                df_spec = df[df['species_h'] == row['species_h']]  # random in-species partner
                row_spec = df_spec.sample(frac=1).iloc[0]
                vh_tokens_is, vl_tokens_is = tokenize_pair(
                    row_spec['sequence_vh'], row_spec['sequence_vl'], device)
            else:
                vh_tokens, vl_tokens = tokenize_pair(
                    row['sequence_vh'], row['sequence_vl'], device)
                vl_tokens_ho, vh_tokens_ho = vl_tokens, vh_tokens  # keep original homo partner
                vh_tokens_he, vl_tokens_he = vh_tokens, vl_tokens  # keep original hetero partner
                vh_tokens_is, vl_tokens_is = vh_tokens, vl_tokens  # keep original in-species partner

            # evaluate cross-heavy/light chain mispairing
            vh_tokens, vl_tokens = tokenize_pair(row['sequence_vh'], row['sequence_vl'], device)
            perplexity = {
                'split': split,
                'cross-L': get_perplexity(model, vh_tokens_cs, vl_tokens) if row['match_species'] else None,
                'cross-H': get_perplexity(model, vl_tokens_cs, vh_tokens) if row['match_species'] else None,
                'in-L': get_perplexity(model, vh_tokens_is, vl_tokens) if row['match_species'] else None,
                'in-H': get_perplexity(model, vl_tokens_is, vh_tokens) if row['match_species'] else None,
                'homo-L': get_perplexity(model, vl_tokens_ho, vl_tokens),
                'homo-H': get_perplexity(model, vh_tokens_ho, vh_tokens),
                'hetero-L': get_perplexity(model, vh_tokens_he, vl_tokens),
                'hetero-H': get_perplexity(model, vl_tokens_he, vh_tokens),
            }
            df_eval.append(perplexity)

    df_eval = pd.DataFrame(df_eval)
    df_eval.to_csv(csv, index=None)

# delta mispairing
df = df_eval.copy()
df['homo-L'] -= df['hetero-L']
df['homo-H'] -= df['hetero-H']
df['cross-L'] -= df['in-L']
df['cross-H'] -= df['in-H']
cols = ['homo-L', 'homo-H', 'cross-L', 'cross-H']

df = df.melt(value_vars=cols, var_name='pairing', value_name='perplexity', id_vars='split')
df['mispairing type'] = df['pairing'].apply(lambda x: 'species' if 'cross' in x else 'chain type')
df['target chain'] = df['pairing'].apply(lambda x: x[-1])
df = df.dropna()

ax = sns.violinplot(
    data=df,
    x='target chain',
    y='perplexity',
    hue='mispairing type',
    hue_order=('chain type', 'species'),
    scale='area',
    scale_hue=False,
    cut=True,
)
ax.set_xlabel('generation target')
ax.set_ylabel('relative mispairing perplexity')
ax.set_xticklabels(('light chain', 'heavy chain'))
handles, labels = ax.get_legend_handles_labels()
ax.legend(framealpha=0.95, handles=handles, labels=['by ' + l for l in labels])

plt.tight_layout(pad=0.3)
plt.savefig(output_dir / f'delta.{args.format}', dpi=300)
plt.close()

df_metrics = []
for (mispairing_type, target_chain), df_ in df.groupby(['mispairing type', 'target chain']):
    # skip training and jump to evaluation
    df_ = df_[df_['split'] == 'test']
    X = df_['perplexity'].values.reshape(-1, 1)
    model = ZeroClassifier()

    accuracies = []
    for _ in range(100):
        X_resampled = resample(X, replace=True)
        y_pred = model.predict(X)
        y_true = np.ones_like(y_pred)
        accuracies.append(accuracy_score(y_true, y_pred))

    accuracies = np.array(accuracies)

    df_metrics.append({
        'mispairing type': mispairing_type,
        'target chain': target_chain,
        'count': len(df_),
        'accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'auroc': None,  # only-one-class does not work in AUROC
        'auroc_std': None,
    })

df_metrics = pd.DataFrame(df_metrics)
df_metrics.to_csv(output_dir / 'delta.csv', index=None)

# (mixing) heavy-light chain mispairing
df = df_eval.copy()
df['homo'] = (df['homo-L'] + df['homo-H']) / 2
df['hetero'] = (df['hetero-L'] + df['hetero-H']) / 2

df = df.melt(value_vars=['hetero', 'homo'], var_name='pairing', value_name='perplexity', id_vars='split')
df['pairing type'] = df['pairing'].apply(lambda x: x.split('-')[0])
df = df.dropna()

ax = sns.violinplot(
    data=df,
    x='pairing type',
    y='perplexity',
    scale='area',
    scale_hue=False,
    inner='quartile',
    cut=True,
)
ax.set_ylim(1, None)
ax.set_xlabel(None)
ax.set_ylabel('perplexity', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, fontsize=16)

plt.tight_layout(pad=0.3)
plt.savefig(output_dir / f'cross_chain.{args.format}', dpi=300)
plt.close()

df_metrics = [{
    'mispairing type': 'chain type',
    **get_metrics(df),
}]

# (mixing) cross species
df = df_eval.copy()
df['cross-species'] = (df['cross-L'] + df['cross-H']) / 2
df['in-species'] = (df['hetero-L'] + df['hetero-H']) / 2

df = df.melt(value_vars=['in-species', 'cross-species'], var_name='pairing', value_name='perplexity', id_vars='split')
df['pairing type'] = df['pairing'].apply(lambda x: 'cross-species' if 'cross' in x else 'in-species')
df = df.dropna()

ax = sns.violinplot(
    data=df,
    x='pairing type',
    y='perplexity',
    scale_hue=False,
    inner='quartile',
    cut=True,
)
ax.set_ylim(1, None)
ax.set_xlabel(None)
ax.set_ylabel('perplexity', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, fontsize=16)

plt.tight_layout(pad=0.3)
plt.savefig(output_dir / f'cross_species.{args.format}', dpi=300)
plt.close()

df_metrics.append({
    'mispairing type': 'species',
    **get_metrics(df, col_true='in'),
})
df_metrics = pd.DataFrame(df_metrics)
df_metrics.to_csv(output_dir / 'shuffled.csv', index=None)
