import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
import torch

from pabt5.dataset import get_antibody_info, dataset2sequences

cli = argparse.ArgumentParser()
cli.add_argument('--output_dir', type=str, default='visualization/ambiguity')
cli.add_argument('--oas_dataset_dir', type=str, default=None)
cli.add_argument('--dataset_split', type=str, default='test')
cli.add_argument('--format', type=str, default='png', help='output figure format')
args = cli.parse_args()


def get_info(sequence: str, assign_germline=True):
    info = get_antibody_info([('', sequence)], assign_germline=assign_germline)[0]
    return info['chain_type'], info['species'], info['is_heavy'], info['v_gene'], info['j_gene']


def main():
    output_dir = Path(args.output_dir)
    csv_raw = output_dir / 'raw.csv'

    if csv_raw.exists():
        df = pd.read_csv(csv_raw)
    else:
        dataset = load_from_disk(args.oas_dataset_dir)[args.dataset_split]

        # compute metadata for each sequence
        pt_metadata = output_dir / f'metadata_{args.dataset_split}.pt'
        if pt_metadata.exists():
            metadata = torch.load(pt_metadata)
        else:
            sequences = dataset2sequences(dataset)
            metadata = {}
            for sequence in tqdm(sequences, desc='get metadata'):
                metadata[sequence] = get_info(sequence)

            torch.save(metadata, pt_metadata)

        # assign metadata to each sequence pair
        df = dataset.to_pandas()
        for col_name in ('chain_type', 'species', 'is_heavy', 'v_gene', 'j_gene'):
            df[f'{col_name}_A'] = df['sequenceA'].map(lambda x: metadata[x][col_name])
            df[f'{col_name}_B'] = df['sequenceB'].map(lambda x: metadata[x][col_name])

        # df['vj_gene_A'] = df['v_gene_A'] + df['j_gene_A']
        # df['vj_gene_B'] = df['v_gene_B'] + df['j_gene_B']
        df.to_csv(csv_raw, index=False)

    # co-ocurrence analysis TODO sunburst plot?
    cols_res = ['species', 'is_heavy', 'v_gene', 'j_gene', 'vj_gene']
    cols_res_A = [f'{col}_A' for col in cols_res]
    cols_res_B = [f'{col}_B' for col in cols_res]

    # TODO plotting
    #  x-axis occurrence in test set, y-axis nunique choices/entropy
    #  category dependent ambiguity (number of unique sequences?)


if __name__ == '__main__':
    main()
