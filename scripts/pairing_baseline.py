from pathlib import Path
from typing import Set

from datasets import load_from_disk, Dataset, DatasetDict
import pandas as pd
import ray

from pabt5.util import write_pkl, read_pkl, squeeze
from pabt5.alignment import SequenceSimilarityNetwork
from pabt5.dataset import get_antibody_info, desymmetrize_dataset


def dataset2sequences(dataset: Dataset) -> Set:
    sequences = set(dataset['sequenceA'])
    sequences.update(dataset['sequenceB'])
    return sequences


def dataset2pairs(dataset: Dataset) -> Set:
    pairs = set()
    for row in dataset:
        pair = (row['sequenceA'], row['sequenceB'])
        pairs.add(pair)
    return pairs


dataset_dict = load_from_disk('../training/dataset_sym')
dataset_dict = DatasetDict({
    'val': dataset_dict['val'],
    'test': dataset_dict['test'],
})
dataset_dict = dataset_dict.map(lambda x: {'sequenceA': squeeze(x['sequenceA']), 'sequenceB': squeeze(x['sequenceB'])})
dataset_dict = desymmetrize_dataset(dataset_dict, num_proc=4)

sequences = {k: dataset2sequences(v) for k, v in dataset_dict.items()}

ray.init()

pkl = Path('sequence_similarity_network.pkl')
if pkl.exists():
    sequence_network = read_pkl(pkl)
else:
    sequence_network = SequenceSimilarityNetwork.from_sequences(
        sequences['test'], sequences['val'],
        num_cpus=0.1,
        clean=False,
    )
    write_pkl(pkl, sequence_network)

# pkl = Path('edge_similarity_network.pkl')
# if pkl.exists():
#     network = read_pkl(pkl)
# else:
#     edges = {k: dataset2pairs(v) for k, v in dataset_dict.items()}
#     network = EdgeSimilarityNetwork.from_edges(
#         edges['test'], edges['val'],
#         num_cpus=0.1,
#         clean=False,
#     )
#     write_pkl(pkl, network)


# whole dataset antibody germline annotation
def annotate_sequence_pair(row):
    sequenceA, sequenceB = squeeze(row['sequenceA']), squeeze(row['sequenceB'])
    sequences = [('A', sequenceA), ('B', sequenceB)]
    infoA, infoB = get_antibody_info(sequences, assign_germline=True)
    return {
        'chain_typeA': infoA['chain_type'],
        'chain_typeB': infoB['chain_type'],
        'v_geneA': infoA['v_gene'],
        'j_geneA': infoA['j_gene'],
        'v_geneB': infoB['v_gene'],
        'j_geneB': infoB['j_gene'],
    }


# dataset_dict = dataset_dict.map(
#     annotate_sequence_pair,
#     batched=False,
#     num_proc=4,
# )
# dataset_dict.save_to_disk('dataset_sym_annotated')

