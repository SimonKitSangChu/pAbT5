import io
import itertools
import json
import logging
from pathlib import Path
import pickle
import time
from typing import Union, List, Optional, Dict, Any, Tuple, Set, Iterable
import warnings

from anarci import anarci
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .util import pair_func, sequences2records, spaceout, squeeze


def is_reflection(example: Dict[str, Any]) -> bool:
    return example['sequenceA'] == example['sequenceB']


def not_reflection(example) -> bool:
    return not is_reflection(example)


def revertAB(example) -> Dict[str, Any]:
    return {'sequenceA': example['sequenceB'], 'sequenceB': example['sequenceA']}


def filter_by_length(example, min_length: int = 8, max_length: int = 2048) -> bool:
    lenA = len(example['sequenceA'])
    lenB = len(example['sequenceB'])
    return (min_length <= lenA <= max_length) and (min_length <= lenB <= max_length)


def dataset2sequences(dataset: Dataset, use_squeeze: bool = True) -> Set[str]:
    sequences_A = set(dataset['sequenceA'])
    sequences_B = set(dataset['sequenceB'])
    sequences = sequences_A.union(sequences_B)

    if use_squeeze:
        sequences = set(squeeze(s) for s in sequences)

    return sequences


def preprocess_function(tokenizer, example: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    if ' ' not in example['sequenceA']:
        example['sequenceA'] = spaceout(example['sequenceA'])
        example['sequenceB'] = spaceout(example['sequenceB'])

    try:
        inputs = {'input_ids': tokenizer.encode(example['sequenceA'], **kwargs)}
        with tokenizer.as_target_tokenizer():
            inputs['labels'] = tokenizer.encode(example['sequenceB'], **kwargs)

    except AttributeError:
        inputs = {
            'input_ids': tokenizer.encode(example['sequenceA'], **kwargs).ids,
            'labels': tokenizer.encode(example['sequenceB'], **kwargs).ids,
        }

    return inputs


def check_dataset_pair_leakage(dataset: DatasetDict) -> bool:
    pairs = {}
    for k, subdataset in dataset.items():
        sequencesA = subdataset['sequenceA']
        sequencesB = subdataset['sequenceB']
        pairs[k] = set(pair_func([a, b]) for a, b in zip(sequencesA, sequencesB))

    for k1, k2 in itertools.combinations(dataset.keys(), 2):
        if any(p in pairs[k1] for p in pairs[k2]):
            return False

    return True


def check_dataset_sequence_leakage(dataset: DatasetDict) -> bool:
    sequences_train = set(dataset['train']['sequenceA'])
    sequences_train = sequences_train.union(
        set(dataset['train']['sequenceB'])
    )

    for example in dataset['test']:
        if example['sequenceA'] in sequences_train or example['sequenceB'] in sequences_train:
            return False

    return True


def load_custom_dataset(tokenizer, n_samples: int = 1024, max_length: int = 1024, test_size: float = 0.5) -> Dataset:
    if n_samples is not None and n_samples < 2:
        raise ValueError(f'requires at least 2 samples but given {n_samples}.')

    amp_factor = n_samples // 2
    sequencesA = [
                     'A' * max_length,
                     'G' * max_length,
                 ] * amp_factor
    sequencesB = [
                     'A' * max_length,
                     'T' * max_length,
                 ] * amp_factor
    scores = [
                 0.5,
                 0.8,
             ] * amp_factor

    data = {
        'id': list(range(len(sequencesA))),
        'sequenceA': sequencesA,
        'sequenceB': sequencesB,
        'score': scores,
    }
    dataset = Dataset.from_dict(data)
    dataset = dataset.map(
        lambda x: preprocess_function(tokenizer, x),
        batched=False,
        num_proc=torch.get_num_threads(),
        desc='tokenize dataset'
    )

    return dataset


def load_oas_dataframe(
        oas_dir: Union[str, Path] = 'data/oas',
        sequenceA: str = 'sequence_alignment_aa_heavy',
        sequenceB: str = 'sequence_alignment_aa_light',
) -> pd.DataFrame:
    df = []
    for csv in sorted(Path(oas_dir).glob('*.csv')):
        df_ = pd.read_csv(csv, skiprows=1)
        header_dict = read_oas_header(csv)
        df_['species'] = header_dict['Species']
        df.append(df_)

    df = pd.concat(df).copy()
    assert sequenceA != sequenceB, 'same column name for sequenceA and sequenceB'
    df['sequenceA'] = df[sequenceA]
    df['sequenceB'] = df[sequenceB]
    return df


def load_oas_dataset(
        oas_dir: Union[str, Path] = 'data/oas',
        sequenceA: str = 'sequence_alignment_aa_heavy',
        sequenceB: str = 'sequence_alignment_aa_light',
        **kwargs,
) -> DatasetDict:
    df = load_oas_dataframe(oas_dir, sequenceA, sequenceB)
    df = df[['sequenceA', 'sequenceB']]
    return create_dataset(df=df, **kwargs)


def check_pair_duplicate(df: pd.DataFrame) -> bool:
    sr = df.apply(lambda x: pair_func([x['sequenceA'], x['sequenceB']]), axis=1)
    sr = sr.duplicated()
    return sr


def filtering_dataset(dataset: Dataset, filtering: Optional[Dict[str, Any]] = None) \
        -> Tuple[Dataset, Dict[str, Any]]:
    filter_criteria = {
        'homomeric_only': None,
        'heteromeric_only': None,
        'min_score': 0,
        'max_length': 1e9,
        'fasta': None,
    }
    if filtering is not None:
        filter_criteria.update(filtering)

    if filter_criteria['fasta'] is None:
        sequences = None
    else:
        sequences = [str(record.seq) for record in SeqIO.parse(filter_criteria['fasta'], 'fasta')]

    for key in filter_criteria.keys():  # warn unused options
        if key not in ('homomeric_only', 'min_score', 'max_length', 'heterometric_only'):
            logging.warning(f'{key} not supported in filtering thus omitted')

    def filtering_fxn(example: Dict[str, Any]) -> bool:
        if filter_criteria['homomeric_only'] and example['sequenceA'] != example['sequenceB']:
            return False
        if 'score' in example:
            if example['score'] < filter_criteria['min_score']:
                return False
        if filter_criteria['max_length'] and \
                (len(example['sequenceA'].replace(' ', '')) > filter_criteria['max_length'] or \
                 len(example['sequenceB'].replace(' ', '')) > filter_criteria['max_length']):
            return False
        if filter_criteria['heteromeric_only'] and (example['sequenceA'] == example['sequenceB']):
            return False
        if sequences is not None:
            if squeeze(example['sequenceA']) not in sequences or \
                    squeeze(example['sequenceB']) not in sequences:
                return False
        return True

    dataset = dataset.filter(filtering_fxn, desc='filtering')
    return dataset, filter_criteria


def pairwise_dataset_split(dataset: Dataset, test_size: Optional[float] = None,
                           split: Union[Tuple[float, float, float], Tuple[float, float], None] = None) -> DatasetDict:
    if test_size is not None and split is not None:
        raise ValueError('cannot specify both test_size and split at the same time')

    if test_size:
        split = (1 - test_size, test_size)
    elif split is None:
        split = (0.9, 0.05, 0.05)

    sequences_ = set(dataset['sequenceA'])
    sequences_ = sequences_.union(set(dataset['sequenceB']))
    sequences_ = np.array(sorted(sequences_))
    n_samples = len(sequences_)

    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(np.arange(n_samples))
    n_samples_train = int(split[0] * n_samples)

    if len(split) == 2:
        sequences = {
            'train': sequences_[idx[:n_samples_train]],
            'test': sequences_[idx[n_samples_train:]],
        }
    elif len(split) == 3:
        n_samples_val = int(split[1] * n_samples)
        sequences = {
            'train': sequences_[idx[:n_samples_train]],
            'val': sequences_[idx[n_samples_train:n_samples_train + n_samples_val]],
            'test': sequences_[idx[n_samples_train + n_samples_val:]],
        }
    else:
        raise ValueError('only accept tuple of 2 or 3 values for split')

    sequences = {k: set(v) for k, v in sequences.items()}
    return stratify_by_sequences(dataset=dataset, sequences=sequences)


def cluster_dataset_split(input_dir: Union[str, Path], dataset: Dataset, test_size: Optional[float] = None,
                          split: Union[Tuple[float, float, float], Tuple[float, float], None] = None,
                          max_identity: float = 0.8) -> DatasetDict:
    if input_dir is None:
        raise ValueError('cluster split expects input_dir argument')

    input_dir = Path(input_dir)
    fasta_cluster = input_dir / 'clusterRes_all_seqs.fasta'
    fasta_sequences = input_dir / 'sequences.fasta'

    if not fasta_cluster.exists() or not fasta_sequences.exists():
        raise FileNotFoundError('to use cluster_dataset_split, user must provide sequences.fasta'
                                'in input_dir.')

    pkl = input_dir / 'identities.pkl'
    if pkl.exists():
        identities = pickle.load(pkl.open('rb'))
    else:
        m8_list = input_dir.glob('*.m8*')
        m8_list = [x for x in m8_list if '.dbtype' not in x.name and '.index' not in x.name]
        m8_list = sorted(m8_list)
        if not m8_list:
            raise FileNotFoundError('no .m8* found under input_dir')

        identities = {}
        for m8 in tqdm(m8_list, desc='load m8'):
            identities.update(identities_from_lines(m8))

        with pkl.open('wb') as f:
            pickle.dump(identities, f)

    print(f'{len(identities)} pairwise identities loaded')

    if test_size is not None and split is not None:
        raise ValueError('cannot specify both test_size and split at the same time')

    if test_size:
        split = (1 - test_size, test_size)
    elif split is None:
        split = (0.9, 0.05, 0.05)

    clusters = {}
    id_clusters = {}

    for record in SeqIO.parse(fasta_cluster, 'fasta'):
        if not record.seq:
            k = record.id
        elif k not in clusters:
            clusters[k] = [record]
            id_clusters[k] = {record.id}
        else:
            clusters[k].append(record)
            id_clusters[k].add(record.id)
    counts = {k: len(v) for k, v in clusters.items()}

    # check sequence leakage across clusters
    # combinations = list(itertools.combinations(id_clusters.keys(), r=2))
    # for k1, k2 in tqdm(combinations, desc='check duplicate across clusters'):
    #    assert not any(id_ in id_clusters[k1] for id_ in id_clusters[k2]), \
    #            f'{k1} and {k2} clusters share some sequence id'

    records = SeqIO.to_dict(SeqIO.parse(fasta_sequences, 'fasta'))
    assert len(records) == sum(counts.values())

    rng = np.random.default_rng(seed=42)
    keys = np.array(sorted(clusters.keys()))
    keys = rng.permutation(keys)

    def _get_i(counts: Dict[str, int], keys: np.array, min_samples: int, i: int = 0) -> int:
        n_samples_ = 0
        while n_samples_ < min_samples:
            n_samples_ += counts[keys[i]]
            i += 1
        return i

    def _get_sequences(clusters: Dict[str, SeqRecord], keys: np.array) -> Set:
        sequences = set()
        for k in keys:
            subset = set(str(record.seq) for record in clusters[k])
            sequences = sequences.union(subset)
        return sequences

    def _get_records(clusters: Dict[str, SeqRecord], keys: np.array) -> List[SeqRecord]:
        records_ = [record for k in keys for record in clusters[k]]
        return records_

    def _get_ids(id_clusters: Dict[str, str], keys: np.array) -> Set:
        ids = set()
        for k in keys:
            ids_ = id_clusters[k]
            ids = ids.union(ids_)
        return ids

    n_sequences = len(records)
    if len(split) == 2:
        i = _get_i(counts=counts, keys=keys, min_samples=int(split[0] * n_sequences))
        keys_train = keys[:i]
        keys_test = keys[i:]
        keys = {'train': keys_train, 'val': keys_test}
    elif len(split) == 3:
        i = _get_i(counts=counts, keys=keys, min_samples=int(split[0] * n_sequences))
        keys_train = keys[:i]

        j = _get_i(counts=counts, keys=keys, min_samples=int(split[1] * n_sequences), i=i)
        keys_val = keys[i:j]
        keys_test = keys[j:]

        keys = {'train': keys_train, 'val': keys_val, 'test': keys_test}
    else:
        raise ValueError('split length must be either 2 or 3')

    sequences = {k: _get_sequences(clusters, v) for k, v in keys.items()}
    ids = {k: _get_ids(id_clusters, v) for k, v in keys.items()}
    assert sum(len(v) for v in sequences.values()) == len(records)

    for k1, k2 in itertools.combinations(keys, r=2):
        # check cluster key leakage
        assert not any(k in keys[k1] for k in keys[k2]), \
            f'cluster leakage between {k1} and {k2} for {k}'
        # check cluster id leakage
        assert not any(id_ in ids[k1] for id_ in ids[k2]), \
            f'cluster id leakage between {k1} and {k2}'

    for k1, k2 in itertools.combinations(sorted(sequences.keys()), r=2):
        # check (extact) sequence leakage
        sequences1 = sequences[k1]
        sequences2 = sequences[k2]
        count12 = sum(s in sequences2 for s in sequences1)
        count21 = sum(s in sequences1 for s in sequences2)
        if count12 or count21:
            raise AssertionError(f'leakages {k1}_{k2} = {count12}\t{k2}_{k1} = {count21}')

        # check sequence leakage by max_identity
        ids1, ids2 = sorted(ids[k1]), sorted(ids[k2])
        for id1, id2 in tqdm(
                itertools.product(ids1, ids2),
                total=len(ids1) * len(ids2),
                desc=f'check sequence leakage by max_identity {k1}-{k2}',
        ):
            identity = identities.get(pair_func((id1, id2)), 0)

            if identity > max_identity:
                logging.warning(f'identify sequence leakage by max_identity {max_identity}. dropping.')
                ids[k1].discard(id1)
                ids[k1].discard(id2)
                ids[k2].discard(id1)
                ids[k2].discard(id2)

                sequence1 = str(records[id1].seq)
                sequence2 = str(records[id2].seq)
                sequences[k1].discard(sequence1)
                sequences[k1].discard(sequence2)
                sequences[k2].discard(sequence1)
                sequences[k2].discard(sequence2)

    for k, sequences_ in keys.items():
        fasta = Path(input_dir) / f'sequences_{k}.fasta'
        records_ = sequences2records(sequences_)
        SeqIO.write(records_, fasta, 'fasta')

    counts = {k: len(v) for k, v in sequences.items()}
    print(f'total sequences count after sequence leakage drop = {counts}: {sum(counts.values())}|{len(records)}')

    return stratify_by_sequences(dataset=dataset, sequences=sequences)


def stratify_by_sequences(dataset: Dataset, sequences: Dict[str, set]) -> DatasetDict:
    def _in_subset(x: Dict[str, Any], keyA: str, keyB: str) -> bool:
        matchA = x['sequenceA'] in sequences[keyA]
        matchA_not = all(x['sequenceA'] not in subsequences \
                         for k, subsequences in sequences.items() if k != keyA)
        matchB = x['sequenceB'] in sequences[keyB]
        matchB_not = all(x['sequenceB'] not in subsequences \
                         for k, subsequences in sequences.items() if k != keyB)
        return matchA and matchA_not and matchB and matchB_not

    dataset_dict = {}
    for k1, k2 in itertools.product(sequences.keys(), repeat=2):
        k = k1 if k1 == k2 else f'{k1}_{k2}'
        subdataset = dataset.filter(function=lambda x: _in_subset(x, k1, k2), desc=f'filter {k}')
        dataset_dict[k] = subdataset

    dataset_dict = DatasetDict(**dataset_dict)
    return dataset_dict


def symmetrize_dataset(dataset: DatasetDict, mark_symmetrized: bool = False) -> DatasetDict:
    for key in sorted(dataset.keys()):
        subdataset = dataset[key]

        subdatasetAA = subdataset.filter(function=is_reflection, desc=f'A->A ({key})')  # symmetrize
        subdatasetAB = subdataset.filter(function=not_reflection, desc=f'A->B ({key})')
        subdatasetBA = subdataset.map(function=revertAB, desc=f'A->B <<>> B->A ({key})')

        if mark_symmetrized:
            subdatasetAA = subdatasetAA.map(lambda x: {'symmetrized': False})  # mark symmetrization
            subdatasetAB = subdatasetAB.map(lambda x: {'symmetrized': False})
            subdatasetBA = subdatasetBA.map(lambda x: {'symmetrized': True})

        dataset[key] = concatenate_datasets([subdatasetAA, subdatasetAB, subdatasetBA])  # concat dataset

    return dataset


def create_dataset(
        df: pd.DataFrame,
        tokenizer,
        dataset_dir: Union[str, Path] = 'data/dataset',
        n_samples: Optional[int] = None,
        test_size: Optional[float] = None,
        filtering: Optional[Dict[str, Any]] = None,
        split_method: str = 'random',
        symmetrize: bool = True,
        split: Optional[Tuple] = None,
        input_dir: Optional[str] = None,
        **kwargs
) -> DatasetDict:
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        dataset = load_from_disk(dataset_dir)
    else:
        sr = check_pair_duplicate(df)
        if sr.any():
            logging.warning('duplicate interaction(s) found; initialize de-duplication')
            df = df[~sr]
            assert not check_pair_duplicate(df).any()

        # construct dataset
        dataset = Dataset.from_pandas(df)
        dataset, filter_criteria = filtering_dataset(dataset=dataset, filtering=filtering)
        n_cores = torch.get_num_threads()

        if split_method == 'random':
            if test_size is None:
                test_size = 0.05
            dataset = dataset.train_test_split(test_size=test_size)
        elif split_method == 'pairwise':
            dataset = pairwise_dataset_split(dataset=dataset, test_size=test_size, split=split)
        elif split_method == 'cluster':
            dataset = cluster_dataset_split(dataset=dataset, test_size=test_size, split=split,
                                            input_dir=input_dir, **kwargs)
        else:
            raise ValueError('split_method only accept one of (random, pairwise)')

        # symmetrize
        if symmetrize:
            dataset = symmetrize_dataset(dataset, mark_symmetrized=False)

        # check dataset leakage
        if not check_dataset_pair_leakage(dataset):
            raise ValueError('sequence pair leakage across subdatasets')

        if split_method in ('pairwise', 'cluster'):
            if not check_dataset_sequence_leakage(dataset):
                raise ValueError('sequence leakage across train and test sets')

        # preprocess dataset
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.map(
            lambda x: preprocess_function(tokenizer, x),
            batched=False,  # variable length
            num_proc=n_cores,
            desc='tokenize dataset'
        )

        # save dataset
        dataset.save_to_disk(dataset_dir)

        # save metadata
        try:
            name_or_path = tokenizer.name_or_path
        except AttributeError:
            name_or_path = None

        with (dataset_dir / 'metadata.json').open('w') as f:
            metadata = {
                'time': time.asctime(),
                'kwargs': {
                    'tokenizer': name_or_path,
                    'n_samples': n_samples,
                    'test_size': test_size,
                    'filter': filter_criteria,
                    'split_method': split_method,
                },
                'statistics': {f'{k}_samples': len(subdataset) for k, subdataset in dataset.items()},
            }
            json.dump(metadata, f, indent=2)

    if n_samples is not None:
        for subset in dataset.keys():
            dataset[subset] = dataset[subset].select(range(n_samples))

    return dataset


def decode2string(tokenizer, outputs, skip_special_tokens: bool = False, squeeze: bool = True) -> List[str]:
    func = tokenizer.batch_decode if hasattr(tokenizer, 'batch_decode') else tokenizer.decode_batch
    outputs_decode = func(outputs, skip_special_tokens)
    if squeeze:
        return [s.replace(' ', '') for s in outputs_decode]
    else:
        return outputs_decode


def desymmetrize_dataset(
        dataset: Union[DatasetDict, Dataset],
):
    def is_ordered(row: Dict[str, Any]) -> bool:
        bool_ = is_heavyA(row['sequenceA'], row['sequenceB'])
        return bool_ is True

    return dataset.filter(
        function=is_ordered,
        desc='desymmetrize dataset'
    )


def read_oas_header(csv: Union[str, Path]) -> Dict[str, Any]:
    csv = Path(csv)
    with csv.open('r') as f:
        line = f.readline()

    line = line.replace('\"\"', '\"')
    line = line[1:-2]

    with io.StringIO(line) as f:
        datum = json.load(f)

    return datum


def get_oas_statistics(dataset_dir: str = 'data/oas') -> Dict[str, Any]:
    keys = ('Species', 'Age', 'BSource', 'Btype', 'Vaccine', 'Disease', 'Orgnaism', 'Longitudinal')
    data = {}

    for csv in Path(dataset_dir).glob('*.csv'):
        datum = read_oas_header(csv)
        counts = datum['Unique sequences']

        for k, v in datum.items():
            if k not in keys:
                continue

            if k in data:
                if v in data[k]:
                    data[k][v] += counts
                else:
                    data[k][v] = counts
            else:
                data[k] = {v: counts}

    return data


def get_cdr_ids(gapped_sequence: str, is_heavy: bool) -> List[List[int]]:
    if is_heavy:
        loop_spans = ((32, 42), (57, 76), (109, 138))
    else:
        loop_spans = ((24, 42), (58, 72), (107, 138))

    resid = i_loop = 0
    loop_ids = [[] for _ in loop_spans]

    for i, aa in enumerate(gapped_sequence, 1):
        resid += aa != '-'
        if loop_spans[i_loop][0] <= i <= loop_spans[i_loop][1] and aa != '-':
            loop_ids[i_loop].append(resid)
        if i == loop_spans[i_loop][1]:
            i_loop += 1
        if i_loop == len(loop_spans):
            break

    return loop_ids


def get_regions(gapped_sequence: str, is_heavy: bool) -> Dict[str, str]:
    if is_heavy:
        return {
            'FR1': gapped_sequence[:32 - 1],
            'CDR1': gapped_sequence[32 - 1:42],
            'FR2': gapped_sequence[42:57 - 1],
            'CDR2': gapped_sequence[57 - 1:76],
            'FR3': gapped_sequence[76:109 - 1],
            'CDR3': gapped_sequence[109 - 1:138],
            'FR4': gapped_sequence[138:],
        }
    else:
        return {
            'FR1': gapped_sequence[:24 - 1],
            'CDR1': gapped_sequence[24 - 1:42],
            'FR2': gapped_sequence[42:58 - 1],
            'CDR2': gapped_sequence[58 - 1:72],
            'FR3': gapped_sequence[72:107 - 1],
            'CDR3': gapped_sequence[107 - 1:138],
            'FR4': gapped_sequence[138:],
        }


def get_identity(x: str, y: str, normalize: bool = True, strict: bool = False) \
                -> Union[int, float]:
    if len(x) != len(y):
        message = f'sequences are of different lengths: {x}, {y}'
        if strict:
            raise ValueError(message)
        else:
            warnings.warn(message)

    identity = 0
    n_non_gap = 0

    for a, b in zip(x, y):
        if a != '-' and b != '-':
            identity += a == b
            n_non_gap += 1

    if n_non_gap == 0:
        warnings.warn(f'no non-gap residues found on {x}, {y}. return None')
        return None

    if normalize:
        return identity / n_non_gap
    else:
        return identity


def get_region_identities(info1: Dict[str, Any], info2: Dict[str, Any]) -> Dict[str, float]:
    regions = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4']
    if 'gapped_sequence' in info1 and 'gapped_sequence' in info2:
        regions.append('gapped_sequence')

    identities = {}
    for region in regions:
        if region not in info1 or region not in info2:
            raise KeyError(f'region {region} not found in neither info1 nor info2')

        seq1 = info1[region]
        seq2 = info2[region]
        identities[f'{region}_identity'] = get_identity(seq1, seq2)

    if 'gapped_sequence_identity' in identities:
        identities['identity'] = identities['gapped_sequence_identity']
        del identities['gapped_sequence_identity']

    return identities


def get_region_lengths(info: Dict[str, Any]) -> Dict[str, int]:
    regions = ['FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4']
    if 'gapped_sequence' in info:
        regions.append('gapped_sequence')

    lengths = {}
    for region in regions:
        if region not in info:
            raise KeyError(f'region {region} not found in info')

        lengths[f'{region}_length'] = len(info[region].replace('-', ''))

    return lengths


def _get_default_datum(
        assign_germline: bool,
        add_cdr_ids: bool,
        output_regions: bool,
        output_region_lengths: bool
) -> Dict[str, None]:
    dafault_datum = {
        'gapped_sequence': None,
        'chain_type': None,
        'species': None,
        'is_heavy': None,
    }
    if assign_germline:
        dafault_datum.update({
            'v_gene': None,
            'j_gene': None,
        })
    if add_cdr_ids:
        dafault_datum.update({
            'cdr_ids': None,
        })
    if output_regions:
        dafault_datum.update({
            'FR1': None,
            'CDR1': None,
            'FR2': None,
            'CDR2': None,
            'FR3': None,
            'CDR3': None,
            'FR4': None,
        })
        if output_region_lengths:
            dafault_datum.update({
                'FR1_length': None,
                'CDR1_length': None,
                'FR2_length': None,
                'CDR2_length': None,
                'FR3_length': None,
                'CDR3_length': None,
                'FR4_length': None,
                'gapped_sequence_length': None,
            })
    return dafault_datum


def get_antibody_info(
        sequences: Iterable[Tuple[str, str]],
        scheme: str = 'aho',
        output: bool = False,
        assign_germline: bool = False,
        output_regions: bool = False,
        output_region_lengths: bool = False,
        add_cdr_ids: bool = False,
        **kwargs
) -> List[Dict[str, Any]]:
    try:
        ncpu = kwargs.pop('ncpu', 1)
        numbering, alignment_details, _ = anarci(
            sequences,
            scheme=scheme,
            output=output,
            assign_germline=assign_germline,
            ncpu=ncpu,
            **kwargs
        )
        assert len(numbering) == len(alignment_details) == len(sequences)
    except:
        return [_get_default_datum(assign_germline, add_cdr_ids, output_regions, output_region_lengths)] * len(sequences)

    data = []
    for i_seq in range(len(sequences)):
        if numbering[i_seq] is None:
            datum = _get_default_datum(assign_germline, add_cdr_ids, output_regions, output_region_lengths)
        else:
            numbering_ = numbering[i_seq][0][0]
            alignment_details_ = alignment_details[i_seq][0]
            if assign_germline:
                germlines = {k: v[0][1] for k, v in alignment_details_['germlines'].items()}
            else:
                germlines = {}

            datum = {
                'gapped_sequence': ''.join(numbering_[i][1] for i in range(len(numbering_))),
                'chain_type': alignment_details_['chain_type'],
                'species': alignment_details_['species'],
                'is_heavy': alignment_details_['chain_type'] == 'H',
                **germlines,
            }
            if add_cdr_ids:
                datum['cdr_ids'] = get_cdr_ids(
                    datum['gapped_sequence'],
                    is_heavy=alignment_details_['chain_type'] == 'H'
                )
            if output_regions:
                datum.update(get_regions(datum['gapped_sequence'], datum['is_heavy']))
                if output_region_lengths:
                    datum.update(get_region_lengths(datum))

        data.append(datum)

    return data


def is_heavyA(sequenceA: str, sequenceB : str) -> Optional[bool]:
    infoA, infoB = get_antibody_info((
        ('A', sequenceA.replace(' ', '')),
        ('B', sequenceB.replace(' ', '')),
    ))
    if infoA['is_heavy'] == infoB['is_heavy']:
        return None
    return infoA['is_heavy']


def is_lightA(sequenceA: str, sequenceB: str) -> Optional[bool]:
    return not is_heavyA(sequenceA, sequenceB)


def coarse_gene(gene):
    if not gene:
        return gene

    gene = gene.split('*')[0]
    gene = gene.split('-')[0]
    return gene
