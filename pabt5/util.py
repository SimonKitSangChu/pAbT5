import hashlib
import json
from pathlib import Path
import pickle
from typing import Union, List, Optional, Dict, Any, Callable

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm


def spaceout(sequence: str) -> str:
    return ' '.join(sequence)


def squeeze(sequence: str) -> str:
    return sequence.replace(' ', '')


def pair_func(strings: List[str]) -> str:
    return '_'.join(sorted(strings))


def string2hash(string: str, method: Callable = hashlib.md5) -> str:
    hashname = string.encode()
    hashname = method(hashname)
    return hashname.hexdigest()


def read_fasta(fasta: Union[str, Path], return_sequence: bool = False) \
        -> List[Union[SeqRecord, str]]:
    if return_sequence:
        return [str(record.seq) for record in SeqIO.parse(fasta, 'fasta')]
    return list(SeqIO.parse(fasta, 'fasta'))


def write_fasta(fasta: Union[str, Path], records: Union[Dict[str, str], List[SeqRecord]]):
    if type(records) == dict:
        record = next(iter(records))
        if type(record) == str:
            records = sequences2records(records)

    SeqIO.write(records, fasta, 'fasta')


def read_pkl(pkl: Union[str, Path], mode: str = 'rb') -> Any:
    with Path(pkl).open(mode) as f:
        return pickle.load(f)


def write_pkl(pkl: Union[str, Path], obj: Any, mode: str = 'wb'):
    with Path(pkl).open(mode) as f:
        pickle.dump(obj, f)


def read_json(js: Union[str, Path], mode: str = 'r') -> Any:
    with Path(js).open(mode) as f:
        return json.load(f)


def write_json(js: Union[str, Path], obj: Any, mode: str = 'w', indent: int = 2):
    with Path(js).open(mode) as f:
        json.dump(obj, f, indent=indent)


def sequences2records(sequences: Union[List[str], Dict[str, str]]) -> List[SeqRecord]:
    if type(sequences) == dict:
        iterator = sequences.items()
    else:
        iterator = enumerate(sequences)

    records = []
    for k, sequence in iterator:
        record = SeqRecord(
            Seq(sequence),
            id=str(k),
            name=string2hash(sequence),
        )
        records.append(record)

    return records


def write_fasta_from_sequences(
    sequences: Dict[str, str],
    fasta: Union[Path, str],
):
    records = sequences2records(sequences)
    SeqIO.write(records, fasta, 'fasta')


def identities_from_lines(m8: Union[str, Path], verbose: bool = False) -> Dict[str, float]:
    identities = {}

    with open(m8, 'r') as f:
        iterator = tqdm(f) if verbose else f
        for line in iterator:
            try:
                id1, id2, _, identity, _, _, _, _, _, _, _ = line.split()
                identities[pair_func([id1, id2])] = float(identity)
            except:
                pass

    if not identities:
        raise ValueError(f'unable to load identities from {m8}')

    return identities
