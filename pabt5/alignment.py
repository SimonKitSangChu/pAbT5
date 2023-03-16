from io import StringIO
import os
from pathlib import Path
import subprocess
import itertools
from typing import Any, List, Union, Iterable, Dict, Optional, Tuple

import torch
from Bio import AlignIO
from Bio.Align.Applications import ClustalwCommandline
from Bio.Blast.Applications import (
    NcbiblastpCommandline,
    NcbimakeblastdbCommandline,
    NcbipsiblastCommandline,
)
from Bio.Blast import NCBIXML
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import logomaker
import pandas as pd
import ray
from tqdm import tqdm

from .util import string2hash, pair_func
from .dataset import get_antibody_info


AA_GAP_LETTERS = list('ACDEFGHIKLMNPQRSTVWY-')
AA_LETTERS = list('ACDEFGHIKLMNPQRSTVWY')


def create_blastdb(fasta: str, sequences: List[SeqRecord]) -> Any:
    fasta = Path(fasta)
    with fasta.open('w') as f:
        SeqIO.write(sequences, f, 'fasta')

    cline = NcbimakeblastdbCommandline(dbtype='prot', input_file=fasta, title=fasta.stem, parse_seqids=True)
    cline()
    return cline


def blastp_seq_db(seq: SeqRecord, db: Union[str, Path]) -> List:
    job_name = string2hash(str(seq.seq))
    fasta = Path(f'.{job_name}.fasta')
    SeqIO.write(seq, fasta, 'fasta')
    return blastp_db(fasta=fasta, db=db)


def blastp_seqs_db(seqs: List[SeqRecord], db: Union[str, Path]) -> List:
    job_name = string2hash(tuple(seqs))
    fasta = Path(f'.{job_name}.fasta')
    SeqIO.write(seqs, fasta, 'fasta')

    alignment = blastp_db(fasta=fasta, db=db)
    fasta.unlink()
    return alignment


def blastp_db(fasta: Union[str, Path], db: Union[str, Path]) -> List:
    cline = NcbiblastpCommandline(query=fasta, db=db, outfmt=5, num_threads=os.cpu_count())
    output = cline()[0]

    blast_result_record = NCBIXML.read(StringIO(output))
    alignments = blast_result_record.alignments
    return alignments


def alignment2hsp(alignment: Any) -> Any:
    scores = [hsp.score for hsp in alignment.hsps]
    assert all(scores[0] >= score for score in scores)

    if alignment.hsps:
        return alignment.hsps[0]
    else:
        return None


def blastp_pair(seq1: Union[str, SeqRecord], seq2: Union[str, SeqRecord], clean: bool = True, outfmt=5, **kwargs) -> List:
    if not issubclass(seq1.__class__, SeqRecord):
        seq1 = SeqRecord(Seq(seq1), id='seq1')
    if not issubclass(seq2.__class__, SeqRecord):
        seq2 = SeqRecord(Seq(seq2), id='seq2')

    hash_ = string2hash(str(seq1.seq) + str(seq2.seq))
    fasta1 = Path(f'.{hash_}_1.fasta')
    fasta2 = Path(f'.{hash_}_2.fasta')
    SeqIO.write(seq1, fasta1, 'fasta')
    SeqIO.write(seq2, fasta2, 'fasta')

    output = NcbiblastpCommandline(query=fasta1, subject=fasta2, outfmt=outfmt, **kwargs)()[0]
    if clean:
        fasta1.unlink(missing_ok=True)
        fasta2.unlink(missing_ok=True)

    blast_result_record = NCBIXML.read(StringIO(output))
    alignments = blast_result_record.alignments
    return alignments


def blastp_pair2identity(seq1: Union[str, SeqRecord], seq2: Union[str, SeqRecord], **kwargs) -> float:
    alignments = blastp_pair(seq1, seq2, **kwargs)
    if alignments:
        hsp = alignment2hsp(alignments[0])
        if hsp:
            return hsp.identities / hsp.align_length
    return 0


def read_psiblast_pssm(pssm: Union[str, Path], clean: bool = True)-> pd.DataFrame:
    aa_ordered = list('ARNDCQEGHILKMFPSTWYV')
    names = [
        'pos',
        'wildtype',
        *aa_ordered,
        *[f'{aa}_perc' for aa in aa_ordered],
        'information',
        'weight'
    ]
    df = pd.read_csv(pssm, skiprows=3, header=None, sep='\s+', skipfooter=5, names=names,
            engine='python')
    df = df.set_index('pos')

    if clean:
        df = df[aa_ordered]

    return df


def psiblast_pssm(seq: SeqRecord, db: Union[str, Path], clean: bool = True, **kwargs) -> pd.DataFrame:
    jobname = string2hash(str(seq.seq))
    fasta = Path(f'.{jobname}.fasta')
    SeqIO.write(seq, fasta, 'fasta')

    db = Path(db)
    os.environ['BLASTDB'] = str(db.parent)

    pssm = Path(f'.{jobname}.pssm')
    cline = NcbipsiblastCommandline(
        query=fasta,
        db=db.name,
        num_iterations=3,
        save_pssm_after_last_round='',
        num_threads=os.cpu_count(),
        out_ascii_pssm=pssm,
        **kwargs,
    )
    output = cline()[0]

    df = read_psiblast_pssm(pssm, clean=clean)
    fasta.unlink()
    pssm.unlink()

    return df


def clustalw_msa(records: List[SeqRecord], return_frequency: bool = False, head_id: Optional[str] = None, **kwargs):
    jobname = string2hash(str(records[0].seq))
    fasta = Path(f'.{jobname}.fasta')
    SeqIO.write(records, fasta, 'fasta')

    aln = Path(f'.{jobname}.aln')
    cline = ClustalwCommandline('clustalw2', infile=fasta, outfile=aln, **kwargs)
    cline()

    alignment = AlignIO.read(aln, 'clustal')
    fasta.unlink()
    aln.unlink()
    Path(f'.{jobname}.dnd').unlink(missing_ok=False)

    # reorder head record to top
    if head_id is not None:
        dic = {record.id: record for record in alignment}
        alignment = [dic.pop(head_id)]
        alignment.extend(list(dic.values()))

    if return_frequency:
        df = msa_matrix(alignment)
        return alignment, df
    else:
        return alignment


def msa_matrix(
    alignment: Iterable[SeqRecord],
    normalize: bool = False,
    strategy: str = 'all',
    drop_gap: bool = True,
    ) -> pd.DataFrame:
    df = [list(record.seq) for record in alignment]
    df = pd.DataFrame(df)
    df = df.T

    for residue in AA_GAP_LETTERS:
        if strategy == 'first':
            df[residue] = df.apply(lambda x: x == residue)[[0]].sum(axis=1)
        elif strategy == 'not first':
            columns = [col for col in df.columns if col != 0]
            df[residue] = df.apply(lambda x: x == residue)[columns].sum(axis=1)
        elif strategy == 'all':
            df[residue] = df.apply(lambda x: x == residue).sum(axis=1)
        else:
            raise ValueError(f'only supports strategy [first|not first|all] but {stratgy} is provided')

    df = df[AA_GAP_LETTERS]
    if normalize:
        df = (df.T / df.sum(axis=1)).T

    df.index.name_y = 'pos'
    if drop_gap:
        return df[AA_LETTERS]
    else:
        return df


def msa_sequence_logo(records: List[SeqRecord]) -> logomaker.Logo:
    alignment = clustalw_msa(records)
    df = msa_matrix(alignment, normalize=True)
    del df['-']

    logo = logomaker.Logo(df, figsize=(0.5 * df.shape[0], 1))
    return logo


def mmseqs_easy_search(
    query_fasta: Union[str, Path],
    target_fasta: Union[str, Path],
    aln_m8: Union[str, Path] = 'aln.m8',
    tmp_dir: Union[str, Path] = 'tmp',
    ):
    cmd = f'mmseqs easy-search {query_fasta} {target_fasta} {aln_m8} {tmp_dir}'
    _ = subprocess.run(cmd.split())


def identities_from_m8(m8: Union[str, Path], low_memory: bool = True) -> Dict[str, float]:
    df_aln = pd.read_csv(m8, sep='\s+', header=None, low_memory=low_memory)
    df_aln = df_aln.dropna()
    df_aln['key'] = df_aln.apply(lambda x: pair_func((str(int(x[0])), str(int(x[1])))), axis='columns')
    df_aln = df_aln.set_index('key')
    return df_aln[2].to_dict()


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


class SequenceSimilarityNetwork(dict):
    @classmethod
    def from_sequences(cls, sequences1: Iterable[str], sequences2: Optional[Iterable[str]],
                       disable_tqdm: bool = False, max_queue_size: Optional[int] = None,
                       num_cpus: float = 1, clean: bool = False) -> 'SequenceSimilarityNetwork':
        sequences1 = set(sequences1)
        sequences2 = set(sequences2) if sequences2 else sequences1

        @ray.remote(num_cpus=num_cpus)
        def fxn(seq1, seq2):
            return (seq1, seq2), blastp_pair2identity(seq1, seq2, clean=clean)

        # ray
        pbar = tqdm(
            itertools.product(sequences1, sequences2),
            disable=disable_tqdm,
            desc='Building network',
            total=len(sequences1) * len(sequences2),
        )
        if max_queue_size is None:
            max_queue_size = torch.get_num_threads() / num_cpus

        result_refs = []
        for seq1, seq2 in pbar:
            pair = (seq1, seq2)
            if len(result_refs) >= max_queue_size:
                ready_refs, result_refs = ray.wait(result_refs)
                ray.get(ready_refs)
            result_refs.append(fxn.remote(*pair))
            ray.get(result_refs)

        network = cls()
        for result_ref in result_refs:
            pair, identity = ray.get(result_ref)
            network[pair] = network[reversed(pair)] = identity

        return network


class EdgeSimilarityNetwork(dict):
    @classmethod
    def from_edges(
            cls,
            edges_from: Iterable[Tuple[str, str]],
            edges_to: Optional[Iterable[Tuple[str, str]]] = None,
            disable_tqdm: bool = False,
            max_queue_size: Optional[int] = None,
            num_cpus: float = 1,
            clean: bool = True,
    ) -> 'EdgeSimilarityNetwork':
        edges_from = set(edges_from)
        edges_to = set(edges_to) if edges_to else edges_from

        @ray.remote(num_cpus=num_cpus, max_retries=5, retry_exceptions=True)
        def fxn(edge_from, edge_to):
            return edge_from, edge_to, blastp_pair2identity(edge_from[0], edge_to[0], clean=clean), \
                blastp_pair2identity(edge_from[1], edge_to[1], clean=clean)

        # ray
        pbar = tqdm(
            itertools.product(edges_from, edges_to),
            disable=disable_tqdm,
            desc='Building network',
            total=len(edges_from) * len(edges_to),
        )
        if max_queue_size is None:
            max_queue_size = torch.get_num_threads() / num_cpus

        result_refs = []
        for edge_from, edge_to in pbar:
            if len(result_refs) >= max_queue_size:
                ready_refs, result_refs = ray.wait(result_refs)
                ray.get(ready_refs)

            result_refs.append(fxn.remote(edge_from, edge_to))

        ray.get(result_refs)

        network = cls()
        for result_ref in result_refs:
            edge_from, edge_to, identity1, identity2 = ray.get(result_ref)
            if edge_from not in network:
                network[edge_from] = {}
            if edge_to not in network:
                network[edge_to] = {}

            network[edge_from][edge_to] = (identity1, identity2)

        return network

    def min_in(self, edge: Tuple[str, str]) -> float:
        return min(min(a, b) for a, b in self[edge].values())

    def max_in(self, edge: Tuple[str, str]) -> float:
        return max(max(a, b) for a, b in self[edge].values())
