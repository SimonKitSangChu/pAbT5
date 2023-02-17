from pabt5.util import *
from pabt5.alignment import *
from datasets import load_from_disk
import sys
import os

from pabt5.dataset import dataset2sequences

dataset = load_from_disk(sys.argv[1])
target_split = 'test'

output_dir = Path('tmp')
output_dir.mkdir(exist_ok=True)

sequences = {}
records = {}

for k in ('train', 'val', 'test'):
    fasta = output_dir / f'sequences_{k}.fasta'
    subset = dataset[k]
    sequences_ = dataset2sequences(subset, use_squeeze=True)
    records_ = sequences2records(sequences_)
    SeqIO.write(records_, fasta, 'fasta')

    sequences[k] = sequences_
    records[k] = records_

os.system(f'mmseqs easy-search {output_dir}/sequences_{target_split}.fasta '
    f'{output_dir}/sequences_train.fasta {output_dir}/alnRes.m8 {output_dir}/tmp')

records_org = SeqIO.to_dict(SeqIO.parse('data/oas/sequences.fasta', 'fasta'))
df_aln = pd.read_csv(output_dir / 'alnRes.m8', sep='\s+', header=None)
df_aln_ = df_aln[df_aln[2] >= 0.8]

