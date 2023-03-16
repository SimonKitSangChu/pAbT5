import argparse
from pathlib import Path
import warnings

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
from tqdm import tqdm

from pabt5.util import read_json, squeeze, spaceout

# cmd options
cli = argparse.ArgumentParser()
cli.add_argument('--input_fasta', type=str, help='input sequences as fasta. supports multiple input sequences', required=True)
cli.add_argument('--checkpoint_dir', type=str, help='path to model checkpoint', required=True)
cli.add_argument('--generate_fasta', type=str, help='generation output in fasta format')
cli.add_argument('--generate_config', type=str, help='generation config in json format. otherwise use default')
cli.add_argument('--temperature', type=float, default=1., help='temperature in pairing partner generation. recommended range from 1.0 to 2.0')
cli.add_argument('--top_p', type=float, default=0.9, help='top_p in pairing partner generation. recommended range from 0.9 to 1.0')
cli.add_argument('--num_return_sequences', type=int, default=10, help='number of sequences to generate')
cli.add_argument('--pt', type=str, help='evaluation output in pt')
cli.add_argument('--output_attentions', action='store_true', help='output attention(s) in evaluation')
cli.add_argument('--output_hidden_states', action='store_true', help='output hidden_embedding(s) in evaluation')
cli.add_argument('--use_cuda', action='store_true', help='default: False. strongly recommended')
cli.add_argument('--full_precision', action='store_true', help='enable full-precision')
args = cli.parse_args()

if args.generate_fasta and args.pt:
    raise ValueError('only support either generation or evaluation')

if (args.output_attentions or args.output_hidden_states) and args.pt is None:
    raise ValueError('need --pt value for attentions and/or embeddings output')

# load model and tokenizer
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
if device == 'cpu':
    warnings.warn('CPU inference mode can be slow. pass --use_cuda if this is unintentional')

model = AutoModelForSeq2SeqLM.from_pretrained(
    args.checkpoint_dir,
    torch_dtype=torch.float32 if args.full_precision else torch.float16
).to(device)
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False, return_tensors='pt')

# inference
if args.generate_fasta:
    fasta = Path(args.generate_fasta)
    if args.generate_config is None:
        generate_config = {
            'max_new_tokens': 130,
            'return_dict_in_generate': True,
            'do_sample': True,
            'top_p': args.top_p,
            'num_return_sequences': args.num_return_sequences,
            'temperature': args.temperature,
        }
    else:
        generate_config = read_json(args.generate_config)

    print(f'generate config: {generate_config}')

    records_in = SeqIO.to_dict(SeqIO.parse(args.input_fasta, 'fasta'))
    records_out = []

    for name, record_in in tqdm(records_in.items(), total=len(records_in), desc='generate'):
        sequence_in = spaceout(str(record_in.seq))
        input_ids = torch.tensor([tokenizer.encode(sequence_in)], device=device)

        generate_output = model.generate(input_ids=input_ids, **generate_config)
        for i, sequence_ids in enumerate(generate_output.sequences, 1):
            sequence_out = squeeze(tokenizer.decode(sequence_ids, skip_special_tokens=True))
            record_out = SeqRecord(
                Seq(sequence_out),
                id=f'{name}_{i}'
            )
            records_out.append(record_out)

    SeqIO.write(records_out, args.generate_fasta, 'fasta')

elif args.pt:
    records_in = SeqIO.to_dict(SeqIO.parse(args.input_fasta, 'fasta'))
    assert len(records_in) % 2 == 0, 'embedding and/or attention generation must be in sequence pair(s)'

    outputs = {}

    records = list(records_in.items())
    for i in tqdm(range(0, len(records), 2), total=len(records)//2, desc='forward'):
        k1, record1 = records[i]
        k2, record2 = records[i+1]
        sequence1 = str(record1.seq)
        sequence2 = str(record2.seq)

        input_ids = torch.tensor([tokenizer.encode(sequence1)], device=device)
        labels = torch.tensor([tokenizer.encode(sequence2)], device=device)
        with torch.no_grad():
            output = model.forward(
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=args.output_hidden_states,
                output_attentions=args.output_attentions,
                return_dict=True,
            )
            outputs[k1 + '_' + k2] = output

    torch.save(outputs, args.pt)

else:
    warnings.warn('neither generate nor output is chosen')
