import argparse
from pathlib import Path
import json
import logging

from transformers import (
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
import torch.utils.data
from tqdm import tqdm
import deepspeed

from pabt5.util import string2hash, write_fasta
from pabt5.model import T5DecoderForCausalLM
from pabt5.dataset import load_custom_dataset, load_oas_dataset
from pabt5.metrics import evaluate_model, evaluate_generate_outputs

# argument parsing
cli = argparse.ArgumentParser()
cli.add_argument('--checkpoint_dirs', nargs='*', type=str, default=[])
cli.add_argument('--output_dir', type=str, default='evaluate')
cli.add_argument('--dataset_dir', type=str, default='data/dataset')
cli.add_argument('--dataset_split', type=str, default='test')
cli.add_argument('--debug', action='store_false', default=False)
cli.add_argument('--use_deepspeed', action='store_true', help='(experimental) use deepspeed for training')
cli.add_argument('--n_samples', type=int)
cli.add_argument('--max_length', type=int, default=1024)
cli.add_argument('--no_cuda', action='store_true', default=False)
cli.add_argument('--use_t5_small', action='store_true',
                 help='train t5-small from scratch for testing; overrides model_type')
cli.add_argument('--model_name', type=str, default='Rostlab/prot_t5_xl_uniref50')
cli.add_argument('--model_type', type=str, default='t5', choices=['t5', 't5-decoder'])
cli.add_argument('--local_batch_size', type=int, default=1)
cli.add_argument('--custom_dataset', default=False, action='store_true', help='train on small artificial dataset')
cli.add_argument('--generate_min_length', default=0, type=int, help='minimum length in generate method')
cli.add_argument('--generate_max_length', default=1024, type=int, help='maximum length in generate method')
cli.add_argument('--decoder_input_start_length', default=0, type=int,
                 help='force starting decoder sequence by n tokens; does not count decoder_start_token_id')
cli.add_argument('--skip_special_tokens', default=True, action='store_true', help='drop special tokens in decoding')
cli.add_argument('--no_skip_special_tokens', action='store_false', dest='skip_special_tokens')
cli.add_argument('--filter_homomeric_only', action='store_true',
                 help='train and evaluate only on homomeric binding')
cli.add_argument('--filter_heteromeric_only', action='store_true',
                 help='train and evaluate only on heteromeric binding')
cli.add_argument('--filter_min_score', type=float, default=0, help='minimum intact score')
cli.add_argument('--filter_max_length', type=int, default=1024,
                 help='different from max_length option; applied to filtering only')
cli.add_argument('--output_attentions', action='store_true', help='output_attentions in pt')
cli.add_argument('--generate', action='store_true', help='generate sequences')
cli.add_argument('--drop_duplicates', action='store_true', help='drop duplicate generate sequences')
cli.add_argument('--cp_test', action='store_true',
                 help='test copy-and-paste behavior by replacing decoder_input_ids from sequenceA instead of sequenceB')
args, unk = cli.parse_known_args()

print(f'Proceed with arguments: {args}')

if unk:
    logging.warning(f'Arguments not recognized thus not used {" ".join(unk)}')

# load tokenizer and model
if args.model_type in ('t5', 't5-decoder'):  # bos, sep, eos, pad = None, None, </s>, <pad>
    tokenizer = T5Tokenizer.from_pretrained(str(args.model_name), do_lower_case=False, return_tensors='pt')
else:
    raise NotImplementedError

device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

# load dataaset
if args.custom_dataset:
    dataset = load_custom_dataset(tokenizer=tokenizer, max_length=1024 if args.max_length is None else args.max_length)
else:
    dataset = load_oas_dataset(
        tokenizer=tokenizer,
        dataset_dir=str(args.dataset_dir),
        n_samples=args.n_samples,
        filtering={
            'homomeric_only': args.filter_homomeric_only,
            'heteromeric_only': args.filter_heteromeric_only,
            'min_score': args.filter_min_score,
            'max_length': args.filter_max_length,
        },
    )

dataset = dataset.remove_columns(['sequenceA', 'sequenceB'])  # drop sequenceA/B for deepspeed batching

# iteerate over checkpoints
for checkpoint_dir in args.checkpoint_dirs:
    # load model
    if args.model_type == 't5':
        model_base = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir, torch_dtype=torch.float16)
        if model_base.config.decoder_start_token_id is None:
            model_base.config.decoder_start_token_id = model_base.config.pad_token_id
    elif args.model_type == 't5-decoder':
        model_base = T5DecoderForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.float16)
        model_base.config.is_decoder_only = True
        model_base.config.is_encoder_decoder = False
    else:
        raise NotImplementedError(f'model_type {args.model_type} is not implemented')

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model_base,
        padding='longest',
        max_length=args.max_length,
        label_pad_token_id=-100,  # default to skip loss
        return_tensors='pt',
    )

    if args.use_deepspeed:
        engine = deepspeed.init_inference(
            model=model_base,
            mp_size=torch.cuda.device_count(),
            dtype=torch.half,
            replace_method='auto',
            replace_with_kernel_inject=True,
        )
        model = engine.module
    elif model_base.is_parallelizable and torch.cuda.is_available():
        model_base.parallelize()
        model = model_base.to(device)
    else:
        model = model_base.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset if args.custom_dataset else dataset[args.dataset_split],
        batch_size=args.local_batch_size,
        pin_memory=not args.no_cuda and torch.cuda.is_available(),
        collate_fn=data_collator,
    )

    # log output in subdirectory
    if checkpoint_dir == args.model_name:
        output_dir = Path('..') / args.output_dir
    else:
        output_dir = Path(checkpoint_dir) / args.output_dir

    output_dir.mkdir(exist_ok=True, parents=True)

    data = {}

    for batch in tqdm(dataloader, desc=str(output_dir)):
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['labels'] = batch['labels'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)

        for input_ids, labels, attention_mask in zip(
                batch['input_ids'].split(1),
                batch['labels'].split(1),
                batch['attention_mask'].split(1),
        ):
            input_sequence = tokenizer.decode(input_ids[0], skip_special_tokens=args.skip_special_tokens).replace(' ',
                                                                                                                  '')
            label_sequence = tokenizer.decode(labels[0], skip_special_tokens=args.skip_special_tokens).replace(' ', '')

            ordered_pair = input_sequence + '_' + label_sequence
            hashname = string2hash(ordered_pair)

            if ordered_pair in data:  # homomeric binding
                continue

            data[ordered_pair] = {
                'input_sequence': input_sequence,
                'label_sequence': label_sequence,
                'hash': string2hash(ordered_pair),
                'generate': {},
                'metric': {},
            }

            fasta = output_dir / f'{data[ordered_pair]["hash"]}.fasta'
            js = output_dir / f'{hashname}.json'
            pt = output_dir / f'{hashname}.pt'

            if js.exists() and fasta.exists() and pt.exists():
                with js.open('r') as f:
                    data[ordered_pair] = json.load(f)
            else:
                # prepare generate parameters
                if args.decoder_input_start_length:
                    decoder_input_ids = input_ids.clone() if args.cp_test else labels.clone()
                    length = args.decoder_input_start_length + 1  # +1 to feed at least the decoder start token
                    length = min(length, labels.shape[-1] - 1)  # do not include eos in decoder_input_ids
                    decoder_input_ids = decoder_input_ids[..., :length]
                else:
                    decoder_input_ids = torch.tensor(  # manually prepend model.config.decoder_start_token_id
                        [[model.config.decoder_start_token_id]] * len(labels),
                        device=device)

                common_config = {
                    'output_scores': True,
                    'min_length': args.generate_min_length,
                    'max_new_tokens': args.generate_max_length,
                    'return_dict_in_generate': True,
                    'decoder_input_ids': decoder_input_ids,
                    'attention_mask': attention_mask,
                    'bos_token_id': tokenizer.bos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                }
                search_common_config = {
                    'do_sample': True,
                    'top_p': 0.9,
                    'num_return_sequences': 10,
                }

                generate_configs = {
                    # 'greedy': common_config,
                    'default': {
                        'temperature': 1.,
                        **common_config,
                        **search_common_config,
                    },
                }
                if not args.generate:  # skip generate
                    generate_configs = {}

                pt_dict = {generate_name: {} for generate_name in generate_configs.keys()}

                # generate sequences
                for generate_name, generate_config in generate_configs.items():
                    if args.model_type == 't5':
                        generate_outputs = model.generate(inputs=input_ids, **generate_config)
                    elif args.model_type == 't5-decoder':
                        decoder_input_ids = generate_config.get('decoder_input_ids')
                        generate_outputs = model.generate(inputs=decoder_input_ids, **generate_config)
                    else:
                        raise ValueError('{args.model_type} not supported')

                    evaluations = evaluate_generate_outputs(
                        model=model,
                        tokenizer=tokenizer,
                        generate_outputs=generate_outputs,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=args.output_attentions,
                        drop_duplicates=args.drop_duplicates,
                        skip_special_tokens=args.skip_special_tokens,
                    )

                    pt_dict[generate_name] = evaluations
                    data[ordered_pair]['generate'][generate_name] = {
                        'generate_sequence': evaluations['generate_sequence'],
                        'loss': evaluations['loss'],
                        'perplexity': evaluations['perplexity'],
                        'accuracy': evaluations['accuracy'],
                        'generate_config': {k: v for k, v in generate_config.items() if
                                            type(v) in (bool, str, float, int)},
                    }

                sequence_dict = {  # store data in fasta
                    'input_sequence': data[ordered_pair]['input_sequence'],
                    'label_sequence': data[ordered_pair]['label_sequence'],
                }
                descriptions = {
                    'input_sequence': '',
                    'label_sequence': '',
                }

                for generate_name, generate_data in data[ordered_pair]['generate'].items():
                    for i, sequence in enumerate(generate_data['generate_sequence']):
                        sequence_dict[f'{generate_name}_{i}'] = sequence

                write_fasta(fasta, sequence_dict)

                with torch.no_grad():
                    evaluation = evaluate_model(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        label_ids=labels,
                        attention_mask=attention_mask,
                        output_attentions=args.output_attentions,
                        skip_special_tokens=args.skip_special_tokens,
                    )
                    pt_dict['label'] = evaluation

                data[ordered_pair]['metric'] = {k: evaluation[k] for k in
                                                ('loss', 'perplexity', 'accuracy')}

                torch.save(pt_dict, pt)
                with js.open('w') as f:
                    json.dump(data[ordered_pair], f, indent=2)

            if args.debug:
                break

        if args.debug:
            break

    del model, model_base, dataloader

    metrics = {}  # aggregate average metrics (e.g. accuracy, perplexity)
    for datum in data.values():
        for k, v in datum['metric'].items():
            try:
                if k in metrics:
                    metrics[k] += v / len(data)
                else:
                    metrics[k] = v / len(data)
            except TypeError:
                pass

    js = output_dir / 'evaluate_average.json'
    with js.open('w') as f:
        json.dump(metrics, f, indent=2)

    js = output_dir / 'evaluate.json'  # store all results in json
    with js.open('w') as f:
        json.dump(data, f, indent=2)
