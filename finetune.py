import argparse
import logging
import os
from pathlib import Path

import torch
from transformers import (
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    Adafactor,
)

from pabt5.util import write_pkl
from pabt5.model import (
    CustomSeq2SeqLM,
    T5DecoderForCausalLM,
)
from pabt5.dataset import load_oas_dataset, load_custom_dataset


# argument parsing
cli = argparse.ArgumentParser()
cli.add_argument('--output_dir', type=str, default=None, help='default to filename')
cli.add_argument('--dataset_dir', type=str, default='data/dataset')
cli.add_argument('--dataset', default='oas', choices=['oas', 'custom'])
cli.add_argument('--debug', default=False, action='store_true')
cli.add_argument('--ds_json', type=str, default=None)
cli.add_argument('--n_samples', type=int)
cli.add_argument('--max_length', type=int, default=1024, help='max_length of sequence in data collator')
cli.add_argument('--no_cuda', default=False, action='store_false')
cli.add_argument('--no_t5_pretraining', default=False, action='store_true', help='train t5-3b from scratch for testing; overrides model_type')
cli.add_argument('--freeze_embedding', default=True, help='freeze shared embedding projection; only supported in pretrained T5')
cli.add_argument('--freeze_encoder', default=False, action='store_true', help='freeze encoder parameters; only supported in pretrained T5')
cli.add_argument('--model_name', type=str, default='Rostlab/prot_t5_xl_uniref50')
cli.add_argument('--model_type', type=str, default='t5', choices=['t5', 't5-decoder'])
cli.add_argument('--lr', type=float, default=5e-5)
cli.add_argument('--local_batch_size', type=int, default=1)
cli.add_argument('--max_epochs', type=int, default=10000)
cli.add_argument('--filter_homomeric_only', default=False, action='store_true', help='train and evaluate only on homomeric binding')
cli.add_argument('--filter_heteromeric_only', default=False, action='store_true', help='train and evaluate only on heteromeric binding')
cli.add_argument('--filter_min_score', type=float, default=0, help='minimum intact score')
cli.add_argument('--filter_max_length', type=int, default=1024, help='different from max_length option; applied to filtering only')
cli.add_argument('--filter_fasta', help='fasta to only include allowed sequences')
args, unk = cli.parse_known_args()

if args.output_dir:
    OUTDIR = Path(args.output_dir)
else:
    OUTDIR = Path(Path(__file__).stem)

OUTDIR.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    filename=str(OUTDIR / 'log'),
    level=logging.INFO
)

logging.info(f'proceed with arguments {args}')
write_pkl(OUTDIR / 'args.pkl', args)
write_pkl(OUTDIR / 'unk.pkl', unk)

if unk:
    logging.warning(f'ignore unrecognized arguments {unk}')

if args.no_cuda:  # invalid to DeepSpeed
    if args.ds_json:
        logging.warning(f'--no_cuda option does not support DeepSpeed runs')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

if args.ds_json and args.max_length is None:
    raise ValueError('must enable max_length for DeepSpeed parallelism')

if args.local_batch_size > 1 and args.max_length is None:
    raise ValueError('must pad seqeunce with max_length for local batch size > 1')

if args.max_length != args.filter_max_length:
    logging.warning('mismatch in max_length and filter_max_length options; sequence(s) will be truncated')

# load tokenzier and model
if args.no_t5_pretraining:
    tokenizer = T5Tokenizer.from_pretrained(str(args.model_name), do_lower_case=False, return_tensors='pt')
    model_base = CustomSeq2SeqLM.from_pretrained(str(args.model_name))
    model = CustomSeq2SeqLM.from_config(model_base.config)
    model.config.decoder_start_token_id = model.config.pad_token_id
    logging.warning('will use warning_steps = 0')
elif args.model_type == 't5':
    tokenizer = T5Tokenizer.from_pretrained(str(args.model_name), do_lower_case=False, return_tensors='pt')
    model = CustomSeq2SeqLM.from_pretrained(args.model_name)
elif args.model_type == 't5-decoder':
    tokenizer = T5Tokenizer.from_pretrained(str(args.model_name), do_lower_case=False, return_tensors='pt')
    model = T5DecoderForCausalLM.from_pretrained(str(args.model_name))
else:
    raise NotImplementedError

# load dataaset
kwargs = {
    'tokenizer': tokenizer,
    'dataset_dir': str(args.dataset_dir),
    'n_samples': args.n_samples,
    'filtering': {
        'homomeric_only': args.filter_homomeric_only,
        'heteromeric_only': args.filter_heteromeric_only,
        'min_score': args.filter_min_score,
        'max_length': args.filter_max_length,
        'fasta': args.filter_fasta,
    },
}

if args.dataset == 'custom':
    dataset = load_custom_dataset(
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        max_length=args.max_length,
        test_size=0.5,
    )
elif args.dataset == 'oas':
    dataset = load_oas_dataset(
        sequenceA='sequence_alignment_aa_heavy',
        sequenceB='sequence_alignment_aa_light',
        symmetrize=True,
        split=(0.9, 0.05, 0.05),
        split_method='pairwise',
        **kwargs
    )
else:
    raise ValueError(f'dataset option {args.dataset} is not supported')

dataset = dataset.remove_columns(['sequenceA', 'sequenceB'])  # drop sequenceA/B for batching
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding='longest',
    max_length=args.max_length,
    label_pad_token_id=-100,  # default to skip loss
    return_tensors='pt',
)

# setup trainer
if args.debug:
    logging.debug('set local and global batch_size to 1')
    local_batch_size = 1
    global_batch_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
else:
    local_batch_size = args.local_batch_size
    global_batch_size = 2048

grad_steps = global_batch_size // local_batch_size
if torch.cuda.is_available():
    grad_steps = grad_steps // torch.cuda.device_count()

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTDIR,
    evaluation_strategy='steps' if args.debug else 'epoch',
    eval_steps=1,
    logging_strategy='steps',
    logging_steps=1,
    save_strategy='steps' if args.debug else 'epoch',
    save_steps=1,
    save_total_limit=None,
    num_train_epochs=1 if args.debug else args.max_epochs,
    per_device_train_batch_size=local_batch_size,
    per_device_eval_batch_size=local_batch_size,
    gradient_accumulation_steps=grad_steps,
    eval_accumulation_steps=1,
    weight_decay=0.,  # 0 weight decay in ProtT5-XL Uniref50
    seed=42,
    data_seed=42 if args.debug else None,
    dataloader_num_workers=torch.get_num_threads(),
    disable_tqdm=False,
    load_best_model_at_end=True,  # EarlyStoppingCallback
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    deepspeed=args.ds_json,
    full_determinism=False,  # disabled for speed up
    ray_scope='last',
    remove_unused_columns=True,
    skip_memory_metrics=not args.debug,  # save memory profile in debug mode
    warmup_steps=0,
)
   
if args.model_type == 't5' and not args.no_t5_pretraining:
    parameters = {
        'shared': model.shared.parameters(),
        'encoder.block': model.encoder.block.parameters(),
        'encoder.final_layer_norm': model.encoder.final_layer_norm.parameters(),
        'encoder.dropout': model.encoder.dropout.parameters(),
        'decoder.block': model.decoder.block.parameters(),
        'decoder.final_layer_norm': model.decoder.final_layer_norm.parameters(),
        'decoder.dropout': model.decoder.dropout.parameters(),
        'lm_head': model.lm_head.parameters(),
    }
    if args.freeze_embedding:
        logging.info('freeze shared embedding')
        del parameters['shared']
    if args.freeze_encoder:
        logging.info('freeze encoder')
        del parameters['encoder.block'], parameters['encoder.final_layer_norm'], parameters['encoder.dropout']
elif args.model_type == 't5-decoder':
    parameters = {'decoder': model.decoder.parameters()}
else:
    parameters = {'all': model.parameters()}

if args.dataset == 'custom':
    train_dataset = dataset
    eval_dataset = dataset
elif 'val' in dataset:
    train_dataset = dataset['train']
    eval_dataset = dataset['val']
else:
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

# recommended AdaFactor hyperparameters at https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
trainer = Seq2SeqTrainer(
    model=model,
    optimizers=[
        Adafactor(
            params=[{'params': params} for params in parameters.values()],
            lr=args.lr,
            clip_threshold=1.0,
            weight_decay=0,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,  # disabled due to pretraining
        ),
        None,
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01),
    ],
)
trainer.train()

best_ckpt_dir = output_dir / 'checkpoint-best'
trainer.model.save_pretrained(best_ckpt_dir)

