#!/bin/bash
format=png
checkpoint_dir=''
dataset_dir=''
uniref90_db=''
output_dir=''

# (optional)
# place all experiment data csv(s) for unsupervised prediction benchmark

# create evaluation
python scripts/evaluate_checkpoints.py \
	--checkpoint_dirs $checkpoint_dir \
	--output_dir evaluate \
	--dataset_dir $dataset_dir \
	--dataset_split test \
	--output_attentions

python scripts/evaluate_checkpoints.py \
	--checkpoint_dirs $checkpoint_dir \
	--output_dir generate_0 \
	--dataset_dir $dataset_dir \
	--dataset_split test \
	--generate \
	--generate_max_length 130 \
	--decoder_input_start_length 0

# visualize alignment (ignore division warning)
python -W ignore scripts/show_alignment.py \
	--blastdb $uniref90_db \
	--evaluation_dir $checkpoint_dir/generate_0 \
	--generate_methods default  \
	--output_dir ${output_dir}/alignment/generate_0 \
	--dump_generate \
	--dump_logits \
	--format $format

# visualize attention maps
python scripts/show_attention.py \
	--output_dir $output_dir \
	--evaluation_dir $checkpoint_dir/evaluate \
	--format $format \
	--overwrite

# visualize embeddings in TSNE
python scripts/show_embeddings.py \
	--checkpoint_dir $checkpoint_dir \
	--generate \
	--oas_dataset_dir $dataset_dir \
	--format $format \
	--output_dir $output_dir/embeddings

# visualize mispairing
python scripts/show_mispairing.py \
	--checkpoint_dir $checkpoint_dir \
	--dataset_dir $dataset_dir \
	--output_dir $output_dir/mispairing_hard \
	--format $format \
	--hard_mode

# visualize generation statistics
python scripts/show_generation.py \
	--checkpoint_dir $checkpoint_dir \
	--dataset_dir $dataset_dir \
	--output_dir $output_dir/generation \
	--format $format

# visualize correlation with experiments
for csv in `ls exp/*csv`; do
	python scripts/show_exp.py \
		--input_csv $csv \
		--checkpoint_dir $checkpoint_dir \
		--use_cuda \
		--output_dir $output_dir/exp/pabt5
done

# visualize oas statistics
python scripts/show_oas.py --format $format --output_dir $output_dir/oas_statistics

