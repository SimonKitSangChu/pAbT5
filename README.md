# pAbT5

Project pAbT5 (paired Antibody T5) is developed to model antibody heavy and light chain (VH and VL)
through encoder-decoder protein language model. Derived from [ProtT5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50), pAbT5 is fine-tuned
on paired [OAS](https://opig.stats.ox.ac.uk/webapps/oas/) database on machine translation task. Specifically, heavy-to-light and light-to-heavy are
regarded as forward and backward translations respectively.

## Publication
[Chu, Simon KS, and Kathy Y. Wei. "Conditional Generation of Paired Antibody Chain Sequences through Encoder-Decoder Language Model." arXiv preprint arXiv:2301.02748 (2023).](https://arxiv.org/abs/2301.02748)

## Installation
pAbT5 depends on [ANARCI](https://github.com/oxpig/ANARCI) which requires manually installation from users. User might have to uncomment for
HMM profile install in ANARCI under `setup.py`. Optionally, user can install [pyrosetta](https://www.pyrosetta.org/) and [DeepAb](https://github.com/RosettaCommons/DeepAb), or [AlphaFold](https://github.com/deepmind/alphafold) for
antibody structure prediction. Part of the repository of [progen2](https://github.com/enijkamp/progen2) is included for benchmark.

```
# create conda environment
conda env create -n pabt5 --file=environment.yml
conda activate -n pabt5
pip install .

# download weights
gdown https://drive.google.com/drive/u/1/folders/1X3M3fcwfpZeHe_HCAiw9KrQzfVsc2Bmh -O model_ckpt --folder

# (optional) download dataset
# gdown https://drive.google.com/drive/u/1/folders/1rO_3vxpkPFeFhlBJSQ2q4HEU-GPep82O -O dataset --folder
```

## Inference
Trained on forward and backward translation, the model learns pairing implicitly. There is no need to specify
its generation target as heavy/light chain. For generation, simply pass sequence(s) as a fasta input.
```commandline
python inference.py --input_fasta example/input.fasta --checkpoint_dir model_ckpt --generate_fasta output.fasta --temperature 1.0 \
    --generate_config optional_generate_config_in_json
```
To generate attention map and hidden states, write the sequences for encoder and decoder as first and second
records, third and fourth, etc. in the fasta input. 
```commandline
python inference.py --input_fasta example/input.fasta --checkpoint_dir model_ckpt --output_attentions --output_hidden_states \
    --pt output.pt
```
User might also use model perplexity as unsupervised prediction. Input in csv format must contain two columns, i.e.
`sequence_h` and `sequence_l`. The scripts will generate three columns depending on the model type.
1. T5: Heavy-to-Light, Light-to-Heavy and bidirectional average
2. T5 (decoder-only), Light, Heavy and Heavy-Light average
3. Progen2: Light, Heavy and Heavy-(GS-linker)-Light
```commandline
python show_exp.py --input_csv input.csv --checkpoint_dir model_ckpt --output_dir out \
    --plotting --statistics  # (optional) on-the-fly scatterplot and correlation analysis
```
User should pass `--is_t5_decoder`, or `--is_progen2` and `--tokenizer` whenever appropriate. 

#### Note: pAbT5 has 3B parameters. Users might consider to either inference on GPU with sufficient memory and possibly on half precision.

## Training
To reproduce the results, `finetune.py` automatically handles dataset generation and finetuning.
```commandline
(cd data/oas; bash download.sh)  # download paired OAS data

python finetune.py --output_dir pAbT5 --dataset_dir dataset --lr 5e-5 --local_batch_size 8  # encoder-decoder model
python finetune.py --output_dir pAbT5_dec --dataset_dir dataset --lr 5e-5 --local_batch_size 8 \  # decoder-only ablation
    --model_type t5-decoder
```

## Licensing
The code, weights and data can only be used for non-commercial purposes.
