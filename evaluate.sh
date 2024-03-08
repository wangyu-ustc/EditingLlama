source ~/.bashrc
conda activate rome

python -m experiments.summarize --dir_name FT --runs run_001 --first_n_cases 2001
python -m experiments.evaluate --alg_name=ROME --model_name=llama2-7b --hparams_fname=llama2-7b.json
python -m experiments.unlearn --alg_name=ROME --model_name=llama2-7b --hparams_fname=llama2-7b-3.json