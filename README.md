# Model-editing baselines implementation

This repo contains the implementations of all the model-editing baselines in the paper **MemoryLLM: Towards Self-Updatable Large Language Models**. We open a sperate folder for model-editing implementations for a clear structure of the code. This repo is primarily based on the repo [rome](https://github.com/kmeng01/rome/tree/main), and the IKE baseline implementation is copied from [IKE](https://github.com/Zce1112zslx/IKE). 

After installing the environment according to the instructions in [rome](https://github.com/kmeng01/rome/tree/main), please update the package `transformers` to the most up-to-date version (We used 4.35.2 when doing the experiments), and install the package `accelerate-0.27.2`. 

# Running the experiments
To run several baselins on ZsRE, use the following code: 
```
# Baseline: FT
python -m experiments.evaluate --alg_name=FT --model_name=llama2-7b --hparams_fname=llama2-7B_unconstr.json --dataset_size_limit 10000 --ds_name zsre
python -m experiments.summarize --dir_name FT --runs run_000 # adjust the variable "--runs" accordingly

# Baseline: FT-L
python -m experiments.evaluate --alg_name=FT --model_name=llama2-7b --hparams_fname=llama2-7B_constr.json --dataset_size_limit 10000 --ds_name zsre
python -m experiments.summarize --dir_name FT --runs run_001 # adjust the variable "--runs" accordingly

# Baseline: ROME
python -m experiments.evaluate --alg_name=ROME --model_name=llama2-7b --hparams_fname=llama2-7b.json --dataset_size_limit 10000 --ds_name zsre
python -m experiments.summarize --dir_name ROME --runs run_000 # adjust the variable "--runs" accordingly
```

To run the baselines on CF, use the following command:
```
# Baseline: FT
python -m experiments.evaluate --alg_name=FT --model_name=llama2-7b --hparams_fname=llama2-7B_unconstr.json --dataset_size_limit 10000 --ds_name cf
python -m experiments.summarize --dir_name FT --runs run_000 # adjust the variable "--runs" accordingly

# Baseline: FT-L
python -m experiments.evaluate --alg_name=FT --model_name=llama2-7b --hparams_fname=llama2-7B_constr.json --dataset_size_limit 10000 --ds_name cf
python -m experiments.summarize --dir_name FT --runs run_001 # adjust the variable "--runs" accordingly

# Baseline: ROME
python -m experiments.evaluate --alg_name=ROME --model_name=llama2-7b --hparams_fname=llama2-7b.json --dataset_size_limit 10000 --ds_name cf
python -m experiments.summarize --dir_name ROME --runs run_000 # adjust the variable "--runs" accordingly

# Baseline: IKE
python edit_evaluate.py --alg_name IKE --ds_name cf  --model_name llama2-7b
python edit_summarize.py --dir_name IKE --runs run_001 # adjust the parameter "--runs"
```