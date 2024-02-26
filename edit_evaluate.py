import json
import os
import shutil
import pandas as pd
from pathlib import Path
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer

from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from baselines.ike.ike import construct_icl_examples

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

RESULTS_DIR = Path("./results")
DATA_DIR = Path("./data")
HPARAMS_DIR = Path("hparams")

with open('./baselines/ike/corpus_idx.txt', 'r') as fIn:
    lines = fIn.readlines()
    lines = [line[:-1] for line in lines]
    corpus_idx = [[int(idx) for idx in line.split()] for line in lines]

def memory_generate(model, tok, prompt):
    input_ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
    attention_masks = torch.ones(input_ids.shape[1] + model.num_blocks * model.num_tokens, dtype=torch.long).unsqueeze(0).to('cuda')
    return tok.decode(model.generate(input_ids, attention_mask=attention_masks)[0], max_new_tokens=20)

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    conserve_memory: bool,
    split_model: bool,
    dir_name: str,
):

    # Determine run directory
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert (
            run_dir.exists()
        ), f"If continuing from run, {continue_from_run} must exist!"
    else:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    
    if hparams_fname is None:
        hparams = None
        hparams_dict = {"model": model_name, 'ds_name': ds_name}
        with open(run_dir / "params.json", "w") as f:
            json.dump(hparams_dict, f, indent=1)
    
    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )

    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    print("Instantiating model")
    if type(model_name) is str:

        if model_name == 'llama2-7b':
            if split_model:
                model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map='auto')
            else:
                model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").cuda()
            tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name

    model.eval()
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    # vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]

    if alg_name == 'IKE':
        ds = ds_class(DATA_DIR, tok=tok)
        demos = ds[2001:]
        example_idx = 0

    else:
        ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)

    # Iterate through dataset
    for record in ds:

        case_id = record["case_id"]
        case_result_path = run_dir / f"case_{case_id}.json"
        if ds_name == 'cf':
            if str(case_id) == '1531': continue
            if str(case_id) == '2001': break
        else:
            if str(case_id) == '10001': break
        if not case_result_path.exists():
            # Compute weight changes + record weights that changed
            start = time()
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )

            if alg_name == 'IKE':
                
                metrics = {
                    "case_id": case_id,
                    "requested_rewrite": record["requested_rewrite"],
                    "pre": ds_eval_method(model, tok, record, snips),
                }

                prompt = record['requested_rewrite']['prompt']
                prompt = record['requested_rewrite']['prompt']
                subject = record['requested_rewrite']['subject']
                prompt_calibrate = prompt.format('SUBJECT')
                prompt = prompt.format(subject)

                target_true = record['requested_rewrite']['target_true']['str']
                target_new = record['requested_rewrite']['target_new']['str']
                
                icl_examples = construct_icl_examples(example_idx, demos, corpus_idx)
                icl_examples.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_new}\n\n')
                example_idx += 1

                # add every icl example to the prompt
                # record['requested_rewrite']['prompt'] = ''.join(icl_examples) + record['requested_rewrite']['prompt']
                record['requested_rewrite']['prompt'] = ''.join(icl_examples) + prompt
                record["paraphrase_prompts"] = [''.join(icl_examples) + x for x in record["paraphrase_prompts"]]
                record['neighborhood_prompts'] = [''.join(icl_examples) + x for x in record['neighborhood_prompts']]

                metrics['post'] = ds_eval_method(model, tok, record, snips)

            else:
                raise NotImplementedError

            print("Evaluation took", time() - start)

            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["Memory", "IKE"],
        default="IKE",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="llama2-7b",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default=None,
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--split_model",
        dest="split_model",
        action="store_true",
        help="Split model into all gpus",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_known_args()[0]

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.conserve_memory,
        args.split_model,
        dir_name=args.alg_name,
    )
