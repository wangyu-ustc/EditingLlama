import json
import os
import shutil
import pandas as pd
from pathlib import Path
from time import time
from typing import Tuple, Union
from scipy.linalg import svd
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from baselines.ft import FTHyperParams, apply_ft_to_model
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model, apply_unlearn_rome_to_model
from util import nethook
from util.globals import *
from rome.compute_existing_u import get_inv_cov

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_unlearn_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    conserve_memory: bool,
    no_roll_back: bool,
    batch_edit: bool,
    batch_edit_num: int,
    dir_name: str,
):
    if not alg_name == 'Memory':
        # Set algorithm-specific variables
        params_class, apply_algo = ALG_DICT[alg_name]

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
        hparams_dict = {"model": model_name}
        with open(run_dir / "params.json", "w") as f:
            json.dump(hparams_dict, f, indent=1)

    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
        hparams = params_class.from_json(params_path)
        if not (run_dir / "params.json").exists():
            # shutil.copyfile(params_path, run_dir / "params.json")
            # dump to run_dir / "params.json"

            with open(params_path, "r") as f:
                hparams_dict = json.load(f)
            with open(run_dir / "params.json", "w") as f:
                hparams_dict['model'] = model_name
                hparams_dict['ds_name'] = ds_name
                json.dump(hparams_dict, f, indent=1)
        
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    print("Instantiating model")
    if type(model_name) is str:
        if 'llama' in model_name:
            from transformers import LlamaForCausalLM, LlamaTokenizer
            if model_name == 'openllama-3b':
                model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2").cuda()
                tok = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
            elif model_name == 'llama2-7b':
                model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").cuda()
                tok = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
            tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    # vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)

    Cs = {}
    for layer in hparams.layers:
        Cs[layer] = get_inv_cov(
                    model,
                    tok,
                    hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples,
                    hparams.mom2_dtype,
                )


    all_indices = []

    if batch_edit:
        all_k_stars = []
        all_v_stars = []

    # Iterate through dataset
    for record in ds:

        # replace 'target_new' with 'target_true'
        record['requested_rewrite']['target_new'] = record['requested_rewrite']['target_true']

        case_id = record["case_id"]
        case_result_path = run_dir / f"case_{case_id}.json"

        # case_id 1531 could cause OOM in A100-40GB
        if ds_name == 'cf' and str(case_id) == '1531': continue

        if not case_result_path.exists():
            # Compute weight changes + record weights that changed
            # Execute evaluation suite
            start = time()

            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )

            outputs = apply_algo(
                model,
                tok,
                [record["requested_rewrite"]],
                hparams,
                copy=False,
                return_orig_weights=True,
                C=Cs,
                return_kstar=(no_roll_back or batch_edit),
                **args_conserve_memory,
            )

            if no_roll_back or batch_edit:
                edited_model, weights_copy, k_stars, v_stars = outputs
                if batch_edit:
                    for key, value in k_stars.items():
                        k_stars[key] = value.detach().cpu().numpy()
                    for key, value in v_stars.items():
                        v_stars[key] = value.detach().cpu().numpy()
                    all_k_stars.append(k_stars)
                    all_v_stars.append(v_stars)
            else:
                edited_model, weights_copy = outputs
            
            exec_time = time() - start
            metrics = {
                "case_id": case_id,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(edited_model, tok, record, snips) if not batch_edit else None,
            }
            print("Execution took", exec_time)

            if no_roll_back:
                for key, value in k_stars.items():
                    Cs[key] = Cs[key] - value.unsqueeze(1) @ value.unsqueeze(0)

            else:
                # roll back to original weights
                for w_name, weights in weights_copy.items():
                    w = nethook.get_parameter(model, w_name)
                    w.data = weights
                    # TODO: not sure why w[...].data = weights is not working

            # "pre": ds_eval_method(model, tok, record, snips)
            metrics['pre'] = ds_eval_method(model, tok, record, snips)

            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)

        if batch_edit and (case_id + 1) % batch_edit_num == 0:
                
        # # debug:
        # if batch_edit:

            k_stars = {}
            v_stars = {}

            for key in all_k_stars[0]:
                k_stars[key] = np.stack([x[key] for x in all_k_stars])

            for key in all_v_stars[0]:
                v_stars[key] = np.stack([x[key] for x in all_v_stars])

            for w_name, weights in weights_copy.items():

                w = nethook.get_parameter(model, w_name)

                weights = weights.cpu()
                k_star = torch.tensor(k_stars[eval(w_name.split(".")[2])])
                v_star = torch.tensor(v_stars[eval(w_name.split(".")[2])])
                C = Cs[eval(w_name.split(".")[2])].cpu()

                # edit weights with k_stars and v_stars
                # new_weights = (weights[weight_name] @ C - v_star.unsqueeze(1) @ k_star.unsqueeze(0)) @ torch.inverse(
                #     C - k_star.unsqueeze(1) @ k_star.unsqueeze(0)
                # )

                new_weights = (weights @ C - v_star.T @ k_star) @ torch.inverse(
                    C - k_star.T @ k_star
                )

                w.data = new_weights.to(w.data.device)

        # if (case_id + 1) % 100 == 0:

            import ipdb; ipdb.set_trace()

            for w_name, _ in weights_copy.items():
                
                weight = nethook.get_parameter(model, w_name).data.cpu().numpy()

                # SVD decomposition:
                # Compute SVD
                U, s, Vt = svd(weight)
            
                s = s/s.max()

                # Determine a reasonable k for approximation
                for i in range(len(s)):
                    if sum(s[:i+1]) / sum(s) >= 0.99:
                        break
                
                print(f"99% Singular values: {i}")
                all_indices.append(i)

            import ipdb; ipdb.set_trace()

            all_indices = []

    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["memory-openllama", "gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", 'openllama-3b', 'llama-7b', 'llama2-7b'],
        default="gpt2-xl",
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
        '--no_roll_back',
        default=False,
        action='store_true',
        help="if set true, then don't roll back after every editing"
    )
    parser.add_argument(
        '--batch_edit',
        default=False,
        action='store_true',
        help='if set true, then store k_star in the results'
    )
    parser.add_argument(
        '--batch_edit_num',
        default=100,
        type=int,
        help='the number of edits to perform at one time'
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.conserve_memory,
        args.no_roll_back,
        args.batch_edit,
        args.batch_edit_num,
        dir_name=args.alg_name,
    )
