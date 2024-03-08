import collections
import json
from pprint import pprint
from typing import List, Optional
from pathlib import Path
import numpy as np
from scipy.stats import hmean


RESULTS_DIR = Path("./results")
DATA_DIR = Path("./data")

def main(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))

        rewrite_prompts_probs = []
        paraphrase_prompts_probs = []
        neighborhood_prompts_probs = []

        rewrite_prompts_entropies = []
        paraphrase_prompts_entropies = []
        neighborhood_prompts_entropies = []

        # put all these into cur_sum

        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            cur_sum['time'].append(0 if 'time' not in data else data['time'])

            # if data['pre']['rewrite_prompts_probs'][0]['target_new'] < 5:
            #     continue

            cur_sum['pre_rewrite_prompts_probs'].append(np.mean([x['target_new'] for x in data['pre']['rewrite_prompts_probs']]))
            cur_sum['post_rewrite_prompts_probs'].append(np.mean([x['target_new'] for x in data['post']['rewrite_prompts_probs']]))
            cur_sum['pre_rewrite_prompts_entropies'].append(np.mean([x['target_new'] for x in data['pre']['rewrite_prompts_entropies']]))
            cur_sum['post_rewrite_prompts_entropies'].append(np.mean([x['target_new'] for x in data['post']['rewrite_prompts_entropies']]))

            cur_sum['pre_paraphrase_prompts_probs'].append(np.mean([x['target_new'] for x in data['pre']['paraphrase_prompts_probs']]))
            cur_sum['post_paraphrase_prompts_probs'].append(np.mean([x['target_new'] for x in data['post']['paraphrase_prompts_probs']]))
            cur_sum['pre_paraphrase_prompts_entropies'].append(np.mean([x['target_new'] for x in data['pre']['paraphrase_prompts_entropies']]))
            cur_sum['post_paraphrase_prompts_entropies'].append(np.mean([x['target_new'] for x in data['post']['paraphrase_prompts_entropies']]))

            cur_sum['pre_neighborhood_prompts_probs'].append(np.mean([x['target_new'] for x in data['pre']['neighborhood_prompts_probs']]))
            cur_sum['post_neighborhood_prompts_probs'].append(np.mean([x['target_new'] for x in data['post']['neighborhood_prompts_probs']]))
            cur_sum['pre_neighborhood_prompts_entropies'].append(np.mean([x['target_new'] for x in data['pre']['neighborhood_prompts_entropies']]))
            cur_sum['post_neighborhood_prompts_entropies'].append(np.mean([x['target_new'] for x in data['post']['neighborhood_prompts_entropies']]))

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}

        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    args = parser.parse_args()

    main(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
    )
