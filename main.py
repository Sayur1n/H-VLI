import argparse
import os
import json
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
import time

from utils import load_and_sample_data, save_json, save_txt
from llm_client import LLMClient
from court_system import CourtroomSystem
from evaluator import calculate_metrics, get_metrics_report

MODELS_TO_RUN = ['qwen3-vl-plus']
AUX_MODEL = "qwen3-vl-plus" 

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_PATH = os.path.join(BASE_PATH, "data/test_set.json")
ANSWER_OUTPUT_PATH = "answers_system"
DATASETNAME = "mmhs"

def process_single_arcade_sample(target_model, aux_model, sid, item, courtroom, args):
    """
    Process a single sample using the ARCADE layered debate logic.
    """
    try:
        full_record = {}
        hier_out = courtroom.run_arcade(target_model, aux_model, sid, item, rounds=args.rounds)
        
        p_res = hier_out.get("prosecutor", [])
        d_res = hier_out.get("defense", [])
        j_res = hier_out.get("verdict", {})
        stage = hier_out.get("final_stage", "unknown")
        
        if not isinstance(j_res, dict):
            j_res = {"label": -1, "reason": str(j_res), "error": "Judge output not dict"}

        full_record.update(j_res if isinstance(j_res, dict) else {"raw": j_res})
        full_record.update({
            "prosecutor_log": p_res,
            "defender_log": d_res,
            "hierarchical_stage": stage,
            "hierarchical_details": hier_out.get("process_details", {}),
            "final_label": item.get("final_label"),
            "source": item.get("source")
        })

        return sid, p_res, d_res, full_record

    except Exception as e:
        err_res = {"error": str(e), "refusal": True, "label": -1}
        return sid, {}, {}, err_res

def process_single_none_sample(model, sid, item, courtroom):
    """
    Process a single sample using the simple baseline (direct classification).
    """
    try:
        res = courtroom.run_baseline_none(model, sid, item)
        res.update({
            "final_label": item.get("final_label"),
            "source": item.get("source")
        })
        return sid, res
    except Exception as e:
        return sid, {"error": str(e), "refusal": True, "label": -1}

def main():
    """
    Main entry point for running ARCADE or Baseline experiments.
    """
    parser = argparse.ArgumentParser(description="ARCADE: Courtroom Multi-Agent Debate System")
    parser.add_argument("--run_mode", type=str, choices=['ARCADE', 'none'], default='ARCADE', help="ARCADE: Layered Debate; none: Baseline")
    parser.add_argument("--class_mode", type=str, choices=['multiclass', 'binary'], default='multiclass', help="Classification standard")
    parser.add_argument("--samples", "-s", type=int, default=10, help="Number of samples to run (0 for all)")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for sampling")
    parser.add_argument("--threads", type=int, default=16, help="Concurrency threads")
    parser.add_argument("--rounds", type=int, default=3, help="Debate rounds in Indirect Stage")
    parser.add_argument("--fill_cache", action="store_true", help="Skip existing samples in cache")
    
    args = parser.parse_args()

    data_map, _ = load_and_sample_data(INPUT_DATA_PATH, args.samples, args.seed)

    client = LLMClient()
    courtroom = CourtroomSystem(client, BASE_PATH, dataset_name=DATASETNAME, class_mode=args.class_mode)
    
    output_base = os.path.join(BASE_PATH, ANSWER_OUTPUT_PATH, args.class_mode)
    os.makedirs(output_base, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in MODELS_TO_RUN:
        print(f"\nTarget Model: {model} | Mode: {args.run_mode}")
        results_dir = os.path.join(output_base, args.run_mode, timestamp, model)
        os.makedirs(results_dir, exist_ok=True)
        final_results = {}

        if args.run_mode == 'none':
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
                futures = {executor.submit(process_single_none_sample, model, sid, item, courtroom): sid for sid, item in data_map.items()}
                for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="None Mode"):
                    sid, res = f.result()
                    final_results[sid] = res
        else:
            tasks = data_map
            if args.fill_cache:
                tasks = {sid: item for sid, item in data_map.items() if not courtroom.check_is_completed(AUX_MODEL, 'ARCADE', sid, args.rounds)}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
                futures = {executor.submit(process_single_arcade_sample, model, AUX_MODEL, sid, item, courtroom, args): sid for sid, item in tasks.items()}
                for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="ARCADE Mode"):
                    sid, pr, dr, full = f.result()
                    final_results[sid] = full

        save_json(final_results, os.path.join(results_dir, f"results_{model}.json"))
        metrics = calculate_metrics(final_results, data_map, mode=args.class_mode)
        report = get_metrics_report(model, args.run_mode, metrics, class_mode=args.class_mode)
        print(report)
        save_txt(report, os.path.join(results_dir, "report.txt"))

if __name__ == "__main__":
    main()
