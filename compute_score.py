import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict

import yaml

_RATE_CARD_PATH = os.path.join(os.path.dirname(__file__), "conf", "llm", "rate_card.yaml")

def load_rate_card(path=_RATE_CARD_PATH):
    """Load rate card YAML. Returns dict: model -> most-recent pricing entry."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Each model has a list of tiers sorted by effective_start_date; pick the latest.
    return {
        model: sorted(tiers, key=lambda t: t["effective_start_date"])[-1]
        for model, tiers in raw.items()
    }

def compute_cost(input_tokens, output_tokens, rate):
    """Return cost in USD given token counts and a rate card entry."""
    n = rate["per_n_tokens"]
    return (input_tokens / n) * rate["text_input_price"] + \
           (output_tokens / n) * rate["text_output_price"]

true = True
false = False
null = None

def get_score(sample):
    return (sample['statistics']['overall_success_rate'], sample['statistics']['verifier_level_pass_rate'])

def process_mode(results_folder, mode):
    file_list = []
    for root, _, files in os.walk(results_folder):
        for file in files:
            if file.endswith('.json'):
                file_list.append(os.path.relpath(os.path.join(root, file), results_folder))

    overall_success_rates = []
    verifier_level_pass_rates = []
    errors = []

    for idx, file_name in enumerate(tqdm(file_list, desc=f"Processing {mode}")):
        if not file_name.endswith('.json'):
            continue
        file_path = os.path.join(results_folder, file_name)
        with open(file_path, 'r') as f:
            # if '+5' not in file_path.lower():
            #     continue
            file_content = f.read()
            result_data = json.loads(file_content)
            has_error = any(run.get("error") for run in result_data.get("runs", []))
            # Discard retry artifacts: files where the run hit HTTP 429 rate limits.
            # The non-429 (successful or otherwise) attempt for the same task is retained.
            if any("429" in str(run.get("error") or "") for run in result_data.get("runs", [])):
                continue
        overall_success_rate, verifier_level_pass_rate = get_score(result_data)
        overall_success_rates.append(overall_success_rate)
        verifier_level_pass_rates.append(verifier_level_pass_rate)
        errors.append(1 if has_error else 0)

    avg_overall_success_rate = sum(overall_success_rates) / len(overall_success_rates) if overall_success_rates else 0
    avg_verifier_level_pass_rate = sum(verifier_level_pass_rates) / len(verifier_level_pass_rates) if verifier_level_pass_rates else 0

    return {
        'mode': mode,
        'total_files': len(overall_success_rates),
        'files_with_errors': sum(errors),
        'avg_overall_success_rate': avg_overall_success_rate * 100.0,
        'avg_verifier_pass_rate': avg_verifier_level_pass_rate * 100.0
    }


def get_token_usage(results_folder):
    """Aggregate input/output token counts per routed model (excludes 429 cases)."""
    usage_by_model = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0})

    for root, _, files in os.walk(results_folder):
        for file in sorted(files):
            if not file.endswith('.json'):
                continue
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                result_data = json.load(f)

            has_429 = any("429" in str(run.get("error") or "") for run in result_data.get("runs", []))
            if has_429:
                continue

            for run in result_data.get("runs", []):
                for msg in run.get("conversation_flow", []):
                    if msg.get("type") != "ai_message":
                        continue
                    usage = msg.get("usage_metadata") or {}
                    if not usage:
                        continue
                    routed_model = (
                        msg.get("response_metadata", {})
                           .get("divyam_response_headers", {})
                           .get("x-routed-model", "unknown")
                    )
                    usage_by_model[routed_model]["input_tokens"]  += usage.get("input_tokens", 0)
                    usage_by_model[routed_model]["output_tokens"] += usage.get("output_tokens", 0)
                    usage_by_model[routed_model]["total_tokens"]  += usage.get("total_tokens", 0)
                    usage_by_model[routed_model]["calls"]         += 1

    return usage_by_model


def get_traffic_split(results_folder):
    """Extract traffic split from x-routed-model header (excludes 429 cases)."""
    traffic_split = defaultdict(int)
    total_calls = 0

    for root, _, files in os.walk(results_folder):
        for file in sorted(files):
            if not file.endswith('.json'):
                continue
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                result_data = json.load(f)

            # Skip 429 cases
            has_429 = any("429" in str(run.get("error") or "") for run in result_data.get("runs", []))
            if has_429:
                continue

            # Extract routed model from conversation_flow
            for run in result_data.get("runs", []):
                conv_flow = run.get("conversation_flow", [])
                for msg in conv_flow:
                    if msg.get("type") == "ai_message":
                        metadata = msg.get("response_metadata", {})
                        divyam_headers = metadata.get("divyam_response_headers", {})
                        routed_model = divyam_headers.get("x-routed-model", "unknown")
                        traffic_split[routed_model] += 1
                        total_calls += 1

    return traffic_split, total_calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', type=str, required=True, help='Path to the input JSON file')

    args = parser.parse_args()
    results_folder = args.results_folder

    # Collect results from all modes
    all_results = []

    # For each mode (folder) in results_folder
    for mode in sorted(os.listdir(results_folder)):
        mode_folder = os.path.join(results_folder, mode)
        if os.path.isdir(mode_folder):
            result = process_mode(mode_folder, mode)
            all_results.append(result)

    # Print results as a nice table
    if all_results:
        print("\n" + "="*100)
        print("FINAL RESULTS")
        print("="*100 + "\n")

        headers = ["Mode", "Total Files", "Files w/ Errors", "Avg Success Rate (%)", "Avg Verifier Pass (%)"]
        table_data = []

        for result in all_results:
            row = [
                result['mode'],
                result['total_files'],
                f"\033[91m{result['files_with_errors']}\033[0m" if result['files_with_errors'] > 0 else "0",
                f"{result['avg_overall_success_rate']:.2f}",
                f"{result['avg_verifier_pass_rate']:.2f}"
            ]
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()

    # Report token usage
    usage_by_model = get_token_usage(results_folder)
    if usage_by_model:
        print("="*100)
        print("TOKEN USAGE (non-429 calls)")
        print("="*100 + "\n")

        try:
            rate_card = load_rate_card()
        except FileNotFoundError:
            rate_card = {}

        total_input = total_output = total_tokens = total_calls_all = 0
        total_cost = 0.0
        usage_data = []
        for model in sorted(usage_by_model.keys()):
            u = usage_by_model[model]
            rate = rate_card.get(model)
            if rate:
                cost = compute_cost(u["input_tokens"], u["output_tokens"], rate)
                cost_str = f"${cost:.4f}"
                total_cost += cost
            else:
                cost_str = "N/A"
            usage_data.append([model, u["calls"], f"{u['input_tokens']:,}", f"{u['output_tokens']:,}", f"{u['total_tokens']:,}", cost_str])
            total_input     += u["input_tokens"]
            total_output    += u["output_tokens"]
            total_tokens    += u["total_tokens"]
            total_calls_all += u["calls"]
        usage_data.append(["TOTAL", total_calls_all, f"{total_input:,}", f"{total_output:,}", f"{total_tokens:,}", f"${total_cost:.4f}"])

        print(tabulate(usage_data, headers=["Model", "Calls", "Input Tokens", "Output Tokens", "Total Tokens", "Cost ($)"], tablefmt="grid"))
        print()

    # Report traffic split
    traffic_split, total_calls = get_traffic_split(results_folder)
    if traffic_split and total_calls > 0:
        print("="*100)
        print("TRAFFIC SPLIT (x-routed-model, non-429 calls)")
        print("="*100 + "\n")

        traffic_data = []
        for model in sorted(traffic_split.keys()):
            count = traffic_split[model]
            pct = 100 * count / total_calls
            traffic_data.append([model, count, f"{pct:.1f}%"])

        print(tabulate(traffic_data, headers=["Model", "Calls", "Percentage"], tablefmt="grid"))
        print()


if __name__ == "__main__":
    main()