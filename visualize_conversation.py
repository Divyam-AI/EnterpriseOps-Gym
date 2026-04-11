"""
Generate a Mermaid gitGraph for a single task trajectory.

Each branch is a distinct LLM (x-routed-model). Each commit is one ai_message turn.
Tool-calling turns are labelled with the tool names; the final turn is "Done".

Usage:
    python visualize_conversation.py <results_folder>
    python visualize_conversation.py <results_folder> --task <task_id>
"""

import argparse
import glob
import json
import os


def _commit_label(turn: dict) -> str:
    tools = turn.get("tools", [])
    if not tools:
        return "Done"
    label = ", ".join(tools[:3])
    if len(tools) > 3:
        label += f" +{len(tools) - 3}"
    return label


def conversation_to_gitgraph(conversation_flow: list) -> str:
    turns = []
    for msg in conversation_flow:
        if msg.get("type") != "ai_message":
            continue
        model = (
            (msg.get("response_metadata") or {})
            .get("divyam_response_headers", {})
            .get("x-routed-model")
            or (msg.get("response_metadata") or {}).get("model_name")
            or "unknown"
        )
        tools = [tc["name"] for tc in (msg.get("tool_calls") or [])]
        turns.append({"model": model, "tools": tools})

    if not turns:
        return ""

    main_model = turns[0]["model"]

    lines = [
        f"%%{{init: {{ 'gitGraph': {{ 'mainBranchName': '{main_model}' }} }} }}%%",
        "gitGraph",
    ]

    current_model = main_model
    known_branches = {main_model}

    for turn in turns:
        model = turn["model"]

        if model != current_model:
            if model not in known_branches:
                lines.append(f"   branch {model}")
                known_branches.add(model)
            else:
                lines.append(f"   checkout {model}")
            current_model = model

        label = _commit_label(turn)
        lines.append(f'   commit id: "{label}"')

    return "\n".join(lines)


def collect_tasks(results_folder: str) -> dict[str, str]:
    """Return {task_id: file_path} for all result files under results_folder."""
    tasks = {}
    for path in sorted(glob.glob(os.path.join(results_folder, "**", "results_*.json"), recursive=True)):
        # Extract task_id from filename: results_<mode>__<domain>__<task_id>.json
        basename = os.path.splitext(os.path.basename(path))[0]  # drop .json
        parts = basename.split("__", 2)
        task_id = parts[2] if len(parts) == 3 else basename
        tasks[task_id] = path
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", help="Folder containing results_*.json files")
    parser.add_argument("--task", help="Task ID to visualize (omit to list all)")
    parser.add_argument("--run", type=int, default=0, help="Run index (default: 0)")
    args = parser.parse_args()

    tasks = collect_tasks(args.results_folder)
    if not tasks:
        print(f"No result files found under: {args.results_folder}")
        return

    if not args.task:
        print(f"Found {len(tasks)} task(s). Re-run with --task <id>:\n")
        for task_id in tasks:
            print(f"  {task_id}")
        return

    if args.task not in tasks:
        print(f"Task '{args.task}' not found. Available tasks:")
        for task_id in tasks:
            print(f"  {task_id}")
        return

    with open(tasks[args.task]) as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if args.run >= len(runs):
        print(f"Run index {args.run} out of range ({len(runs)} run(s) available).")
        return

    cf = runs[args.run].get("conversation_flow", [])
    if not cf:
        print("No conversation_flow in this run (likely errored).")
        return

    print(conversation_to_gitgraph(cf))


if __name__ == "__main__":
    main()
