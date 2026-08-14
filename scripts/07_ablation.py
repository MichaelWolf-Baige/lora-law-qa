"""
07_ablation.py — Run systematic ablation experiments.

Experiments from configs/ablation_config.yaml:
  1. Rank comparison: r = 4, 8, 16, 32
  2. Data size: 500, 1000, 2000, 3000 samples
  3. Method: LoRA vs DoRA
  4. Learning rate: 5e-5, 1e-4, 2e-4, 5e-4
  5. Target modules: attention-only, all-attention, all-linear

Each experiment runs a shortened training (2 epochs, 2000 samples default),
evaluates on the test set, and saves results for comparison.

Usage:
    python scripts/07_ablation.py --experiment rank_comparison
    python scripts/07_ablation.py --experiment all  # Run ALL experiments
    python scripts/07_ablation.py --experiment method_comparison --dry_run
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


def load_ablation_config(config_path: str = "configs/ablation_config.yaml") -> dict:
    """Load ablation experiment matrix."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_command(exp_config: dict, var_name: str, var_value, output_name: str) -> list[str]:
    """Build the training command for a single ablation run."""
    fixed = exp_config["fixed"]

    cmd = [
        sys.executable, "scripts/04_train_lora.py",
        "--output_dir", f"outputs/ablation/{output_name}",
        "--no_wandb",  # Don't use WandB for ablation runs
    ]

    # Fixed parameters
    if "lora_r" in fixed:
        cmd += ["--lora_r", str(fixed["lora_r"])]
    if "use_dora" in fixed:
        cmd += ["--use_dora"] if fixed["use_dora"] else []
    if "learning_rate" in fixed:
        cmd += ["--learning_rate", str(fixed["learning_rate"])]
    if "num_train_epochs" in fixed:
        cmd += ["--epochs", str(fixed["num_train_epochs"])]
    if "max_train_samples" in fixed:
        cmd += ["--max_train_samples", str(fixed["max_train_samples"])]

    # Variable parameter
    if var_name == "target_modules":
        # target_modules is a list, needs special handling
        # We pass it via a temporary config override
        pass  # Handle separately
    elif var_name == "use_dora":
        cmd += ["--use_dora"] if var_value else []
    else:
        cmd += [f"--{var_name}", str(var_value)]

    return cmd


def run_experiment(
    exp_config: dict,
    var_name: str,
    var_values: list,
    exp_name: str,
    dry_run: bool = False,
) -> list[dict]:
    """Run a single ablation experiment across all variable values."""
    results = []
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name} ({exp_config.get('description', '')})")
    print(f"Variable: {var_name}")
    print(f"Values: {var_values}")
    print(f"{'='*60}")

    for i, value in enumerate(var_values):
        run_name = f"{exp_name}_{var_name}_{value}"
        run_name = run_name.replace("[", "").replace("]", "").replace(",", "_").replace(" ", "")
        print(f"\n--- Run {i+1}/{len(var_values)}: {var_name}={value} ---")

        # For target_modules, create a temp config
        if var_name == "target_modules":
            # Write a temporary training config
            temp_config = copy.deepcopy(exp_config["fixed"])
            temp_config["target_modules"] = value
            temp_config_path = f"outputs/ablation/temp_config_{run_name}.yaml"
            os.makedirs("outputs/ablation", exist_ok=True)
            with open(temp_config_path, "w") as f:
                yaml.dump(temp_config, f)
            cmd = [
                sys.executable, "scripts/04_train_lora.py",
                "--config", temp_config_path,
                "--output_dir", f"outputs/ablation/{run_name}",
                "--no_wandb",
            ]
        else:
            cmd = [
                sys.executable, "scripts/04_train_lora.py",
                "--output_dir", f"outputs/ablation/{run_name}",
                "--no_wandb",
            ]
            # Apply fixed params
            for k, v in exp_config["fixed"].items():
                if k == "target_modules":
                    continue  # Skip, use default
                if isinstance(v, bool):
                    if v:
                        cmd.append(f"--{k}")
                else:
                    cmd += [f"--{k}", str(v)]

            # Apply variable
            if var_name == "use_dora":
                if value:
                    cmd.append("--use_dora")
            else:
                cmd += [f"--{var_name}", str(value)]

        print(f"  Command: {' '.join(cmd)}")

        if dry_run:
            print("  [DRY RUN] Skipping execution")
            results.append({
                "run_name": run_name,
                "variable": var_name,
                "value": str(value),
                "status": "skipped",
            })
            continue

        t0 = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            elapsed = time.time() - t0

            if result.returncode == 0:
                print(f"  ✅ Completed in {elapsed:.0f}s")
                status = "success"
            else:
                print(f"  ❌ Failed (exit code {result.returncode})")
                print(f"  STDERR: {result.stderr[-500:]}")
                status = "failed"

        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout (>1 hour)")
            status = "timeout"
        except Exception as e:
            print(f"  ❌ Error: {e}")
            status = "error"

        results.append({
            "run_name": run_name,
            "variable": var_name,
            "value": str(value),
            "status": status,
            "elapsed": time.time() - t0,
        })

    return results


def generate_ablation_report(all_results: dict, output_dir: str):
    """Generate a summary report of all ablation experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "experiments": all_results,
        "findings": {},
    }

    # Generate findings for each experiment
    for exp_name, runs in all_results.items():
        successful = [r for r in runs if r["status"] == "success"]
        if len(successful) < 2:
            report["findings"][exp_name] = "Insufficient successful runs for comparison"
            continue

        # Collect metrics from each run's output
        run_metrics = []
        for run in successful:
            # Try to read training metadata
            metadata_path = Path(
                f"outputs/ablation/{run['run_name']}/lora_weights/{run['run_name']}/training_metadata.json"
            )
            # Try alternative path
            alt_path = Path(f"outputs/ablation/{run['run_name']}/training_metadata.json")
            for mp in [metadata_path, alt_path]:
                if mp.exists():
                    with open(mp) as f:
                        meta = json.load(f)
                    run_metrics.append({
                        "value": run["value"],
                        "trainable_params": meta.get("trainable_params", 0),
                        "trainable_ratio": meta.get("trainable_ratio", "N/A"),
                    })
                    break

        if run_metrics:
            report["findings"][exp_name] = {
                "description": f"Compared {len(run_metrics)} values of {successful[0]['variable']}",
                "metrics": run_metrics,
            }

    # Save report
    report_path = output_dir / "ablation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Ablation report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    for exp_name, runs in all_results.items():
        successful = [r for r in runs if r["status"] == "success"]
        failed = [r for r in runs if r["status"] != "success"]
        print(f"\n  {exp_name}:")
        print(f"    Total runs: {len(runs)}")
        print(f"    Successful: {len(successful)}")
        if failed:
            print(f"    Failed: {len(failed)}")
        print(f"    Values tested: {[r['value'] for r in runs]}")


def main():
    parser = argparse.ArgumentParser(description="Run LoRA ablation experiments")
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "rank_comparison", "data_size", "method_comparison",
                 "learning_rate", "target_modules"],
        help="Which experiment to run"
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation")
    args = parser.parse_args()

    ablation_cfg = load_ablation_config()
    all_results = {}

    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = ablation_cfg["experiments"]
    else:
        for exp in ablation_cfg["experiments"]:
            if exp["name"] == args.experiment:
                experiments_to_run = [exp]
                break

    if not experiments_to_run:
        print(f"ERROR: Experiment '{args.experiment}' not found in config")
        sys.exit(1)

    for exp in experiments_to_run:
        results = run_experiment(
            exp_config=exp,
            var_name=exp["variable"],
            var_values=exp["values"],
            exp_name=exp["name"],
            dry_run=args.dry_run,
        )
        all_results[exp["name"]] = results

        # Small pause between experiments to let GPU cool down
        if not args.dry_run and len(experiments_to_run) > 1:
            print("\n  Cooling down for 10 seconds...")
            time.sleep(10)

    # Generate report
    generate_ablation_report(all_results, args.output_dir)


if __name__ == "__main__":
    main()
