import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import List


BEST_F1_PATTERN = re.compile(r"Best val_f1=([0-9]*\.?[0-9]+)")


@dataclass
class TrialResult:
    command: List[str]
    return_code: int
    best_f1: float
    log_path: str


def run_trial(cmd: List[str], log_path: str) -> TrialResult:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    best_f1 = -1.0
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        m = BEST_F1_PATTERN.search(content)
        if m:
            best_f1 = float(m.group(1))

    return TrialResult(command=cmd, return_code=proc.returncode, best_f1=best_f1, log_path=log_path)


def main():
    parser = argparse.ArgumentParser(description="One-click strategy search for best validation F1.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="splits/train_split.json")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for each trial.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="search_runs")
    parser.add_argument("--max_trials", type=int, default=0, help="0 means run all.")
    args = parser.parse_args()

    presets = ["baseline", "full_strong", "baseline4", "srresnet_baseline"]
    optimizers = ["adamw", "adam", "sgd"]
    schedulers = ["cosine", "onecycle"]
    aug_flags = [
        [],
        ["--use_hflip"],
        ["--use_hflip", "--use_affine"],
        ["--use_hflip", "--use_affine", "--use_color_jitter"],
    ]

    all_trials = list(itertools.product(presets, optimizers, schedulers, aug_flags))
    if args.max_trials > 0:
        all_trials = all_trials[:args.max_trials]

    results: List[TrialResult] = []
    print(f"Total trials: {len(all_trials)}")
    for idx, (preset, optimizer, scheduler, aug) in enumerate(all_trials, start=1):
        run_name = f"{idx:03d}_{preset}_{optimizer}_{scheduler}_aug{len(aug)}"
        log_path = os.path.join(args.out_dir, "logs", run_name + ".log")
        ckpt_dir = os.path.join(args.out_dir, "ckpts", run_name)
        cmd = [
            sys.executable, "train.py",
            "--exp_preset", preset,
            "--optimizer", optimizer,
            "--scheduler", scheduler,
            "--data_root", args.data_root,
            "--split", args.split,
            "--num_classes", str(args.num_classes),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--size", str(args.size),
            "--ckpt_dir", ckpt_dir,
        ] + aug

        print(f"[{idx}/{len(all_trials)}] Running: {' '.join(cmd)}")
        result = run_trial(cmd, log_path=log_path)
        print(f"  -> rc={result.return_code}, best_f1={result.best_f1:.4f}, log={result.log_path}")
        results.append(result)

    success_results = [r for r in results if r.return_code == 0 and r.best_f1 >= 0]
    if not success_results:
        print("No successful trials found.")
        return

    success_results.sort(key=lambda x: x.best_f1, reverse=True)
    best = success_results[0]
    print("\n=== Search finished ===")
    print(f"Best F1: {best.best_f1:.4f}")
    print(f"Best command: {' '.join(best.command)}")
    print(f"Best log: {best.log_path}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best": asdict(best),
                "results": [asdict(r) for r in success_results],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
