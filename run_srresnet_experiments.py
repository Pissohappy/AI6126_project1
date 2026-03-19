#!/usr/bin/env python
"""
SRResNet Grid Search Experiment Script

This script runs a phased grid search for SRResNet-based face segmentation.
It supports:
- Dual GPU parallel execution
- Checkpoint/resume functionality
- Phased search (each phase builds on the best config from previous phase)

Usage:
    python run_srresnet_experiments.py --epochs 50 --gpu_ids 1,3
    python run_srresnet_experiments.py --phase 1 --max_trials 2 --epochs 5  # Quick test
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Tuple

BEST_F1_PATTERN = re.compile(r"Best val_f1=([0-9]*\.?[0-9]+)")
PARAM_LIMIT = 1_821_085


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    phase: int
    name: str
    base_ch: int = 32
    num_blocks: int = 16
    lr: float = 2e-4
    loss_type: str = "baseline"  # baseline, dice, focal, focal_dice, ohem_dice, focal_ohem_dice
    aug_flags: List[str] = field(default_factory=list)
    balance_type: str = "none"  # none, balanced_w3, balanced_w5, auto_weights
    inference_type: str = "baseline"  # baseline, tta, tta_slide
    epochs: int = 50
    batch_size: int = 4
    size: int = 512

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrialResult:
    """Result of a single experiment run."""
    config: ExperimentConfig
    command: List[str]
    return_code: int
    best_f1: float
    log_path: str
    ckpt_path: str
    error_msg: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d


class ExperimentRunner:
    """Manages experiment execution and result tracking."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = args.out_dir
        self.logs_dir = os.path.join(args.out_dir, "logs")
        self.ckpts_dir = os.path.join(args.out_dir, "ckpts")
        self.gpu_ids = [int(g) for g in args.gpu_ids.split(",") if g.strip()]
        self.gpu_lock = Lock()
        self.gpu_available = {gid: True for gid in self.gpu_ids}
        self.results_lock = Lock()
        self.completed_configs: Dict[str, float] = {}  # config_key -> best_f1

        # Create directories
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.ckpts_dir, exist_ok=True)

        # Load existing results if resuming
        if args.resume:
            self._load_existing_results()

    def _load_existing_results(self):
        """Load existing experiment results for resume functionality."""
        for phase in range(1, 7):
            results_path = os.path.join(self.out_dir, f"phase{phase}_results.json")
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for result in data.get("results", []):
                        config = result.get("config", {})
                        key = self._config_key(config)
                        self.completed_configs[key] = result.get("best_f1", -1.0)
        print(f"Resumed with {len(self.completed_configs)} completed experiments")

    def _config_key(self, config: dict) -> str:
        """Generate a unique key for a config."""
        return json.dumps(config, sort_keys=True)

    def _acquire_gpu(self) -> int:
        """Acquire an available GPU, blocking if none available."""
        while True:
            with self.gpu_lock:
                for gid, available in self.gpu_available.items():
                    if available:
                        self.gpu_available[gid] = False
                        return gid
            import time
            time.sleep(1)

    def _release_gpu(self, gpu_id: int):
        """Release a GPU back to the pool."""
        with self.gpu_lock:
            self.gpu_available[gpu_id] = True

    def build_command(self, config: ExperimentConfig) -> List[str]:
        """Build the command line for running a single experiment."""
        cmd = [
            sys.executable, "train.py",
            "--exp_preset", "custom",
            "--model_arch", "srresnet",
            "--base_ch", str(config.base_ch),
            "--sr_num_blocks", str(config.num_blocks),
            "--data_root", self.args.data_root,
            "--split", self.args.split,
            "--num_classes", str(self.args.num_classes),
            "--epochs", str(config.epochs),
            "--batch_size", str(config.batch_size),
            "--lr", str(config.lr),
            "--size", str(config.size),
            "--num_workers", str(self.args.num_workers),
        ]

        # Loss type configuration
        if config.loss_type == "dice":
            cmd.append("--use_dice")
        elif config.loss_type == "focal":
            cmd.append("--use_focal")
        elif config.loss_type == "focal_dice":
            cmd.extend(["--use_focal", "--use_dice", "--focal_weight", "0.5"])
        elif config.loss_type == "ohem_dice":
            cmd.extend(["--use_ohem", "--use_dice"])
        elif config.loss_type == "focal_ohem_dice":
            cmd.extend(["--use_ohem", "--use_focal", "--use_dice", "--focal_weight", "0.5"])

        # Augmentation flags
        aug_map = {
            "hflip": "--use_hflip",
            "affine": "--use_affine",
            "color_jitter": "--use_color_jitter",
            "random_crop": "--use_random_crop",
        }
        for aug in config.aug_flags:
            if aug in aug_map:
                cmd.append(aug_map[aug])

        # Class balance
        if config.balance_type == "balanced_w3":
            cmd.extend(["--use_balanced_sampling", "--rare_weight", "3.0"])
        elif config.balance_type == "balanced_w5":
            cmd.extend(["--use_balanced_sampling", "--rare_weight", "5.0"])
        elif config.balance_type == "auto_weights":
            cmd.append("--auto_class_weights")

        # Inference tricks
        if config.inference_type in ("tta", "tta_slide"):
            cmd.append("--use_tta")
        if config.inference_type == "tta_slide":
            cmd.extend(["--slide_inference", "--slide_stride", "224"])

        return cmd

    def run_experiment(self, config: ExperimentConfig, gpu_id: int) -> TrialResult:
        """Run a single experiment on a specific GPU."""
        run_name = f"p{config.phase}_{config.name}"
        log_path = os.path.join(self.logs_dir, run_name + ".log")
        ckpt_dir = os.path.join(self.ckpts_dir, run_name)

        cmd = self.build_command(config)
        cmd.extend(["--ckpt_dir", ckpt_dir])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[GPU {gpu_id}] Starting: {run_name}")

        try:
            with open(log_path, "w", encoding="utf-8") as f:
                proc = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    timeout=3600 * 12  # 12 hour timeout
                )

            best_f1 = -1.0
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
                m = BEST_F1_PATTERN.search(content)
                if m:
                    best_f1 = float(m.group(1))

            result = TrialResult(
                config=config,
                command=cmd,
                return_code=proc.returncode,
                best_f1=best_f1,
                log_path=log_path,
                ckpt_path=os.path.join(ckpt_dir, "best.pth"),
            )

            status = "SUCCESS" if proc.return_code == 0 else "FAILED"
            print(f"[GPU {gpu_id}] {status}: {run_name} -> f1={best_f1:.4f}")

        except subprocess.TimeoutExpired:
            result = TrialResult(
                config=config,
                command=cmd,
                return_code=-1,
                best_f1=-1.0,
                log_path=log_path,
                ckpt_path=os.path.join(ckpt_dir, "best.pth"),
                error_msg="Timeout (>12h)",
            )
            print(f"[GPU {gpu_id}] TIMEOUT: {run_name}")

        except Exception as e:
            result = TrialResult(
                config=config,
                command=cmd,
                return_code=-1,
                best_f1=-1.0,
                log_path=log_path,
                ckpt_path=os.path.join(ckpt_dir, "best.pth"),
                error_msg=str(e),
            )
            print(f"[GPU {gpu_id}] ERROR: {run_name} -> {e}")

        finally:
            self._release_gpu(gpu_id)

        return result

    def run_phase_parallel(self, configs: List[ExperimentConfig]) -> List[TrialResult]:
        """Run multiple experiments in parallel across available GPUs."""
        results = []
        pending = []

        # Filter out already completed configs
        new_configs = []
        for cfg in configs:
            key = self._config_key(cfg.to_dict())
            if key in self.completed_configs:
                print(f"Skipping completed: {cfg.name}")
                results.append(TrialResult(
                    config=cfg,
                    command=[],
                    return_code=0,
                    best_f1=self.completed_configs[key],
                    log_path="",
                    ckpt_path="",
                    error_msg="Resumed from cache",
                ))
            else:
                new_configs.append(cfg)

        if not new_configs:
            return results

        # Apply max_trials limit
        if self.args.max_trials > 0:
            new_configs = new_configs[:self.args.max_trials]

        print(f"Running {len(new_configs)} experiments on GPUs {self.gpu_ids}")

        with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
            futures = {}
            for cfg in new_configs:
                gpu_id = self._acquire_gpu()
                future = executor.submit(self.run_experiment, cfg, gpu_id)
                futures[future] = cfg

            for future in as_completed(futures):
                result = future.result()
                with self.results_lock:
                    results.append(result)

        return results

    def save_phase_results(self, phase: int, results: List[TrialResult], best_config: Optional[dict]):
        """Save phase results to JSON."""
        data = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(results),
            "successful": sum(1 for r in results if r.return_code == 0),
            "best_config": best_config,
            "results": [r.to_dict() for r in results],
        }

        results_path = os.path.join(self.out_dir, f"phase{phase}_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved phase {phase} results to {results_path}")

    def save_summary(self, all_results: Dict[int, List[TrialResult]], best_config: dict):
        """Save overall summary to JSON and CSV."""
        # JSON summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "best_config": best_config,
            "phases": {
                phase: {
                    "total": len(results),
                    "successful": sum(1 for r in results if r.return_code == 0),
                    "best_f1": max((r.best_f1 for r in results if r.return_code == 0), default=-1.0),
                }
                for phase, results in all_results.items()
            }
        }

        summary_path = os.path.join(self.out_dir, "best_config.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # CSV summary
        csv_path = os.path.join(self.out_dir, "experiment_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "phase", "name", "base_ch", "num_blocks", "lr", "loss_type",
                "aug_flags", "balance_type", "inference_type", "best_f1", "return_code"
            ])
            for phase, results in sorted(all_results.items()):
                for r in results:
                    writer.writerow([
                        r.config.phase,
                        r.config.name,
                        r.config.base_ch,
                        r.config.num_blocks,
                        r.config.lr,
                        r.config.loss_type,
                        "|".join(r.config.aug_flags),
                        r.config.balance_type,
                        r.config.inference_type,
                        r.best_f1,
                        r.return_code,
                    ])

        print(f"Saved summary to {summary_path} and {csv_path}")


def generate_phase_configs(phase: int, best_config: dict, epochs: int, batch_size: int) -> List[ExperimentConfig]:
    """Generate experiment configs for a specific phase."""
    configs = []

    # Extract base settings from best config
    base_ch = best_config.get("base_ch", 32)
    num_blocks = best_config.get("num_blocks", 16)
    lr = best_config.get("lr", 2e-4)
    loss_type = best_config.get("loss_type", "baseline")
    aug_flags = best_config.get("aug_flags", [])
    balance_type = best_config.get("balance_type", "none")

    if phase == 1:
        # Phase 1: Architecture search (base_ch: 32/48, num_blocks: 12/16)
        for ch, nb in [(32, 12), (32, 16), (48, 12), (48, 16)]:
            configs.append(ExperimentConfig(
                phase=phase,
                name=f"ch{ch}_blocks{nb}",
                base_ch=ch,
                num_blocks=nb,
                lr=lr,
                loss_type=loss_type,
                aug_flags=aug_flags.copy(),
                balance_type=balance_type,
                inference_type="baseline",
                epochs=epochs,
                batch_size=batch_size,
            ))

    elif phase == 2:
        # Phase 2: Learning rate search
        for lr_val in [1e-4, 2e-4, 3e-4]:
            configs.append(ExperimentConfig(
                phase=phase,
                name=f"lr{lr_val:.0e}".replace("-", "m"),
                base_ch=base_ch,
                num_blocks=num_blocks,
                lr=lr_val,
                loss_type=loss_type,
                aug_flags=aug_flags.copy(),
                balance_type=balance_type,
                inference_type="baseline",
                epochs=epochs,
                batch_size=batch_size,
            ))

    elif phase == 3:
        # Phase 3: Loss function search
        loss_types = ["baseline", "dice", "focal", "focal_dice", "ohem_dice", "focal_ohem_dice"]
        for lt in loss_types:
            configs.append(ExperimentConfig(
                phase=phase,
                name=f"loss_{lt}",
                base_ch=base_ch,
                num_blocks=num_blocks,
                lr=lr,
                loss_type=lt,
                aug_flags=aug_flags.copy(),
                balance_type=balance_type,
                inference_type="baseline",
                epochs=epochs,
                batch_size=batch_size,
            ))

    elif phase == 4:
        # Phase 4: Data augmentation search
        aug_options = [
            ("no_aug", []),
            ("hflip", ["hflip"]),
            ("hflip_affine", ["hflip", "affine"]),
            ("hflip_affine_jitter", ["hflip", "affine", "color_jitter"]),
            ("full", ["hflip", "affine", "color_jitter", "random_crop"]),
        ]
        for name, flags in aug_options:
            configs.append(ExperimentConfig(
                phase=phase,
                name=f"aug_{name}",
                base_ch=base_ch,
                num_blocks=num_blocks,
                lr=lr,
                loss_type=loss_type,
                aug_flags=flags,
                balance_type=balance_type,
                inference_type="baseline",
                epochs=epochs,
                batch_size=batch_size,
            ))

    elif phase == 5:
        # Phase 5: Class balance search
        balance_types = ["none", "balanced_w3", "balanced_w5", "auto_weights"]
        for bt in balance_types:
            configs.append(ExperimentConfig(
                phase=phase,
                name=f"balance_{bt}",
                base_ch=base_ch,
                num_blocks=num_blocks,
                lr=lr,
                loss_type=loss_type,
                aug_flags=aug_flags.copy(),
                balance_type=bt,
                inference_type="baseline",
                epochs=epochs,
                batch_size=batch_size,
            ))

    elif phase == 6:
        # Phase 6: Inference tricks search
        inference_types = ["baseline", "tta", "tta_slide"]
        for it in inference_types:
            configs.append(ExperimentConfig(
                phase=phase,
                name=f"infer_{it}",
                base_ch=base_ch,
                num_blocks=num_blocks,
                lr=lr,
                loss_type=loss_type,
                aug_flags=aug_flags.copy(),
                balance_type=balance_type,
                inference_type=it,
                epochs=epochs,
                batch_size=batch_size,
            ))

    return configs


def find_best_result(results: List[TrialResult]) -> Tuple[Optional[TrialResult], dict]:
    """Find the best result and extract its config."""
    successful = [r for r in results if r.return_code == 0 and r.best_f1 >= 0]
    if not successful:
        return None, {}

    best = max(successful, key=lambda r: r.best_f1)
    best_config = {
        "base_ch": best.config.base_ch,
        "num_blocks": best.config.num_blocks,
        "lr": best.config.lr,
        "loss_type": best.config.loss_type,
        "aug_flags": best.config.aug_flags,
        "balance_type": best.config.balance_type,
        "inference_type": best.config.inference_type,
        "best_f1": best.best_f1,
    }
    return best, best_config


def main():
    parser = argparse.ArgumentParser(description="SRResNet Grid Search Experiments")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="splits/train_split.json")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for each experiment")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="srresnet_search")
    parser.add_argument("--gpu_ids", type=str, default="1,3", help="Comma-separated GPU IDs to use")
    parser.add_argument("--phase", type=int, default=0, help="Phase to run (0=all, 1-6=specific)")
    parser.add_argument("--max_trials", type=int, default=0, help="Max trials per phase (0=unlimited)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    args = parser.parse_args()

    print("=" * 60)
    print("SRResNet Grid Search Experiment")
    print("=" * 60)
    print(f"Output directory: {args.out_dir}")
    print(f"GPUs: {args.gpu_ids}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Phase: {args.phase if args.phase > 0 else 'all'}")
    print("=" * 60)

    runner = ExperimentRunner(args)
    all_results: Dict[int, List[TrialResult]] = {}

    # Default best config (baseline)
    best_config = {
        "base_ch": 32,
        "num_blocks": 16,
        "lr": 2e-4,
        "loss_type": "baseline",
        "aug_flags": [],
        "balance_type": "none",
        "inference_type": "baseline",
        "best_f1": 0.0,
    }

    # Load best config from previous phases if resuming
    if args.resume:
        for phase in range(6, 0, -1):
            results_path = os.path.join(args.out_dir, f"phase{phase}_results.json")
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("best_config"):
                        best_config = data["best_config"]
                        print(f"Loaded best config from phase {phase}: f1={best_config.get('best_f1', 0):.4f}")
                        break

    # Determine which phases to run
    phases_to_run = [args.phase] if args.phase > 0 else [1, 2, 3, 4, 5, 6]

    for phase in phases_to_run:
        print(f"\n{'=' * 60}")
        print(f"Phase {phase}")
        print(f"{'=' * 60}")

        # Generate configs for this phase
        configs = generate_phase_configs(phase, best_config, args.epochs, args.batch_size)
        print(f"Generated {len(configs)} configs for phase {phase}")

        # Run experiments
        results = runner.run_phase_parallel(configs)
        all_results[phase] = results

        # Find best result
        best_result, new_best_config = find_best_result(results)
        if best_result:
            best_config = new_best_config
            print(f"\nPhase {phase} best: {best_result.config.name} -> f1={best_result.best_f1:.4f}")
        else:
            print(f"\nPhase {phase} failed: no successful experiments")

        # Save phase results
        runner.save_phase_results(phase, results, best_config)

    # Save final summary
    runner.save_summary(all_results, best_config)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Best config: {json.dumps(best_config, indent=2)}")
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
