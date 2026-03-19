#!/usr/bin/env python
"""
Grid Search Experiment Script for Face Segmentation

This script runs a phased grid search for face segmentation models.
It supports:
- Multiple architectures: MiniUNet (plain/residual) and SRResNet
- Dual GPU parallel execution
- Checkpoint/resume functionality
- Phased search (each phase builds on the best config from previous phase)
- Weights & Biases integration for experiment tracking

Usage:
    python run_srresnet_experiments.py --epochs 50 --gpu_ids 1,3
    python run_srresnet_experiments.py --phase 1 --max_trials 2 --epochs 5  # Quick test
    python run_srresnet_experiments.py --arch all  # Search all architectures
    python run_srresnet_experiments.py --arch srresnet  # Only SRResNet
    python run_srresnet_experiments.py --arch miniunet  # Only MiniUNet
    python run_srresnet_experiments.py --use_wandb --wandb_project my-project  # Enable W&B
"""

import argparse
import csv
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
    # Architecture
    model_arch: str = "srresnet"  # miniunet, srresnet
    backbone: str = "plain"  # plain, residual (for miniunet)
    base_ch: int = 32
    num_blocks: int = 16  # sr_num_blocks for SRResNet
    # MiniUNet specific
    use_dsconv: bool = False
    use_attention: bool = False
    use_context: bool = False
    use_aux_head: bool = False
    # Training
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

    def arch_name(self) -> str:
        """Return a short architecture name for display."""
        if self.model_arch == "srresnet":
            return f"srresnet_ch{self.base_ch}_b{self.num_blocks}"
        else:
            flags = []
            if self.backbone == "residual":
                flags.append("res")
            if self.use_dsconv:
                flags.append("ds")
            if self.use_attention:
                flags.append("att")
            if self.use_context:
                flags.append("ctx")
            if self.use_aux_head:
                flags.append("aux")
            suffix = "_".join(flags) if flags else "plain"
            return f"miniunet_{suffix}"


@dataclass
class TrialResult:
    """Result of a single experiment run."""
    config: ExperimentConfig
    command: List[str]
    return_code: int
    best_f1: float
    log_path: str
    ckpt_path: str
    n_params: int = 0
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
            "--model_arch", config.model_arch,
            "--base_ch", str(config.base_ch),
            "--data_root", self.args.data_root,
            "--split", self.args.split,
            "--num_classes", str(self.args.num_classes),
            "--epochs", str(config.epochs),
            "--batch_size", str(config.batch_size),
            "--lr", str(config.lr),
            "--size", str(config.size),
            "--num_workers", str(self.args.num_workers),
        ]

        # Architecture-specific args
        if config.model_arch == "srresnet":
            cmd.extend(["--sr_num_blocks", str(config.num_blocks)])
        else:  # miniunet
            cmd.extend(["--backbone", config.backbone])
            if config.use_dsconv:
                cmd.append("--use_dsconv")
            if config.use_attention:
                cmd.append("--use_attention")
            if config.use_context:
                cmd.append("--use_context")
            if config.use_aux_head:
                cmd.append("--use_aux_head")

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
            n_params = 0
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
                m = BEST_F1_PATTERN.search(content)
                if m:
                    best_f1 = float(m.group(1))
                # Extract param count
                params_m = re.search(r"Model params: ([\d,]+)", content)
                if params_m:
                    n_params = int(params_m.group(1).replace(",", ""))

            result = TrialResult(
                config=config,
                command=cmd,
                return_code=proc.returncode,
                best_f1=best_f1,
                log_path=log_path,
                ckpt_path=os.path.join(ckpt_dir, "best.pth"),
                n_params=n_params,
            )

            status = "SUCCESS" if proc.returncode == 0 else "FAILED"
            print(f"[GPU {gpu_id}] {status}: {run_name} -> f1={best_f1:.4f} params={n_params:,}")

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
                "phase", "name", "model_arch", "backbone", "base_ch", "num_blocks",
                "use_dsconv", "use_attention", "use_context", "use_aux_head",
                "lr", "loss_type", "aug_flags", "balance_type", "inference_type",
                "best_f1", "n_params", "return_code"
            ])
            for phase, results in sorted(all_results.items()):
                for r in results:
                    writer.writerow([
                        r.config.phase,
                        r.config.name,
                        r.config.model_arch,
                        r.config.backbone,
                        r.config.base_ch,
                        r.config.num_blocks,
                        r.config.use_dsconv,
                        r.config.use_attention,
                        r.config.use_context,
                        r.config.use_aux_head,
                        r.config.lr,
                        r.config.loss_type,
                        "|".join(r.config.aug_flags),
                        r.config.balance_type,
                        r.config.inference_type,
                        r.best_f1,
                        r.n_params,
                        r.return_code,
                    ])

        print(f"Saved summary to {summary_path} and {csv_path}")


def generate_arch_configs(arch_filter: str) -> List[ExperimentConfig]:
    """Generate architecture search configs based on filter."""
    configs = []

    if arch_filter in ("all", "srresnet"):
        # SRResNet variants
        for ch, nb in [(32, 12), (32, 16), (48, 12), (48, 16)]:
            configs.append(ExperimentConfig(
                phase=1,
                name=f"srresnet_ch{ch}_b{nb}",
                model_arch="srresnet",
                base_ch=ch,
                num_blocks=nb,
            ))

    if arch_filter in ("all", "miniunet"):
        # MiniUNet variants - Quick search
        # Baseline plain
        configs.append(ExperimentConfig(
            phase=1,
            name="miniunet_plain",
            model_arch="miniunet",
            backbone="plain",
            base_ch=32,
        ))
        # Residual backbone (important for parameter limit)
        configs.append(ExperimentConfig(
            phase=1,
            name="miniunet_residual",
            model_arch="miniunet",
            backbone="residual",
            base_ch=32,
        ))
        # With attention
        configs.append(ExperimentConfig(
            phase=1,
            name="miniunet_res_att",
            model_arch="miniunet",
            backbone="residual",
            base_ch=32,
            use_attention=True,
        ))
        # With dsconv + attention (efficient)
        configs.append(ExperimentConfig(
            phase=1,
            name="miniunet_res_ds_att",
            model_arch="miniunet",
            backbone="residual",
            base_ch=32,
            use_dsconv=True,
            use_attention=True,
        ))
        # Full strong config
        configs.append(ExperimentConfig(
            phase=1,
            name="miniunet_res_ds_att_ctx_aux",
            model_arch="miniunet",
            backbone="residual",
            base_ch=32,
            use_dsconv=True,
            use_attention=True,
            use_context=True,
            use_aux_head=True,
        ))

    return configs


def generate_phase_configs(phase: int, best_config: dict, epochs: int, batch_size: int,
                          arch_filter: str = "all") -> List[ExperimentConfig]:
    """Generate experiment configs for a specific phase."""
    configs = []

    # Extract base settings from best config
    model_arch = best_config.get("model_arch", "srresnet")
    backbone = best_config.get("backbone", "plain")
    base_ch = best_config.get("base_ch", 32)
    num_blocks = best_config.get("num_blocks", 16)
    use_dsconv = best_config.get("use_dsconv", False)
    use_attention = best_config.get("use_attention", False)
    use_context = best_config.get("use_context", False)
    use_aux_head = best_config.get("use_aux_head", False)
    lr = best_config.get("lr", 2e-4)
    loss_type = best_config.get("loss_type", "baseline")
    aug_flags = best_config.get("aug_flags", [])
    balance_type = best_config.get("balance_type", "none")

    if phase == 1:
        # Phase 1: Architecture search
        configs = generate_arch_configs(arch_filter)
        # Set epochs and batch_size
        for cfg in configs:
            cfg.epochs = epochs
            cfg.batch_size = batch_size

    elif phase == 2:
        # Phase 2: Learning rate search
        for lr_val in [1e-4, 2e-4, 3e-4]:
            configs.append(ExperimentConfig(
                phase=phase,
                name=f"lr{lr_val:.0e}".replace("-", "m").replace("e0", ""),
                model_arch=model_arch,
                backbone=backbone,
                base_ch=base_ch,
                num_blocks=num_blocks,
                use_dsconv=use_dsconv,
                use_attention=use_attention,
                use_context=use_context,
                use_aux_head=use_aux_head,
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
                model_arch=model_arch,
                backbone=backbone,
                base_ch=base_ch,
                num_blocks=num_blocks,
                use_dsconv=use_dsconv,
                use_attention=use_attention,
                use_context=use_context,
                use_aux_head=use_aux_head,
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
                model_arch=model_arch,
                backbone=backbone,
                base_ch=base_ch,
                num_blocks=num_blocks,
                use_dsconv=use_dsconv,
                use_attention=use_attention,
                use_context=use_context,
                use_aux_head=use_aux_head,
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
                model_arch=model_arch,
                backbone=backbone,
                base_ch=base_ch,
                num_blocks=num_blocks,
                use_dsconv=use_dsconv,
                use_attention=use_attention,
                use_context=use_context,
                use_aux_head=use_aux_head,
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
                model_arch=model_arch,
                backbone=backbone,
                base_ch=base_ch,
                num_blocks=num_blocks,
                use_dsconv=use_dsconv,
                use_attention=use_attention,
                use_context=use_context,
                use_aux_head=use_aux_head,
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
        "model_arch": best.config.model_arch,
        "backbone": best.config.backbone,
        "base_ch": best.config.base_ch,
        "num_blocks": best.config.num_blocks,
        "use_dsconv": best.config.use_dsconv,
        "use_attention": best.config.use_attention,
        "use_context": best.config.use_context,
        "use_aux_head": best.config.use_aux_head,
        "lr": best.config.lr,
        "loss_type": best.config.loss_type,
        "aug_flags": best.config.aug_flags,
        "balance_type": best.config.balance_type,
        "inference_type": best.config.inference_type,
        "best_f1": best.best_f1,
        "n_params": best.n_params,
    }
    return best, best_config


def main():
    parser = argparse.ArgumentParser(description="Grid Search Experiments for Face Segmentation")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="splits/train_split.json")
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for each experiment")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="grid_search_results")
    parser.add_argument("--gpu_ids", type=str, default="1,3", help="Comma-separated GPU IDs to use")
    parser.add_argument("--arch", type=str, default="all",
                        choices=["all", "srresnet", "miniunet"],
                        help="Architecture to search: all, srresnet, or miniunet")
    parser.add_argument("--phase", type=int, default=0, help="Phase to run (0=all, 1-6=specific)")
    parser.add_argument("--max_trials", type=int, default=0, help="Max trials per phase (0=unlimited)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    args = parser.parse_args()

    print("=" * 60)
    print("Grid Search Experiment for Face Segmentation")
    print("=" * 60)
    print(f"Output directory: {args.out_dir}")
    print(f"GPUs: {args.gpu_ids}")
    print(f"Architecture filter: {args.arch}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Phase: {args.phase if args.phase > 0 else 'all'}")
    print("=" * 60)

    runner = ExperimentRunner(args)
    all_results: Dict[int, List[TrialResult]] = {}

    # Default best config (baseline)
    best_config = {
        "model_arch": "srresnet",
        "backbone": "plain",
        "base_ch": 32,
        "num_blocks": 16,
        "use_dsconv": False,
        "use_attention": False,
        "use_context": False,
        "use_aux_head": False,
        "lr": 2e-4,
        "loss_type": "baseline",
        "aug_flags": [],
        "balance_type": "none",
        "inference_type": "baseline",
        "best_f1": 0.0,
        "n_params": 0,
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
        configs = generate_phase_configs(phase, best_config, args.epochs, args.batch_size, args.arch)
        print(f"Generated {len(configs)} configs for phase {phase}")

        # Run experiments
        results = runner.run_phase_parallel(configs)
        all_results[phase] = results

        # Find best result
        best_result, new_best_config = find_best_result(results)
        if best_result:
            best_config = new_best_config
            print(f"\nPhase {phase} best: {best_result.config.name} -> f1={best_result.best_f1:.4f}")
            print(f"  Architecture: {best_result.config.arch_name()}")
            print(f"  Params: {best_result.n_params:,}")
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
