# AI6126 Project 1 (Face Parsing)

## OpenMMLab-style training/inference tricks

`train.py` now supports the following practical tricks (with configurable switches):

- **OHEM**: `--use_ohem --ohem_kept_ratio 0.25`
- **Multi-loss joint training**: CE/OHEM + Focal + Dice
  - `--ce_weight`
  - `--focal_weight`
  - `--dice_weight` (enabled when preset/flag uses dice)
- **Class balance**
  - image-level balanced sampler: `--use_balanced_sampling --rare_classes ... --rare_weight ...`
  - pixel-level auto class weights: `--auto_class_weights`
- **More standard data pipeline**
  - image normalization: `--mean 0.5,0.5,0.5 --std 0.5,0.5,0.5`
  - optional random crop: `--use_random_crop --crop_size 448`
- **TTA / slide inference (validation stage)**
  - `--use_tta`
  - `--slide_inference --crop_size 448 --slide_stride 224`

### Example: enable all requested tricks

```bash
python train.py \
  --exp_preset custom \
  --model_arch miniunet \
  --use_dsconv --use_attention --use_context --use_aux_head \
  --use_ohem --ohem_kept_ratio 0.25 \
  --use_dice --dice_weight 0.4 \
  --use_focal --focal_weight 0.5 --focal_gamma 2.0 \
  --use_balanced_sampling --rare_classes 3,14,15,16 --rare_weight 5.0 \
  --auto_class_weights \
  --use_hflip --use_affine --use_color_jitter --use_random_crop --crop_size 448 \
  --mean 0.5,0.5,0.5 --std 0.5,0.5,0.5 \
  --use_tta --slide_inference --slide_stride 224
```

## New support: Baseline Network 4

You can now run a `baseline4` preset that keeps the model **from scratch** (no pretrained weights) while enabling a stronger configuration under the fairness constraint (`< 1,821,085` trainable params checked in `train.py`).

### Run Baseline-4

```bash
python train.py \
  --exp_preset baseline4 \
  --data_root data \
  --split splits/train_split.json \
  --epochs 80 \
  --batch_size 4 \
  --lr 3e-4 \
  --use_hflip --use_affine --use_color_jitter
```

## Backbone options

- `--backbone plain` (default): original blocks
- `--backbone residual`: residual double-conv blocks (still from scratch)

`baseline4` uses `--backbone residual` by default.

## Optimization tips (no pretrain, param-limited)

1. Keep `--exp_preset full_strong` and run longer (`80~120` epochs).
2. Try `--exp_preset baseline4 --focal_gamma 1.5` for rare classes.
3. For CE runs, use mild label smoothing: `--label_smoothing 0.05`.
4. Use balanced sampling with tuned `--rare_weight` (`3.0~6.0`).
5. Keep `--scheduler cosine --min_lr 1e-6`.

> Note: All options remain compatible with the fairness parameter cap guard in `train.py`.

## New support: SRResNet baseline (actual SRResNet)

You can now benchmark a real SRResNet baseline (residual trunk + optional PixelShuffle upsampling blocks):

```bash
python train.py \
  --exp_preset srresnet_baseline \
  --data_root data \
  --split splits/train_split.json \
  --epochs 80 \
  --batch_size 4 \
  --lr 3e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --use_hflip --use_affine --use_color_jitter
```

For custom SRResNet runs:

```bash
python train.py --exp_preset custom --model_arch srresnet --sr_num_blocks 16 --sr_upscale_factor 1
```

## New support: one-click strategy search

Run a single command to search different presets + optimizers + schedulers + augmentation combinations and automatically report the best strategy:

```bash
python search_best_strategy.py \
  --data_root data \
  --split splits/train_split.json \
  --epochs 20 \
  --batch_size 4
```

Useful options:

- `--max_trials 8`: quick sanity search
- `--out_dir search_runs`: where logs/checkpoints/summary are saved

The script writes:

- per-trial logs under `search_runs/logs/`
- per-trial checkpoints under `search_runs/ckpts/`
- ranked summary file `search_runs/summary.json`
