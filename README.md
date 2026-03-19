# AI6126 Project 1 (Face Parsing)

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
