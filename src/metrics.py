# src/metrics.py
import torch


@torch.no_grad()
def f1_score_multiclass(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    ignore_background: bool = True,
) -> float:
    """
    logits: (B, C, H, W)
    target: (B, H, W) long
    Returns macro F1 over classes.
    """
    pred = logits.argmax(dim=1)  # (B,H,W)

    # flatten
    pred = pred.view(-1)
    target = target.view(-1)

    if ignore_index >= 0:
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]

    f1_list = []
    class_range = range(1, num_classes) if ignore_background else range(num_classes)

    for c in class_range:
        pred_c = pred == c
        targ_c = target == c

        tp = (pred_c & targ_c).sum().item()
        fp = (pred_c & (~targ_c)).sum().item()
        fn = ((~pred_c) & targ_c).sum().item()

        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1_list.append(f1)

    if len(f1_list) == 0:
        return 0.0
    return float(sum(f1_list) / len(f1_list))