"""Pixel-metrikker per billede på thresholdede forudsigelser (sigmoid(logits) > 0.5)."""
import torch

METRICS = ("dice", "iou", "accuracy", "sensitivity", "specificity")


def confusion_counts(logits, target, valid=None):
    """TP, FP, FN, TN per billede (B x 4). `valid` (fx DRIVE's FOV) begrænser hvilke pixels der tælles."""
    pred = logits > 0  # svarer til sigmoid(logits) > 0.5
    true = target > 0.5
    if valid is None:
        valid = torch.ones_like(true)
    dims = (1, 2)
    tp = (pred & true & valid).sum(dims)
    fp = (pred & ~true & valid).sum(dims)
    fn = (~pred & true & valid).sum(dims)
    tn = (~pred & ~true & valid).sum(dims)
    return torch.stack([tp, fp, fn, tn], dim=1).double().cpu()


def metrics_from_counts(counts):
    """Dict med én værdi per billede for hver metrik. 0/0 giver NaN og udelades i gennemsnit."""
    tp, fp, fn, tn = counts.unbind(1)
    return {
        "dice": 2 * tp / (2 * tp + fp + fn),
        "iou": tp / (tp + fp + fn),
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }


class MetricTracker:
    """Samler confusion counts over et helt sæt og giver gennemsnit over billeder."""

    def __init__(self):
        self.counts = []

    def update(self, logits, target, valid=None):
        self.counts.append(confusion_counts(logits, target, valid))

    def per_image(self):
        return metrics_from_counts(torch.cat(self.counts))

    def compute(self):
        return {name: values.nanmean().item() for name, values in self.per_image().items()}
