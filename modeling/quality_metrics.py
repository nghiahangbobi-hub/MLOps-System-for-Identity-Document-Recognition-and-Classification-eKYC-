"""Document and quality evaluation metrics for the fixed-split comparison."""

import torch
from torch.nn import functional as F


def classification_metrics(targets, predictions, labels):
    classes = len(labels)
    counts = torch.bincount(targets * classes + predictions, minlength=classes ** 2)
    matrix = counts.reshape(classes, classes)
    true = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    total = targets.numel()
    return {
        "samples": total,
        "accuracy": true.sum().item() / total if total else None,
        "macro_f1": (2 * true / (support + predicted).clamp(min=1)).mean().item() if total else None,
        "per_class_recall": {
            label: true[i].item() / support[i].item() if support[i] else None
            for i, label in enumerate(labels)
        },
        "per_class_support": dict(zip(labels, support.tolist())),
        "labels": list(labels),
        "confusion_true_rows_predicted_columns": matrix.tolist(),
    }


def calibration(logits, targets, temperature, bins=15):
    probabilities = (logits.to(torch.float64) / temperature).softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets)
    assignments = (confidence * bins).long().clamp(max=bins - 1)
    records = []
    for i in range(bins):
        mask = assignments == i
        count = mask.sum().item()
        records.append({
            "count": count,
            "mean_confidence": confidence[mask].mean().item() if count else None,
            "accuracy": correct[mask].to(torch.float64).mean().item() if count else None,
        })
    ece = sum(row["count"] * abs(row["accuracy"] - row["mean_confidence"])
              for row in records if row["count"]) / targets.numel()
    return {"ece": ece, "equal_width_bins": records,
            "nll": F.cross_entropy(logits.to(torch.float64) / temperature, targets).item()}, confidence, correct


def fit_temperature(logits, targets):
    # A bounded, deterministic validation-only search includes the original scale.
    candidates = torch.logspace(-1, 1, 81, dtype=torch.float64)
    logits = logits.to(torch.float64)
    return min(candidates.tolist(), key=lambda value: (
        F.cross_entropy(logits / value, targets).item(), abs(value - 1)
    ))


def select_threshold(confidence, correct, target_accuracy):
    """Retain the largest validation prefix that meets the target, including ties."""
    values, order = confidence.sort(descending=True)
    running_correct = correct[order].cumsum(0)
    positions = torch.arange(1, values.numel() + 1)
    tie_boundaries = torch.cat((values[:-1] != values[1:], torch.tensor([True])))
    valid = (running_correct >= target_accuracy * positions) & tie_boundaries
    if valid.any():
        return values[valid][-1].item(), False
    return 1.0, True


def coverage(confidence, correct, threshold, review_all):
    accepted = confidence >= threshold
    if review_all:
        accepted.fill_(False)
    count = accepted.sum().item()
    return {"retained": count, "coverage": count / confidence.numel(),
            "retained_accuracy": correct[accepted].double().mean().item() if count else None}
