"""Metric helpers for validation and reporting."""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    """Compute a compact metric bundle for training and validation."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=list(range(len(class_names))),
        ).tolist(),
    }
