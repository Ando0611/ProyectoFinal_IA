"""Tests para src/metrics.py."""

import torch

from src.metrics import AttrMetric, SegMetric


def test_seg_metric_perfect_prediction():
    metric = SegMetric(num_classes=3)
    pred = torch.tensor([[[0, 1, 2], [1, 1, 2]]])
    true = pred.clone()
    metric.update(pred, true)

    per_class, mean_iou = metric.compute_iou()
    assert per_class == [1.0, 1.0, 1.0]
    assert mean_iou == 1.0


def test_seg_metric_partial_overlap():
    metric = SegMetric(num_classes=2)
    # Predijo todo como clase 1; mitad del GT es 0, mitad es 1.
    pred = torch.tensor([[[1, 1], [1, 1]]])
    true = torch.tensor([[[0, 0], [1, 1]]])
    metric.update(pred, true)

    per_class, mean_iou = metric.compute_iou()
    # Clase 0: TP=0, FP=2, FN=2 -> IoU=0
    # Clase 1: TP=2, FP=2, FN=0 -> IoU=0.5
    assert per_class[0] == 0.0
    assert per_class[1] == 0.5
    assert mean_iou == 0.25


def test_seg_metric_absent_class_is_skipped():
    metric = SegMetric(num_classes=3)
    pred = torch.tensor([[[0, 1], [1, 0]]])
    true = pred.clone()
    metric.update(pred, true)

    per_class, mean_iou = metric.compute_iou()
    # Clase 2 nunca aparece ni en GT ni en pred: None
    assert per_class[2] is None
    assert mean_iou == 1.0


def test_attr_metric_basic():
    metric = AttrMetric(num_attrs=2)
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])  # certeza casi total
    truth = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    metric.update(logits, truth)

    report = metric.compute()
    # Atributo 0: predicciones [1, 0], verdaderos [1, 0] -> acc=1
    # Atributo 1: predicciones [0, 1], verdaderos [0, 1] -> acc=1
    names = list(report.keys())
    names.remove("_mean")
    for name in names:
        assert report[name]["acc"] == 1.0
        assert report[name]["f1"] == 1.0
    assert report["_mean"]["acc"] == 1.0
