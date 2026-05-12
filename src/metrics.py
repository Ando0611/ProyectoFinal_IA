"""Métricas para el modelo multi-tarea.

Segmentación:
    - Confusion matrix por clase acumulada en un epoch.
    - mIoU = promedio de IoU por clase (ignora clases sin GT).
    - per-class IoU para reportar dónde falla el modelo.

Atributos:
    - Accuracy por atributo.
    - F1 (macro) por atributo.
"""

from collections import defaultdict

import torch
import torch.nn.functional as F

from src.config import ATTRS, NUM_SEG_CLASSES


class SegMetric:
    """Acumula confusion matrix sobre múltiples batches."""

    def __init__(self, num_classes=NUM_SEG_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.long
        )

    def update(self, pred_mask, true_mask):
        """
        pred_mask, true_mask: [B, H, W] long, mismos shapes.
        """
        valid = (true_mask >= 0) & (true_mask < self.num_classes)
        t = true_mask[valid]
        p = pred_mask[valid]

        # confusion[i, j] = #píxeles con etiqueta i predichos como j
        indices = t * self.num_classes + p
        binc = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion += binc.reshape(self.num_classes, self.num_classes).cpu()

    def compute_iou(self):
        """Devuelve (per_class_iou: list[float|None], mean_iou: float)."""
        tp = torch.diag(self.confusion).float()
        fp = self.confusion.sum(dim=0).float() - tp
        fn = self.confusion.sum(dim=1).float() - tp

        per_class = []
        usable = []
        for k in range(self.num_classes):
            denom = tp[k] + fp[k] + fn[k]
            if denom == 0:
                per_class.append(None)
            else:
                iou = (tp[k] / denom).item()
                per_class.append(iou)
                usable.append(iou)

        mean_iou = sum(usable) / len(usable) if usable else 0.0
        return per_class, mean_iou


class AttrMetric:
    """Métricas binarias por atributo (accuracy + F1)."""

    def __init__(self, num_attrs=None, threshold=0.5):
        self.num_attrs = num_attrs if num_attrs is not None else len(ATTRS)
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_attrs, dtype=torch.long)
        self.fp = torch.zeros(self.num_attrs, dtype=torch.long)
        self.tn = torch.zeros(self.num_attrs, dtype=torch.long)
        self.fn = torch.zeros(self.num_attrs, dtype=torch.long)

    def update(self, attr_logits, attrs_true):
        probs = torch.sigmoid(attr_logits)
        preds = (probs >= self.threshold).long().cpu()
        truth = attrs_true.long().cpu()

        for k in range(self.num_attrs):
            p = preds[:, k]
            t = truth[:, k]
            self.tp[k] += int(((p == 1) & (t == 1)).sum())
            self.fp[k] += int(((p == 1) & (t == 0)).sum())
            self.tn[k] += int(((p == 0) & (t == 0)).sum())
            self.fn[k] += int(((p == 0) & (t == 1)).sum())

    def compute(self):
        """Devuelve dict por atributo + promedios."""
        report = {}
        accs, f1s = [], []
        for k, name in enumerate(ATTRS[: self.num_attrs]):
            tp, fp, tn, fn = self.tp[k], self.fp[k], self.tn[k], self.fn[k]
            total = tp + fp + tn + fn
            acc = float(tp + tn) / float(total) if total > 0 else 0.0
            precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
            recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            report[name] = {"acc": acc, "f1": f1, "precision": precision, "recall": recall}
            accs.append(acc)
            f1s.append(f1)

        report["_mean"] = {
            "acc": sum(accs) / len(accs) if accs else 0.0,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        }
        return report
