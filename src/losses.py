"""Pérdida combinada para U-Net multi-tarea.

- Segmentación: cross-entropy categórica sobre los logits pixel-level.
- Atributos: binary cross-entropy con logits, promediada sobre el batch
             y los NUM_ATTRS atributos.
- Total: seg_loss + lambda_attr * attr_loss
"""

import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_attr=1.0, seg_class_weights=None):
        super().__init__()
        self.lambda_attr = lambda_attr
        self.seg_loss = nn.CrossEntropyLoss(weight=seg_class_weights)
        self.attr_loss = nn.BCEWithLogitsLoss()

    def forward(self, seg_logits, attr_logits, mask, attrs):
        """
        seg_logits:  [B, NUM_SEG_CLASSES, H, W]
        attr_logits: [B, NUM_ATTRS]
        mask:        [B, H, W]    long
        attrs:       [B, NUM_ATTRS]  float (0.0/1.0)
        """
        l_seg = self.seg_loss(seg_logits, mask)
        l_attr = self.attr_loss(attr_logits, attrs)
        total = l_seg + self.lambda_attr * l_attr
        return total, l_seg.detach(), l_attr.detach()
