"""Smoke test del U-Net multi-tarea: que las shapes salgan correctas."""

import torch

from src.config import NUM_ATTRS, NUM_SEG_CLASSES
from src.model import UNetMultiTask


def test_unet_output_shapes_256():
    model = UNetMultiTask(base_channels=8)  # versión enana para que el test sea rápido
    model.train(False)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        seg_logits, attr_logits = model(x)
    assert seg_logits.shape == (2, NUM_SEG_CLASSES, 256, 256)
    assert attr_logits.shape == (2, NUM_ATTRS)


def test_unet_output_shapes_128():
    model = UNetMultiTask(base_channels=8)
    model.train(False)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        seg_logits, attr_logits = model(x)
    assert seg_logits.shape == (1, NUM_SEG_CLASSES, 128, 128)
    assert attr_logits.shape == (1, NUM_ATTRS)


def test_unet_backward_runs():
    model = UNetMultiTask(base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    seg, attr = model(x)
    loss = seg.mean() + attr.mean()
    loss.backward()
    # Si llega aquí sin excepción, el grafo se construyó OK.
