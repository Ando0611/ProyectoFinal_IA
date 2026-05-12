"""Dataset PyTorch para CelebAMask-HQ preprocesado.

Espera la estructura que produce `data/preprocess_celebamaskhq.py`:

    data/celeba-hq/
      images/N.jpg
      masks/N.png         (uint8, valores 0..18)
      attributes.txt      (formato CelebA: +1/-1)

Cada muestra devuelve:
    image:  tensor float32 [3, H, W] normalizado a [0, 1]
    mask:   tensor long    [H, W]    con valores 0..NUM_SEG_CLASSES-1
    attrs:  tensor float32 [NUM_ATTRS] con valores 0.0 / 1.0
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

from src.config import (
    ATTRS,
    ATTRIBUTES_FILE,
    IMAGES_DIR,
    IMAGE_SIZE,
    MASKS_DIR,
    NUM_SEG_CLASSES,
)


def _load_attributes(attr_path):
    """Lee attributes.txt al formato dict[image_id] -> tensor [NUM_ATTRS]."""
    attr_path = Path(attr_path)
    with open(attr_path) as f:
        f.readline()  # número total de imágenes
        header = f.readline().split()
        rows = [line.strip().split() for line in f if line.strip()]

    name_to_col = {name: i for i, name in enumerate(header)}
    out = {}
    for row in rows:
        filename, values = row[0], row[1:]
        stem = Path(filename).stem
        if not stem.isdigit():
            continue
        image_id = int(stem)

        attrs = []
        for name in ATTRS:
            if name == "Beard":
                # En CelebA viene "No_Beard": +1 = no tiene barba.
                # Invertimos para que 1.0 = barba presente.
                raw = values[name_to_col["No_Beard"]]
                attrs.append(0.0 if raw == "1" else 1.0)
            else:
                raw = values[name_to_col[name]]
                attrs.append(1.0 if raw == "1" else 0.0)

        out[image_id] = torch.tensor(attrs, dtype=torch.float32)
    return out


class CelebAMaskHQDataset(Dataset):
    """Dataset multi-tarea de CelebAMask-HQ.

    Aumentos en entrenamiento:
        - flip horizontal aleatorio (con swap consistente l_eye <-> r_eye, etc.)
        - color jitter ligero

    En modo eval se aplica solo resize + normalización.
    """

    # Pares de clases que deben intercambiarse al hacer flip horizontal,
    # para que la máscara siga siendo consistente con la imagen.
    _FLIP_PAIRS = [
        ("l_brow", "r_brow"),
        ("l_eye", "r_eye"),
        ("l_ear", "r_ear"),
    ]

    def __init__(
        self,
        images_dir=IMAGES_DIR,
        masks_dir=MASKS_DIR,
        attributes_file=ATTRIBUTES_FILE,
        image_size=IMAGE_SIZE,
        ids=None,
        augment=False,
    ):
        from src.config import SEG_NAME_TO_ID

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment

        if not self.images_dir.is_dir():
            raise RuntimeError(
                f"No existe {self.images_dir}. Ejecuta primero "
                "`python -m data.preprocess_celebamaskhq ...`"
            )

        self.attrs = _load_attributes(attributes_file)

        # IDs disponibles = intersección de imágenes, máscaras y atributos
        img_ids = {int(p.stem) for p in self.images_dir.glob("*.jpg")}
        mask_ids = {int(p.stem) for p in self.masks_dir.glob("*.png")}
        attr_ids = set(self.attrs.keys())
        available = sorted(img_ids & mask_ids & attr_ids)

        if ids is not None:
            ids = sorted(set(ids) & set(available))
            self.ids = ids
        else:
            self.ids = available

        if len(self.ids) == 0:
            raise RuntimeError(
                f"Sin pares (imagen, máscara, atributos) en {self.images_dir.parent}"
            )

        # Pre-calcular el swap de IDs para flip horizontal
        self._flip_remap = list(range(NUM_SEG_CLASSES))
        for a, b in self._FLIP_PAIRS:
            ia, ib = SEG_NAME_TO_ID[a], SEG_NAME_TO_ID[b]
            self._flip_remap[ia] = ib
            self._flip_remap[ib] = ia
        self._flip_remap = torch.tensor(self._flip_remap, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        img = Image.open(self.images_dir / f"{image_id}.jpg").convert("RGB")
        mask = Image.open(self.masks_dir / f"{image_id}.png")

        # Resize: bilinear para imagen, nearest para máscara
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        if mask.size != (self.image_size, self.image_size):
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_tensor = TF.to_tensor(img)            # [3, H, W] en [0, 1]
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))  # [H, W]
        attrs = self.attrs[image_id].clone()        # [NUM_ATTRS]

        if self.augment:
            image_tensor, mask_tensor, attrs = self._augment(
                image_tensor, mask_tensor, attrs
            )

        return image_tensor, mask_tensor, attrs

    def _augment(self, image, mask, attrs):
        # Flip horizontal con probabilidad 0.5
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
            mask = self._flip_remap[mask]
            # Los atributos no cambian al voltear horizontalmente

        # Color jitter ligero (solo sobre la imagen)
        if random.random() < 0.5:
            brightness = 1.0 + (random.random() - 0.5) * 0.3
            contrast = 1.0 + (random.random() - 0.5) * 0.3
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)

        return image, mask, attrs


def split_ids(all_ids, train_ratio, seed):
    """Reproducible 70/30 (o similar) split entre train y valid."""
    ids = list(all_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * train_ratio)
    return ids[:cut], ids[cut:]
