"""Preproceso del dataset CelebAMask-HQ.

CelebAMask-HQ entrega las anotaciones de segmentación como **un PNG
binario por parte**:
    CelebAMask-HQ-mask-anno/0/00000_skin.png
    CelebAMask-HQ-mask-anno/0/00000_l_eye.png
    ...

Este script combina todos esos PNG binarios en una **única máscara PNG
por imagen** con valores enteros 0..18 (ver `src/config.SEG_CLASSES`).
También normaliza el archivo de atributos y separa los rostros usados
para entrenamiento / validación.

Uso:
    python -m data.preprocess_celebamaskhq \\
        --raw /ruta/al/CelebAMask-HQ \\
        --out data/celeba-hq \\
        --limit 5000

`--raw` apunta al directorio que contiene `CelebA-HQ-img/`,
`CelebAMask-HQ-mask-anno/` y `CelebAMask-HQ-attribute-anno.txt`.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import SEG_CLASSES, SEG_NAME_TO_ID


# Las anotaciones están particionadas en 15 subdirectorios (0..14),
# cada uno con 2000 imágenes consecutivas.
ANNO_BUCKET_SIZE = 2000


def build_mask(anno_dir, image_id, height=512, width=512):
    """Combina los PNGs binarios de partes en una máscara entera única.

    CelebAMask-HQ provee los PNGs en resolución 512x512.
    El orden de overlay importa: las partes superiores (pelo, sombrero,
    lentes) deben pintarse encima del rostro.
    """
    bucket = image_id // ANNO_BUCKET_SIZE
    bucket_dir = anno_dir / str(bucket)

    mask = np.zeros((height, width), dtype=np.uint8)

    # Saltamos "background" (ya es 0 por defecto)
    for class_name in SEG_CLASSES[1:]:
        part_file = bucket_dir / f"{image_id:05d}_{class_name}.png"
        if not part_file.exists():
            continue
        binary = np.array(Image.open(part_file).convert("L"))
        mask[binary > 0] = SEG_NAME_TO_ID[class_name]

    return mask


def parse_attribute_header(path):
    """Devuelve la lista de nombres de atributos en el orden del archivo."""
    with open(path) as f:
        f.readline()           # primera línea: número de imágenes
        return f.readline().split()


def parse_attribute_rows(path):
    """Itera (filename, dict[name -> 0|1]) por cada fila del archivo."""
    names = parse_attribute_header(path)
    with open(path) as f:
        f.readline()
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 1 + len(names):
                continue
            filename = parts[0]
            values = {n: 1 if v == "1" else 0 for n, v in zip(names, parts[1:])}
            yield filename, values


def parse_args():
    parser = argparse.ArgumentParser(description="Preproceso de CelebAMask-HQ")
    parser.add_argument(
        "--raw", type=str, required=True,
        help="Directorio raíz de CelebAMask-HQ descomprimido",
    )
    parser.add_argument(
        "--out", type=str, default="data/celeba-hq",
        help="Directorio destino (default: data/celeba-hq)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Procesar solo los primeros N rostros (útil para pruebas).",
    )
    parser.add_argument(
        "--mask-size", type=int, default=512,
        help="Resolución de las máscaras combinadas (default 512).",
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Resolución a la que se redimensiona la imagen (default 512).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw = Path(args.raw)
    src_images = raw / "CelebA-HQ-img"
    src_masks = raw / "CelebAMask-HQ-mask-anno"
    src_attrs = raw / "CelebAMask-HQ-attribute-anno.txt"

    for path in (src_images, src_masks, src_attrs):
        if not path.exists():
            raise SystemExit(f"No existe: {path}")

    out = Path(args.out)
    out_images = out / "images"
    out_masks = out / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # 1) Copiar/redimensionar imágenes + combinar máscaras
    image_files = sorted(
        src_images.glob("*.jpg"),
        key=lambda p: int(p.stem),
    )
    if args.limit:
        image_files = image_files[: args.limit]

    print(f"Procesando {len(image_files)} imágenes...")
    for img_path in tqdm(image_files, unit="img"):
        image_id = int(img_path.stem)

        # Imagen
        img = Image.open(img_path).convert("RGB")
        if img.size != (args.image_size, args.image_size):
            img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
        img.save(out_images / f"{image_id}.jpg", quality=92)

        # Máscara combinada
        mask = build_mask(
            src_masks, image_id,
            height=args.mask_size, width=args.mask_size,
        )
        Image.fromarray(mask, mode="L").save(out_masks / f"{image_id}.png")

    # 2) Filtrar archivo de atributos a los IDs procesados
    processed_ids = {int(p.stem) for p in out_images.glob("*.jpg")}
    names = parse_attribute_header(src_attrs)
    out_attrs = out / "attributes.txt"

    with open(out_attrs, "w") as f:
        f.write(f"{len(processed_ids)}\n")
        f.write(" ".join(names) + "\n")
        for filename, values in parse_attribute_rows(src_attrs):
            stem = Path(filename).stem
            if not stem.isdigit():
                continue
            if int(stem) not in processed_ids:
                continue
            row = [filename] + [
                "1" if values[n] == 1 else "-1" for n in names
            ]
            f.write(" ".join(row) + "\n")

    print(f"\nListo:")
    print(f"  Imágenes : {out_images} ({len(processed_ids)} archivos)")
    print(f"  Máscaras : {out_masks}")
    print(f"  Atributos: {out_attrs}")


if __name__ == "__main__":
    main()
