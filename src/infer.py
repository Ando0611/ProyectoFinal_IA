"""Inferencia con U-Net multi-tarea entrenado.

Salida:
    - Máscara colorida superpuesta sobre la imagen original
    - Texto con la predicción de atributos (sexo, edad, sonrisa, etc.)

Uso:
    python -m src.infer --checkpoint models/unet_multitask_best.pt --image foto.jpg
    python -m src.infer --checkpoint models/unet_multitask_best.pt --dir fotos/ --save reports/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from src.config import (
    ATTRS,
    ATTR_THRESHOLD,
    IMAGE_SIZE,
    MODELS_DIR,
    SEG_CLASSES,
    SEG_PALETTE,
)
from src.model import UNetMultiTask, get_device


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    model = UNetMultiTask(base_channels=cfg.get("base_channels", 32))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.train(False)
    return model, checkpoint


@torch.no_grad()
def predict(model, pil_image, device, image_size=IMAGE_SIZE):
    """Devuelve (mask_np uint8 [H,W], attr_probs np[NUM_ATTRS]).

    H,W coinciden con image_size; el caller decide si re-escala al tamaño original.
    """
    img = pil_image.convert("RGB")
    if img.size != (image_size, image_size):
        img_resized = img.resize((image_size, image_size), Image.BILINEAR)
    else:
        img_resized = img
    tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    seg_logits, attr_logits = model(tensor)
    mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    attr_probs = torch.sigmoid(attr_logits).squeeze(0).cpu().numpy()
    return mask, attr_probs


def colorize_mask(mask):
    """Convierte máscara [H,W] de IDs en RGB usando SEG_PALETTE."""
    palette = np.array(SEG_PALETTE, dtype=np.uint8)
    return palette[mask]


def overlay(pil_image, mask, alpha=0.55, image_size=IMAGE_SIZE):
    """Devuelve la imagen original con la máscara coloreada superpuesta."""
    img = pil_image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    img_arr = np.array(img)
    color_mask = colorize_mask(mask)
    # No mezclamos sobre el fondo (id=0) para mantener la imagen visible
    visible = mask > 0
    out = img_arr.copy()
    out[visible] = (
        alpha * color_mask[visible] + (1 - alpha) * img_arr[visible]
    ).astype(np.uint8)
    return Image.fromarray(out)


def describe_attributes(attr_probs, threshold=ATTR_THRESHOLD):
    """Convierte el vector de probabilidades en frases legibles."""
    p = {name: float(prob) for name, prob in zip(ATTRS, attr_probs)}

    # Sexo
    sexo = "Hombre" if p["Male"] >= threshold else "Mujer"
    edad = "joven" if p["Young"] >= threshold else "mayor"

    adornos = []
    if p["Smiling"] >= threshold:
        adornos.append("sonriendo")
    if p["Eyeglasses"] >= threshold:
        adornos.append("con lentes")
    if p["Beard"] >= threshold:
        adornos.append("con barba")
    if p["Mustache"] >= threshold:
        adornos.append("con bigote")

    descripcion = f"{sexo}, {edad}"
    if adornos:
        descripcion += ", " + ", ".join(adornos)

    detalle = ", ".join(
        f"{name}={p[name]*100:.0f}%" for name in ATTRS
    )
    return descripcion, detalle


def mesh_overlay(pil_image, image_size, darken=0.55, with_seg_outline=None):
    """Aplica el wireframe sci-fi (MediaPipe FaceMesh) sobre una imagen PIL.

    Si `with_seg_outline` es una máscara uint8 [H,W], además dibuja los
    bordes finos de las regiones segmentadas.
    """
    from src.visualize_mesh import draw_face_mesh, draw_seg_outline
    import cv2

    img_rgb = np.array(pil_image.convert("RGB").resize(
        (image_size, image_size), Image.BILINEAR
    ))
    frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    draw_face_mesh(frame_bgr, static=True, darken=darken)
    if with_seg_outline is not None:
        draw_seg_outline(frame_bgr, with_seg_outline)
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia U-Net multi-tarea")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(MODELS_DIR / "unet_multitask_best.pt"),
    )
    parser.add_argument("--image", type=str, help="Imagen para inferir")
    parser.add_argument("--dir", type=str, help="Directorio con imágenes")
    parser.add_argument(
        "--save", type=str, default=None,
        help="Guarda visualizaciones en este directorio en lugar de mostrarlas",
    )
    parser.add_argument("--threshold", type=float, default=ATTR_THRESHOLD)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--style", type=str, default="seg",
        choices=["seg", "mesh", "both"],
        help="seg = máscara coloreada; mesh = wireframe sci-fi; both = ambos",
    )
    parser.add_argument(
        "--darken", type=float, default=0.55,
        help="Oscurece la imagen antes de pintar el mesh (solo mesh/both)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    model, ckpt = load_model(args.checkpoint, device)
    print(f"Checkpoint: epoch={ckpt.get('epoch')} mIoU={ckpt.get('mean_iou', 0):.4f}")

    if args.image:
        paths = [Path(args.image)]
    elif args.dir:
        d = Path(args.dir)
        paths = sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")) + sorted(d.glob("*.png"))
        if not paths:
            raise RuntimeError(f"No hay imágenes en {args.dir}")
    else:
        raise SystemExit("Usa --image PATH o --dir DIR")

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        pil = Image.open(path)
        mask, probs = predict(model, pil, device, image_size=args.image_size)
        desc, detalle = describe_attributes(probs, threshold=args.threshold)

        print(f"\n{path.name}: {desc}")
        print(f"  {detalle}")

        if args.style == "seg":
            overlay_img = overlay(pil, mask, image_size=args.image_size)
        elif args.style == "mesh":
            overlay_img = mesh_overlay(
                pil, image_size=args.image_size, darken=args.darken,
            )
        else:  # both
            overlay_img = mesh_overlay(
                pil, image_size=args.image_size, darken=args.darken,
                with_seg_outline=mask,
            )

        original = pil.convert("RGB").resize(
            (args.image_size, args.image_size), Image.BILINEAR
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original)
        axes[0].set_title(path.name)
        axes[0].axis("off")
        axes[1].imshow(overlay_img)
        axes[1].set_title(desc)
        axes[1].axis("off")
        fig.tight_layout()

        if save_dir:
            out = save_dir / f"{path.stem}_pred.jpg"
            fig.savefig(out, bbox_inches="tight", dpi=120)
            plt.close(fig)
            print(f"  guardado: {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
