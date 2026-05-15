"""Inferencia con U-Net multi-tarea entrenado.

Salida:
    - Máscara colorida superpuesta sobre la imagen original
    - Texto con la predicción de atributos (sexo, edad, sonrisa, etc.)

Mejoras vs. la versión inicial:
    - Detección y recorte de cara antes de entregársela al U-Net
      (mucho mejor precisión cuando la cara no llena el frame).
    - Corrección de prior por atributo: el modelo se entrenó con
      una distribución sesgada de CelebA (~78% jóvenes); restamos
      logit(prior) al logit para que la decisión 0.5 sea matemáticamente
      justa, independientemente de qué tan común sea el atributo.
    - Test-Time Augmentation con flip horizontal: promedia logits
      de la imagen original y su espejo. Reduce ruido y sesgo lateral.
    - Multi-cara: una predicción por rostro detectado.

Uso:
    python -m src.infer --checkpoint models/unet_multitask_best.pt --image foto.jpg
    python -m src.infer --checkpoint models/unet_multitask_best.pt --dir fotos/ --save reports/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from src.config import (
    ATTRS,
    ATTR_THRESHOLD,
    ATTR_THRESHOLDS,
    ATTR_PRIORS,
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


def apply_prior_correction(probs, priors=None):
    """Corrige el sesgo de prior de la sigmoid del modelo.

    El modelo se entrenó con BCE sobre una distribución donde, por
    ejemplo, 78% de las imágenes son "jóvenes". Eso desplaza la
    sigmoid hacia la clase mayoritaria. Si restamos `logit(prior)`
    al logit antes de tomar la decisión, equivalemos el comportamiento
    a un dataset 50/50 — eliminando el sesgo de clase de forma
    matemáticamente limpia.

    `probs` es np[NUM_ATTRS] con valores en (0, 1).
    Devuelve probs corregidas, también en (0, 1).
    """
    if priors is None:
        priors = ATTR_PRIORS
    p_train = np.array([priors[a] for a in ATTRS], dtype=np.float64)
    eps = 1e-6
    p = np.clip(probs.astype(np.float64), eps, 1 - eps)
    logits = np.log(p / (1 - p))
    logits_corr = logits - np.log(p_train / (1 - p_train))
    return (1.0 / (1.0 + np.exp(-logits_corr))).astype(np.float32)


@torch.no_grad()
def _forward_tensor(model, tensor, use_tta):
    """tensor: [1, 3, H, W]. Devuelve (mask uint8 [H, W], attr_logits np[NUM_ATTRS])."""
    seg_logits, attr_logits = model(tensor)
    mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    if use_tta:
        tensor_flipped = torch.flip(tensor, dims=[-1])
        _, attr_logits_flipped = model(tensor_flipped)
        attr_logits = (attr_logits + attr_logits_flipped) * 0.5

    logits_np = attr_logits.squeeze(0).cpu().numpy()
    return mask, logits_np


def _logits_to_probs(logits_np, correct_prior):
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    if correct_prior:
        probs = apply_prior_correction(probs)
    return probs.astype(np.float32)


@torch.no_grad()
def predict(model, pil_image, device, image_size=IMAGE_SIZE,
            use_tta=True, correct_prior=True):
    """Inferencia sobre una imagen PIL completa (sin detección/recorte previo).

    Devuelve (mask uint8 [image_size, image_size], probs np[NUM_ATTRS]).
    """
    img = pil_image.convert("RGB")
    if img.size != (image_size, image_size):
        img_resized = img.resize((image_size, image_size), Image.BILINEAR)
    else:
        img_resized = img
    tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
    mask, logits = _forward_tensor(model, tensor, use_tta=use_tta)
    return mask, _logits_to_probs(logits, correct_prior)


@torch.no_grad()
def predict_face_crop_bgr(model, frame_bgr, bbox, device,
                          image_size=IMAGE_SIZE,
                          use_tta=True, correct_prior=True):
    """Predice sobre el recorte de cara `bbox` dentro de un frame BGR (OpenCV).

    `bbox=(x, y, w, h)` en píxeles. Devuelve (mask uint8 [image_size, image_size],
    probs np[NUM_ATTRS]).
    """
    import cv2

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"bbox inválido: {bbox}")
    crop_bgr = frame_bgr[y:y + h, x:x + w]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(
        crop_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR,
    )
    tensor = TF.to_tensor(crop_resized).unsqueeze(0).to(device)
    mask, logits = _forward_tensor(model, tensor, use_tta=use_tta)
    return mask, _logits_to_probs(logits, correct_prior)


def colorize_mask(mask):
    """Convierte máscara [H,W] de IDs en RGB usando SEG_PALETTE."""
    palette = np.array(SEG_PALETTE, dtype=np.uint8)
    return palette[mask]


def overlay(pil_image, mask, alpha=0.55, image_size=IMAGE_SIZE):
    """Devuelve la imagen original con la máscara coloreada superpuesta."""
    img = pil_image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    img_arr = np.array(img)
    color_mask = colorize_mask(mask)
    visible = mask > 0
    out = img_arr.copy()
    out[visible] = (
        alpha * color_mask[visible] + (1 - alpha) * img_arr[visible]
    ).astype(np.uint8)
    return Image.fromarray(out)


def describe_attributes(attr_probs, threshold=None):
    p = {name: float(prob) for name, prob in zip(ATTRS, attr_probs)}
    thr = ATTR_THRESHOLDS if threshold is None else {a: threshold for a in ATTRS}

    sexo = "Hombre" if p["Male"] >= thr["Male"] else "Mujer"
    edad = "joven"  if p["Young"] >= thr["Young"] else "mayor"

    adornos = []
    if p["Smiling"]    >= thr["Smiling"]:    adornos.append("sonriendo")
    if p["Eyeglasses"] >= thr["Eyeglasses"]: adornos.append("con lentes")
    if p["Beard"]      >= thr["Beard"]:      adornos.append("con barba")
    if p["Mustache"]   >= thr["Mustache"]:   adornos.append("con bigote")

    descripcion = f"{sexo}, {edad}"
    if adornos:
        descripcion += ", " + ", ".join(adornos)
    detalle = ", ".join(f"{name}={p[name]*100:.0f}%" for name in ATTRS)
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


def predict_image_with_face_detect(
    model, pil_image, device, image_size=IMAGE_SIZE,
    use_tta=True, correct_prior=True,
):
    """Detecta caras en la imagen y predice por recorte.

    Devuelve lista de dicts: [{"bbox": (x,y,w,h), "mask": np, "probs": np}].
    Si no detecta ninguna cara, hace fallback a la imagen completa
    (igual que `predict`) y devuelve una sola entrada con bbox = full image.
    """
    import cv2
    from src.visualize_mesh import detect_faces

    pil_image = pil_image.convert("RGB")
    img_rgb = np.array(pil_image)
    frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    faces = detect_faces(frame_bgr, static=True)
    if not faces:
        mask, probs = predict(
            model, pil_image, device, image_size=image_size,
            use_tta=use_tta, correct_prior=correct_prior,
        )
        H, W = img_rgb.shape[:2]
        return [{"bbox": (0, 0, W, H), "mask": mask, "probs": probs}]

    results = []
    for f in faces:
        mask, probs = predict_face_crop_bgr(
            model, frame_bgr, f["bbox"], device,
            image_size=image_size, use_tta=use_tta, correct_prior=correct_prior,
        )
        results.append({"bbox": f["bbox"], "mask": mask, "probs": probs})
    return results


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
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Si se pasa, usa este threshold uniforme para todos los atributos. "
             "Si no, se usan los per-atributo definidos en config.ATTR_THRESHOLDS.",
    )
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
    parser.add_argument(
        "--no-face-detect", action="store_true",
        help="No detectar/recortar cara; corre el U-Net sobre la imagen completa "
             "(modo antiguo, peor precisión si la cara no llena el frame).",
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Desactiva test-time augmentation (flip horizontal).",
    )
    parser.add_argument(
        "--no-prior-correction", action="store_true",
        help="Desactiva la corrección de prior. Útil para depurar el sesgo crudo.",
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

    use_tta = not args.no_tta
    correct_prior = not args.no_prior_correction

    for path in paths:
        pil = Image.open(path)

        if args.no_face_detect:
            mask, probs = predict(
                model, pil, device, image_size=args.image_size,
                use_tta=use_tta, correct_prior=correct_prior,
            )
            results = [{"bbox": None, "mask": mask, "probs": probs}]
        else:
            results = predict_image_with_face_detect(
                model, pil, device, image_size=args.image_size,
                use_tta=use_tta, correct_prior=correct_prior,
            )

        print(f"\n{path.name}: {len(results)} cara(s)")
        descs = []
        for i, r in enumerate(results):
            desc, detalle = describe_attributes(r["probs"], threshold=args.threshold)
            print(f"  [{i+1}] {desc}")
            print(f"      {detalle}")
            descs.append(desc)

        # Visualización: si hay 1 cara o no detectó, usa el flujo original.
        # Con múltiples caras, mostramos la primera (las demás solo en texto).
        primary = results[0]
        primary_mask = primary["mask"]

        if args.style == "seg":
            overlay_img = overlay(pil, primary_mask, image_size=args.image_size)
        elif args.style == "mesh":
            overlay_img = mesh_overlay(
                pil, image_size=args.image_size, darken=args.darken,
            )
        else:  # both
            overlay_img = mesh_overlay(
                pil, image_size=args.image_size, darken=args.darken,
                with_seg_outline=primary_mask,
            )

        original = pil.convert("RGB").resize(
            (args.image_size, args.image_size), Image.BILINEAR
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original)
        axes[0].set_title(path.name)
        axes[0].axis("off")
        axes[1].imshow(overlay_img)
        axes[1].set_title(" | ".join(descs))
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
