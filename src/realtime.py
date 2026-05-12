"""Demo en tiempo real (webcam).

Muestra cada frame al lado de su máscara superpuesta y escribe el texto
descriptivo de atributos (Sexo, edad, sonrisa, lentes...) en pantalla.

Uso:
    python -m src.realtime
    python -m src.realtime --checkpoint models/unet_multitask_best.pt --every 2
"""

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.config import ATTR_THRESHOLD, IMAGE_SIZE, MODELS_DIR, SEG_PALETTE
from src.infer import describe_attributes, load_model
from src.model import get_device


def colorize_mask_bgr(mask):
    """Convierte máscara IDs en imagen BGR (OpenCV)."""
    palette_rgb = np.array(SEG_PALETTE, dtype=np.uint8)
    rgb = palette_rgb[mask]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def predict_bgr(model, frame_bgr, device, image_size):
    """frame_bgr: HxWx3 uint8. Devuelve (mask uint8 image_size, attr_probs np)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    seg_logits, attr_logits = model(tensor)
    mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    probs = torch.sigmoid(attr_logits).squeeze(0).cpu().numpy()
    return mask, probs


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam U-Net multi-tarea")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(MODELS_DIR / "unet_multitask_best.pt"),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=ATTR_THRESHOLD)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--every", type=int, default=1,
        help="Procesa 1 de cada N frames (>=1). Sube el valor si el FPS es bajo.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.55,
        help="Opacidad de la máscara superpuesta (0..1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    model, ckpt = load_model(args.checkpoint, device)
    print(f"Checkpoint epoch={ckpt.get('epoch')} mIoU={ckpt.get('mean_iou', 0):.4f}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.camera}")

    last_mask = None
    last_desc = ""
    last_detalle = ""
    frame_idx = 0
    t_prev = time.time()
    fps = 0.0

    print("Pulsa Q o Esc para salir.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % args.every == 0:
                mask, probs = predict_bgr(model, frame, device, args.image_size)
                last_mask = mask
                last_desc, last_detalle = describe_attributes(
                    probs, threshold=args.threshold
                )

            display = frame.copy()
            h, w = display.shape[:2]

            if last_mask is not None:
                color_mask = colorize_mask_bgr(last_mask)
                color_mask = cv2.resize(color_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                visible = cv2.resize(
                    (last_mask > 0).astype(np.uint8) * 255, (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
                visible3 = cv2.merge([visible, visible, visible]) > 0
                blended = cv2.addWeighted(display, 1 - args.alpha, color_mask, args.alpha, 0)
                display = np.where(visible3, blended, display)

            # Texto descriptivo arriba
            cv2.putText(
                display, last_desc, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                display, last_detalle, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA,
            )

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(
                display, f"FPS: {fps:5.1f}", (w - 130, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

            cv2.imshow("U-Net multi-tarea - rostro y atributos", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
