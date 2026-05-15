"""Demo en tiempo real (webcam) — versión multi-cara.

Pipeline:
    1. MediaPipe FaceMesh detecta una o varias caras (`--max-faces`).
    2. Cada cara se recorta del frame con padding, se redimensiona a
       `image_size×image_size` y se le entrega individualmente al U-Net.
       Esto da mucha más precisión cuando la cara no llena el frame
       (estás lejos de la cámara) que pasarle el frame completo.
    3. Los logits de atributos se promedian con su flip horizontal (TTA)
       y luego se corrigen por prior (ver src.infer.apply_prior_correction)
       para eliminar el sesgo de clase del dataset de entrenamiento.
    4. Cada cara muestra su descripción debajo de su bbox.

Estilos de visualización:
  - `seg`  : máscara segmentada coloreada por cara (look "filtro").
  - `mesh` : wireframe sci-fi de 478 landmarks sobre todas las caras.
  - `both` : mesh + bordes de la segmentación de cada cara.

Uso:
    python -m src.realtime
    python -m src.realtime --max-faces 6 --style both
    python -m src.realtime --no-tta --no-prior-correction  # comportamiento crudo
"""

import argparse
import time

import cv2
import numpy as np
import torch

from src.config import IMAGE_SIZE, MODELS_DIR, SEG_PALETTE
from src.infer import (
    apply_prior_correction,
    describe_attributes,
    load_model,
    predict_face_crop_bgr,
)
from src.model import get_device
from src.visualize_mesh import (
    DEFAULT_MAX_FACES,
    detect_faces,
    draw_face_mesh,
    draw_seg_outline,
)


def colorize_mask_bgr(mask):
    """Convierte máscara IDs en imagen BGR (OpenCV)."""
    palette_rgb = np.array(SEG_PALETTE, dtype=np.uint8)
    rgb = palette_rgb[mask]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_seg_at_bbox(display, mask, bbox, alpha):
    """Pinta la máscara coloreada solo dentro del bbox (no rellena toda la imagen)."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return
    color_mask = cv2.resize(
        colorize_mask_bgr(mask), (w, h), interpolation=cv2.INTER_NEAREST,
    )
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    visible = (mask_resized > 0)
    if not visible.any():
        return
    roi = display[y:y + h, x:x + w]
    blended = cv2.addWeighted(roi, 1 - alpha, color_mask, alpha, 0)
    visible3 = np.repeat(visible[..., None], 3, axis=2)
    roi[:] = np.where(visible3, blended, roi)


def draw_face_label(display, text, bbox, color=(120, 255, 200)):
    """Dibuja el texto descriptivo arriba del bbox de la cara."""
    if not text:
        return
    x, y, w, h = bbox
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4

    # Posición preferida: arriba del bbox. Si no cabe, debajo.
    label_y_top = y - 6
    if label_y_top - th - pad < 0:
        label_y_top = y + h + th + pad + 6
        label_y_top = min(label_y_top, display.shape[0] - pad)

    x1 = max(0, x)
    y1 = label_y_top - th - pad
    x2 = min(display.shape[1], x + tw + 2 * pad)
    y2 = label_y_top + pad

    # Caja de fondo semitransparente
    overlay = display.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, dst=display)
    cv2.putText(
        display, text, (x1 + pad, label_y_top - pad // 2),
        font, scale, color, thickness, cv2.LINE_AA,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam U-Net multi-tarea (multi-cara)")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(MODELS_DIR / "unet_multitask_best.pt"),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Si se pasa, usa este threshold uniforme para todos los atributos. "
             "Si no, se usan los per-atributo definidos en config.ATTR_THRESHOLDS.",
    )
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--every", type=int, default=1,
        help="Corre el U-Net 1 de cada N frames (>=1). El mesh sigue dibujándose "
             "a tasa completa para fluidez.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.55,
        help="Opacidad de la máscara superpuesta (style=seg)",
    )
    parser.add_argument(
        "--style", type=str, default="mesh",
        choices=["seg", "mesh", "both"],
        help="seg = relleno por cara; mesh = wireframe sci-fi; both = ambos",
    )
    parser.add_argument(
        "--darken", type=float, default=0.55,
        help="Oscurece el frame antes de pintar (solo en mesh/both). 1.0 = sin oscurecer.",
    )
    parser.add_argument(
        "--max-faces", type=int, default=DEFAULT_MAX_FACES,
        help=f"Máximo de caras a detectar simultáneamente (default {DEFAULT_MAX_FACES}).",
    )
    parser.add_argument(
        "--pad", type=float, default=0.25,
        help="Padding del bbox de cada cara (proporción del tamaño del rostro). "
             "El U-Net se entrenó con caras que incluyen pelo/cuello, así que "
             "necesitamos un poco de contexto.",
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Desactiva test-time augmentation (flip horizontal). Más rápido pero más sesgado.",
    )
    parser.add_argument(
        "--no-prior-correction", action="store_true",
        help="Desactiva la corrección de prior. Útil para ver el sesgo crudo del modelo.",
    )
    return parser.parse_args()


def predict_for_face(model, frame_bgr, bbox, device, image_size,
                     use_tta, correct_prior, threshold):
    """Run U-Net en el recorte y produce (mask, probs, desc)."""
    mask, probs = predict_face_crop_bgr(
        model, frame_bgr, bbox, device,
        image_size=image_size,
        use_tta=use_tta,
        correct_prior=correct_prior,
    )
    desc, detalle = describe_attributes(probs, threshold=threshold)
    return mask, probs, desc, detalle


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    model, ckpt = load_model(args.checkpoint, device)
    print(f"Checkpoint epoch={ckpt.get('epoch')} mIoU={ckpt.get('mean_iou', 0):.4f}")
    print(f"Estilo: {args.style} | max_faces: {args.max_faces} | "
          f"TTA: {not args.no_tta} | prior-correction: {not args.no_prior_correction}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.camera}")

    # Cada elemento: {"bbox", "mask", "desc", "detalle"}.
    # Se actualiza cada `--every` frames; entre tanto, se reutiliza para
    # no introducir lag en el renderizado.
    last_results = []
    last_status = "Iniciando..."
    frame_idx = 0
    t_prev = time.time()
    fps = 0.0
    use_tta = not args.no_tta
    correct_prior = not args.no_prior_correction

    print("Pulsa Q o Esc para salir.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())

            # Detectamos caras cada frame (es barato y mantiene el mesh fluido)
            if brightness < 15:
                faces = []
                last_status = "Sin deteccion (camara cubierta)"
            else:
                faces = detect_faces(
                    frame, static=False, max_faces=args.max_faces, pad=args.pad,
                )
                last_status = f"{len(faces)} cara(s)" if faces else "Sin deteccion"

            # Corremos U-Net cada N frames (lo caro)
            if frame_idx % args.every == 0:
                results = []
                for f in faces:
                    try:
                        mask, probs, desc, detalle = predict_for_face(
                            model, frame, f["bbox"], device, args.image_size,
                            use_tta=use_tta, correct_prior=correct_prior,
                            threshold=args.threshold,
                        )
                    except ValueError:
                        continue
                    results.append({
                        "bbox": f["bbox"],
                        "mask": mask,
                        "desc": desc,
                        "detalle": detalle,
                    })
                last_results = results

            # ---- Render ----
            display = frame.copy()
            h, w = display.shape[:2]

            if args.style in ("mesh", "both"):
                # Reusamos los landmarks ya detectados (evita correr MediaPipe 2x)
                draw_face_mesh(display, faces=faces, darken=args.darken)

            if args.style == "seg":
                for r in last_results:
                    render_seg_at_bbox(display, r["mask"], r["bbox"], args.alpha)
            elif args.style == "both":
                for r in last_results:
                    draw_seg_outline(display, r["mask"], bbox=r["bbox"])

            for r in last_results:
                draw_face_label(display, r["desc"], r["bbox"])

            # Status global arriba a la izquierda + detalle de la primera cara abajo
            cv2.putText(
                display, last_status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 200), 2, cv2.LINE_AA,
            )
            if last_results:
                cv2.putText(
                    display, last_results[0]["detalle"], (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA,
                )

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(
                display, f"FPS: {fps:5.1f}", (w - 130, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 200), 2, cv2.LINE_AA,
            )

            cv2.imshow("U-Net multi-tarea - multi-cara", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
