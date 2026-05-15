"""Wireframe estilo sci-fi sobre el rostro usando MediaPipe FaceMesh.

Soporta múltiples caras simultáneas: el FaceMesh se inicializa con
`max_num_faces` configurable (default 4) y todas las funciones iteran
sobre cada rostro detectado.

También expone `detect_faces` para obtener bounding boxes a partir de
los landmarks, lo cual permite recortar cada cara y entregarla al
U-Net por separado (mucho mejor precisión a distancia).
"""

import cv2
import numpy as np


# Paleta del efecto (BGR porque OpenCV usa BGR)
LINE_COLOR = (220, 180, 90)         # cian claro / azul-celeste
DOT_OUTER_COLOR = (255, 210, 130)   # halo del punto
DOT_INNER_COLOR = (255, 255, 255)   # núcleo brillante
LINE_THICKNESS = 1
DOT_OUTER_RADIUS = 2
DOT_INNER_RADIUS = 1

DEFAULT_MAX_FACES = 4

# Cache de FaceMesh por (static, max_faces). MediaPipe es pesado de
# inicializar, así que reusamos la instancia mientras la config no cambie.
_FACE_MESH_CACHE = {}


def _get_face_mesh(static, max_faces=DEFAULT_MAX_FACES):
    """Devuelve un FaceMesh reusable, inicializado en modo video o foto."""
    import mediapipe as mp

    key = (bool(static), int(max_faces))
    inst = _FACE_MESH_CACHE.get(key)
    if inst is not None:
        return inst

    if static:
        inst = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    else:
        inst = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    _FACE_MESH_CACHE[key] = inst
    return inst


def _process(frame_bgr, static, max_faces):
    """Corre FaceMesh y devuelve `results`."""
    face_mesh = _get_face_mesh(static=static, max_faces=max_faces)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return face_mesh.process(rgb)


def detect_faces(frame_bgr, static=False, max_faces=DEFAULT_MAX_FACES, pad=0.25):
    """Detecta caras y devuelve lista de dicts:

        [{"bbox": (x, y, w, h), "landmarks": [(x_px, y_px), ...]}, ...]

    `bbox` ya viene expandido con `pad` (porcentaje relativo al tamaño del
    rostro) y recortado a los límites del frame. `landmarks` son las 478
    coordenadas en píxeles (en el sistema del frame original).
    """
    results = _process(frame_bgr, static=static, max_faces=max_faces)
    if not results.multi_face_landmarks:
        return []

    H, W = frame_bgr.shape[:2]
    faces = []
    for face_landmarks in results.multi_face_landmarks:
        coords = [
            (int(lm.x * W), int(lm.y * H))
            for lm in face_landmarks.landmark
        ]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min
        h = y_max - y_min

        # Expandimos el bbox: el U-Net se entrenó con caras que
        # incluyen cuello/pelo/hombros, no solo el polígono facial.
        px = int(w * pad)
        py = int(h * pad)
        x0 = max(0, x_min - px)
        y0 = max(0, y_min - py)
        x1 = min(W, x_max + px)
        y1 = min(H, y_max + py)

        faces.append({
            "bbox": (x0, y0, x1 - x0, y1 - y0),
            "landmarks": coords,
        })
    return faces


def has_face(frame_bgr, static=False, max_faces=DEFAULT_MAX_FACES):
    """True si MediaPipe detecta al menos un rostro."""
    results = _process(frame_bgr, static=static, max_faces=max_faces)
    return bool(results.multi_face_landmarks)


def _darken_frame(frame_bgr, darken):
    """Oscurece + tinte azul sutil para el look sci-fi."""
    if darken >= 1.0:
        return
    frame_bgr[:] = np.clip(
        frame_bgr.astype(np.float32) * darken, 0, 255
    ).astype(np.uint8)
    tint = np.zeros_like(frame_bgr)
    tint[..., 0] = 25  # canal B
    frame_bgr[:] = cv2.add(frame_bgr, tint)


def draw_face_mesh(
    frame_bgr, static=False, darken=0.55, max_faces=DEFAULT_MAX_FACES,
    faces=None,
):
    """Dibuja el wireframe **in place** sobre todas las caras detectadas.

    Si `faces` se pasa (resultado de `detect_faces`), se usa directamente
    y no se vuelve a correr MediaPipe (útil cuando ya detectamos caras
    arriba para recortar y pasarlas al U-Net).
    """
    import mediapipe as mp

    if faces is None:
        results = _process(frame_bgr, static=static, max_faces=max_faces)
        face_coords = []
        if results.multi_face_landmarks:
            H, W = frame_bgr.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                face_coords.append([
                    (int(lm.x * W), int(lm.y * H))
                    for lm in face_landmarks.landmark
                ])
    else:
        face_coords = [f["landmarks"] for f in faces]

    _darken_frame(frame_bgr, darken)

    if not face_coords:
        return frame_bgr

    tess = mp.solutions.face_mesh.FACEMESH_TESSELATION

    for coords in face_coords:
        for i, j in tess:
            cv2.line(
                frame_bgr, coords[i], coords[j],
                LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA,
            )
        for x, y in coords:
            cv2.circle(frame_bgr, (x, y), DOT_OUTER_RADIUS, DOT_OUTER_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (x, y), DOT_INNER_RADIUS, DOT_INNER_COLOR, -1, cv2.LINE_AA)

    return frame_bgr


def draw_seg_outline(frame_bgr, mask, color=(220, 180, 90), thickness=1, bbox=None):
    """Dibuja solo los **bordes** de cada región segmentada (no rellena).

    Si `bbox=(x,y,w,h)` se pasa, la máscara se reescala a (w,h) y se
    pinta solo dentro de esa ROI (útil cuando el U-Net se corrió sobre
    un recorte de la cara). Sin bbox, se reescala al frame completo
    como antes.
    """
    H, W = frame_bgr.shape[:2]
    if bbox is None:
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        _paint_outline(frame_bgr, mask, color, thickness)
        return frame_bgr

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return frame_bgr
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    roi = frame_bgr[y:y + h, x:x + w]
    _paint_outline(roi, mask_resized, color, thickness)
    return frame_bgr


def _paint_outline(roi_bgr, mask, color, thickness):
    """Helper: pinta los bordes de `mask` sobre `roi_bgr` (modificándolo)."""
    dx = np.abs(np.diff(mask.astype(np.int16), axis=1, prepend=mask[:, :1]))
    dy = np.abs(np.diff(mask.astype(np.int16), axis=0, prepend=mask[:1, :]))
    edges = ((dx > 0) | (dy > 0)).astype(np.uint8) * 255
    if thickness > 1:
        kernel = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    mask3 = cv2.merge([edges, edges, edges]) > 0
    overlay = np.full_like(roi_bgr, color, dtype=np.uint8)
    roi_bgr[:] = np.where(mask3, overlay, roi_bgr)
