"""Wireframe estilo sci-fi sobre el rostro usando MediaPipe FaceMesh.

Complementa la salida del U-Net (segmentación + atributos) con un mesh
de 468 landmarks conectados, similar al típico visual de "AI face
recognition". El mesh NO sustituye a la red multi-tarea: solo le
agrega el efecto visual encima del frame.

Mediapipe se importa de forma perezosa para que el resto del proyecto
(tests, train) no dependa de él.
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


_FACE_MESH_VIDEO = None
_FACE_MESH_STATIC = None


def _get_face_mesh(static):
    """Devuelve un FaceMesh reusable, inicializado en modo video o foto."""
    global _FACE_MESH_VIDEO, _FACE_MESH_STATIC
    import mediapipe as mp

    if static:
        if _FACE_MESH_STATIC is None:
            _FACE_MESH_STATIC = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        return _FACE_MESH_STATIC

    if _FACE_MESH_VIDEO is None:
        _FACE_MESH_VIDEO = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _FACE_MESH_VIDEO


def draw_face_mesh(frame_bgr, static=False, darken=0.55):
    """Dibuja el wireframe sobre `frame_bgr` (uint8 HxWx3) **in place**.

    - `static=True` para imágenes sueltas, `False` para video.
    - `darken=0.55` oscurece la imagen primero para que el mesh resalte.
      Pon `darken=1.0` para no oscurecer.
    """
    import mediapipe as mp

    face_mesh = _get_face_mesh(static=static)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if darken < 1.0:
        # Oscurece y agrega un sutil tinte azul para acercarnos al look sci-fi
        frame_bgr[:] = np.clip(frame_bgr.astype(np.float32) * darken, 0, 255).astype(np.uint8)
        tint = np.zeros_like(frame_bgr)
        tint[..., 0] = 25  # canal B
        frame_bgr[:] = cv2.add(frame_bgr, tint)

    if not results.multi_face_landmarks:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    tess = mp.solutions.face_mesh.FACEMESH_TESSELATION

    for face_landmarks in results.multi_face_landmarks:
        coords = [
            (int(lm.x * w), int(lm.y * h))
            for lm in face_landmarks.landmark
        ]

        for i, j in tess:
            cv2.line(
                frame_bgr, coords[i], coords[j],
                LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA,
            )

        for x, y in coords:
            cv2.circle(frame_bgr, (x, y), DOT_OUTER_RADIUS, DOT_OUTER_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (x, y), DOT_INNER_RADIUS, DOT_INNER_COLOR, -1, cv2.LINE_AA)

    return frame_bgr


def draw_seg_outline(frame_bgr, mask, color=(220, 180, 90), thickness=1):
    """Dibuja solo los **bordes** de cada región segmentada (no rellena).

    Útil para combinar con el mesh: en lugar de pintar la piel/cabello con
    colores opacos, solo se trazan las fronteras entre clases.
    `mask` es uint8 [H,W] con valores 0..N. Se redimensiona a frame_bgr.
    """
    h, w = frame_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Gradiente discreto: el borde aparece donde cambia la clase
    dx = np.abs(np.diff(mask.astype(np.int16), axis=1, prepend=mask[:, :1]))
    dy = np.abs(np.diff(mask.astype(np.int16), axis=0, prepend=mask[:1, :]))
    edges = ((dx > 0) | (dy > 0)).astype(np.uint8) * 255

    if thickness > 1:
        kernel = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    mask3 = cv2.merge([edges, edges, edges]) > 0
    overlay = np.full_like(frame_bgr, color, dtype=np.uint8)
    frame_bgr[:] = np.where(mask3, overlay, frame_bgr)
    return frame_bgr
