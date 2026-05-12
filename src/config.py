"""Configuración central del proyecto.

Reconocimiento facial multi-tarea sobre CelebAMask-HQ:
- Segmentación pixel-level de 19 partes faciales
- Clasificación global de 6 atributos binarios
"""

import os
from pathlib import Path


# Raíces
#
# Las rutas de datos/modelos/logs se pueden sobrescribir con variables
# de entorno. Útil en Kaggle / Colab donde los datos viven en
# /kaggle/input/... y las salidas deben ir a /kaggle/working/...
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PROJECT_DATA_DIR", PROJECT_ROOT / "data" / "celeba-hq"))
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
ATTRIBUTES_FILE = DATA_DIR / "attributes.txt"
MODELS_DIR = Path(os.environ.get("PROJECT_MODELS_DIR", PROJECT_ROOT / "models"))
LOGS_DIR = Path(os.environ.get("PROJECT_LOGS_DIR", PROJECT_ROOT / "logs"))


# Clases de segmentación (19 incluyendo fondo)
#
# Estos nombres y orden coinciden con la convención oficial de
# CelebAMask-HQ. El índice 0 es fondo; cada píxel de la máscara
# tiene un entero 0..18.
SEG_CLASSES = [
    "background",  # 0
    "skin",        # 1
    "l_brow",      # 2
    "r_brow",      # 3
    "l_eye",       # 4
    "r_eye",       # 5
    "eye_g",       # 6  (gafas / lentes)
    "l_ear",       # 7
    "r_ear",       # 8
    "ear_r",       # 9  (aretes)
    "nose",        # 10
    "mouth",       # 11 (interior de la boca)
    "u_lip",       # 12 (labio superior)
    "l_lip",       # 13 (labio inferior)
    "neck",        # 14
    "neck_l",      # 15 (collar)
    "cloth",       # 16 (ropa)
    "hair",        # 17
    "hat",         # 18
]
NUM_SEG_CLASSES = len(SEG_CLASSES)

# Mapa nombre -> id, usado por el preprocesador para combinar las máscaras
# binarias individuales que provee CelebAMask-HQ.
SEG_NAME_TO_ID = {name: i for i, name in enumerate(SEG_CLASSES)}


# Atributos binarios a predecir (subset relevante de los 40 de CelebA)
#
# CelebA usa convención +1 / -1 en el archivo de atributos; el dataset
# normaliza a 1 / 0 internamente. "No_Beard" se invierte a "Beard"
# para que 1 = atributo presente en todos los casos.
ATTRS = [
    "Male",         # 1 = hombre,  0 = mujer
    "Young",        # 1 = joven,   0 = no joven
    "Smiling",      # 1 = sonriendo
    "Eyeglasses",   # 1 = con lentes
    "Mustache",     # 1 = bigote visible
    "Beard",        # 1 = barba (invertido de No_Beard en CelebA)
]
NUM_ATTRS = len(ATTRS)


# Paleta para visualizar la segmentación
# (RGB; un color distinto por clase, con el fondo en negro)
SEG_PALETTE = [
    (0, 0, 0),         # background
    (204, 0, 0),       # skin
    (76, 153, 0),      # l_brow
    (204, 204, 0),     # r_brow
    (51, 51, 255),     # l_eye
    (204, 0, 204),     # r_eye
    (0, 255, 255),     # eye_g
    (255, 204, 204),   # l_ear
    (102, 51, 0),      # r_ear
    (255, 0, 0),       # ear_r
    (102, 204, 0),     # nose
    (255, 255, 0),     # mouth
    (0, 0, 153),       # u_lip
    (0, 0, 204),       # l_lip
    (255, 51, 153),    # neck
    (0, 204, 204),     # neck_l
    (0, 51, 0),        # cloth
    (255, 153, 51),    # hair
    (0, 204, 0),       # hat
]


# Dataset
IMAGE_SIZE = 256          # H = W = 256 para entrenamiento
TRAIN_RATIO = 0.85
SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 0


# Entrenamiento
NUM_EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_ATTR = 1.0         # peso de la pérdida de atributos vs segmentación


# Inferencia
ATTR_THRESHOLD = 0.5
