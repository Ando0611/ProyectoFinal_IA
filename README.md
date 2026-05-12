# Reconocimiento facial con segmentación + atributos

Proyecto académico de Inteligencia Artificial (UP).
**U-Net multi-tarea desde cero (PyTorch)** sobre **CelebAMask-HQ** que aprende simultáneamente:

1. **Segmentación pixel-level** de 19 partes faciales (piel, cejas, ojos, gafas, orejas, aretes, nariz, boca, labios, cuello, collar, ropa, pelo, sombrero, fondo).
2. **Clasificación global** de 6 atributos binarios: `Male`, `Young`, `Smiling`, `Eyeglasses`, `Mustache`, `Beard`.

La demo en tiempo real muestra la máscara coloreada sobre la cara y describe a la persona en una línea: *"Mujer, joven, sonriendo, con lentes."*

## Arquitectura

```
Encoder (5 niveles, base_channels=32)
   DoubleConv -> MaxPool (×4) -> bottleneck DoubleConv
Decoder (4 niveles)
   UpSample bilinear + skip + DoubleConv
Cabeza de segmentación
   Conv 1×1  ->  [B, 19, H, W]   (logits por píxel)
Cabeza de clasificación (sobre el bottleneck)
   GlobalAvgPool -> Linear(c5, 128) -> ReLU -> Dropout -> Linear(128, 6)
```

Todo está implementado desde cero en `src/model.py` (U-Net), `src/losses.py` (CE + λ·BCE) y `src/metrics.py` (mIoU + accuracy/F1 por atributo). Sin pesos pre-entrenados.

## Requisitos

- macOS (Apple Silicon recomendado) o Linux
- Python 3.11+
- ~5 GB de espacio para el dataset preprocesado
- Webcam (opcional)

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verificar PyTorch + MPS:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

### Entrenar en Kaggle (GPU gratis)

Si tu máquina no tiene GPU, abre [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) en Kaggle:

1. Crear un Notebook, **Settings → Accelerator → GPU (T4 o P100)**.
2. **Add Input → Search Datasets →** `CelebAMask-HQ` y agregarlo.
3. **File → Import Notebook →** subir `notebooks/kaggle_train.ipynb`.
4. **Run All**.

El notebook clona el repo, localiza el dataset, preprocesa, entrena y muestra ejemplos de inferencia. Las rutas se configuran vía las variables de entorno `PROJECT_DATA_DIR`, `PROJECT_MODELS_DIR` y `PROJECT_LOGS_DIR` para que los outputs caigan en `/kaggle/working/` (lo único persistente al cerrar la sesión).

## Dataset — CelebAMask-HQ

Ver [`docs/DATASET.md`](docs/DATASET.md) para el flujo completo. Resumen:

1. Descargar `CelebAMask-HQ.zip` desde el Google Drive oficial (gratis para uso académico).
2. Descomprimir en un directorio temporal.
3. Correr el preprocesador para combinar máscaras y dejar la estructura final:

```bash
python -m data.preprocess_celebamaskhq \
    --raw /ruta/a/CelebAMask-HQ \
    --out data/celeba-hq \
    --limit 5000
```

Esto produce:

```
data/celeba-hq/
  images/N.jpg       (512×512)
  masks/N.png        (512×512, valores 0..18)
  attributes.txt
```

> Para entrenar más rápido en el M4, empieza con `--limit 3000` o `5000`.

## Entrenamiento

```bash
python -m src.train                                  # 15 épocas con defaults
python -m src.train --epochs 30 --batch-size 4
python -m src.train --lambda-attr 0.5                # baja el peso de atributos
python -m src.train --base-channels 16               # modelo más pequeño / más rápido
```

Defaults:

- Optimizador: Adam lr=1e-3, weight_decay=1e-4
- Loss: CE(seg) + 1.0 · BCE(atributos)
- Split: 85/15 train/valid determinista (seed=42)
- Augmentación: flip horizontal con swap de `l_*` ↔ `r_*` + color jitter ligero

Por época se reportan: loss total, loss de segmentación, loss de atributos, **mIoU** sobre las 19 clases, **accuracy** y **F1** por atributo.

Salidas:

- `models/unet_multitask_best.pt` (mejor mIoU)
- `logs/history.json`

## Inferencia sobre imágenes

```bash
python -m src.infer --image foto.jpg
python -m src.infer --dir fotos/ --save reports/predicciones
```

Muestra (o guarda) un panel con: imagen original + imagen con la máscara colorida superpuesta, y debajo el texto descriptivo de atributos.

## Demo en vivo (webcam)

```bash
python -m src.realtime
python -m src.realtime --camera 1 --every 2 --alpha 0.6
```

Pulsa `Q` o `Esc` para salir. Si el FPS está bajo, sube `--every` (procesa 1 de cada N frames).

## Tests

```bash
pytest -v
```

Cubren las shapes de la red, el cálculo de IoU y de accuracy/F1 por atributo.

## Estructura

```
.
├── data/
│   ├── preprocess_celebamaskhq.py  (combina máscaras binarias en una sola)
│   └── celeba-hq/
│       ├── images/      N.jpg
│       ├── masks/       N.png    (0..18)
│       └── attributes.txt
├── docs/
│   └── DATASET.md
├── src/
│   ├── config.py       (clases, atributos, hiperparámetros)
│   ├── dataset.py      (CelebAMaskHQDataset + aumentaciones)
│   ├── model.py        (U-Net multi-tarea desde cero)
│   ├── losses.py       (CE + λ·BCE)
│   ├── metrics.py      (mIoU + accuracy/F1)
│   ├── train.py
│   ├── infer.py
│   └── realtime.py
├── tests/
│   ├── test_model.py
│   └── test_metrics.py
└── requirements.txt
```

## Nota técnica

El backbone es un U-Net **entrenado desde cero** (He init, sin pesos pre-entrenados). Con `--limit 5000` y 15 épocas en M4 (MPS), se llega a un mIoU razonable en ~30–60 minutos. Para más calidad, sube a 10,000+ imágenes y 25+ épocas.

## Cita del dataset

> Lee, C.-H., Liu, Z., Wu, L., & Luo, P. (2020). *MaskGAN: Towards Diverse and Interactive Facial Image Manipulation.* CVPR.
