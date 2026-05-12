# Dataset — CelebAMask-HQ

Usamos **[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)**, una versión profesionalmente segmentada de CelebA con 30,000 rostros en alta resolución.

## Qué contiene

- **30,000 imágenes** 1024×1024 (`CelebA-HQ-img/0.jpg` … `29999.jpg`)
- **19 máscaras pixel-level por imagen**: piel, cejas, ojos, gafas, orejas, aretes, nariz, boca, labios, cuello, collar, ropa, pelo, sombrero, fondo
- **40 atributos binarios por imagen**: `Male`, `Young`, `Smiling`, `Eyeglasses`, `Mustache`, `No_Beard`, etc.

## Cita

> Lee, C.-H., Liu, Z., Wu, L., & Luo, P. (2020). *MaskGAN: Towards Diverse and Interactive Facial Image Manipulation.* CVPR.

## Descarga

El dataset pesa ~3 GB. La descarga oficial está en Google Drive (uso académico y no comercial):

1. Ve a [https://github.com/switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ).
2. Sigue el link "Google Drive" en la sección **Downloads**.
3. Descarga **`CelebAMask-HQ.zip`** y descomprímelo en un directorio temporal:
   ```
   CelebAMask-HQ/
     CelebA-HQ-img/
       0.jpg ... 29999.jpg
     CelebAMask-HQ-mask-anno/
       0/ 1/ ... 14/
       (cada bucket tiene 2000 imágenes con sus máscaras por parte)
     CelebAMask-HQ-attribute-anno.txt
   ```

> Tip: para un primer entrenamiento, no necesitas las 30,000 imágenes. Con 3,000–5,000 ya converge bien en M4.

## Preproceso

CelebAMask-HQ entrega **un PNG binario por parte** (p. ej. `0/00000_skin.png`, `0/00000_l_eye.png`). Hay que combinarlos en un único PNG por imagen con IDs 0..18.

Eso lo hace `data/preprocess_celebamaskhq.py`:

```bash
python -m data.preprocess_celebamaskhq \
    --raw /ruta/a/CelebAMask-HQ \
    --out data/celeba-hq \
    --limit 5000
```

Esto produce:

```
data/celeba-hq/
  images/        N.jpg     (512×512, ya redimensionado)
  masks/         N.png     (512×512, valores 0..18)
  attributes.txt           (CelebA estilo: +1/-1)
```

`--limit N` procesa solo las primeras N imágenes (útil para probar el pipeline antes de procesar las 30,000).

## Estructura final esperada por el código

```
data/celeba-hq/
├── images/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── masks/
│   ├── 0.png        # valores 0..18 según src.config.SEG_CLASSES
│   ├── 1.png
│   └── ...
└── attributes.txt
```

## Las 19 clases de segmentación

| ID | Nombre  | Qué es                  |
|----|---------|-------------------------|
| 0  | background | fondo / piel no facial |
| 1  | skin    | piel del rostro         |
| 2  | l_brow  | ceja izquierda          |
| 3  | r_brow  | ceja derecha            |
| 4  | l_eye   | ojo izquierdo           |
| 5  | r_eye   | ojo derecho             |
| 6  | eye_g   | gafas / lentes          |
| 7  | l_ear   | oreja izquierda         |
| 8  | r_ear   | oreja derecha           |
| 9  | ear_r   | aretes                  |
| 10 | nose    | nariz                   |
| 11 | mouth   | interior de la boca     |
| 12 | u_lip   | labio superior          |
| 13 | l_lip   | labio inferior          |
| 14 | neck    | cuello                  |
| 15 | neck_l  | collar                  |
| 16 | cloth   | ropa                    |
| 17 | hair    | pelo                    |
| 18 | hat     | sombrero                |

## Los 6 atributos predichos

Del archivo `attributes.txt` solo usamos un subset relevante:

- `Male` (sexo binario)
- `Young` (edad binaria)
- `Smiling`
- `Eyeglasses`
- `Mustache`
- `Beard` (= `NOT No_Beard`; invertimos para que 1 = barba presente)
