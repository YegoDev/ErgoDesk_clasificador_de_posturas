from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Configuración
# ============================

# Carpeta donde están TODAS las imágenes .jpg
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # sube un nivel desde /eda
DATA_DIR = PROJECT_ROOT / "imagenes_jpg"

print("Directorio raíz del proyecto:", PROJECT_ROOT)
print("Directorio de imágenes:", DATA_DIR)

# ============================
# 1. Recoger estadísticas
# ============================

rows = []

for img_path in DATA_DIR.rglob("*.jpg"):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[AVISO] No se pudo leer: {img_path}")
        continue

    h, w = img.shape[:2]
    orientation = "vertical" if h >= w else "horizontal"

    # Brillo promedio (canal V en HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[:, :, 2].mean())

    rows.append({
        "path": str(img_path),
        "filename": img_path.name,
        "width": w,
        "height": h,
        "orientation": orientation,
        "aspect_ratio": w / h,
        "brightness": brightness,
    })

df = pd.DataFrame(rows)
print("\n============================")
print("Total de imágenes leídas:", len(df))
print("============================\n")

if df.empty:
    print("No se encontraron imágenes. Revisa la ruta:", DATA_DIR)
    exit(0)

print("Primeras filas del dataframe:")
print(df.head(), "\n")


# ============================
# 2. Histogramas de tamaños
# ============================

def plot_hist_sizes(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["width"], bins=20)
    axes[0].set_title("Distribución de ancho (px)")
    axes[0].set_xlabel("width")
    axes[0].set_ylabel("frecuencia")

    axes[1].hist(df["height"], bins=20)
    axes[1].set_title("Distribución de alto (px)")
    axes[1].set_xlabel("height")
    axes[1].set_ylabel("frecuencia")

    plt.tight_layout()
    plt.show()

    print("Ancho mínimo / máximo:", df["width"].min(), df["width"].max())
    print("Alto mínimo / máximo:", df["height"].min(), df["height"].max())
    print()

plot_hist_sizes(df)

# ============================
# 3. Aspect ratio & orientación
# ============================

print("Orientación de las imágenes:")
print(df["orientation"].value_counts(), "\n")

def plot_aspect_ratio(df):
    plt.figure(figsize=(6, 4))
    plt.hist(df["aspect_ratio"], bins=20)
    plt.title("Distribución del aspect ratio (width / height)")
    plt.xlabel("aspect_ratio (w/h)")
    plt.ylabel("frecuencia")
    plt.tight_layout()
    plt.show()

    print("Aspect ratio - min / max:", df["aspect_ratio"].min(), df["aspect_ratio"].max())
    print()

plot_aspect_ratio(df)

# ============================
# 4. Muestra aleatoria de imágenes
# ============================

def show_random_images(df, n=16, window_title="Muestra aleatoria"):
    sample = df.sample(min(n, len(df)), random_state=42)

    cols = 4
    rows = int(np.ceil(len(sample) / cols))

    plt.figure(figsize=(4*cols, 4*rows))
    plt.suptitle(window_title, fontsize=14)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        img = cv2.imread(row["path"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(row["filename"], fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

print("Mostrando muestra aleatoria de imágenes...")
show_random_images(df, n=16, window_title="Muestra aleatoria de imágenes")

# ============================
# 5. Buscar imágenes problemáticas
# ============================

# Umbrales (ajústalos según lo que veas)
MIN_WIDTH  = 400
MIN_HEIGHT = 400

MAX_WIDTH  = 3000
MAX_HEIGHT = 3000

MIN_BRIGHTNESS = 40     # muy oscuras
MAX_BRIGHTNESS = 230    # muy claras (quemadas)

MIN_ASPECT = 0.5        # demasiado vertical
MAX_ASPECT = 2.0        # demasiado horizontal

small_imgs = df[(df["width"] < MIN_WIDTH) | (df["height"] < MIN_HEIGHT)]
large_imgs = df[(df["width"] > MAX_WIDTH) | (df["height"] > MAX_HEIGHT)]
dark_imgs  = df[df["brightness"] < MIN_BRIGHTNESS]
bright_imgs = df[df["brightness"] > MAX_BRIGHTNESS]
weird_ratio = df[(df["aspect_ratio"] < MIN_ASPECT) | (df["aspect_ratio"] > MAX_ASPECT)]

print("Imágenes muy pequeñas:", len(small_imgs))
print("Imágenes muy grandes :", len(large_imgs))
print("Imágenes muy oscuras :", len(dark_imgs))
print("Imágenes muy claras  :", len(bright_imgs))
print("Aspect ratio raro    :", len(weird_ratio))
print()

def show_images_from_df(subdf, title, n=9):
    if len(subdf) == 0:
        print(f"No se encontraron imágenes para: {title}")
        return

    sample = subdf.sample(min(n, len(subdf)), random_state=42)
    cols = 3
    rows = int(np.ceil(len(sample) / cols))

    plt.figure(figsize=(4*cols, 4*rows))
    plt.suptitle(title, fontsize=14)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        img = cv2.imread(row["path"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(row["filename"], fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Ver algunas de cada tipo problemático
show_images_from_df(small_imgs,  "Imágenes muy pequeñas")
show_images_from_df(large_imgs,  "Imágenes muy grandes")
show_images_from_df(dark_imgs,   "Imágenes muy oscuras")
show_images_from_df(bright_imgs, "Imágenes muy claras")
show_images_from_df(weird_ratio, "Imágenes con aspect ratio raro")

# ============================
# 6. Guardar estadísticas
# ============================

out_csv = PROJECT_ROOT / "image_stats.csv"
df.to_csv(out_csv, index=False)
print(f"\nEstadísticas guardadas en: {out_csv}")
print("EDA de imágenes finalizado.")