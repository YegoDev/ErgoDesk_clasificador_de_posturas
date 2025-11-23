"""
Script: resize_images.py
Descripción: Redimensiona todas las imágenes del dataset a 512x512
             manteniendo el aspect ratio y agregando padding centrado.

Ubicación esperada:
    Proyecto/
        preprocessing/resize_images.py
"""

from pathlib import Path
from PIL import Image

# ======================================================
# 🔧 CONFIGURACIÓN SEGÚN TU PROYECTO
# ======================================================

# Carpeta de entrada: tus imágenes finales
INPUT_DIR = Path("imagenes_etiquetadas/Dudosa")

# Carpeta de salida: nuevas imágenes 512x512
OUTPUT_DIR = Path("imagenes_jpg/imagenes_resized_512")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tamaño objetivo
TARGET_SIZE = (512, 512)

# Extensiones válidas
VALID_EXT = {".jpg", ".jpeg", ".png"}


# ======================================================
# 🔧 FUNCIONES
# ======================================================

def resize_with_padding(img: Image.Image, target_size=(512, 512), fill_color=(0, 0, 0)):
    """
    Redimensiona una imagen manteniendo aspect ratio.
    Luego agrega padding para llegar exactamente a target_size.
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size

    # Escala manteniendo proporción
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    # Redimensionado sin distorsión
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Lienzo final (background)
    new_img = Image.new("RGB", (target_w, target_h), fill_color)

    # Centrar la imagen
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    new_img.paste(img_resized, (offset_x, offset_y))

    return new_img


def process_folder(input_dir: Path, output_dir: Path, target_size=(512, 512)):
    """
    Procesa recursivamente una carpeta completa.
    """
    for path in input_dir.rglob("*"):
        if path.suffix.lower() in VALID_EXT:
            try:
                img = Image.open(path).convert("RGB")
                img_out = resize_with_padding(img, target_size)

                # Guardar con mismo nombre pero en JPG
                out_name = path.stem + ".jpg"
                out_path = output_dir / out_name

                img_out.save(out_path, "JPEG", quality=95)

                print(f"Procesada: {path.name} -> {out_path}")
            except Exception as e:
                print(f"❌ Error con {path}: {e}")


# ======================================================
# ▶️ EJECUCIÓN
# ======================================================

if __name__ == "__main__":
    print("🔧 Redimensionando imágenes a 512x512…")
    process_folder(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)
    print("✔ Listo. Imágenes guardadas en:", OUTPUT_DIR.absolute())
