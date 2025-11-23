import cv2
from pathlib import Path
import shutil

# Directorio de entrada (todas las imágenes sin etiqueta)
RAW_DIR = Path("imagenes_jpg")

# Directorios de salida (etiquetados)
OUT_DIR = Path("imagenes_etiquetadas")
OUT_CORRECTA   = OUT_DIR / "Correcta"
OUT_INCORRECTA = OUT_DIR / "Incorrecta"
OUT_DUDOSA     = OUT_DIR / "Dudosa"

# Crear carpetas de salida si no existen
for d in [OUT_CORRECTA, OUT_INCORRECTA, OUT_DUDOSA]:
    d.mkdir(parents=True, exist_ok=True)

def move_image(src_path: Path, dest_dir: Path):
    dest_path = dest_dir / src_path.name
    shutil.move(str(src_path), str(dest_path))
    print(f"Movido: {src_path.name} -> {dest_dir.name}")

def mostrar_redimensionada(img, window_name="Etiqueta postura"):
    """
    Muestra la imagen en pantalla pero reducida si es muy grande.
    No modifica la imagen original, solo la versión que se muestra.
    """
    h, w = img.shape[:2]
    max_dim = 900  # tamaño máximo en píxeles para ancho/alto en pantalla

    scale = min(max_dim / w, max_dim / h, 1.0)  # nunca agrandar, solo reducir

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_show = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_show = img

    cv2.imshow(window_name, img_show)

def main():
    # Listar imágenes pendientes por etiquetar (solo .jpg)
    image_paths = sorted(list(RAW_DIR.rglob("*.jpg")))

    print("\n====================================================")
    print(f"Imágenes pendientes por etiquetar: {len(image_paths)}")
    print("Controles:")
    print("  C / c -> postura CORRECTA")
    print("  I / i -> postura INCORRECTA")
    print("  S / s -> DUDOSA (no usar para entrenar)")
    print("  ESC   -> SALIR del proceso")
    print("====================================================\n")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] No se pudo leer: {img_path}")
            continue

        # Mostrar versión redimensionada
        mostrar_redimensionada(img, "Etiqueta postura (C=correcta, I=incorrecta, S=dudosa, ESC=salir)")
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            print("Salida manual. Puedes retomar luego sin problema.")
            break
        elif key in [ord('c'), ord('C')]:
            move_image(img_path, OUT_CORRECTA)
        elif key in [ord('i'), ord('I')]:
            move_image(img_path, OUT_INCORRECTA)
        elif key in [ord('s'), ord('S')]:
            move_image(img_path, OUT_DUDOSA)
        else:
            print("Tecla no reconocida, la imagen se deja en data_jpg sin mover.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()