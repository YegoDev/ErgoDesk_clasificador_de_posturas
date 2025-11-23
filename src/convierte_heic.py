from pathlib import Path
from PIL import Image
import pillow_heif

# Activar soporte HEIC en Pillow
pillow_heif.register_heif_opener()

def convert_heic_folder(src_root="imagenes_crudas", dst_root="imagenes_jpg"):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    # Convertir HEIC → JPEG
    for heic_path in list(src_root.rglob("*.HEIC")) + list(src_root.rglob("*.heic")):
        rel = heic_path.relative_to(src_root).with_suffix(".jpg")
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(heic_path).convert("RGB")
        img.save(dst_path, "JPEG", quality=95)
        print(f"[HEIC] {heic_path.name}  →  {dst_path}")

    # Convertir también JPG/PNG a JPEG “normalizado”
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in src_root.rglob(ext):
            rel = img_path.relative_to(src_root).with_suffix(".jpg")
            dst_path = dst_root / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if not dst_path.exists():
                img = Image.open(img_path).convert("RGB")
                img.save(dst_path, "JPEG", quality=95)
                print(f"[IMG]  {img_path.name}  →  {dst_path}")

if __name__ == "__main__":
    convert_heic_folder()
