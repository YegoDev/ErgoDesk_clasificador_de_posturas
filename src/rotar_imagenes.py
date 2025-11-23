import cv2
from pathlib import Path

DATA_DIR = Path("imagenes_jpg")  # o la carpeta donde tengas tus jpg

def force_vertical():
    for img_path in DATA_DIR.rglob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            print("Error leyendo:", img_path)
            continue

        h, w = img.shape[:2]

        # Si es horizontal (más ancha que alta), la rotamos 90° antihorario
        if w > h:
            rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(str(img_path), rot)
            print("Rotada ->", img_path.name)

if __name__ == "__main__":
    force_vertical()
