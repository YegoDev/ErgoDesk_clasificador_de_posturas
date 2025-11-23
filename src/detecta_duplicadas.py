import hashlib
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("imagenes_jpg")  # o data_labeled si ya etiquetaste

def file_hash(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def find_exact_duplicates():
    hashes = defaultdict(list)

    for img_path in DATA_DIR.rglob("*.jpg"):
        h = file_hash(img_path)
        hashes[h].append(img_path)

    dup_groups = [paths for paths in hashes.values() if len(paths) > 1]

    if not dup_groups:
        print("No se encontraron duplicados exactos.")
        return

    print("Grupos de imágenes duplicadas (exactas):")
    for group in dup_groups:
        print("\n---")
        for p in group:
            print(p)

if __name__ == "__main__":
    find_exact_duplicates()