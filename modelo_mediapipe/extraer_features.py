from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

DATA_LABELED = Path("imagenes_etiquetadas")
OUTPUT_CSV = Path("modelo_mediapipe/posture_features.csv")

mp_pose = mp.solutions.pose
POSE_LANDMARK = mp_pose.PoseLandmark

print("Carpetas disponibles en imagenes_etiquetadas:",
      [p.name for p in DATA_LABELED.iterdir()])

# =========================================
# 2. Funciones auxiliares de geometría
# =========================================
def angle_with_vertical(p1, p2):
    """
    Ángulo entre el vector p1->p2 y la vertical (0, -1).
    Retorna el ángulo en grados.
    """
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]

    dot = vx * 0 + vy * (-1)  # producto con (0, -1)
    norm_v = np.sqrt(vx**2 + vy**2)
    if norm_v == 0:
        return np.nan

    cos_theta = dot / norm_v
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def distance(p1, p2):
    """Distancia euclidiana entre dos puntos (x, y)."""
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

# =========================================
# 3. Extraer features de UNA imagen
# =========================================
def extract_pose_features(image_path, pose_model):
    """
    Extrae features de postura desde una imagen usando MediaPipe Pose.
    Retorna un dict con ángulos y distancias, o None si no hay pose.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"[WARN] No se pudo leer: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose_model.process(img_rgb)

    if not results.pose_landmarks:
        print(f"[WARN] Sin landmarks en: {image_path}")
        return None

    lm = results.pose_landmarks.landmark

    shoulder = lm[POSE_LANDMARK.LEFT_SHOULDER]
    hip      = lm[POSE_LANDMARK.LEFT_HIP]
    ear      = lm[POSE_LANDMARK.LEFT_EAR]
    nose     = lm[POSE_LANDMARK.NOSE]

    shoulder_xy = (shoulder.x, shoulder.y)
    hip_xy      = (hip.x, hip.y)
    ear_xy      = (ear.x, ear.y)
    nose_xy     = (nose.x, nose.y)

    # 1) Ángulo de tronco: cadera -> hombro vs vertical
    trunk_angle = angle_with_vertical(hip_xy, shoulder_xy)

    # 2) Ángulo de cuello: hombro -> oreja vs vertical
    neck_angle = angle_with_vertical(shoulder_xy, ear_xy)

    # 3) Distancia hombro-cadera
    shoulder_hip_dist = distance(shoulder_xy, hip_xy)

    # 4) Distancia nariz-oreja
    nose_ear_dist = distance(nose_xy, ear_xy)

    # 5) Proyección horizontal de la cabeza respecto a la cadera
    head_hip_dx = abs(ear_xy[0] - hip_xy[0])

    # 6) Ratio de “cabeza adelantada”
    head_forward_ratio = (
        head_hip_dx / shoulder_hip_dist if shoulder_hip_dist > 0 else np.nan
    )

    return {
        "filename": image_path.name,
        "trunk_angle_deg": trunk_angle,
        "neck_angle_deg": neck_angle,
        "shoulder_hip_dist": shoulder_hip_dist,
        "nose_ear_dist": nose_ear_dist,
        "head_forward_ratio": head_forward_ratio,
    }

# =========================================
# 4. Recorrer Correcta / Incorrecta y crear el CSV
# =========================================
def main():
    rows = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:

        for label_name in ["Correcta", "Incorrecta"]:
            folder = DATA_LABELED / label_name
            if not folder.exists():
                print(f"[WARN] Carpeta no existe: {folder}")
                continue

            label_value = 0 if label_name == "Correcta" else 1

            print(f"\nProcesando carpeta: {folder}")
            for img_path in folder.glob("*.jpg"):
                feats = extract_pose_features(img_path, pose)
                if feats is None:
                    continue

                feats["label_name"] = label_name
                feats["label"] = label_value
                rows.append(feats)

    df = pd.DataFrame(rows)
    print("\nPreview del DataFrame:")
    print(df.head())
    print("Total de filas:", len(df))

    # Limpieza rápida (por si hay NaNs)
    df = df.dropna().reset_index(drop=True)
    print("Filas después de dropna:", len(df))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV guardado en: {OUTPUT_CSV.resolve()}")
    

if __name__ == "__main__":
    main()