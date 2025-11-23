from pathlib import Path
import argparse

import cv2
import mediapipe as mp
import numpy as np
import joblib

# ============================
# 1. Configuración
# ============================

# Ruta al modelo entrenado (ajusta si cambia)
MODEL_PATH = Path("modelo_mediapipe/best_model.pkl")

# 0 = Correcta, 1 = Incorrecta (mismas etiquetas que en entrenamiento)
IDX_TO_LABEL = {0: "Correcta", 1: "Incorrecta"}

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_LANDMARK = mp_pose.PoseLandmark


# ============================
# 2. Funciones auxiliares
# ============================

def angle_with_vertical(p1, p2):
    """
    Ángulo entre el vector p1->p2 y la vertical (0, -1).
    Retorna el ángulo en grados.
    """
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]

    dot = vx * 0 + vy * (-1)
    norm_v = np.sqrt(vx**2 + vy**2)
    if norm_v == 0:
        return np.nan

    cos_theta = dot / norm_v
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def distance(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def extract_features_from_image(img_bgr, pose_model):
    """
    Extrae el vector de features para una imagen BGR (OpenCV).
    Devuelve (features_dict, landmarks) o (None, None) si falla.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose_model.process(img_rgb)

    if not results.pose_landmarks:
        return None, None

    lm = results.pose_landmarks.landmark

    # Mismo lado que en el entrenamiento: izquierdo
    shoulder = lm[POSE_LANDMARK.LEFT_SHOULDER]
    hip      = lm[POSE_LANDMARK.LEFT_HIP]
    ear      = lm[POSE_LANDMARK.LEFT_EAR]
    nose     = lm[POSE_LANDMARK.NOSE]

    shoulder_xy = (shoulder.x, shoulder.y)
    hip_xy      = (hip.x, hip.y)
    ear_xy      = (ear.x, ear.y)
    nose_xy     = (nose.x, nose.y)

    trunk_angle = angle_with_vertical(hip_xy, shoulder_xy)
    neck_angle  = angle_with_vertical(shoulder_xy, ear_xy)

    shoulder_hip_dist = distance(shoulder_xy, hip_xy)
    nose_ear_dist     = distance(nose_xy, ear_xy)
    head_hip_dx       = abs(ear_xy[0] - hip_xy[0])

    head_forward_ratio = (
        head_hip_dx / shoulder_hip_dist if shoulder_hip_dist > 0 else np.nan
    )

    feats = {
        "trunk_angle_deg": trunk_angle,
        "neck_angle_deg": neck_angle,
        "shoulder_hip_dist": shoulder_hip_dist,
        "nose_ear_dist": nose_ear_dist,
        "head_forward_ratio": head_forward_ratio,
    }

    return feats, results.pose_landmarks


def features_to_array(feats_dict):
    """
    Convierte el dict de features en el vector ordenado que espera el modelo.
    El orden debe coincidir con el usado en entrenamiento.
    """
    return np.array([
        feats_dict["trunk_angle_deg"],
        feats_dict["neck_angle_deg"],
        feats_dict["shoulder_hip_dist"],
        feats_dict["nose_ear_dist"],
        feats_dict["head_forward_ratio"],
    ], dtype=float).reshape(1, -1)


# ============================
# 3. Función principal de inferencia
# ============================

def predict_posture(image_path: Path, show: bool = False):
    # Cargar modelo
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Cargar imagen
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:

        feats, landmarks = extract_features_from_image(img_bgr, pose)
        if feats is None:
            print("No se pudieron obtener landmarks de la imagen.")
            return

        X = features_to_array(feats)

        # Predicción
        y_pred = model.predict(X)[0]
        label_name = IDX_TO_LABEL[int(y_pred)]

        prob = None
        # Si el modelo soporta predict_proba (LogReg, RF, SVM con probability=True)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            prob = float(proba[int(y_pred)])

        print(f"Imagen: {image_path.name}")
        print(f"Clasificación: {label_name} (clase {int(y_pred)})")
        if prob is not None:
            print(f"Probabilidad modelo: {prob:.3f}")

        print("\nFeatures:")
        for k, v in feats.items():
            print(f"  {k}: {v:.4f}")

        # Visualización opcional con esqueleto
        if show and landmarks is not None:
            img_vis = img_bgr.copy()
            mp_drawing.draw_landmarks(img_vis, landmarks, mp_pose.POSE_CONNECTIONS)

            # --- Resize para que se vea bien ---
            h, w = img_vis.shape[:2]
            scale = 0.4   # ⇦ Ajusta este factor (0.3, 0.4, 0.5)
            img_small = cv2.resize(img_vis, (int(w*scale), int(h*scale)))
            
            cv2.imshow(f"Postura: {label_name}", img_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# ============================
# 4. CLI
# ============================

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(
    #    description="Inferencia de postura (Correcta/Incorrecta) usando MediaPipe + modelo entrenado."
    #)
    #parser.add_argument("image_path", type=str, help="Ruta a la imagen a evaluar")
    #parser.add_argument(
    #    "--show",
    #    action="store_true",
    #    help="Mostrar imagen con esqueleto dibujado",
    #)
#
    #args = parser.parse_args()
#
    img_path = Path("Validaciones/correcto10.jpg")
    #predict_posture(Path(args.image_path), show=args.show)
    predict_posture(img_path, show=True)
