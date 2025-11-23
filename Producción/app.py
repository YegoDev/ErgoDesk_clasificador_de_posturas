import os
import io
from pathlib import Path

import cv2
import numpy as np
import requests
import gradio as gr
from PIL import Image
import mediapipe as mp

# ============================================================
# 1. CONFIGURACIÓN DE CUSTOM VISION (variables de entorno)
# ============================================================

PREDICTION_KEY = os.getenv("CUSTOM_VISION_PREDICTION_KEY")
ENDPOINT = os.getenv("CUSTOM_VISION_ENDPOINT")  # ej: https://southcentralus.api.cognitive.microsoft.com/
PROJECT_ID = os.getenv("CUSTOM_VISION_PROJECT_ID")
PUBLISHED_NAME = os.getenv("CUSTOM_VISION_PUBLISHED_NAME")

if not all([PREDICTION_KEY, ENDPOINT, PROJECT_ID, PUBLISHED_NAME]):
    raise RuntimeError(
        "⚠️ Faltan variables de entorno: CUSTOM_VISION_PREDICTION_KEY, "
        "CUSTOM_VISION_ENDPOINT, CUSTOM_VISION_PROJECT_ID, CUSTOM_VISION_PUBLISHED_NAME"
    )

PREDICT_URL = (
    f"{ENDPOINT}customvision/v3.0/Prediction/"
    f"{PROJECT_ID}/classify/iterations/{PUBLISHED_NAME}/image"
)

GOOD_LABEL = "Correcta"
BAD_LABEL = "Incorrecta"


# ============================================================
# 2. MEDIA PIPE (landmarks)
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_LM = mp_pose.PoseLandmark


def analyze_landmarks(img_bgr):
    """
    Corre MediaPipe Pose sobre la imagen.
    Devuelve: (features_dict, pose_landmarks) o (None, None)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        res = pose.process(img_rgb)

    if not res.pose_landmarks:
        return None, None

    lm = res.pose_landmarks.landmark

    shoulder = lm[POSE_LM.LEFT_SHOULDER]
    hip      = lm[POSE_LM.LEFT_HIP]
    ear      = lm[POSE_LM.LEFT_EAR]

    shoulder_xy = (shoulder.x, shoulder.y)
    hip_xy      = (hip.x, hip.y)
    ear_xy      = (ear.x, ear.y)

    def angle_with_vertical(p1, p2):
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        dot = vy * (-1)
        norm = np.sqrt(vx**2 + vy**2)
        if norm == 0:
            return np.nan
        cos_theta = dot / norm
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    trunk_angle = angle_with_vertical(hip_xy, shoulder_xy)
    neck_angle  = angle_with_vertical(shoulder_xy, ear_xy)

    feats = {
        "trunk_angle_deg": trunk_angle,
        "neck_angle_deg": neck_angle,
    }

    return feats, res.pose_landmarks


def draw_landmarks_resized(img_bgr, landmarks, scale=0.5):
    img_copy = img_bgr.copy()
    mp_drawing.draw_landmarks(img_copy, landmarks, mp_pose.POSE_CONNECTIONS)
    h, w = img_copy.shape[:2]
    img_small = cv2.resize(img_copy, (int(w * scale), int(h * scale)))
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    return img_rgb


# ============================================================
# 3. CUSTOM VISION (predicción)
# ============================================================

def predict_custom_vision(pil_image: Image.Image):
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/octet-stream",
    }

    response = requests.post(PREDICT_URL, headers=headers, data=img_bytes)
    response.raise_for_status()
    data = response.json()

    preds = data.get("predictions", [])
    if not preds:
        return None, []

    best = max(preds, key=lambda x: x["probability"])
    return best, preds


# ============================================================
# 4. FUNCIÓN PRINCIPAL DE ANÁLISIS
# ============================================================

def analizar_postura(pil_image: Image.Image):
    if pil_image is None:
        return None, "<p style='color:red;'>No se recibió una imagen válida.</p>"

    # --- 1) Custom Vision ---
    try:
        best, preds = predict_custom_vision(pil_image)
    except Exception as e:
        return None, f"<p style='color:red;'>Error llamando a Custom Vision: {e}</p>"

    etiqueta = best["tagName"]
    prob = best["probability"]

    # --- 2) Landmarks ---
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    feats, landmarks = analyze_landmarks(img_bgr)

    if landmarks is not None:
        img_overlay = draw_landmarks_resized(img_bgr, landmarks, scale=0.45)
    else:
        img_overlay = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- 3) Semáforo ---
    if etiqueta == GOOD_LABEL:
        color = "green"
        emoji = "🟢"
        texto = "Postura Correcta"
    elif etiqueta == BAD_LABEL:
        color = "red"
        emoji = "🔴"
        texto = "Postura Incorrecta"
    else:
        color = "gray"
        emoji = "🟡"
        texto = etiqueta

    html = f"""
    <div style="text-align:center; font-size: 1.3rem;">
        <div style="font-size:3rem;">{emoji}</div>
        <p style="color:{color}; font-weight:bold;">{texto}</p>
        <p>Probabilidad modelo: <b>{prob:.2f}</b></p>
    """

    if feats:
        html += f"""
        <p style="font-size:0.9rem;">
        Ángulo tronco: {feats['trunk_angle_deg']:.1f}°<br/>
        Ángulo cuello: {feats['neck_angle_deg']:.1f}°
        </p>
        """

    html += "</div>"

    return img_overlay, html


# ============================================================
# 5. INTERFAZ GRADIO
# ============================================================

with gr.Blocks() as demo:
    gr.Markdown("## 🪑 Clasificador de Postura Sentada (Custom Vision + MediaPipe)")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Sube una imagen lateral")
        with gr.Column():
            output_img = gr.Image(label="Esqueleto y postura", interactive=False)
            output_html = gr.HTML()

    btn = gr.Button("Analizar postura")

    btn.click(
        fn=analizar_postura,
        inputs=input_img,
        outputs=[output_img, output_html],
    )

if __name__ == "__main__":
    demo.launch()