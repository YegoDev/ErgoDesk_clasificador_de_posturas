from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================
# 1. Configuración
# ============================
CSV_PATH = Path("modelo_mediapipe/posture_features.csv")
MODEL_PATH = Path("modelo_mediapipe/best_model.pkl")

FEATURES = [
    "trunk_angle_deg",
    "neck_angle_deg",
    "shoulder_hip_dist",
    "nose_ear_dist",
    "head_forward_ratio",
]

# ============================
# 2. Cargar datos
# ============================
df = pd.read_csv(CSV_PATH)

X = df[FEATURES].values
y = df["label"].values  # 0 = Correcta, 1 = Incorrecta

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ============================
# 3. Definir modelos
# ============================

models = {}

# Modelo 1: Regresión Logística
models["logreg"] = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ]
)

# Modelo 2: Random Forest (sin escalado)
models["rf"] = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
)

# Modelo 3: SVM RBF
models["svm"] = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=False)),
    ]
)

# ============================
# 4. Entrenamiento y evaluación
# ============================

results = []  # guardamos métricas y modelo

for name, model in models.items():
    print("\n==============================")
    print(f"Entrenando modelo: {name}")
    print("==============================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy: {acc:.3f}")
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Correcta", "Incorrecta"],
            digits=3,
        )
    )
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    results.append(
        {
            "name": name,
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    )

# ============================
# 5. Seleccionar mejor modelo
# ============================

best = max(results, key=lambda r: r["accuracy"])
best_name = best["name"]
best_acc = best["accuracy"]
best_model = best["model"]

print("\n======================================")
print(f"Mejor modelo: {best_name} con accuracy = {best_acc:.3f}")
print("Guardando en:", MODEL_PATH)
print("======================================")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)

# ============================
# 6. Gráficas de comparación
# ============================

model_names = [r["name"] for r in results]
accuracies = [r["accuracy"] for r in results]
precisions = [r["precision"] for r in results]
recalls = [r["recall"] for r in results]
f1_scores = [r["f1"] for r in results]

# --- 6.1 Accuracy por modelo ---
plt.figure(figsize=(6, 4))
plt.bar(model_names, accuracies)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparación de Accuracy por modelo")
plt.tight_layout()
plt.show()

# --- 6.2 Precision / Recall / F1 por modelo ---

metrics = {
    "Precision": precisions,
    "Recall": recalls,
    "F1-score": f1_scores,
}

x = np.arange(len(model_names))
width = 0.25

plt.figure(figsize=(8, 5))

plt.bar(x - width, metrics["Precision"], width, label="Precision")
plt.bar(x,         metrics["Recall"],    width, label="Recall")
plt.bar(x + width, metrics["F1-score"],  width, label="F1-score")

plt.xticks(x, model_names)
plt.ylim(0, 1)
plt.ylabel("Valor")
plt.title("Comparación de métricas por modelo")
plt.legend()
plt.tight_layout()
plt.show()