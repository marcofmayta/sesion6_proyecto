import os
import json
import pickle
from sklearn.model_selection import cross_val_score
from src.modelo import cargar_datos

modelo_path = "outputs/modelo.pkl"
if not os.path.exists(modelo_path):
    raise FileNotFoundError(
        f"No se encontró el modelo en {modelo_path}. "
        "¿Corrió el job 'entrenar' primero?"
    )

print(f"Cargando modelo desde {modelo_path}...")
with open(modelo_path, "rb") as f:
    modelo = pickle.load(f)

print("Cargando datos para validación cruzada...")
X, y = cargar_datos()

print("Corriendo validación cruzada (5-fold)...")
scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")

print(f"  Scores por fold: {[f'{s:.4f}' for s in scores]}")
print(f"  Media: {scores.mean():.4f}")
print(f"  Desviación estándar: {scores.std():.4f}")

reporte_path = "outputs/reporte_validacion.txt"
with open(reporte_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("REPORTE DE VALIDACIÓN CRUZADA\n")
    f.write("=" * 60 + "\n\n")
    f.write("Método: 5-Fold Cross Validation\n")
    f.write(f"Scores: {scores.tolist()}\n")
    f.write(f"Media de accuracy: {scores.mean():.4f}\n")
    f.write(f"Desviación estándar: {scores.std():.4f}\n")

metricas_val_path = "outputs/metricas_validacion.json"
with open(metricas_val_path, "w", encoding="utf-8") as f:
    json.dump({
        "cv_scores": scores.tolist(),
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "n_folds": 5,
    }, f, indent=2)

print(f"\nReporte de validación guardado en: {reporte_path}")
print(f"Métricas de validación guardadas en: {metricas_val_path}")
print("\n¡Evaluación adicional completada!")
