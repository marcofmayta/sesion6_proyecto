import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from src.modelo import cargar_datos, entrenar_modelo, evaluar_modelo
import pickle

os.makedirs("outputs", exist_ok=True)

print("Cargando datos...")
X, y = cargar_datos()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train: {X_train.shape[0]} muestras")
print(f"  Test:  {X_test.shape[0]} muestras")

print("Entrenando modelo...")
modelo = entrenar_modelo(X_train, y_train)

print("Evaluando modelo...")
metricas = evaluar_modelo(modelo, X_test, y_test)
print(f"  Accuracy: {metricas['accuracy']:.4f}")

timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
reporte_path = "outputs/report.txt"

with open(reporte_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("REPORTE DE ENTRENAMIENTO — Proyecto DS Sesión 5\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Timestamp (UTC): {timestamp}\n")
    f.write(f"Muestras de entrenamiento: {X_train.shape[0]}\n")
    f.write(f"Muestras de prueba: {X_test.shape[0]}\n")
    f.write(f"Accuracy: {metricas['accuracy']:.4f}\n\n")
    f.write("--- Reporte de clasificación ---\n")
    f.write(metricas["reporte_texto"])
    f.write("\n")

metricas_json_path = "outputs/metricas.json"
metricas_json = {
    "timestamp": timestamp,
    "accuracy": metricas["accuracy"],
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
}

with open(metricas_json_path, "w", encoding="utf-8") as f:
    json.dump(metricas_json, f, indent=2)

modelo_path = "outputs/modelo.pkl"
with open(modelo_path, "wb") as f:
    pickle.dump(modelo, f)

print(f"Modelo guardado en: {modelo_path}")
print(f"Reporte guardado en: {reporte_path}")
print(f" Métricas guardadas en: {metricas_json_path}")
print("\n¡Entrenamiento completado exitosamente!\nFIN DEL PROGRAMA.")
