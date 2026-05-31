import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def cargar_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    config = cargar_config(config_path)

    print("Cargando modelo...")
    with open("outputs/model.pkl", "rb") as f:
        modelo = pickle.load(f)

    print("Cargando datos de test...")
    X_test = np.load("outputs/X_test.npy")
    y_test = np.load("outputs/y_test.npy")

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    reporte_texto = classification_report(y_test, y_pred)

    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  F1 macro : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print("\nReporte de clasificación:")
    print(reporte_texto)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metricas = {
        "timestamp": timestamp,
        "accuracy": round(accuracy, 6),
        "f1_macro": round(f1, 6),
        "roc_auc_ovr": round(roc_auc, 6),
        "n_test": int(len(y_test)),
        "model_name": config["model"]["name"],
        "n_estimators": config["model"]["n_estimators"],
    }

    output_path = config["evaluation"]["output_path"]
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)

    print(f"\nMétricas guardadas en: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de evaluación")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
