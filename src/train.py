import argparse
# import os
# import json
import pickle
from pathlib import Path

import yaml
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def cargar_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_datos(config: dict):
    source = config["data"]["source"]

    if source == "sklearn_iris":
        iris = load_iris()
        X, y = iris.data, iris.target
    else:
        # credenciales de os.environ en caso de necesitar conectarse a una base de datos
        # db_password = os.environ.get("DB_PASSWORD", "")
        raise ValueError(f"Fuente de datos no reconocida: {source}")

    return X, y


def entrenar(config: dict, X_train, y_train):
    params = config["model"]
    modelo = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    modelo.fit(X_train, y_train)
    return modelo


def main(config_path: str):
    config = cargar_config(config_path)

    Path("outputs").mkdir(exist_ok=True)

    print("Cargando datos...")
    X, y = cargar_datos(config)
    print(f"  Dataset: {X.shape[0]} muestras, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("Entrenando modelo...")
    modelo = entrenar(config, X_train, y_train)

    print("Guardando modelo...")
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(modelo, f)

    import numpy as np
    np.save("outputs/X_test.npy", X_test)
    np.save("outputs/y_test.npy", y_test)

    print("Entrenamiento completado.")
    print("  → outputs/model.pkl")
    print("  → outputs/X_test.npy")
    print("  → outputs/y_test.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de entrenamiento")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    args = parser.parse_args()
    main(args.config)
