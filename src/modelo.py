# src/modelo.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def cargar_datos():
    iris = load_iris()
    return iris.data, iris.target


def entrenar_modelo(X_train, y_train, n_estimators=100, random_state=42):
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, output_dict=False)
    return {
        "accuracy": accuracy,
        "reporte_texto": reporte,
    }
