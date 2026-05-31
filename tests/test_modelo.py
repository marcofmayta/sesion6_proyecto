# import pytest
from src.modelo import cargar_datos, entrenar_modelo, evaluar_modelo
from sklearn.model_selection import train_test_split


def test_cargar_datos_retorna_arrays():
    X, y = cargar_datos()
    assert X is not None
    assert y is not None
    assert len(X) > 0
    assert len(y) > 0


def test_dimensiones_correctas():
    X, y = cargar_datos()
    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_entrenar_modelo_retorna_objeto():
    from sklearn.ensemble import RandomForestClassifier
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modelo = entrenar_modelo(X_train, y_train)
    assert isinstance(modelo, RandomForestClassifier)


def test_accuracy_mayor_a_0_9():
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modelo = entrenar_modelo(X_train, y_train)
    metricas = evaluar_modelo(modelo, X_test, y_test)
    assert metricas["accuracy"] > 0.90, (
        f"Accuracy esperada >0.90, obtenida: {metricas['accuracy']:.4f}"
    )


def test_evaluar_retorna_reporte_texto():
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modelo = entrenar_modelo(X_train, y_train)
    metricas = evaluar_modelo(modelo, X_test, y_test)
    assert "reporte_texto" in metricas
    assert isinstance(metricas["reporte_texto"], str)
    assert len(metricas["reporte_texto"]) > 0
