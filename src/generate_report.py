import argparse
import json
# from datetime import datetime, timezone
from pathlib import Path

import yaml


def cargar_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generar_html(metricas: dict) -> str:
    timestamp_display = metricas.get("timestamp", "N/A")
    accuracy = metricas.get("accuracy", 0)
    f1 = metricas.get("f1_macro", 0)
    roc_auc = metricas.get("roc_auc_ovr", 0)
    modelo = metricas.get("model_name", "N/A")
    n_estimators = metricas.get("n_estimators", "N/A")
    n_test = metricas.get("n_test", "N/A")

    if accuracy >= 0.95:
        color_acc = "#28a745"
        label_acc = "Excelente"
    elif accuracy >= 0.85:
        color_acc = "#ffc107"
        label_acc = "Bueno"
    else:
        color_acc = "#dc3545"
        label_acc = "Revisar"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reporte de Métricas — Proyecto DS</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f6f8fa;
      color: #24292e;
      padding: 2rem;
    }}
    .container {{
      max-width: 700px;
      margin: 0 auto;
      background: white;
      border-radius: 8px;
      border: 1px solid #e1e4e8;
      overflow: hidden;
    }}
    .header {{
      background: #24292e;
      color: white;
      padding: 1.5rem 2rem;
    }}
    .header h1 {{ font-size: 1.4rem; margin-bottom: 0.3rem; }}
    .header p  {{ font-size: 0.85rem; color: #8b949e; }}
    .body {{ padding: 2rem; }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .metric-card {{
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      padding: 1.2rem;
      text-align: center;
    }}
    .metric-value {{
      font-size: 2rem;
      font-weight: bold;
      color: #0366d6;
    }}
    .metric-label {{
      font-size: 0.8rem;
      color: #6a737d;
      margin-top: 0.3rem;
    }}
    .badge {{
      display: inline-block;
      padding: 0.2rem 0.6rem;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: bold;
      color: white;
      background: {color_acc};
      margin-top: 0.5rem;
    }}
    .info-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    .info-table td {{
      padding: 0.6rem 0;
      border-bottom: 1px solid #f1f3f4;
    }}
    .info-table td:first-child {{ color: #6a737d; width: 40%; }}
    .info-table td:last-child {{ font-weight: 500; }}
    .footer {{
      padding: 1rem 2rem;
      background: #f6f8fa;
      border-top: 1px solid #e1e4e8;
      font-size: 0.8rem;
      color: #6a737d;
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Reporte de Métricas del Modelo</h1>
      <p>Generado automáticamente por GitHub Actions</p>
    </div>
    <div class="body">
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-value">{accuracy:.4f}</div>
          <div class="metric-label">Accuracy</div>
          <span class="badge">{label_acc}</span>
        </div>
        <div class="metric-card">
          <div class="metric-value">{f1:.4f}</div>
          <div class="metric-label">F1 Macro</div>
        </div>
        <div class="metric-card">
          <div class="metric-value">{roc_auc:.4f}</div>
          <div class="metric-label">ROC-AUC (OvR)</div>
        </div>
      </div>

      <h3 style="margin-bottom: 1rem; font-size: 0.95rem; color: #6a737d;">
        Detalles del experimento
      </h3>
      <table class="info-table">
        <tr><td>Modelo</td><td>{modelo}</td></tr>
        <tr><td>N° estimadores</td><td>{n_estimators}</td></tr>
        <tr><td>Muestras de test</td><td>{n_test}</td></tr>
        <tr><td>Timestamp (UTC)</td><td>{timestamp_display}</td></tr>
      </table>
    </div>
    <div class="footer">
      Proyecto DS · Curso Git &amp; GitHub · Ing. Marco Mayta
    </div>
  </div>
</body>
</html>
"""
    return html


def main(config_path: str):
    config = cargar_config(config_path)

    metrics_path = config["evaluation"]["output_path"]
    print(f"Leyendo métricas desde: {metrics_path}")
    with open(metrics_path, encoding="utf-8") as f:
        metricas = json.load(f)

    html = generar_html(metricas)

    report_path = config["report"]["output_path"]
    Path(report_path).parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Reporte guardado en: {report_path}")

    pages_path = config["report"]["pages_path"]
    Path(pages_path).parent.mkdir(exist_ok=True)
    with open(pages_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Reporte copiado a Pages: {pages_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de reporte HTML")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
