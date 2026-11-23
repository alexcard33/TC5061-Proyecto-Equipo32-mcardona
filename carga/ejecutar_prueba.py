"""Corre la prueba de carga con Locust y al mismo tiempo registra docker stats.

Requisitos: el sistema levantado con docker compose y el entorno .venv-carga activo.

    python carga/ejecutar_prueba.py --etiqueta w1

Todo queda en reports/carga/<etiqueta>/:
    locust_stats.csv, locust_stats_history.csv, locust_failures.csv, locust_exceptions.csv
    locust_reporte.html   (reporte HTML de Locust, va como anexo)
    docker_stats.csv      (CPU y memoria de cada contenedor durante la prueba)

La etiqueta sirve para distinguir corridas con distinta configuración, por ejemplo w1 con
API_WORKERS=1 y w2 con API_WORKERS=2, y luego compararlas en el análisis FinOps.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess  # nosec B404 - solo se invocan docker y locust con argumentos fijos
import sys
import threading
import time

RAIZ = Path(__file__).resolve().parents[1]
LOCUSTFILE = RAIZ / "carga" / "locustfile.py"
CONTENEDORES = ["mibici-api", "mibici-mlflow"]

_FACTOR_MIB = {
    "b": 1 / 1024**2,
    "kib": 1 / 1024,
    "mib": 1,
    "gib": 1024,
    "kb": 1000 / 1024**2,
    "mb": 1000**2 / 1024**2,
    "gb": 1000**3 / 1024**2,
}


def a_mib(texto: str) -> float:
    """Convierte cosas como '153.2MiB' o '2GiB' a MiB."""
    m = re.match(r"\s*([\d.]+)\s*([a-zA-Z]+)", texto)
    if not m:
        return float("nan")
    return float(m.group(1)) * _FACTOR_MIB.get(m.group(2).lower(), float("nan"))


def a_numero(texto: str) -> float:
    try:
        return float(texto.strip().rstrip("%"))
    except ValueError:
        return float("nan")


def registrar_docker_stats(destino: Path, intervalo: float, detener: threading.Event) -> None:
    """Toma una muestra de docker stats cada `intervalo` segundos hasta que se pida detener."""
    with destino.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(
            ["timestamp", "contenedor", "cpu_pct", "mem_mib", "mem_limite_mib", "mem_pct", "pids"]
        )
        while not detener.is_set():
            inicio = time.time()
            r = subprocess.run(  # nosec B603 B607
                ["docker", "stats", "--no-stream", "--format", "{{json .}}", *CONTENEDORES],
                capture_output=True,
                text=True,
                check=False,
            )
            for linea in r.stdout.splitlines():
                if not linea.strip():
                    continue
                d = json.loads(linea)
                uso, _, limite = d.get("MemUsage", "").partition("/")
                escritor.writerow(
                    [
                        f"{inicio:.1f}",
                        d.get("Name", ""),
                        a_numero(d.get("CPUPerc", "")),
                        round(a_mib(uso), 1),
                        round(a_mib(limite), 1),
                        a_numero(d.get("MemPerc", "")),
                        d.get("PIDs", ""),
                    ]
                )
            f.flush()
            detener.wait(max(0.0, intervalo - (time.time() - inicio)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--etiqueta", default="w1", help="Nombre de la corrida (carpeta de salida).")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="URL base de la API.")
    parser.add_argument("--intervalo", type=float, default=2.0, help="Segundos entre muestras.")
    args = parser.parse_args()

    salida = RAIZ / "reports" / "carga" / args.etiqueta
    salida.mkdir(parents=True, exist_ok=True)

    detener = threading.Event()
    hilo = threading.Thread(
        target=registrar_docker_stats,
        args=(salida / "docker_stats.csv", args.intervalo, detener),
        daemon=True,
    )
    hilo.start()
    time.sleep(args.intervalo * 2)  # unas muestras en reposo antes de empezar la carga

    comando = [
        sys.executable, "-m", "locust",
        "-f", str(LOCUSTFILE),
        "--headless",
        "--host", args.host,
        "--csv", str(salida / "locust"),
        "--csv-full-history",
        "--html", str(salida / "locust_reporte.html"),
    ]  # fmt: skip
    print("Ejecutando:", " ".join(comando), flush=True)
    try:
        # PYTHONUTF8 hace que Locust escriba sus CSV en UTF-8 también en Windows.
        entorno = {**os.environ, "PYTHONUTF8": "1"}
        codigo = subprocess.run(  # nosec B603
            comando, cwd=RAIZ, env=entorno, check=False
        ).returncode
    finally:
        time.sleep(args.intervalo * 2)  # y unas muestras después, para ver la recuperación
        detener.set()
        hilo.join()

    print(f"\nResultados en {salida}")
    # Locust regresa 1 cuando hubo peticiones fallidas; eso es un resultado, no un error del script.
    return 0 if codigo in (0, 1) else codigo


if __name__ == "__main__":
    sys.exit(main())
