"""Revisa el resultado de una prueba de carga de Locust contra umbrales fijos.

Lo usa el pipeline de CI/CD después de correr la versión corta de la prueba de carga, pero
también se puede correr a mano sobre cualquier corrida:

    python carga/validar_carga.py reports/carga/ci/locust_stats.csv

Toma la fila "Aggregated" del CSV que deja Locust (--csv) y revisa dos cosas:

- que el porcentaje de peticiones fallidas no pase de CARGA_MAX_ERRORES_PCT (por defecto 0),
- que el percentil 95 del tiempo de respuesta no pase de CARGA_MAX_P95_MS (por defecto 500).

Imprime una tabla en Markdown con el resultado y, si corre en GitHub Actions, la agrega al
resumen del job (GITHUB_STEP_SUMMARY). Termina con código 1 si algún umbral no se cumple, para
que el pipeline se detenga antes de publicar la imagen.

Solo usa la librería estándar para no tener que instalar pandas en el runner.
"""

import csv
import os
from pathlib import Path
import sys

FILA_TOTAL = "Aggregated"


def leer_total(ruta: Path) -> dict:
    """Fila Aggregated del CSV de estadísticas de Locust."""
    with ruta.open(newline="", encoding="utf-8", errors="replace") as f:
        for fila in csv.DictReader(f):
            if fila.get("Name") == FILA_TOTAL:
                return fila
    raise ValueError(f"{ruta} no tiene la fila {FILA_TOTAL!r}; ¿la prueba alcanzó a correr?")


def evaluar(total: dict, max_errores_pct: float, max_p95_ms: float) -> dict:
    peticiones = int(total["Request Count"])
    fallas = int(total["Failure Count"])
    errores_pct = 100 * fallas / peticiones if peticiones else 100.0
    p95 = float(total["95%"])
    return {
        "peticiones": peticiones,
        "fallas": fallas,
        "rps": float(total["Requests/s"]),
        "p50_ms": float(total["50%"]),
        "p95_ms": p95,
        "p99_ms": float(total["99%"]),
        "errores_pct": errores_pct,
        # Sin peticiones no hay nada que validar: se toma como falla.
        "ok_errores": peticiones > 0 and errores_pct <= max_errores_pct,
        "ok_p95": peticiones > 0 and p95 <= max_p95_ms,
    }


def tabla(r: dict, max_errores_pct: float, max_p95_ms: float) -> str:
    def marca(ok: bool) -> str:
        return "✅" if ok else "❌"

    errores = (
        f"| Errores | {r['fallas']} ({r['errores_pct']:.2f}%) | "
        f"≤ {max_errores_pct:g}% | {marca(r['ok_errores'])} |"
    )
    return "\n".join(
        [
            "### Prueba de carga (Locust)",
            "",
            "| Métrica | Valor | Umbral | Resultado |",
            "|---|---|---|---|",
            f"| Peticiones | {r['peticiones']} | > 0 | {marca(r['peticiones'] > 0)} |",
            f"| Peticiones por segundo | {r['rps']:.2f} | - | - |",
            f"| p50 | {r['p50_ms']:.0f} ms | - | - |",
            f"| p95 | {r['p95_ms']:.0f} ms | ≤ {max_p95_ms:.0f} ms | {marca(r['ok_p95'])} |",
            f"| p99 | {r['p99_ms']:.0f} ms | - | - |",
            errores,
        ]
    )


def main(argv: list[str]) -> int:
    # En la consola de Windows (cp1252) los símbolos de la tabla no se pueden imprimir.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ruta = Path(argv[1] if len(argv) > 1 else "reports/carga/ci/locust_stats.csv")
    max_errores_pct = float(os.getenv("CARGA_MAX_ERRORES_PCT", "0"))
    max_p95_ms = float(os.getenv("CARGA_MAX_P95_MS", "500"))

    resultado = evaluar(leer_total(ruta), max_errores_pct, max_p95_ms)
    texto = tabla(resultado, max_errores_pct, max_p95_ms)
    print(texto)

    resumen = os.getenv("GITHUB_STEP_SUMMARY")
    if resumen:
        with open(resumen, "a", encoding="utf-8") as f:
            f.write(texto + "\n")

    if resultado["ok_errores"] and resultado["ok_p95"]:
        print("\nLa prueba de carga cumple los umbrales.")
        return 0
    print("\nLa prueba de carga NO cumple los umbrales.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
