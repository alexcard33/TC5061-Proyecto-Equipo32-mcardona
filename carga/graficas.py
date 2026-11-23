"""Gráficas y resumen por escalón de las pruebas de carga.

    python carga/graficas.py --etiqueta w1
    python carga/graficas.py --etiqueta w1 --comparar w2

Lee reports/carga/<etiqueta>/locust_stats_history.csv y docker_stats.csv, y escribe en la
misma carpeta:
    latencia.png        p50, p95 y p99 a lo largo de la prueba
    rendimiento.png     peticiones por segundo e inferencias por segundo
    errores.png         tasa de error
    recursos.png        CPU y memoria de cada contenedor (docker stats)
    resumen_escalones.csv   una fila por escalón de usuarios, base del análisis FinOps

Con --comparar se agrega reports/carga/comparacion_<etiquetas>.png con las corridas lado a
lado (por ejemplo, la API con 1 worker contra la API con 2 workers).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
CARGA = RAIZ / "reports" / "carga"

TAMANO_LOTE = 10
NOMBRE_UNA = "/predict [1 estación]"
NOMBRE_LOTE = f"/predict [lote de {TAMANO_LOTE}]"
# Segundos al inicio de cada escalón que no entran en los promedios, mientras arrancan los
# usuarios nuevos y la latencia se acomoda.
DESCARTE_S = 10
COLORES = {"mibici-api": "#2563eb", "mibici-mlflow": "#9ca3af"}


def cargar_historial(carpeta: Path, duracion: int) -> pd.DataFrame:
    h = pd.read_csv(carpeta / "locust_stats_history.csv")
    for col in h.columns:
        if col not in ("Type", "Name"):
            h[col] = pd.to_numeric(h[col], errors="coerce")

    agregado = h[h["Name"] == "Aggregated"].copy()
    for nombre, columna in ((NOMBRE_UNA, "rps_una"), (NOMBRE_LOTE, "rps_lote")):
        parte = h.loc[h["Name"] == nombre, ["Timestamp", "Requests/s"]]
        parte = parte.drop_duplicates("Timestamp", keep="last")
        agregado = agregado.merge(
            parte.rename(columns={"Requests/s": columna}), on="Timestamp", how="left"
        )
    agregado[["rps_una", "rps_lote"]] = agregado[["rps_una", "rps_lote"]].fillna(0.0)
    # Una petición de lote son 10 inferencias: esto es lo que importa para el costo.
    agregado["inferencias_s"] = agregado["rps_una"] + TAMANO_LOTE * agregado["rps_lote"]

    agregado = agregado[agregado["User Count"] > 0]
    agregado = agregado.drop_duplicates("Timestamp", keep="last").sort_values("Timestamp")
    agregado["t"] = agregado["Timestamp"] - agregado["Timestamp"].min()
    agregado["escalon"] = (agregado["t"] // duracion).astype(int)
    agregado["t_en_escalon"] = agregado["t"] - agregado["escalon"] * duracion
    agregado["usuarios"] = agregado.groupby("escalon")["User Count"].transform("max")
    rps = agregado["Requests/s"].where(agregado["Requests/s"] > 0)
    agregado["error_pct"] = 100 * agregado["Failures/s"] / rps
    return agregado


def cargar_docker(carpeta: Path, t0: float, duracion: int) -> pd.DataFrame:
    d = pd.read_csv(carpeta / "docker_stats.csv")
    d["t"] = d["timestamp"] - t0
    d["escalon"] = (d["t"] // duracion).where(d["t"] >= 0)
    d["t_en_escalon"] = d["t"] - d["escalon"] * duracion
    return d


def marcar_escalones(ax, historial: pd.DataFrame) -> None:
    """Líneas verticales al inicio de cada escalón con el número de usuarios arriba."""
    inicios = historial.groupby("escalon").agg(t=("t", "min"), usuarios=("usuarios", "first"))
    for _, fila in inicios.iterrows():
        ax.axvline(fila["t"], color="#d1d5db", lw=0.8, zorder=0)
        ax.annotate(
            f"{int(fila['usuarios'])} u",
            xy=(fila["t"], 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -12),
            textcoords="offset points",
            fontsize=8,
            color="#6b7280",
        )


def guardar(fig, ruta: Path) -> None:
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("Gráfica:", ruta.relative_to(RAIZ))


def graficar(historial: pd.DataFrame, docker: pd.DataFrame, carpeta: Path, etiqueta: str):
    t = historial["t"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col, color in (("50%", "#16a34a"), ("95%", "#f59e0b"), ("99%", "#dc2626")):
        ax.plot(t, historial[col], label=f"p{col.rstrip('%')}", color=color, lw=1.4)
    ax.set_title(f"Latencia de la API bajo carga ({etiqueta})")
    ax.set_xlabel("Tiempo de prueba (s)")
    ax.set_ylabel("Latencia (ms)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    marcar_escalones(ax, historial)
    guardar(fig, carpeta / "latencia.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, historial["Requests/s"], label="Peticiones/s", color="#2563eb", lw=1.4)
    ax.plot(t, historial["inferencias_s"], label="Inferencias/s", color="#7c3aed", lw=1.4)
    ax.set_title(f"Rendimiento de la API ({etiqueta})")
    ax.set_xlabel("Tiempo de prueba (s)")
    ax.set_ylabel("Por segundo")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    marcar_escalones(ax, historial)
    guardar(fig, carpeta / "rendimiento.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, historial["error_pct"].fillna(0), color="#dc2626", lw=1.4)
    ax.set_title(f"Tasa de error ({etiqueta})")
    ax.set_xlabel("Tiempo de prueba (s)")
    ax.set_ylabel("Peticiones fallidas (%)")
    ax.set_ylim(bottom=0, top=max(5, float(historial["error_pct"].max() or 0) * 1.2))
    ax.grid(alpha=0.3)
    marcar_escalones(ax, historial)
    guardar(fig, carpeta / "errores.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for nombre, grupo in docker.groupby("contenedor"):
        color = COLORES.get(nombre)
        ax1.plot(grupo["t"], grupo["cpu_pct"], label=nombre, color=color, lw=1.4)
        ax2.plot(grupo["t"], grupo["mem_mib"], label=nombre, color=color, lw=1.4)
    ax1.set_title(f"Consumo de recursos de los contenedores ({etiqueta})")
    ax1.set_ylabel("CPU (%; 100 = 1 núcleo)")
    ax2.set_ylabel("Memoria (MiB)")
    ax2.set_xlabel("Tiempo de prueba (s)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        marcar_escalones(ax, historial)
    guardar(fig, carpeta / "recursos.png")


def resumir(historial: pd.DataFrame, docker: pd.DataFrame) -> pd.DataFrame:
    estable = historial[historial["t_en_escalon"] >= DESCARTE_S]
    g = estable.groupby("escalon")
    resumen = pd.DataFrame(
        {
            "usuarios": g["usuarios"].first(),
            "rps": g["Requests/s"].mean(),
            "inferencias_s": g["inferencias_s"].mean(),
            "p50_ms": g["50%"].median(),
            "p95_ms": g["95%"].median(),
            "p99_ms": g["99%"].median(),
            "error_pct": 100 * g["Failures/s"].sum() / g["Requests/s"].sum(),
        }
    )
    d = docker[(docker["t_en_escalon"] >= DESCARTE_S) & docker["escalon"].isin(resumen.index)]
    for contenedor, sufijo in (("mibici-api", "api"), ("mibici-mlflow", "mlflow")):
        c = d[d["contenedor"] == contenedor].groupby("escalon")
        resumen[f"cpu_{sufijo}_prom"] = c["cpu_pct"].mean()
        resumen[f"cpu_{sufijo}_max"] = c["cpu_pct"].max()
        resumen[f"mem_{sufijo}_mib_max"] = c["mem_mib"].max()
    return resumen.reset_index(drop=True).round(2)


def procesar(etiqueta: str, duracion: int) -> pd.DataFrame:
    carpeta = CARGA / etiqueta
    historial = cargar_historial(carpeta, duracion)
    docker = cargar_docker(carpeta, historial["Timestamp"].min(), duracion)
    graficar(historial, docker, carpeta, etiqueta)

    reposo = docker[docker["t"] < 0].groupby("contenedor")[["cpu_pct", "mem_mib"]].mean()
    if not reposo.empty:
        print("\nConsumo en reposo (antes de la carga):")
        print(reposo.round(2).to_string())

    resumen = resumir(historial, docker)
    resumen.to_csv(carpeta / "resumen_escalones.csv", index=False)
    print(f"\nResumen por escalón ({etiqueta}):")
    print(resumen.to_string(index=False))
    return resumen


def comparar(resumenes: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    paneles = (
        ("inferencias_s", "Inferencias por segundo"),
        ("p95_ms", "Latencia p95 (ms)"),
        ("cpu_api_prom", "CPU promedio de la API (%)"),
    )
    for ax, (columna, titulo) in zip(axes, paneles):
        for etiqueta, resumen in resumenes.items():
            ax.plot(resumen["usuarios"], resumen[columna], marker="o", label=etiqueta)
        ax.set_title(titulo)
        ax.set_xlabel("Usuarios concurrentes")
        ax.grid(alpha=0.3)
        ax.legend()
    guardar(fig, CARGA / f"comparacion_{'_vs_'.join(resumenes)}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gráficas de las pruebas de carga.")
    parser.add_argument("--etiqueta", default="w1")
    parser.add_argument("--comparar", nargs="*", default=[], help="Otras etiquetas a comparar.")
    parser.add_argument(
        "--segundos-escalon", type=int, default=60, help="Igual a CARGA_SEGUNDOS_ESCALON."
    )
    args = parser.parse_args()

    resumenes = {args.etiqueta: procesar(args.etiqueta, args.segundos_escalon)}
    for otra in args.comparar:
        resumenes[otra] = procesar(otra, args.segundos_escalon)
    if len(resumenes) > 1:
        comparar(resumenes)


if __name__ == "__main__":
    main()
