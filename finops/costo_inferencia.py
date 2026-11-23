"""Análisis FinOps: costo por inferencia de la API de MiBici.

    python finops/costo_inferencia.py

Toma lo que se midió en las pruebas de carga (reports/carga/w1 y w2) y lo traduce a dinero
con precios de referencia de AWS (us-east-1, Linux, bajo demanda salvo que se indique otra
cosa). Escribe en reports/finops/:

    capacidad.csv            inferencias/s que aguanta cada configuración sin romper el SLO
    costo_por_inferencia.csv costo por millón de inferencias con cada opción de despliegue
    escenarios_trafico.csv   réplicas, costo mensual y costo efectivo según la demanda real
    costo_por_millon.png     comparación de opciones a utilización plena
    costo_vs_demanda.png     cómo se dispara el costo por inferencia cuando sobra capacidad

Supuestos principales (se discuten en el reporte):
- Un núcleo de la máquina local equivale a una vCPU de la nube. El docker-compose limita la
  API a 2 CPUs, igual que una instancia de 2 vCPU.
- El SLO es p95 <= 200 ms sin errores. La capacidad de cada configuración es la mayor
  cantidad de inferencias por segundo que se midió cumpliéndolo.
- Una petición de lote de 10 cuenta como 10 inferencias.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
CARGA = RAIZ / "reports" / "carga"
SALIDA = RAIZ / "reports" / "finops"

SLO_P95_MS = 200
HORAS_MES = 730
TAMANO_LOTE = 10
ESTACIONES_MIBICI = 484  # estaciones activas en los datos de 2026 (ver reporte de drift)

# Precios de referencia en USD, AWS us-east-1, Linux. Consultados en noviembre de 2026.
# EC2: páginas de precios de c7i en doit.com/compute y calculator.holori.com.
# Fargate y ALB: página oficial de precios de AWS y vantage.sh / cloudzero.com.
PRECIOS = {
    "c7i.large_bajo_demanda": 0.0892,  # 2 vCPU, 4 GiB
    "c7i.large_savings_plan_1a": 0.0645,  # Compute Savings Plan 1 año sin pago adelantado
    "c7i.large_spot": 0.0292,  # varía por zona y hora; puede interrumpirse
    "c7i.xlarge_bajo_demanda": 0.1785,  # 4 vCPU, 8 GiB
    "fargate_vcpu_h": 0.04048,  # Linux/x86
    "fargate_gb_h": 0.004445,
    "fargate_arm_vcpu_h": 0.03238,  # Linux/ARM (Graviton)
    "fargate_arm_gb_h": 0.00356,
    "alb_h": 0.0225,  # Application Load Balancer, sin contar LCU
}


def fargate(vcpu: float, gb: float, arm: bool = False) -> float:
    if arm:
        return vcpu * PRECIOS["fargate_arm_vcpu_h"] + gb * PRECIOS["fargate_arm_gb_h"]
    return vcpu * PRECIOS["fargate_vcpu_h"] + gb * PRECIOS["fargate_gb_h"]


def leer_csv(ruta: Path) -> pd.DataFrame:
    # Locust en Windows escribe en cp1252 (ver carga/graficas.py).
    try:
        return pd.read_csv(ruta, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(ruta, encoding="cp1252")


def costo_por_millon(precio_hora: float, inferencias_s: float) -> float:
    return precio_hora / (inferencias_s * 3600) * 1_000_000


def capacidad(etiqueta: str) -> dict:
    r = leer_csv(CARGA / etiqueta / "resumen_escalones.csv")
    ok = r[(r["p95_ms"] <= SLO_P95_MS) & (r["error_pct"] == 0)]
    fila = ok.loc[ok["inferencias_s"].idxmax()]
    maxima = r.loc[r["inferencias_s"].idxmax()]
    # Segundos de CPU que gasta la API por cada inferencia, en el punto de capacidad.
    cpu_ms = (fila["cpu_api_prom"] / 100) / fila["inferencias_s"] * 1000
    return {
        "configuracion": etiqueta,
        "usuarios_en_slo": int(fila["usuarios"]),
        "inferencias_s_slo": fila["inferencias_s"],
        "p95_ms_slo": fila["p95_ms"],
        "cpu_api_pct_slo": fila["cpu_api_prom"],
        "cpu_ms_por_inferencia": round(cpu_ms, 2),
        "mem_api_mib_max": r["mem_api_mib_max"].max(),
        "mem_mlflow_mib_max": r["mem_mlflow_mib_max"].max(),
        "inferencias_s_max": maxima["inferencias_s"],
        "p95_ms_en_max": maxima["p95_ms"],
    }


def latencia_por_tipo(etiqueta: str) -> pd.DataFrame:
    s = leer_csv(CARGA / etiqueta / "locust_stats.csv")
    s = s[s["Name"] != "Aggregated"][["Name", "Request Count", "Median Response Time", "95%"]]
    s.insert(0, "configuracion", etiqueta)
    return s


def opciones(cap: dict[str, dict]) -> pd.DataFrame:
    w1, w2 = cap["w1"]["inferencias_s_slo"], cap["w2"]["inferencias_s_slo"]
    filas = [
        ("w1 en EC2 c7i.large (como el compose: API + MLflow)", PRECIOS["c7i.large_bajo_demanda"], w1),
        ("w2 en EC2 c7i.large", PRECIOS["c7i.large_bajo_demanda"], w2),
        ("w2 en EC2 c7i.large + Savings Plan 1 año", PRECIOS["c7i.large_savings_plan_1a"], w2),
        ("w2 en EC2 c7i.large Spot", PRECIOS["c7i.large_spot"], w2),
        ("w1 en Fargate 1 vCPU / 2 GB (sin MLflow)", fargate(1, 2), w1),
        ("w2 en Fargate 2 vCPU / 4 GB (sin MLflow)", fargate(2, 4), w2),
        ("w2 en Fargate ARM 2 vCPU / 4 GB (sin MLflow)", fargate(2, 4, arm=True), w2),
    ]  # fmt: skip
    df = pd.DataFrame(filas, columns=["opcion", "usd_hora", "inferencias_s"])
    df["usd_mes"] = df["usd_hora"] * HORAS_MES
    df["usd_por_millon"] = [costo_por_millon(p, i) for p, i in zip(df.usd_hora, df.inferencias_s)]
    # Piso teórico: solo los segundos de CPU que de verdad consume cada inferencia, a precio
    # de vCPU de Fargate. Es lo que costaría si no hubiera capacidad ociosa.
    piso = cap["w2"]["cpu_ms_por_inferencia"] / 1000 * PRECIOS["fargate_vcpu_h"] / 3600 * 1e6
    nan = float("nan")
    df.loc[len(df)] = ["Piso: solo la CPU consumida (w2, vCPU Fargate)", nan, nan, nan, piso]
    return df.round(4)


def escenarios(cap: dict[str, dict]) -> pd.DataFrame:
    """Réplicas y costo según la demanda promedio, con la configuración w2 en c7i.large."""
    capacidad_replica = cap["w2"]["inferencias_s_slo"]
    demandas = [
        ("Pronóstico diario de todas las estaciones", ESTACIONES_MIBICI / 86_400),
        ("Tablero que refresca las estaciones cada 15 min", ESTACIONES_MIBICI * 96 / 86_400),
        ("App pública con consultas frecuentes", 50.0),
        ("Demanda al límite de una réplica", capacidad_replica),
        ("Demanda alta (horizontal)", 1_000.0),
    ]
    filas = []
    for nombre, demanda in demandas:
        replicas = max(1, math.ceil(demanda / capacidad_replica))
        balanceador = PRECIOS["alb_h"] if replicas > 1 else 0.0
        usd_hora = replicas * PRECIOS["c7i.large_bajo_demanda"] + balanceador
        filas.append(
            {
                "escenario": nombre,
                "inferencias_s": round(demanda, 3),
                "inferencias_mes": round(demanda * 3600 * HORAS_MES),
                "replicas": replicas,
                "utilizacion_pct": round(100 * demanda / (replicas * capacidad_replica), 2),
                "usd_mes": round(usd_hora * HORAS_MES, 2),
                "usd_por_millon": round(costo_por_millon(usd_hora, demanda), 4),
            }
        )
    return pd.DataFrame(filas)


def graficar(costos: pd.DataFrame, cap: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    datos = costos.iloc[::-1]
    colores = ["#16a34a" if "Piso" in o else "#2563eb" for o in datos["opcion"]]
    ax.barh(datos["opcion"], datos["usd_por_millon"], color=colores)
    for y, valor in enumerate(datos["usd_por_millon"]):
        ax.text(valor, y, f" ${valor:.3f}", va="center", fontsize=8)
    ax.set_xlabel("USD por millón de inferencias (a utilización plena, con SLO p95 <= 200 ms)")
    ax.set_title("Costo por inferencia según la opción de despliegue")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(SALIDA / "costo_por_millon.png", dpi=150)
    plt.close(fig)

    capacidad_replica = cap["w2"]["inferencias_s_slo"]
    demanda = [10**e for e in [x / 10 for x in range(-30, 35)]]
    siempre_encendido = []
    for d in demanda:
        replicas = max(1, math.ceil(d / capacidad_replica))
        usd_hora = replicas * PRECIOS["c7i.large_bajo_demanda"] + (
            PRECIOS["alb_h"] if replicas > 1 else 0
        )
        siempre_encendido.append(costo_por_millon(usd_hora, d))
    piso = costos.iloc[-1]["usd_por_millon"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        demanda,
        siempre_encendido,
        color="#2563eb",
        lw=1.8,
        label="EC2 c7i.large siempre encendida (w2)",
    )
    ax.axhline(piso, color="#16a34a", ls="--", label="Piso: solo la CPU consumida")
    tablero = ESTACIONES_MIBICI * 96 / 86_400
    ax.axvline(tablero, color="#9ca3af", ls=":", label="Tablero cada 15 min")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Demanda promedio (inferencias por segundo)")
    ax.set_ylabel("USD por millón de inferencias")
    ax.set_title("El costo por inferencia depende de cuánto se usa la capacidad pagada")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "costo_vs_demanda.png", dpi=150)
    plt.close(fig)


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    cap = {e: capacidad(e) for e in ("w1", "w2")}

    tabla_cap = pd.DataFrame(cap.values()).round(2)
    tabla_cap.to_csv(SALIDA / "capacidad.csv", index=False)
    print(f"Capacidad por configuración (SLO p95 <= {SLO_P95_MS} ms):")
    print(tabla_cap.to_string(index=False))

    latencias = pd.concat([latencia_por_tipo(e) for e in ("w1", "w2")])
    latencias.to_csv(SALIDA / "latencia_por_tipo.csv", index=False)
    print("\nLatencia por tipo de petición (toda la prueba):")
    print(latencias.to_string(index=False))

    costos = opciones(cap)
    costos.to_csv(SALIDA / "costo_por_inferencia.csv", index=False)
    print("\nCosto por millón de inferencias a utilización plena:")
    print(costos.to_string(index=False))

    tabla_esc = escenarios(cap)
    tabla_esc.to_csv(SALIDA / "escenarios_trafico.csv", index=False)
    print("\nEscenarios de demanda (w2 en c7i.large bajo demanda):")
    print(tabla_esc.to_string(index=False))

    graficar(costos, cap)
    print(f"\nResultados en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
