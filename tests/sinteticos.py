"""Generador de viajes sintéticos con semilla fija para las pruebas."""

import numpy as np
import pandas as pd

from mibici.config import SEED


def generar_viajes(
    n_estaciones: int = 6,
    dias: int = 60,
    inicio: str = "2025-01-01",
    viajes_por_dia: int = 120,
    seed: int = SEED,
) -> pd.DataFrame:
    
    rng = np.random.default_rng(seed)
    n = dias * viajes_por_dia
    base = pd.Timestamp(inicio)
    dia = rng.integers(0, dias, n)
    origen = rng.integers(1, n_estaciones + 1, n)
    destino = rng.integers(1, n_estaciones + 1, n)
    hora = np.where(origen % 2 == 0, rng.normal(8, 1.5, n), rng.normal(18, 1.5, n))
    hora = np.clip(hora, 0, 23.9)
    inicio_viaje = base + pd.to_timedelta(dia, unit="D") + pd.to_timedelta(hora, unit="h")
    duracion = pd.to_timedelta(rng.integers(120, 3600, n), unit="s")
    viajes = pd.DataFrame(
        {
            "viaje_id": np.arange(n),
            "inicio_viaje": inicio_viaje.floor("s"),
            "fin_viaje": (inicio_viaje + duracion).floor("s"),
            "origen_id": origen,
            "destino_id": destino,
        }
    )
    return viajes.sort_values("inicio_viaje", ignore_index=True)
