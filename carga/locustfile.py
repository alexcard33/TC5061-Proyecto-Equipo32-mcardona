"""Escenario de carga para la API de MiBici con Locust.

Cada usuario virtual representa un cliente de la API, por ejemplo el tablero con el que el
equipo de operación decide a qué estaciones mandar camiones de rebalanceo. Ese cliente:

- pide el pronóstico de una sola estación (el caso más común, peso 6),
- pide un lote de 10 estaciones, como cuando se refresca una zona completa del mapa (peso 3),
- y de vez en cuando revisa que el servicio esté vivo con /health (peso 1).

Las observaciones se generan al azar a partir de las del archivo de ejemplo de las pruebas,
para que el modelo no reciba siempre exactamente la misma entrada.

La carga sube por escalones (ver EscalonesCarga) para encontrar el punto donde la API se
satura. Se corre con carga/ejecutar_prueba.py, que además registra docker stats.
"""

import json
import os
from pathlib import Path
import random

from locust import HttpUser, LoadTestShape, between, task

RAIZ = Path(__file__).resolve().parents[1]
EJEMPLOS = json.loads(
    (RAIZ / "tests" / "fixtures" / "predict_ejemplo.json").read_text(encoding="utf-8")
)["instancias"]

TAMANO_LOTE = 10


def observacion_aleatoria() -> dict:
    """Una estación-día válida para el esquema de la API, con valores parecidos a los reales."""
    base = random.choice(EJEMPLOS)
    dia = random.randint(0, 6)
    obs = dict(base)
    obs["dia_semana"] = dia
    obs["fin_semana"] = 1 if dia >= 5 else 0
    obs["feriado"] = 1 if random.random() < 0.03 else 0
    for campo in ("lag_1", "lag_2", "lag_7", "lag_14"):
        obs[campo] = max(0, round(base[campo] * random.uniform(0.6, 1.4)))
    for campo in ("media_7d", "media_28d", "media_mismo_dia_4sem"):
        obs[campo] = round(base[campo] * random.uniform(0.8, 1.2), 2)
    obs["dias_activa"] = random.randint(30, 3000)
    obs["capacidad"] = random.choice([12, 15, 18, 20, 24])
    return obs


class UsuarioOperacion(HttpUser):
    # Pausa entre peticiones de un mismo usuario, como alguien que revisa el resultado.
    wait_time = between(0.5, 1.5)

    @task(6)
    def predecir_una_estacion(self):
        self._predecir(1, "/predict [1 estación]")

    @task(3)
    def predecir_lote(self):
        self._predecir(TAMANO_LOTE, f"/predict [lote de {TAMANO_LOTE}]")

    @task(1)
    def salud(self):
        self.client.get("/health", name="/health")

    def _predecir(self, n: int, nombre: str) -> None:
        cuerpo = {"instancias": [observacion_aleatoria() for _ in range(n)]}
        with self.client.post("/predict", json=cuerpo, name=nombre, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            try:
                predicciones = r.json().get("predicciones", [])
            except ValueError:
                r.failure("La respuesta no es JSON")
                return
            if len(predicciones) != n:
                r.failure(f"Se esperaban {n} predicciones y llegaron {len(predicciones)}")
            else:
                r.success()


def _escalones() -> list[int]:
    texto = os.getenv("CARGA_USUARIOS", "5,10,25,50,100,150")
    return [int(x) for x in texto.split(",") if x.strip()]


class EscalonesCarga(LoadTestShape):
    """Sube la carga por escalones de usuarios y mantiene cada uno un tiempo fijo.

    Por defecto: 5, 10, 25, 50, 100 y 150 usuarios, 60 segundos cada uno (6 minutos).
    Se puede cambiar con las variables CARGA_USUARIOS (lista separada por comas) y
    CARGA_SEGUNDOS_ESCALON.
    """

    usuarios = _escalones()
    duracion = int(os.getenv("CARGA_SEGUNDOS_ESCALON", "60"))

    def tick(self):
        indice = int(self.get_run_time() // self.duracion)
        if indice >= len(self.usuarios):
            return None  # fin de la prueba
        objetivo = self.usuarios[indice]
        # Tasa de arranque: alcanzar el escalón en unos 5 segundos.
        return objetivo, max(1, objetivo / 5)
