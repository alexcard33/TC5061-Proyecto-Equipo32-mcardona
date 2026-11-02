import matplotlib

matplotlib.use("Agg")  # sin ventanas: las pruebas pueden correr en CI o por SSH

import pandas as pd
import pytest

from tests.sinteticos import generar_viajes


@pytest.fixture
def viajes_sinteticos() -> pd.DataFrame:
    return generar_viajes()


@pytest.fixture
def capacidad_sintetica() -> pd.Series:
    return pd.Series({i: 15 + i for i in range(1, 7)}, name="dpcapacity")
