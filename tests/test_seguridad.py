"""Chequeos de seguridad básicos del repositorio.

1. Ningún archivo sensible está versionado (``.env``, bases de MLflow, datos crudos,
   llaves) y el ``.gitignore`` los cubre.
2. Ni los archivos versionados ni el historial de Git contienen credenciales con
   formatos conocidos (llaves de AWS, tokens de GitHub, llaves privadas...).
3. La configuración sensible se lee de variables de entorno y existe la plantilla
   ``.env.example`` sin valores reales.

Las pruebas que consultan Git se omiten si ``git`` no está disponible.
"""

import importlib
import re
import shutil
import subprocess  # nosec B404 - solo git con argumentos fijos

import pytest

from mibici import config

RAIZ = config.PROJ_ROOT
GIT = shutil.which("git")
requiere_git = pytest.mark.skipif(
    GIT is None or not (RAIZ / ".git").exists(), reason="Requiere git y el repositorio clonado"
)

# Formatos de credenciales conocidos (se buscan en archivos y en el historial).
PATRONES_SECRETOS = {
    "llave de acceso AWS": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "llave privada": re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    "token de GitHub": re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|github_pat_\w{22,}"),
    "token de Slack": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "llave de OpenAI/Anthropic": re.compile(r"\bsk-(ant-)?[A-Za-z0-9_-]{32,}"),
    "contraseña en texto plano": re.compile(
        r"(?i)\b(password|passwd|pwd|secret|api_key|apikey|access_token)\b\s*[:=]\s*"
        r"['\"][^'\"\s]{6,}['\"]"
    ),
    "URL con usuario y contraseña": re.compile(r"[a-z]+://[^/\s:@]+:[^/\s:@]+@[^/\s]+"),
}

# Archivos que nunca deben versionarse.
PROHIBIDOS = [
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.env\.(?!example$)[^/]+$"),
    re.compile(r"\.(pem|key|p12|db|sqlite)$"),
    re.compile(r"^data/(raw|interim|processed|external)/"),
    re.compile(r"^(mlruns|mlartifacts)/"),
    re.compile(r"(^|/)credentials[^/]*\.json$"),
]


def _git(*args: str) -> str:
    return subprocess.run(  # nosec B603
        [GIT, *args], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=True,
    ).stdout  # fmt: skip


def _buscar_secretos(lineas):
    hallazgos = []
    for i, linea in enumerate(lineas):
        if len(linea) > 2000:  # imágenes en base64 de los notebooks
            continue
        for nombre, patron in PATRONES_SECRETOS.items():
            if patron.search(linea):
                hallazgos.append((nombre, i, linea.strip()[:120]))
    return hallazgos


# ---------- Los detectores funcionan ----------
@pytest.mark.parametrize(
    "linea",
    [
        "aws_key = 'AKIAABCDEFGHIJKLMNOP'",
        "-----BEGIN RSA PRIVATE KEY-----",
        "token: ghp_" + "a" * 36,
        'password = "Sup3rS3creta!"',
        "MLFLOW_TRACKING_URI=https://usuario:clave123@mlflow.ejemplo.com",
    ],
)
def test_detector_encuentra_secretos(linea):
    assert _buscar_secretos([linea])


@pytest.mark.parametrize(
    "linea",
    [
        'MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")',
        "MLFLOW_TRACKING_PASSWORD=",
        "password = os.getenv('PASSWORD')",
        "https://www.mibici.net/site/assets/files/97653/datos_abiertos_2025_01.csv",
    ],
)
def test_detector_ignora_codigo_seguro(linea):
    assert not _buscar_secretos([linea])


# ---------- 1. Archivos sensibles fuera del repositorio ----------
@requiere_git
def test_no_hay_archivos_sensibles_versionados():
    versionados = _git("ls-files").splitlines()
    prohibidos = [f for f in versionados if any(p.search(f) for p in PROHIBIDOS)]
    assert prohibidos == []


@requiere_git
@pytest.mark.parametrize(
    "ruta",
    [
        ".env", ".env.local", "llave.pem", "mlflow.db", "mlruns/0/meta.yaml",
        "data/raw/viajes_2025_01.csv", "data/processed/viajes.parquet", ".venv/pyvenv.cfg",
        ".venv-validacion/pyvenv.cfg", "credentials.json",
    ],
)  # fmt: skip
def test_gitignore_cubre_archivos_sensibles(ruta):
    resultado = subprocess.run(  # nosec B603
        [GIT, "check-ignore", "-q", "--no-index", ruta], cwd=RAIZ, check=False
    )
    assert resultado.returncode == 0, f"{ruta} no está en .gitignore"


@requiere_git
def test_env_example_no_esta_ignorado():
    resultado = subprocess.run(  # nosec B603
        [GIT, "check-ignore", "-q", "--no-index", ".env.example"], cwd=RAIZ, check=False
    )
    assert resultado.returncode == 1


# ---------- 2. Sin credenciales en archivos ni en el historial ----------
@requiere_git
def test_archivos_versionados_sin_credenciales():
    hallazgos = []
    for archivo in _git("ls-files").splitlines():
        ruta = RAIZ / archivo
        if not ruta.is_file() or archivo.startswith("tests/test_seguridad"):
            continue
        try:
            texto = ruta.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binarios (pdf, docx, parquet...)
        hallazgos += [(archivo, *h) for h in _buscar_secretos(texto.splitlines())]
    assert hallazgos == []


@requiere_git
def test_historial_de_git_sin_credenciales():
    """Revisa todas las versiones de todos los archivos en todas las ramas."""
    diff = _git("log", "--all", "-p", "--no-color", "--", ".", ":(exclude)tests/test_seguridad.py")
    agregadas = [
        ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")
    ]
    assert _buscar_secretos(agregadas) == []


# ---------- 3. Configuración por variables de entorno ----------
def test_tracking_uri_se_lee_de_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://servidor-de-prueba:5000")
    try:
        assert importlib.reload(config).MLFLOW_TRACKING_URI == "http://servidor-de-prueba:5000"
    finally:
        monkeypatch.delenv("MLFLOW_TRACKING_URI")
        importlib.reload(config)


def test_env_example_existe_sin_valores_secretos():
    plantilla = RAIZ / ".env.example"
    assert plantilla.exists()
    for linea in plantilla.read_text(encoding="utf-8").splitlines():
        if linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        if any(p in clave for p in ("PASSWORD", "SECRET", "KEY", "TOKEN")):
            assert valor.strip() == "", f"{clave} tiene un valor en la plantilla"
