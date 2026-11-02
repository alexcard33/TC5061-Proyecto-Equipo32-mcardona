# Chequeos de seguridad — Equipo 32

Paso 3 de la Actividad 7. Resume las herramientas, los hallazgos y lo que se hizo con
cada uno. Las salidas completas de cada herramienta están en esta misma carpeta.

## Cómo reproducirlo

Con el entorno principal (`.venv`) activado y desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt            # instala bandit y pip-audit
bandit -r mibici -f txt -o reports/security/bandit.txt
pip-audit -r requirements.txt -o reports/security/pip_audit.txt
pip-audit -r requirements-validacion.txt -o reports/security/pip_audit_validacion.txt
python -m pytest tests/test_seguridad.py -v
```

## 1. Análisis estático del código (bandit)

| | Antes | Después |
|---|---:|---:|
| Severidad alta / media | 0 / 0 | 0 / 0 |
| Severidad baja | 13 | **0** |

Los 13 hallazgos (B404, B603, B607) venían de la misma función, copiada en P1, P2 y P3,
que ejecuta `git rev-parse` y `git status` para registrar el commit en MLflow:

- **B607** (ejecutable con ruta parcial): **corregido**. Ahora se usa la ruta absoluta de
  git que da `shutil.which("git")`, así que un `git` falso en el PATH no se ejecuta.
- **B404 / B603** (uso de `subprocess`): **riesgo aceptado y documentado** con
  `# nosec`. Los argumentos son fijos, no se usa `shell=True` y ningún dato externo llega
  al comando.
- Las tres copias se unificaron en `mibici.modeling.mlflow_utils.obtener_git_commit`.

## 2. Auditoría de dependencias (pip-audit)

| Entorno | Paquete | Vulnerabilidad | Acción |
|---|---|---|---|
| Principal (`requirements.txt`) | mlflow 3.17.1 | CVE-2026-53995 | **Corregido**: actualizado a 3.17.2. Resultado final: *No known vulnerabilities found*. |
| Validación (`requirements-validacion.txt`) | pyarrow 21.0.0 | PYSEC-2026-113 | **Corregido**: actualizado a 23.0.1, compatible con numpy 1.26. |
| Validación | setuptools 80.10.2 | PYSEC-2026-3447, GHSA-xmj3-9gf7-h52w | **Riesgo aceptado** (ver abajo). |

Por qué se acepta setuptools < 81: DeepChecks 0.19.1 (última versión) importa
`pkg_resources`, que se eliminó en setuptools 81. Las dos vulnerabilidades afectan la
construcción de distribuciones de código fuente (`sdist`) y la instalación de wheels por
la ruta heredada de *eggs*; el proyecto no hace ninguna de las dos. Además, el entorno de
validación es local, aislado del de entrenamiento y no se despliega. Se revisará cuando
DeepChecks publique una versión sin `pkg_resources`.

## 3. Credenciales y datos sensibles

- **Variables de entorno.** La URI de MLflow y cualquier credencial se leen del entorno o
  de un `.env` local (`mibici/config.py` usa `python-dotenv`). La plantilla `.env.example`
  se versiona sin valores; `.env` nunca se versiona.
- **`.gitignore`.** Excluye `.env` y `.env.*` (salvo la plantilla), llaves (`*.pem`,
  `*.key`, `*.p12`, `credentials*.json`), la base de MLflow (`mlflow.db`, `mlruns/`,
  `mlartifacts/`), los datos (`data/raw`, `data/processed`...) y los entornos virtuales.
- **Datos personales.** La ingesta (`mibici/dataset.py`) elimina el pseudónimo de usuario,
  el género y el año de nacimiento antes de guardar los Parquet; ningún artefacto de
  MLflow los contiene. Probado en `tests/test_dataset.py`.
- **Pruebas automáticas** (`tests/test_seguridad.py`):
  - ningún archivo sensible está versionado y el `.gitignore` cubre cada tipo;
  - los archivos versionados **y todo el historial de Git (todas las ramas)** no
    contienen credenciales con formatos conocidos (llaves de AWS, llaves privadas,
    tokens de GitHub/Slack/OpenAI, contraseñas en texto plano, URLs con usuario y
    contraseña);
  - los detectores encuentran secretos de ejemplo y no marcan código seguro;
  - la URI de MLflow se toma de la variable de entorno y `.env.example` no trae valores.
