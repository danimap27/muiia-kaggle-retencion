"""
Configuración global del proyecto Kaggle de retención de clientes.

Aquí se definen rutas relativas (necesarias para que Hercules no se queje de
permisos), semilla, columnas del dataset y parámetros de validación cruzada.
Cualquier script importa de aquí sus constantes para que un cambio de seed o
de splits no obligue a tocar los demás módulos.
"""
from pathlib import Path

# Rutas relativas al directorio raíz del proyecto
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RES_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"
LOG_DIR = ROOT / "logs"

for d in (RES_DIR, MODELS_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

SEED = 42
N_SPLITS = 5
N_JOBS = -1
SCORING = "f1"

TARGET = "Exited"
ID_COL = "CustomerId"
DROP_COLS = ["Surname"]

NUMERICAL_RAW = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL_RAW = ["Geography", "Gender"]
