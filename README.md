# Proyecto Kaggle — Retención de clientes

Trabajo final del módulo de Ciencia de Datos y Aprendizaje Automático del MUIIA (UIMP). Se predice la variable `Exited` (1 = fuga, 0 = permanencia) sobre un dataset de 8000 clientes bancarios. Métrica de la competición: **F1**.

El proyecto se ejecuta en el clúster **Hércules (CICA)** mediante el [hercules-framework](https://github.com/danimap27/hercules-framework): `config.yaml` define el sweep, `runner.py` traduce cada combinación a un comando, y `core/manager.py` lanza y monitoriza los array jobs SLURM.

## Estructura

```
proyecto-kaggle/
├── data/                       # train.csv, test.csv, sample_submission.csv (no versionados)
├── src/
│   ├── config.py               # rutas, semilla, columnas
│   ├── preprocessing.py        # ColumnTransformer + ingeniería de variables
│   ├── eda.py                  # análisis exploratorio
│   ├── cv.py                   # CV estratificada 5-fold y métricas
│   ├── models.py               # catálogo de 14 modelos
│   ├── train_baseline.py       # CV (acepta --only <modelo>)
│   ├── hyperopt.py             # Optuna por modelo
│   ├── threshold_tune.py       # umbral óptimo F1 OOF
│   ├── stacking.py             # meta-modelo (XGB+LGBM+Cat+RF -> LogReg)
│   └── predict.py              # genera submission.csv
├── core/                       # hercules-framework (no editar)
│   ├── manager.py              # HUB interactivo (R/F/M/T)
│   ├── slurm_generic.sh        # plantilla array job
│   ├── generate_tables.py
│   └── deploy.sh
├── config.yaml                 # sweep + fases + etiquetas LaTeX
├── runner.py                   # adapter del framework
├── requirements.txt
└── README.md
```

## Pipeline

1. **EDA** (`src/eda.py`).
2. **Preprocesamiento** (`src/preprocessing.py`):
   - Imputación: mediana para numéricas, moda para categóricas.
   - Escalado: `StandardScaler` (sólo modelos lineales/MLP/SVM/KNN).
   - OneHot para `Geography` y `Gender`.
   - 9 variables derivadas (cocientes, indicadores binarios).
   - Todo dentro de un `ColumnTransformer` para evitar fugas en CV.
3. **Catálogo** (`src/models.py`): LDA, Logística, NB, KNN, SVM RBF, MLP, árbol, Random Forest, Extra Trees, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost.
4. **Optuna** (`src/hyperopt.py`): TPE + poda mediana, 30-80 trials por modelo, callback con ETA.
5. **Threshold tuning** (`src/threshold_tune.py`): umbral óptimo F1 sobre probabilidades OOF.
6. **Stacking** (`src/stacking.py`): XGB+LGBM+Cat+RF como base, Logística como meta.
7. **Predict** (`src/predict.py`): aplica el umbral optimizado si existe.

## Ejecución en Hércules

### 1. Una sola vez: entorno conda

```bash
salloc --mem=8G -c 4 -t 02:00:00 srun --pty /bin/bash -i
cd ~/proyecto-kaggle
module load Miniconda3
conda create -n kaggle-retencion python=3.11 -y
source activate kaggle-retencion
pip install -r requirements.txt
exit
```

### 2. Datos

Descargar `train.csv`, `test.csv`, `sample_submission.csv` y dejarlos en `data/`.

### 3. Lanzar y monitorizar con manager

```bash
cd ~/proyecto-kaggle
python core/manager.py
```

Menú interactivo:
- `R` — refresca los `cmds_*.txt` desde `config.yaml`.
- `1`, `2`, ... — lanza una fase como array job.
- `F` — lanza todas las fases en cadena con dependencias SLURM.
- `M` — monitor en vivo (F1 medio por fase, refresco 2 s).
- `C` — comprueba qué runs ya están completas (resume).
- `T` — genera tablas LaTeX desde los CSV.
- `X` — sale.

### Fases (definidas en `config.yaml`)

| # | Fase | Tareas | Cuándo |
|---|---|---|---|
| 1 | eda | 1 | Una vez al principio |
| 2 | baseline | 14 (un modelo por tarea) | Tras EDA |
| 3 | hyperopt | 6 (xgb, lgbm, cat, histgb, rf, mlp) | Tras baseline |
| 4 | threshold | 4 (cat, histgb, xgb, lgbm) | Tras hyperopt |
| 5 | stacking | 1 | Tras hyperopt |
| 6 | predict | 5 (cat, histgb, xgb, lgbm, stacking) | Final |

Total: 31 runs.

### Ejecución manual sin manager

```bash
# Generar comandos
python runner.py --export-commands

# Lanzar una fase concreta como array
sbatch --array=1-14 --export=CMD_FILE=cmds_2_baseline.txt \
       --job-name=baseline core/slurm_generic.sh

# Lanzar una orden suelta
python runner.py --run-id hyperopt__cat
```

## Reproducibilidad

- Semilla fija (`SEED=42`) en config, modelos y splits.
- 5-fold CV estratificada idéntica para todas las comparaciones.
- `requirements.txt` con versiones exactas (Python 3.11).
