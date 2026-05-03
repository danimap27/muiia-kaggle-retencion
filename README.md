# Proyecto Kaggle — Retención de clientes

Trabajo final del módulo de Ciencia de Datos y Aprendizaje Automático del MUIIA (UIMP). El objetivo es predecir la variable `Exited` (1 = fuga, 0 = permanencia) sobre el dataset de retención de clientes bancarios. La métrica de la competición es **F1**.

El código está pensado para ejecutarse en el **clúster Hércules del CICA** mediante SLURM y un entorno conda. Todas las rutas son relativas al directorio raíz del proyecto.

## Estructura

```
proyecto-kaggle/
├── data/                       # train.csv, test.csv, sample_submission.csv (NO versionados)
├── src/
│   ├── config.py               # rutas, semilla, columnas
│   ├── preprocessing.py        # ColumnTransformer + ingeniería de variables
│   ├── eda.py                  # análisis exploratorio
│   ├── cv.py                   # CV estratificada y métricas
│   ├── models.py               # catálogo de 14 modelos
│   ├── train_baseline.py       # CV de todo el catálogo
│   ├── hyperopt.py             # Optuna por modelo
│   ├── stacking.py             # meta-modelo
│   └── predict.py              # genera submission.csv
├── slurm/                      # scripts SLURM listos para sbatch
├── logs/                       # salida de los jobs
├── results/                    # CSV/JSON con métricas y predicciones
├── models/                     # modelos serializados (joblib)
├── figures/                    # gráficos EDA
├── memoria/                    # memoria PDF (max 10 págs)
├── requirements.txt
└── README.md
```

## Pipeline de modelado

1. **EDA** (`src/eda.py`): distribución del target, histogramas por clase, correlaciones, tasa de fuga por país y género.
2. **Preprocesamiento** (`src/preprocessing.py`):
   - Imputación: mediana para numéricas, moda para categóricas.
   - Escalado: `StandardScaler` (sólo para modelos lineales/MLP/SVM/KNN).
   - Codificación: `OneHotEncoder(handle_unknown='ignore')` para `Geography` y `Gender`.
   - Ingeniería de variables: `BalanceSalaryRatio`, `TenureByAge`, `CreditScoreByAge`, `BalancePerProduct`, `SalaryPerProduct`, `IsZeroBalance`, `IsHighProducts`, `IsSenior`, `ProductsActivity`.
   - Todo en un único `ColumnTransformer` dentro de un `Pipeline` para que la CV no produzca fugas de datos.
3. **Catálogo de modelos** (`src/models.py`): LDA, Logística, Naive Bayes, KNN, SVM RBF, MLP, árbol, Random Forest, Extra Trees, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost.
4. **Búsqueda de hiperparámetros** (`src/hyperopt.py`): Optuna con TPE y poda mediana, 80 trials por modelo (XGBoost, LightGBM, CatBoost, Random Forest, HistGB y MLP).
5. **Stacking** (`src/stacking.py`): mete los tres mejores boostings + Random Forest como base y Logística como meta-aprendiz.
6. **Predicción final** (`src/predict.py`): reentrena el modelo elegido sobre todo train y produce `submission_<modelo>.csv`.

## Ejecución en Hércules

> El nodo de login bloquea ejecuciones largas. Cualquier cosa más allá de `sbatch` y comprobaciones rápidas se hace en una sesión interactiva con `salloc`.

### 1. Preparar entorno (una sola vez)

```bash
salloc --mem=8G -c 4 -t 02:00:00 srun --pty /bin/bash -i
cd ~/proyecto-kaggle
bash slurm/00_setup_env.sh
exit
```

### 2. Subir datos

Descargar `train.csv`, `test.csv` y `sample_submission.csv` desde la página de la competición y dejarlos en `data/`.

### 3. Lanzar la cadena de jobs

```bash
sbatch slurm/01_eda.slurm                      # análisis exploratorio
sbatch slurm/02_baseline.slurm                 # CV de los 14 modelos
sbatch --array=0-5 slurm/03_hyperopt.slurm     # Optuna en paralelo
sbatch slurm/04_stacking.slurm                 # meta-modelo
sbatch slurm/05_predict.slurm lgbm             # submission.csv del mejor LightGBM
sbatch slurm/05_predict.slurm stacking         # alternativa con stacking
```

Para monitorizar:

```bash
squeue -u $USER
tail -f logs/baseline_*.out
```

## Resultados esperados

- `results/models_cv.csv` — tabla con F1, accuracy, precision, recall y AUC por modelo (5-fold CV).
- `results/optuna_<modelo>.json` — mejores hiperparámetros y F1 alcanzado.
- `results/stacking_cv.json` — métricas del stacking.
- `results/submission_<modelo>.csv` — fichero final para subir a Kaggle.
- `figures/eda_*.png` — gráficos del análisis exploratorio.

## Reproducibilidad

- Semilla fija (`SEED=42`) en config, modelos y splits.
- 5-fold CV estratificada idéntica para todas las comparaciones.
- Entorno conda con Python 3.11 y `pip install -r requirements.txt` (versiones exactas).
