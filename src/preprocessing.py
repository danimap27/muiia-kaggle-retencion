"""
Pipeline de preprocesamiento e ingeniería de variables.

La idea es que todo lo que toque a los datos viva dentro de un único
``ColumnTransformer`` envuelto en un ``Pipeline`` de scikit-learn. Así, cuando
se entrena con validación cruzada, el imputador, el escalador y el OneHot solo
ven el fold de entrenamiento (estimación honesta sin fugas).

Las nuevas variables (Balance/Salario, Productos por edad, etc.) se calculan
con un ``FunctionTransformer`` antes de pasar al transformer numérico/categórico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from .config import NUMERICAL_RAW, CATEGORICAL_RAW, ID_COL, DROP_COLS, TARGET


# ---------- Ingeniería de variables ----------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas derivadas añadidas.

    Las variables nuevas se eligen tras la EDA: cocientes que capturan la
    saturación financiera del cliente, indicadores binarios para situaciones
    extremas (saldo cero, edad de jubilación) y agrupaciones por tramos.
    """
    out = df.copy()

    # Cocientes financieros: pequeño epsilon para evitar divisiones por cero.
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    out["TenureByAge"] = out["Tenure"] / (out["Age"] + 1.0)
    out["CreditScoreByAge"] = out["CreditScore"] / (out["Age"] + 1.0)
    out["BalancePerProduct"] = out["Balance"] / (out["NumOfProducts"] + 1.0)
    out["SalaryPerProduct"] = out["EstimatedSalary"] / (out["NumOfProducts"] + 1.0)

    # Indicadores binarios.
    out["IsZeroBalance"] = (out["Balance"] == 0).astype(int)
    out["IsHighProducts"] = (out["NumOfProducts"] >= 3).astype(int)
    out["IsSenior"] = (out["Age"] >= 60).astype(int)

    # Productos x actividad: un cliente con varios productos pero inactivo
    # suele ser candidato a fuga.
    out["ProductsActivity"] = out["NumOfProducts"] * (out["IsActiveMember"] + 1)

    return out


NUMERICAL_ENG = NUMERICAL_RAW + [
    "BalanceSalaryRatio", "TenureByAge", "CreditScoreByAge",
    "BalancePerProduct", "SalaryPerProduct",
    "IsZeroBalance", "IsHighProducts", "IsSenior", "ProductsActivity",
]


def split_X_y(df: pd.DataFrame):
    """Separa target, ID y predictoras. Aplica add_features a las predictoras."""
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    drop = [c for c in DROP_COLS + [ID_COL, TARGET] if c in df.columns]
    X = df.drop(columns=drop)
    X = add_features(X)
    return X, y


# ---------- Pipeline preprocesador ----------

def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Devuelve un ColumnTransformer con imputación + escalado + OneHot.

    Parameters
    ----------
    scale : bool
        Si False se omite el StandardScaler. Útil para árboles, que no se
        benefician del escalado y son insensibles a la magnitud absoluta.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERICAL_ENG),
            ("cat", cat_pipe, CATEGORICAL_RAW),
        ],
        remainder="drop",
    )
