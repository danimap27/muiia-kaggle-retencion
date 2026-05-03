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

    Variables generadas:
    - Cocientes financieros (BalanceSalaryRatio, ...).
    - Transformaciones logarítmicas (BalanceLog, EstimatedSalaryLog).
    - Indicadores binarios (IsZeroBalance, IsSenior, ...).
    - Bucket de edad (AgeGroup) y de credit score (CreditGroup).
    - Longitud del apellido como proxy de origen cultural.
    - Interacciones (Geography_Gender, AgeProducts).
    """
    out = df.copy()

    # Cocientes financieros (epsilon evita división por cero).
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    out["TenureByAge"] = out["Tenure"] / (out["Age"] + 1.0)
    out["CreditScoreByAge"] = out["CreditScore"] / (out["Age"] + 1.0)
    out["BalancePerProduct"] = out["Balance"] / (out["NumOfProducts"] + 1.0)
    out["SalaryPerProduct"] = out["EstimatedSalary"] / (out["NumOfProducts"] + 1.0)

    # Logaritmos de variables muy asimétricas.
    out["BalanceLog"] = np.log1p(out["Balance"])
    out["EstimatedSalaryLog"] = np.log1p(out["EstimatedSalary"])

    # Indicadores binarios (preserva NaN).
    out["IsZeroBalance"] = np.where(out["Balance"].isna(), np.nan,
                                     (out["Balance"] == 0).astype(float))
    out["IsHighProducts"] = np.where(out["NumOfProducts"].isna(), np.nan,
                                      (out["NumOfProducts"] >= 3).astype(float))
    out["IsSenior"] = np.where(out["Age"].isna(), np.nan,
                                (out["Age"] >= 60).astype(float))
    out["IsYoung"] = np.where(out["Age"].isna(), np.nan,
                               (out["Age"] < 30).astype(float))
    out["IsLowCredit"] = np.where(out["CreditScore"].isna(), np.nan,
                                   (out["CreditScore"] < 500).astype(float))
    out["IsLongTenure"] = np.where(out["Tenure"].isna(), np.nan,
                                    (out["Tenure"] >= 8).astype(float))

    # Productos x actividad: cliente con varios productos pero inactivo es
    # candidato a fuga.
    out["ProductsActivity"] = out["NumOfProducts"] * (out["IsActiveMember"] + 1)
    out["AgeProducts"] = out["Age"] * out["NumOfProducts"]
    out["CreditTenure"] = out["CreditScore"] * out["Tenure"]

    # Bucket de edad y credit score como categorías ordinales.
    age_bins = [0, 25, 35, 45, 55, 65, 200]
    out["AgeGroup"] = pd.cut(out["Age"], bins=age_bins, labels=False).astype(float)
    cs_bins = [0, 500, 600, 700, 800, 900]
    out["CreditGroup"] = pd.cut(out["CreditScore"], bins=cs_bins, labels=False).astype(float)

    # Longitud del apellido (proxy débil de origen).
    if "Surname" in out.columns:
        out["SurnameLen"] = out["Surname"].astype(str).str.len().astype(float)
    else:
        out["SurnameLen"] = np.nan

    # Interacción categórica.
    out["Geo_Gender"] = (out["Geography"].astype(str) + "_" +
                         out["Gender"].astype(str))

    return out


NUMERICAL_ENG = NUMERICAL_RAW + [
    "BalanceSalaryRatio", "TenureByAge", "CreditScoreByAge",
    "BalancePerProduct", "SalaryPerProduct",
    "BalanceLog", "EstimatedSalaryLog",
    "IsZeroBalance", "IsHighProducts", "IsSenior", "IsYoung",
    "IsLowCredit", "IsLongTenure",
    "ProductsActivity", "AgeProducts", "CreditTenure",
    "AgeGroup", "CreditGroup", "SurnameLen",
]
CATEGORICAL_ENG = CATEGORICAL_RAW + ["Geo_Gender"]


def split_X_y(df: pd.DataFrame):
    """Separa target, ID y predictoras. Aplica add_features antes de descartar
    columnas auxiliares (Surname se usa para SurnameLen y luego se elimina)."""
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    X = df.drop(columns=[c for c in [ID_COL, TARGET] if c in df.columns])
    X = add_features(X)
    drop = [c for c in DROP_COLS if c in X.columns]
    X = X.drop(columns=drop)
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
            ("cat", cat_pipe, CATEGORICAL_ENG),
        ],
        remainder="drop",
    )
