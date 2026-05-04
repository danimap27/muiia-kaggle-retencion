"""
Pipeline de preprocesamiento e ingeniería de variables (versión ampliada).

Añade respecto a la versión inicial:
- Indicadores binarios de missing por columna con NaN.
- Frecuencia y target encoding de Surname (alta cardinalidad) con CV-safe fit.
- Imputación opcional KNN o IterativeImputer.
- Yeo-Johnson PowerTransformer sobre Balance/EstimatedSalary.
- RobustScaler como alternativa a StandardScaler.
- Más interacciones y bins más finos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, RobustScaler,
    FunctionTransformer, PowerTransformer,
)

from .config import NUMERICAL_RAW, CATEGORICAL_RAW, ID_COL, DROP_COLS, TARGET


# ---------- Transformer estado-ful para Surname (frecuencia + target enc) ----

class SurnameStats(BaseEstimator, TransformerMixin):
    """Calcula SurnameFreq (count) y SurnameTE (target encoding con smoothing)
    a partir del fold de entrenamiento. Espera columna 'Surname' en X y
    target y en .fit. CV-safe: solo ve el fold de train."""

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        s = X["Surname"].astype(str).fillna("__NA__")
        self.global_mean_ = float(np.mean(y))
        counts = s.value_counts()
        means = pd.Series(y.values if hasattr(y, "values") else y, index=s.index).groupby(s).mean()
        # Smoothing: (n*mean + m*global) / (n+m)
        smoothed = (counts * means + self.smoothing * self.global_mean_) / (counts + self.smoothing)
        self.freq_map_ = counts.to_dict()
        self.te_map_ = smoothed.to_dict()
        return self

    def transform(self, X):
        s = X["Surname"].astype(str).fillna("__NA__")
        freq = s.map(self.freq_map_).fillna(0).astype(float).values
        te = s.map(self.te_map_).fillna(self.global_mean_).astype(float).values
        return np.column_stack([freq, te])

    def get_feature_names_out(self, input_features=None):
        return np.array(["SurnameFreq", "SurnameTE"])


# ---------- Ingeniería de variables ----------

MISSING_COLS = ["CreditScore", "Balance", "NumOfProducts",
                "EstimatedSalary", "HasCrCard", "Surname"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas derivadas añadidas."""
    out = df.copy()

    # Missing indicators (1 si NaN, 0 si presente).
    for c in MISSING_COLS:
        if c in out.columns:
            out[f"{c}_NA"] = out[c].isna().astype(float)

    # Cocientes financieros.
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    out["TenureByAge"] = out["Tenure"] / (out["Age"] + 1.0)
    out["CreditScoreByAge"] = out["CreditScore"] / (out["Age"] + 1.0)
    out["BalancePerProduct"] = out["Balance"] / (out["NumOfProducts"] + 1.0)
    out["SalaryPerProduct"] = out["EstimatedSalary"] / (out["NumOfProducts"] + 1.0)
    out["CreditPerProduct"] = out["CreditScore"] / (out["NumOfProducts"] + 1.0)

    # Logaritmos.
    out["BalanceLog"] = np.log1p(out["Balance"])
    out["EstimatedSalaryLog"] = np.log1p(out["EstimatedSalary"])

    # Indicadores binarios.
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
    out["NoProducts"] = np.where(out["NumOfProducts"].isna(), np.nan,
                                  (out["NumOfProducts"] <= 1).astype(float))

    # Interacciones.
    out["ProductsActivity"] = out["NumOfProducts"] * (out["IsActiveMember"] + 1)
    out["AgeProducts"] = out["Age"] * out["NumOfProducts"]
    out["CreditTenure"] = out["CreditScore"] * out["Tenure"]
    out["AgeActive"] = out["Age"] * (out["IsActiveMember"] + 1)
    out["AgeActiveProducts"] = out["Age"] * (out["IsActiveMember"] + 1) * out["NumOfProducts"]
    out["BalanceActive"] = out["Balance"] * (out["IsActiveMember"] + 1)
    out["AgeCredit"] = out["Age"] * out["CreditScore"]

    # Bins.
    age_bins = [0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 200]
    out["AgeGroup"] = pd.cut(out["Age"], bins=age_bins, labels=False).astype(float)
    cs_bins = [0, 450, 550, 650, 750, 850, 1000]
    out["CreditGroup"] = pd.cut(out["CreditScore"], bins=cs_bins, labels=False).astype(float)
    bal_bins = [-1, 0, 50000, 100000, 150000, 1e9]
    out["BalanceGroup"] = pd.cut(out["Balance"], bins=bal_bins, labels=False).astype(float)

    # Longitud apellido.
    if "Surname" in out.columns:
        out["SurnameLen"] = out["Surname"].astype(str).str.len().astype(float)
    else:
        out["SurnameLen"] = np.nan

    # Interacciones categóricas.
    out["Geo_Gender"] = (out["Geography"].astype(str) + "_" +
                         out["Gender"].astype(str))
    out["Geo_Active"] = (out["Geography"].astype(str) + "_" +
                         out["IsActiveMember"].astype(str))

    return out


# Conjuntos de columnas. Mantener lista actualizada.
NUMERICAL_ENG = NUMERICAL_RAW + [
    "BalanceSalaryRatio", "TenureByAge", "CreditScoreByAge",
    "BalancePerProduct", "SalaryPerProduct", "CreditPerProduct",
    "BalanceLog", "EstimatedSalaryLog",
    "IsZeroBalance", "IsHighProducts", "IsSenior", "IsYoung",
    "IsLowCredit", "IsLongTenure", "NoProducts",
    "ProductsActivity", "AgeProducts", "CreditTenure",
    "AgeActive", "AgeActiveProducts", "BalanceActive", "AgeCredit",
    "AgeGroup", "CreditGroup", "BalanceGroup", "SurnameLen",
] + [f"{c}_NA" for c in MISSING_COLS]
CATEGORICAL_ENG = CATEGORICAL_RAW + ["Geo_Gender", "Geo_Active"]
SKEWED_NUM = ["Balance", "EstimatedSalary", "BalanceSalaryRatio",
              "BalancePerProduct", "SalaryPerProduct", "CreditTenure",
              "AgeProducts", "AgeActiveProducts", "BalanceActive", "AgeCredit"]


def split_X_y(df: pd.DataFrame):
    """Separa target, ID y predictoras. Mantiene Surname para SurnameStats."""
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    X = df.drop(columns=[c for c in [ID_COL, TARGET] if c in df.columns])
    X = add_features(X)
    # Surname se mantiene (lo consume SurnameStats); Surname se descartará al final del CT.
    return X, y


# ---------- Pipeline preprocesador ----------

def make_preprocessor(
    scale: bool = True,
    imputer: str = "median",
    power_transform: bool = True,
    robust: bool = False,
    use_surname_stats: bool = False,
) -> ColumnTransformer:
    """Construye un ColumnTransformer flexible.

    Parameters
    ----------
    scale : bool
        Aplica scaler a numéricas (no a árboles).
    imputer : {"median", "knn", "iterative"}
        Estrategia de imputación numérica.
    power_transform : bool
        Aplica Yeo-Johnson a columnas asimétricas (Balance, Salary, ratios).
    robust : bool
        Usa RobustScaler en lugar de StandardScaler.
    use_surname_stats : bool
        Añade SurnameFreq y SurnameTE como features.
    """
    # Imputador numérico.
    if imputer == "knn":
        num_imp = KNNImputer(n_neighbors=7)
    elif imputer == "iterative":
        num_imp = IterativeImputer(random_state=0, max_iter=10)
    else:
        num_imp = SimpleImputer(strategy="median")

    num_steps = [("imputer", num_imp)]
    if scale:
        num_steps.append(("scaler", RobustScaler() if robust else StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    # Pipeline para columnas asimétricas: imputar -> Yeo-Johnson -> scaler.
    skew_steps = [("imputer", SimpleImputer(strategy="median")),
                  ("power", PowerTransformer(method="yeo-johnson", standardize=True))]
    skew_pipe = Pipeline(steps=skew_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    # Distribuir columnas: las asimétricas van por skew_pipe si power_transform.
    if power_transform:
        skew_cols = [c for c in SKEWED_NUM if c in NUMERICAL_ENG]
        plain_num = [c for c in NUMERICAL_ENG if c not in skew_cols]
        transformers.append(("skew", skew_pipe, skew_cols))
        transformers.append(("num", num_pipe, plain_num))
    else:
        transformers.append(("num", num_pipe, NUMERICAL_ENG))
    transformers.append(("cat", cat_pipe, CATEGORICAL_ENG))

    if use_surname_stats:
        transformers.append(("surname", SurnameStats(smoothing=20.0), ["Surname"]))

    return ColumnTransformer(transformers=transformers, remainder="drop")
