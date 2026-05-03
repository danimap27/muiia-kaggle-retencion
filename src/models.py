"""
Catálogo de modelos. Cada modelo es un Pipeline completo
(preprocesamiento -> clasificador) para que la CV lo trate como una sola unidad.

Los árboles y boostings no necesitan escalado: les pasamos un preprocesador
sin StandardScaler para no inflar el coste innecesariamente.
"""
from __future__ import annotations

from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier

from .preprocessing import make_preprocessor
from .config import SEED


def _wrap(name: str, clf, scale: bool = True) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale=scale)),
            ("classifier", clf),
        ]
    )


def get_models() -> Dict[str, Pipeline]:
    """Devuelve un diccionario nombre -> pipeline listo para CV."""
    models: Dict[str, Pipeline] = {}

    # Lineales / clásicos. class_weight="balanced" porque la clase positiva es ~20%.
    models["LDA"] = _wrap("LDA", LinearDiscriminantAnalysis(), scale=True)
    models["LogReg"] = _wrap(
        "LogReg",
        LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED,
        ),
        scale=True,
    )
    models["GaussianNB"] = _wrap("GaussianNB", GaussianNB(), scale=True)
    models["SVC_RBF"] = _wrap(
        "SVC_RBF",
        SVC(C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=SEED),
        scale=True,
    )
    models["KNN"] = _wrap(
        "KNN", KNeighborsClassifier(n_neighbors=15, weights="distance"), scale=True,
    )
    models["MLP"] = _wrap(
        "MLP",
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                      random_state=SEED, early_stopping=True),
        scale=True,
    )

    # Árboles y ensambles. No necesitan StandardScaler.
    models["DecisionTree"] = _wrap(
        "DecisionTree",
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED, max_depth=10),
        scale=False,
    )
    models["RandomForest"] = _wrap(
        "RandomForest",
        RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        scale=False,
    )
    models["ExtraTrees"] = _wrap(
        "ExtraTrees",
        ExtraTreesClassifier(
            n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        scale=False,
    )
    models["GradBoost"] = _wrap(
        "GradBoost",
        GradientBoostingClassifier(random_state=SEED),
        scale=False,
    )
    models["HistGB"] = _wrap(
        "HistGB",
        HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"),
        scale=False,
    )

    # Boostings externos (XGBoost, LightGBM, CatBoost).
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = _wrap(
            "XGBoost",
            XGBClassifier(
                n_estimators=600, max_depth=5, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", n_jobs=-1, random_state=SEED,
                tree_method="hist",
            ),
            scale=False,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = _wrap(
            "LightGBM",
            LGBMClassifier(
                n_estimators=800, max_depth=-1, num_leaves=63, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
                n_jobs=-1, random_state=SEED, verbose=-1,
            ),
            scale=False,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = _wrap(
            "CatBoost",
            CatBoostClassifier(
                iterations=800, depth=6, learning_rate=0.05,
                random_seed=SEED, verbose=0, auto_class_weights="Balanced",
            ),
            scale=False,
        )
    except ImportError:
        pass

    return models
