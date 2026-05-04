"""
Catálogo de modelos. Pipelines (preproc -> clasificador), con variantes SMOTE
para los top modelos cuando aporta mejora del recall en clase positiva.
"""
from __future__ import annotations

from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .preprocessing import make_preprocessor
from .config import SEED


# Class imbalance: 79.6% / 20.4% → scale_pos_weight ≈ 3.91.
SPW = 3.91


def _wrap(name: str, clf, scale: bool = True, **prep_kw) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale=scale, **prep_kw)),
            ("classifier", clf),
        ]
    )


def _wrap_smote(name: str, clf, scale: bool = True, **prep_kw) -> ImbPipeline:
    """Pipeline con SMOTE entre preproc y clasificador. SMOTE solo se aplica
    durante fit (CV-safe: solo ve el fold de train)."""
    return ImbPipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale=scale, **prep_kw)),
            ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
            ("classifier", clf),
        ]
    )


def get_models() -> Dict[str, Pipeline]:
    """Devuelve dict nombre -> pipeline listo para CV."""
    models: Dict[str, Pipeline] = {}

    # ── Lineales / clásicos ──────────────────────────────────────────────
    models["LDA"] = _wrap("LDA", LinearDiscriminantAnalysis(), scale=True)
    models["QDA"] = _wrap("QDA", QuadraticDiscriminantAnalysis(reg_param=0.1), scale=True)
    models["LogReg"] = _wrap(
        "LogReg",
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           solver="lbfgs", random_state=SEED),
        scale=True,
    )
    models["LogRegL1"] = _wrap(
        "LogRegL1",
        LogisticRegression(penalty="l1", C=1.0, max_iter=2000,
                           class_weight="balanced", solver="liblinear",
                           random_state=SEED),
        scale=True,
    )
    models["Ridge"] = _wrap("Ridge",
        RidgeClassifier(class_weight="balanced", random_state=SEED), scale=True)
    models["SGD"] = _wrap("SGD",
        SGDClassifier(loss="log_loss", class_weight="balanced",
                      max_iter=1000, random_state=SEED, n_jobs=-1), scale=True)
    models["GaussianNB"] = _wrap("GaussianNB", GaussianNB(), scale=True)
    models["BernoulliNB"] = _wrap("BernoulliNB", BernoulliNB(), scale=True)
    models["LinearSVC"] = _wrap("LinearSVC",
        CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000),
            cv=3), scale=True)
    models["SVC_RBF"] = _wrap("SVC_RBF",
        SVC(C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=SEED), scale=True)
    models["KNN"] = _wrap("KNN",
        KNeighborsClassifier(n_neighbors=15, weights="distance"), scale=True)
    models["MLP"] = _wrap("MLP",
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                      random_state=SEED, early_stopping=True), scale=True)

    # ── Árboles / ensambles ──────────────────────────────────────────────
    models["DecisionTree"] = _wrap("DecisionTree",
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED, max_depth=10),
        scale=False)
    models["RandomForest"] = _wrap("RandomForest",
        RandomForestClassifier(n_estimators=500,
                               class_weight="balanced_subsample",
                               random_state=SEED, n_jobs=-1),
        scale=False)
    models["ExtraTrees"] = _wrap("ExtraTrees",
        ExtraTreesClassifier(n_estimators=500,
                             class_weight="balanced_subsample",
                             random_state=SEED, n_jobs=-1),
        scale=False)
    models["GradBoost"] = _wrap("GradBoost",
        GradientBoostingClassifier(random_state=SEED), scale=False)
    models["HistGB"] = _wrap("HistGB",
        HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"),
        scale=False)
    models["AdaBoost"] = _wrap("AdaBoost",
        AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=SEED),
        scale=False)
    models["Bagging"] = _wrap("Bagging",
        BaggingClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        scale=False)

    # ── Boostings externos ───────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = _wrap("XGBoost",
            XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9,
                          scale_pos_weight=SPW, eval_metric="logloss",
                          n_jobs=-1, random_state=SEED, tree_method="hist"),
            scale=False)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = _wrap("LightGBM",
            LGBMClassifier(n_estimators=800, num_leaves=63, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9,
                           class_weight="balanced",
                           n_jobs=-1, random_state=SEED, verbose=-1),
            scale=False)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = _wrap("CatBoost",
            CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                               random_seed=SEED, verbose=0,
                               auto_class_weights="Balanced"),
            scale=False)
    except ImportError:
        pass

    # ── Variantes SMOTE de los top boostings ─────────────────────────────
    try:
        from xgboost import XGBClassifier
        models["XGBoost_SMOTE"] = _wrap_smote("XGBoost_SMOTE",
            XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9,
                          eval_metric="logloss", n_jobs=-1, random_state=SEED,
                          tree_method="hist"),
            scale=False)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM_SMOTE"] = _wrap_smote("LightGBM_SMOTE",
            LGBMClassifier(n_estimators=800, num_leaves=63, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9,
                           n_jobs=-1, random_state=SEED, verbose=-1),
            scale=False)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        models["CatBoost_SMOTE"] = _wrap_smote("CatBoost_SMOTE",
            CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05,
                               random_seed=SEED, verbose=0),
            scale=False)
    except ImportError:
        pass
    models["LogReg_SMOTE"] = _wrap_smote("LogReg_SMOTE",
        LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
        scale=True)

    return models
