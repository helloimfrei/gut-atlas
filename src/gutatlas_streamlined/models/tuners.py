"""
Bayesian optimization tuners for XGBoost, LightGBM, and Logistic Regression.

Provides unified interface for hyperparameter tuning with BayesSearchCV.
"""

import json
from pathlib import Path

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression

from skopt.space import Real, Integer, Categorical
from skopt.searchcv import BayesSearchCV

from sklearn.model_selection import KFold, StratifiedKFold


# Default search spaces for each model type
XGBOOST_SEARCH_SPACE = {
    "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
    "reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
    "reg_alpha": Real(1e-3, 10.0, prior="log-uniform"),
    "n_estimators": Integer(50, 800),
    "max_depth": Integer(3, 8),
}

LIGHTGBM_SEARCH_SPACE = {
    "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
    "reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
    "reg_alpha": Real(1e-3, 10.0, prior="log-uniform"),
    "n_estimators": Integer(50, 800),
    "max_depth": Integer(3, 8),
    "num_leaves": Integer(20, 150),
}

LOGREG_SEARCH_SPACE = {
    "C": Real(1e-3, 10.0, prior="log-uniform"),
    "l1_ratio": Real(0.0, 1.0),
    "penalty": Categorical(["elasticnet"]),
    "solver": Categorical(["saga"]),
}


class BaseTuner:
    """
    Base tuner class for all models.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        The model to tune
    cv : sklearn cross-validation splitter
        The cross-validation strategy
    search_space : dict, optional
        The hyperparameter search space. If None, use default.
    n_iter : int, default=10
        Number of iterations for Bayesian optimization
    n_jobs : int, default=-1
        Number of parallel jobs
    scoring : str, optional
        Scoring metric for optimization
    verbose : int, default=1
        Verbosity level
    """

    def __init__(
        self,
        estimator,
        cv,
        search_space=None,
        n_iter=10,
        n_jobs=-1,
        scoring=None,
        verbose=1,
    ):
        self.estimator = estimator
        self.cv = cv
        self.bo_search_space = search_space
        self.scoring = scoring

        self.opt = BayesSearchCV(
            self.estimator,
            self.bo_search_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=self.cv,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    def fit(self, X_train, y_train):
        """Fit the tuner."""
        self.opt.fit(X_train, y_train)
        return self

    def best_params(self):
        """Get best parameters."""
        return self.opt.best_params_

    def best_score(self):
        """Get best score."""
        return (self.opt.scoring, self.opt.best_score_)

    def best_estimator(self):
        """Get best estimator."""
        return self.opt.best_estimator_


class XGBBinClassTuner(BaseTuner):
    """
    Bayesian optimization tuner for XGBoost binary classification.

    Parameters
    ----------
    cv_splits : int, default=5
        Number of cross-validation folds
    search_space : dict, optional
        Hyperparameter search space. If None, uses XGBOOST_SEARCH_SPACE.
    n_iter : int, default=10
        Number of Bayesian optimization iterations
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, default=42
        Random seed
    scoring : str, default="roc_auc"
        Scoring metric
    eval_metric : str, default="logloss"
        XGBoost internal evaluation metric
    tree_method : str, default="auto"
        Tree construction algorithm
    verbose : int, default=1
        Verbosity level
    **xgb_kwargs :
        Additional arguments for XGBClassifier
    """

    def __init__(
        self,
        cv_splits=5,
        search_space=None,
        n_iter=10,
        n_jobs=-1,
        random_state=42,
        scoring="roc_auc",
        eval_metric="logloss",
        tree_method="auto",
        verbose=1,
        **xgb_kwargs,
    ):
        estimator = XGBClassifier(
            eval_metric=eval_metric,
            tree_method=tree_method,
            random_state=random_state,
            **xgb_kwargs,
        )
        cv = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=random_state
        )
        super().__init__(
            estimator=estimator,
            cv=cv,
            search_space=search_space or XGBOOST_SEARCH_SPACE,
            n_iter=n_iter,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=verbose,
        )


class LGBMBinClassTuner(BaseTuner):
    """
    Bayesian optimization tuner for LightGBM binary classification.

    Parameters
    ----------
    cv_splits : int, default=5
        Number of cross-validation folds
    search_space : dict, optional
        Hyperparameter search space. If None, uses LIGHTGBM_SEARCH_SPACE.
    n_iter : int, default=10
        Number of Bayesian optimization iterations
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, default=42
        Random seed
    scoring : str, default="roc_auc"
        Scoring metric
    verbose : int, default=1
        Verbosity level
    **lgbm_kwargs :
        Additional arguments for LGBMClassifier
    """

    def __init__(
        self,
        cv_splits=5,
        search_space=None,
        n_iter=10,
        n_jobs=-1,
        random_state=42,
        scoring="roc_auc",
        verbose=1,
        **lgbm_kwargs,
    ):
        estimator = LGBMClassifier(
            random_state=random_state,
            **lgbm_kwargs,
        )
        cv = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=random_state
        )
        super().__init__(
            estimator=estimator,
            cv=cv,
            search_space=search_space or LIGHTGBM_SEARCH_SPACE,
            n_iter=n_iter,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=verbose,
        )


class LogRegBinClassTuner(BaseTuner):
    """
    Bayesian optimization tuner for Logistic Regression binary classification.

    Parameters
    ----------
    cv_splits : int, default=5
        Number of cross-validation folds
    search_space : dict, optional
        Hyperparameter search space. If None, uses LOGREG_SEARCH_SPACE.
    n_iter : int, default=10
        Number of Bayesian optimization iterations
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, default=42
        Random seed
    scoring : str, default="roc_auc"
        Scoring metric
    max_iter : int, default=1000
        Maximum iterations for convergence
    verbose : int, default=1
        Verbosity level
    **logreg_kwargs :
        Additional arguments for LogisticRegression
    """

    def __init__(
        self,
        cv_splits=5,
        search_space=None,
        n_iter=10,
        n_jobs=-1,
        random_state=42,
        scoring="roc_auc",
        max_iter=1000,
        verbose=1,
        **logreg_kwargs,
    ):
        estimator = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            **logreg_kwargs,
        )
        cv = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=random_state
        )
        super().__init__(
            estimator=estimator,
            cv=cv,
            search_space=search_space or LOGREG_SEARCH_SPACE,
            n_iter=n_iter,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=verbose,
        )
