import json
from pathlib import Path

from lightgbm import LGBMClassifier, LGBMRegressor

from skopt.space import Real, Integer
from skopt.searchcv import BayesSearchCV

from sklearn.model_selection import KFold, StratifiedKFold


# default search space for tuning
default_search_space = {
    "learning_rate": Real(1e-3, 0.3, prior="log-uniform"),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
    "reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
    "reg_alpha": Real(1e-3, 10.0, prior="log-uniform"),
    "n_estimators": Integer(50, 800),
    "max_depth": Integer(3, 8),
    "num_leaves": Integer(20, 150),
}


## base tuner class


class BOTuner:
    """
    Base tuner class for LightGBM models.

    Parameters
    ----------
    estimator : lgbm.LGBMClassifier or lgbm.LGBMRegressor
        The LightGBM model to tune
    cv : int or sklearn.model_selection._split.KFold
        The cross-validation strategy
    search_space : dict, optional
        The hyperparameter search space. If None, use default_search_space.
    n_iter : int, optional
        The number of iterations for the hyperparameter search
    n_jobs : int, optional
        The number of jobs to run in parallel
    scoring : str, optional
        The scoring metric to use for the hyperparameter search
    verbose : int, optional
        The verbosity level of the hyperparameter search
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
        self.bo_search_space = search_space or default_search_space
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
        self.opt.fit(X_train, y_train)
        return self

    def best_params(self):
        return self.opt.best_params_

    def best_score(self):
        return (self.opt.scoring, self.opt.best_score_)

    def best_estimator(self):
        return self.opt.best_estimator_

    def save_model(self, dir_path, model_name="best_model.txt"):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        model_path = dir_path / model_name
        self.opt.best_estimator_.booster_.save_model(str(model_path))

    def save_params(self, dir_path, model_name="best_model_params.json"):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        params_path = dir_path / model_name
        with open(params_path, "w") as f:
            json.dump(self.opt.best_params_, f, indent=2)


## LGBM Regressor


class LGBMRegTuner(BOTuner):
    """
    Bayesian optimization tuner for LightGBM regression models.

    This class wraps an ``LGBMRegressor`` with ``BayesSearchCV`` to perform
    hyperparameter optimization. It uses ``KFold`` cross-validation and is
    configured with regression-friendly defaults.

    Parameters
    ----------
    cv_splits : int, default=5
        Number of cross-validation folds (KFold).
    search_space : dict, optional
        Hyperparameter search space. If None, ``default_search_space`` is used.
    n_iter : int, default=10
        Number of parameter settings sampled by BayesSearchCV.
    n_jobs : int, default=-1
        Number of parallel jobs for training and CV. ``-1`` uses all CPUs.
    random_state : int, default=42
        Random seed for reproducibility.
    scoring : str, default="neg_mean_squared_error"
        Scoring function for model selection during optimization.
    verbose : int, default=1
        Verbosity level passed to BayesSearchCV.
    **lgbm_kwargs :
        Additional keyword arguments passed directly to ``LGBMRegressor``.
    """

    def __init__(
        self,
        cv_splits=5,
        search_space=None,
        n_iter=10,
        n_jobs=-1,
        random_state=42,
        scoring="neg_mean_squared_error",
        verbose=1,
        **lgbm_kwargs,
    ):
        estimator = LGBMRegressor(random_state=random_state, **lgbm_kwargs)
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        super().__init__(
            estimator=estimator,
            cv=cv,
            search_space=search_space,
            n_iter=n_iter,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=verbose,
        )


## LGBM Binary Classifier


class LGBMBinClassTuner(BOTuner):
    """
    Bayesian optimization tuner for LightGBM classification models.

    This class wraps an ``LGBMClassifier`` with ``BayesSearchCV`` to perform
    hyperparameter optimization. It uses ``StratifiedKFold`` cross-validation
    and classification-friendly defaults.

    Parameters
    ----------
    cv_splits : int, default=5
        Number of cross-validation folds (StratifiedKFold).
    search_space : dict, optional
        Hyperparameter search space. If None, ``default_search_space`` is used.
    n_iter : int, default=10
        Number of parameter settings sampled by BayesSearchCV.
    n_jobs : int, default=-1
        Number of parallel jobs for training and CV. ``-1`` uses all CPUs.
    random_state : int, default=42
        Random seed for reproducibility.
    scoring : str, default="roc_auc"
        Scoring function for model selection during optimization.
    verbose : int, default=1
        Verbosity level passed to BayesSearchCV.
    **lgbm_kwargs :
        Additional keyword arguments passed directly to ``LGBMClassifier``.
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
            search_space=search_space,
            n_iter=n_iter,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=verbose,
        )
