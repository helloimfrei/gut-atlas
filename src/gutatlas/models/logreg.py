import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from skopt.space import Real, Categorical
from skopt.searchcv import BayesSearchCV

from sklearn.model_selection import KFold, StratifiedKFold
import joblib


# default search space for tuning
default_search_space = {
    "C": Real(1e-3, 10.0, prior="log-uniform"),
    "l1_ratio": Real(0.0, 1.0),
    "penalty": Categorical(["elasticnet"]),
    "solver": Categorical(["saga"]),
}


## base tuner class


class BOTuner:
    """
    Base tuner class for Logistic Regression models.

    Parameters
    ----------
    estimator : sklearn.linear_model.LogisticRegression
        The Logistic Regression model to tune
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

    def save_model(self, dir_path, model_name="best_model.pkl"):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        model_path = dir_path / model_name
        joblib.dump(self.opt.best_estimator_, model_path)

    def save_params(self, dir_path, model_name="best_model_params.json"):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        params_path = dir_path / model_name
        with open(params_path, "w") as f:
            json.dump(self.opt.best_params_, f, indent=2)


## LogReg Binary Classifier


class LogRegBinClassTuner(BOTuner):
    """
    Bayesian optimization tuner for Logistic Regression classification models.

    This class wraps a ``LogisticRegression`` with ``BayesSearchCV`` to perform
    hyperparameter optimization. It uses ``StratifiedKFold`` cross-validation
    and classification-friendly defaults. Configured for ElasticNet regularization.

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
    max_iter : int, default=1000
        Maximum number of iterations for the solver to converge.
    verbose : int, default=1
        Verbosity level passed to BayesSearchCV.
    **logreg_kwargs :
        Additional keyword arguments passed directly to ``LogisticRegression``.
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
            search_space=search_space,
            n_iter=n_iter,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=verbose,
        )
