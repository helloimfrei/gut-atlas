import xgboost as xgb
from skopt.searchcv import BayesSearchCV
from sklearn.model_selection import KFold
from skopt.space import Real, Integer
from pathlib import Path
import json



default_search_space = {
    'learning_rate': Real(1e-3, 0.3, prior="log-uniform"),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'reg_lambda': Real(1e-3, 10.0, prior="log-uniform"),
    'reg_alpha': Real(1e-3, 10.0, prior="log-uniform"),
    'n_estimators': Integer(50, 800),
    'max_depth': Integer(3, 8)
}
class XGBRegressorBOTuner:
    def __init__(self, cv_splits = 5, search_space=None, n_iter=10, n_jobs=-1, random_state=42):
        self.regressor = xgb.XGBRegressor()
        self.cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        if search_space is None:
            bo_search_space = default_search_space
        else:
            bo_search_space = search_space
        self.opt = BayesSearchCV(
            self.regressor,
            bo_search_space,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",
            cv=self.cv,
            verbose=1,
            n_jobs=n_jobs
        )

    def fit(self, X_train, y_train):
        self.opt.fit(X_train, y_train)
        return self

    def best_params(self):
        return self.opt.best_params_

    def best_score(self):
        return self.opt.best_score_

    def best_estimator(self):
        return self.opt.best_estimator_

    def save_model(self, dir_path: str | Path, model_name: str = "best_model.json") -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        model_path = dir_path / model_name
        self.opt.best_estimator_.save_model(model_path)

    def save_params(self, dir_path: str | Path, model_name: str = "best_model_params.json") -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        params_path = dir_path / model_name
        with open(params_path, "w") as f:
            json.dump(self.opt.best_params_, f, indent=2)



