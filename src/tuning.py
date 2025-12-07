import optuna
import mlflow

import xgboost as xgb
import catboost as cb
import lightgbm as lgb

from sklearn.metrics import roc_auc_score

import pandas as pd

from config import RangeFloat, RangeInt, settings


class Tuning:
    def __init__(self, model_type) -> None:
        self.model_type = model_type
        self.train_df = None
        self.val_df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def load_dataset(self):
        if self.train_df is not None:
            return print("There is data already loaded... ")

        print("Loading the datasets... ")
        self.train_df = pd.read_parquet(
            settings.paths.data_processed / "train_df.parquet"
        )
        self.val_df = pd.read_parquet(settings.paths.data_processed / "val_df.parquet")

        self.X_train = self.train_df.drop(columns=settings.drop_cols)
        self.y_train = self.train_df[settings.target]
        self.X_val = self.val_df.drop(columns=settings.drop_cols)
        self.y_val = self.val_df[settings.target]
        print(f"   -> Train: {self.X_train.shape} | Val: {self.X_val.shape}")

    def get_objective(self):
        self.load_dataset()

        def objective(trial):
            config_space = getattr(settings.optuna, f"{self.model_type}_space")

            params = {}

            for name, field_value in config_space:
                if isinstance(field_value, RangeFloat):
                    params[name] = trial.suggest_float(
                        name, field_value.low, field_value.high, log=field_value.log
                    )

                elif isinstance(field_value, RangeInt):
                    params[name] = trial.suggest_int(
                        name, field_value.low, field_value.high
                    )

            fixed_params = {
                "n_estimators": 5000,
                "early_stopping_rounds": 100,
                "verbosity": 0,
                "n_jobs": -1,
            }

            if self.model_type == "xgboost":
                fixed_params.update(
                    {"tree_method": "hist", "device": "cuda", "eval_metric": "auc"}  # type: ignore
                )
                model = xgb.XGBClassifier(**params, **fixed_params)
                fit_params = {"eval_set": [(self.X_val, self.y_val)], "verbose": False}

            elif self.model_type == "lightgbm":
                fixed_params.update({"device": "gpu", "metric": "auc"})  # type: ignore
                model = lgb.LGBMClassifier(**params, **fixed_params)  # type: ignore
                fit_params = {
                    "eval_set": [(self.X_val, self.y_val)],
                    "callbacks": [lgb.early_stopping(100, verbose=False)],
                }

            elif self.model_type == "catboost":
                fixed_params.update({"task_type": "GPU", "eval_metric": "AUC"})  # type: ignore
                model = cb.CatBoostClassifier(**params, **fixed_params)
                fit_params = {
                    "eval_set": (self.X_val, self.y_val),
                    "early_stopping_rounds": 100,
                }

            with mlflow.start_run(nested=True):
                mlflow.set_tag("model_type", self.model_type)
                mlflow.set_tag("type", "tuning_trial")

                mlflow.log_params(params)

                model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    verbose=False,
                )

                preds = model.predict_proba(self.X_val)[:, 1]
                auc = roc_auc_score(self.y_val, preds)

                mlflow.log_metric("val_auc", auc)

                return auc

        return objective


def run_tuning(model_name: str):
    tuner = Tuning(model_name)

    mlflow.set_tracking_uri(settings.mlflow.uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    print(f"Starting Optuna for {model_name}...")

    study = optuna.create_study(direction="maximize")
    study.optimize(tuner.get_objective(), n_trials=settings.optuna.n_trials)

    print(f"Best AUC ({model_name}): {study.best_value:.5f}")
    print(f"Best Params: {study.best_params}")
