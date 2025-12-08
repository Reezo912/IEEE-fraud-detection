import optuna
import mlflow

import xgboost as xgb
import catboost as cb
import lightgbm as lgb

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import RangeFloat, RangeInt, settings


class Tuning:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        self.train_df = None
        self.val_df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def load_dataset(self):
        if self.train_df is not None:
            print("There is data already loaded... ")
            return

        print("Loading the datasets... ")
        self.train_df = pd.read_parquet(
            settings.paths.data_processed / "train_df.parquet"
        )
        self.val_df = pd.read_parquet(settings.paths.data_processed / "val_df.parquet")

        # Safety Patch: Object -> Category for XGBoost/LGBM
        for col in self.train_df.columns:
            if self.train_df[col].dtype == "object":
                self.train_df[col] = self.train_df[col].astype("category")
                self.val_df[col] = self.val_df[col].astype("category")

        self.X_train = self.train_df.drop(columns=settings.drop_cols)
        self.y_train = self.train_df[settings.target]
        self.X_val = self.val_df.drop(columns=settings.drop_cols)
        self.y_val = self.val_df[settings.target]
        print(f"   -> Train: {self.X_train.shape} | Val: {self.X_val.shape}")

    def get_objective(self):
        self.load_dataset()

        def objective(trial):
            print(f"     -> Trial {trial.number} for {self.model_type}...")

            base_config = getattr(settings, self.model_type).model_dump()

            config_space = getattr(settings.optuna, f"{self.model_type}_space")

            trial_params = {}
            for name, field_value in config_space:
                if isinstance(field_value, RangeFloat):
                    trial_params[name] = trial.suggest_float(
                        name, field_value.low, field_value.high, log=field_value.log
                    )
                elif isinstance(field_value, RangeInt):
                    trial_params[name] = trial.suggest_int(
                        name, field_value.low, field_value.high
                    )

            final_params = {**base_config, **trial_params}

            fit_params = {}

            if self.model_type == "xgboost":
                final_params["verbosity"] = 0
                model = xgb.XGBClassifier(**final_params)
                fit_params = {"eval_set": [(self.X_val, self.y_val)], "verbose": False}

            elif self.model_type == "lightgbm":
                final_params["verbosity"] = -1
                model = lgb.LGBMClassifier(**final_params)
                fit_params = {
                    "eval_set": [(self.X_val, self.y_val)],
                    "callbacks": [lgb.early_stopping(100, verbose=False)],
                }

            elif self.model_type == "catboost":
                final_params["verbose"] = False
                model = cb.CatBoostClassifier(**final_params)
                fit_params = {
                    "eval_set": (self.X_val, self.y_val),
                    "early_stopping_rounds": 100,
                }

            with mlflow.start_run(nested=True):
                mlflow.set_tag("model_type", self.model_type)
                mlflow.set_tag("type", "tuning_trial")

                mlflow.log_params(trial_params)

                model.fit(self.X_train, self.y_train, **fit_params)

                preds = model.predict_proba(self.X_val)[:, 1]
                auc = roc_auc_score(self.y_val, preds)

                mlflow.log_metric("val_auc", auc)
                return auc

        return objective


def run_tuning(model_name: str):
    tuner = Tuning(model_name)

    mlflow.set_tracking_uri(settings.mlflow.uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    print(f"\nStarting Optuna for {model_name.upper()}...")

    with mlflow.start_run(run_name=f"Tuning_Master_{model_name}") as parent_run:
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("type", "tuning_master")

        study = optuna.create_study(direction="maximize")
        study.optimize(tuner.get_objective(), n_trials=settings.optuna.n_trials)

        print(f"\n{'=' * 60}")
        print(f"Best AUC ({model_name}): {study.best_value:.5f}")
        print(f"Best Params: {study.best_params}")
        print(f"{'=' * 60}")

        # Loguear ganadores
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_auc", study.best_value)
