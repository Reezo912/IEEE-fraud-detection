from config import settings

import pandas as pd
import mlflow
from mlflow.models import infer_signature

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.metrics import roc_auc_score


"""
This script contains the logic for the model building and the training. 
It also includes the X / y split
"""


class ModelBuilder:
    """ "
    Builder for optimized models
    Automatically looks for best params in mlflow
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.experiment = mlflow.set_experiment(settings.mlflow.experiment_name)

    def _cast_params(self, raw_params: dict) -> dict:
        """Clean the param output of MLflow"""
        clean = {}
        # Lista negra de params internos que no queremos pasar al constructor
        ignored_keys = [
            "device",
            "task_type",
            "verbose",
            "verbosity",
            "n_jobs",
            "early_stopping_rounds",
        ]

        for k, v in raw_params.items():
            if k in ignored_keys or v is None:
                continue

            try:
                if isinstance(v, str):
                    if v.lower() == "true":
                        clean[k] = True
                    elif v.lower() == "false":
                        clean[k] = False
                    elif "." in v:
                        clean[k] = float(v)
                    else:
                        clean[k] = int(v)
                else:
                    clean[k] = v
            except ValueError:
                clean[k] = v

        local_cfg = getattr(settings, self.model_name)
        clean["device"] = local_cfg.device if hasattr(local_cfg, "device") else None
        if hasattr(local_cfg, "task_type"):
            clean["task_type"] = local_cfg.task_type  # Catboost

        return clean

    def get_best_params(self) -> dict:
        print("Looking for best params for {self.model_name")

        query = f"tags.model_type = '{self.model_name}' and tags.type = 'tuning_trial'"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=query,
            order_by=["metrics.val_auc DESC"],
            max_results=1,
            output_format="pandas",
        )

        if runs.empty:
            print("No runs found... ")
            return getattr(settings, self.model_name).model_dump()

        # cleaning params. that mlflow returns as default
        best_run = runs.iloc[0]
        print(
            f"Params found in Run ID: {best_run.run_id} (AUC: {best_run['metrics.val_auc']:.4f})"
        )

        params_dict = {
            k.replace("params.", ""): v
            for k, v in best_run.to_dict().items()
            if k.startswith("params.")
        }

        return self._cast_params(params_dict)

    def build(self):
        """ "instantiates model_name with best params"""
        final_params = self.get_best_params()

        if self.model_name == "xgboost":
            return xgb.XGBClassifier(
                **final_params, n_jobs=-1, random_state=42
            ), final_params

        elif self.model_name == "lightgbm":
            return lgb.LGBMClassifier(
                **final_params, n_jobs=-1, verbose=-1
            ), final_params

        elif self.model_name == "catboost":
            return cb.CatBoostClassifier(**final_params, verbose=False), final_params

        else:
            raise ValueError("There is no model_name with that name... ")


class Trainer:
    def __init__(self) -> None:
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

    def train_single(self, model_name: str):
        self.load_dataset()
        print(f"Starting training for {model_name}... ")

        builder = ModelBuilder(model_name)
        model, params = builder.build()

        mlflow.set_tracking_uri(settings.mlflow.uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)
        full_name = f"{settings.mlflow.catalog}.{settings.mlflow.d_brick_schema}.{model_name}_prod"

        with mlflow.start_run(run_name=f"PROD_{model_name}"):
            mlflow.log_params(params)

            if model_name == "xgboost":
                model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    early_stopping_rounds=100,
                )
            elif model_name == "lightgbm":
                model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                    early_stopping_rounds=100,
                )
            elif model_name == "catboost":
                model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                )

            y_proba = model.predict_proba(self.X_val)[:, 1]
            auc = roc_auc_score(self.y_val, y_proba)
            mlflow.log_metric("auc_roc", auc)
            print(f"Best AUC: {auc:.2f}")

            signature = infer_signature(self.X_val.head(), y_proba[:5])
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=self.X_train.head(),
                registered_model_name=full_name,
            )
            print("Model registered in: ", full_name)

    def run(self, models_to_train: list[str] = None):
        if models_to_train is None:
            models_to_train = ["xgboost", "lightgbm", "catboost"]

        for m in models_to_train:
            self.train_single(m)


def train_model(model_name: str):
    trainer = Trainer()
    if model_name == "all":
        trainer.run()
    else:
        trainer.run([model_name])

