from typing import final
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score
from config import settings


class EnsambleValidator:
    def __init__(self) -> None:
        self.val_df = None
        self.X_val = None
        self.y_val = None

        self.models_to_load = ["xgboost", "lightgbm", "catboost"]

    def load_validation_data(self):
        self.val_df = pd.read_parquet(settings.paths.data_processed / "val_df.parquet")
        self.X_val = self.val_df.drop(columns=settings.drop_cols)
        self.y_val = self.val_df[settings.target]

    def get_predictions(self):
        mlflow.set_tracking_uri(settings.mlflow.uri)
        preds_dict = {}

        for model in self.models_to_load:
            model_uri = f"models:/{settings.mlflow.catalog}.{settings.mlflow.schema}.{model}_prod/latest"
            print(f"    Loading {model} from: {model_uri}")

            try:
                model = mlflow.sklearn.load_model(model_uri)

                prob = model.predict_proba(self.X_val)[:, 1]
                preds_dict[model] = prob

                auc = roc_auc_score(self.X_val)[:, 1]
                print(f"    {model} AUC: {auc:.5f}")

            except Exception as e:
                print(f"    Error loading {model}: {e}")
                print("     (Model should be trained beforehand)")

        return preds_dict

    def run(self):
        self.load_validation_data()
        preds = self.get_predictions()

        if not preds:
            print("Could load any model data, aborting...       ")
            return

        print("\n Calculating ensemble average...       ")

        final_pred = 0
        for p in preds.values():
            final_pred += p

        final_pred /= len(preds)

        ensemble_auc = roc_auc_score(self.y_val, final_pred)
        print(f"ENSEMBLE AUC: {ensemble_auc:.5f}")

        best_single = max([roc_auc_score(self.y_val, p) for p in preds.values])
        lift = ensemble_auc - best_single
        print(f"Improve over best single model: +{lift:.5f}")


def run_ensemble_validation():
    EnsambleValidator().run()
