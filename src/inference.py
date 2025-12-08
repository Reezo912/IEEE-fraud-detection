import pandas as pd
import numpy as np
import mlflow
from src.config import settings


def generate_submission():
    print("\n Starting inference process...")

    # 1. Loading data
    test_path = settings.paths.data_processed / "test_df.parquet"

    if not test_path.exists():
        raise FileNotFoundError(
            f"{test_path} doesnt exist. Execute 'preprocess' first."
        )

    print(f"   -> Loading test data: {test_path}")
    test_df = pd.read_parquet(test_path)

    if settings.id_col in test_df.columns:
        ids = test_df[settings.id_col]
        X_test = test_df.drop(columns=[settings.id_col])
    else:
        print(" Couldnt find column ID, using index")
        ids = test_df.index
        X_test = test_df

    # 2. Loading ensemble models
    models_to_use = ["xgboost", "lightgbm", "catboost"]
    predictions = []

    mlflow.set_tracking_uri(settings.mlflow.uri)

    for model_name in models_to_use:
        full_name = f"{settings.mlflow.catalog}.{settings.mlflow.d_brick_schema}.{model_name}_prod"
        model_uri = f"models:/{full_name}@Champion"
        print(f"   -> loading model: {model_name} ({model_uri})...")

        try:
            model = mlflow.sklearn.load_model(model_uri)

            prob = model.predict_proba(X_test)[:, 1]
            predictions.append(prob)
            print("       Prediction generated")

        except Exception as e:
            print(f"       Error Loading {model_name}: {e}")
            print("         Continuing without this model...")

    if not predictions:
        raise RuntimeError("Couldnt load any model. Training must be run first")

    # 3. Ensemble
    print("   Calculating ensemble average...")
    final_pred = np.mean(predictions, axis=0)

    # 4. Save inference
    settings.paths.submissions.mkdir(parents=True, exist_ok=True)
    output_file = settings.paths.submissions / "submission.csv"

    submission = pd.DataFrame({settings.id_col: ids, settings.target: final_pred})

    submission.to_csv(output_file, index=False)
    print(f"\n Finished succesfully, output in: {output_file}")
    print(f"   (it has {len(submission)} rows)")


if __name__ == "__main__":
    generate_submission()
