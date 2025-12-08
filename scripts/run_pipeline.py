import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
print(f"DEBUG: Added to path-> {ROOT_DIR}")

try:
    from src.preprocess import run_ETL

    print("imported ETL")
    from src.tuning import run_tuning

    print("imported tuning")
    from src.training import train_model

    print("imported training")
    from src.ensemble import run_ensemble_validation

    print("imported ensemble")
    from src.inference import generate_submission

    print("imported inference")

except ImportError as e:
    print(f" Import Error: {e}")
    print("Make sure to execute this in project root and 'src' is correctly set up.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fraud detection (IEEE-CIS) pipeline",
        epilog="Ejemplo: python scripts/run_pipeline.py train --model xgboost",
    )

    # Subcomands (preprocess, tune, train, ensemble, all)
    subparsers = parser.add_subparsers(
        dest="stage", required=True, help="Step to execute"
    )

    # Execute all the Pipeline
    parser_all = subparsers.add_parser(
        "all", help="Executes all the pipeline (ETL -> Train -> Ensemble)"
    )

    # ---------------------------------------------------------
    # STEP 1: PREPROCESSING
    # ---------------------------------------------------------
    parser_prep = subparsers.add_parser("preprocess", help="Executes ETL with Spark")

    # ---------------------------------------------------------
    # STEP 2: TUNING (Optuna)
    # ---------------------------------------------------------
    parser_tune = subparsers.add_parser(
        "tune", help="Looks for best hyperparameters with Optuna"
    )
    parser_tune.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["xgboost", "lightgbm", "catboost", "all"],
        help="Model you want to optimize (default: all)",
    )

    # ---------------------------------------------------------
    # STEP 3: TRAINING
    # ---------------------------------------------------------
    parser_train = subparsers.add_parser(
        "train", help="Trains and registers the final models"
    )
    parser_train.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["xgboost", "lightgbm", "catboost", "all"],
        help="Model to train (Use 'all' to train the 3 of them in the same run)",
    )

    # ---------------------------------------------------------
    # FASE 4: VALIDATION (Ensemble)
    # ---------------------------------------------------------
    parser_ens = subparsers.add_parser("ensemble", help="Validate ensemble")

    # ---------------------------------------------------------
    # STEP 5: INFERENCE
    # ---------------------------------------------------------
    parser_inf = subparsers.add_parser("inference", help="predicts test dataset")

    args = parser.parse_args()

    print(f"\n **---** Starting stage: {args.stage.upper()} \n" + "=" * 40)

    if args.stage == "preprocess":
        run_ETL()

    elif args.stage == "tune":
        if args.model == "all":
            print("Stating tuning for all models (XGB -> LGBM -> CAT ->)... ")
            for m in ["xgboost", "lightgbm", "catboost"]:
                print(f"\n--- Tuning {m} ---")
                run_tuning(m)
        else:
            run_tuning(args.model)

    elif args.stage == "train":
        train_model(args.model)

    elif args.stage == "ensemble":
        run_ensemble_validation()

    elif args.stage == "inference":
        generate_submission()

    elif args.stage == "all":
        print("Executing FULL Pipeline...       ")

        # 1. Preprocess
        run_ETL()

        for m in ["xgboost", "lightgbm", "catboost"]:
            print(f"\n--- Tuning {m} ---")
            run_tuning(m)

        # 2. Train
        train_model("all")

        # 3. Ensemble
        run_ensemble_validation()

    print("\n" + "=" * 40 + "\nProcess ended succesfully")


if __name__ == "__main__":
    main()
