import argparse

from preprocess import run_ETL
from training import train_model
from tuning import run_tuning
from ensemble import run_ensemble_validation
from inference import generate_submission


def main():
    parser = argparse.ArgumentParser(description="Pipeline Modular IEEE Fraud")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    # 1. ETL
    subparsers.add_parser("preprocess", help="Ejecuta ETL Spark")

    # 2. TUNING
    tune_parser = subparsers.add_parser("tune", help="Ejecuta Optuna")
    tune_parser.add_argument(
        "--model", type=str, required=True, choices=["xgboost", "lightgbm", "catboost"]
    )

    # 3. TRAIN
    train_parser = subparsers.add_parser("train", help="Entrena modelo final")
    train_parser.add_argument("--model", type=str, default="all")

    # 4. ENSEMBLE
    subparsers.add_parser("ensemble", help="Valida ensemble")

    # 5. PREDICT
    subparsers.add_parser("predict", help="Genera submission")

    args = parser.parse_args()

    if args.stage == "preprocess":
        run_ETL()

    elif args.stage == "tune":
        run_tuning(args.model)

    elif args.stage == "train":
        train_model(args.model)

    elif args.stage == "ensemble":
        run_ensemble_validation()

    elif args.stage == "predict":
        generate_submission()


if __name__ == "__main__":
    main()
