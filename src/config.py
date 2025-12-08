from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent.resolve()

load_dotenv()


class PathsConfig(BaseModel):
    data_raw: Path = BASE_DIR / "data" / "raw"
    data_processed: Path = BASE_DIR / "data" / "processed"
    submissions: Path = BASE_DIR / "data" / "submissions"


class MLflowConfig(BaseModel):
    uri: str = "databricks"
    experiment_name: str = "/Shared/Fraud_detection"
    catalog: str = "workspace"
    d_brick_schema: str = "default"


class XGBParams(BaseModel):
    n_estimators: int = 5000
    learning_rate: float = 0.01
    max_depth: int = 10
    tree_method: str = "hist"
    device: str = "cuda"  # Tienes una 4080
    eval_metric: str = "auc"
    early_stopping_rounds: int = 100


class LGBMParams(BaseModel):
    n_estimators: int = 5000
    learning_rate: float = 0.01
    num_leaves: int = 256
    device: str = "cpu"
    metric: str = "auc"


class CatParams(BaseModel):
    iterations: int = 5000
    learning_rate: float = 0.01
    depth: int = 10
    task_type: str = "GPU"
    eval_metric: str = "AUC"


class RangeFloat(BaseModel):
    low: float
    high: float
    log: bool = False


class RangeInt(BaseModel):
    low: int
    high: int


class XGBConfigSearch(BaseModel):
    learning_rate: RangeFloat = RangeFloat(low=0.005, high=0.1, log=True)
    max_depth: RangeInt = RangeInt(low=6, high=15)
    subsample: RangeFloat = RangeFloat(low=0.6, high=0.95)
    colsample_bytree: RangeFloat = RangeFloat(low=0.6, high=0.95)
    reg_alpha: RangeFloat = RangeFloat(low=0.1, high=10.0, log=True)
    reg_lambda: RangeFloat = RangeFloat(low=0.1, high=10.0, log=True)
    scale_pos_weight: RangeFloat = RangeFloat(low=1.0, high=15.0)


class LGBMConfigSearch(BaseModel):
    learning_rate: RangeFloat = RangeFloat(low=0.005, high=0.1, log=True)
    num_leaves: RangeInt = RangeInt(low=20, high=300)
    max_depth: RangeInt = RangeInt(low=-1, high=15)
    subsample: RangeFloat = RangeFloat(low=0.6, high=0.95)
    colsample_bytree: RangeFloat = RangeFloat(low=0.6, high=0.95)
    reg_alpha: RangeFloat = RangeFloat(low=0.1, high=10.0, log=True)
    reg_lambda: RangeFloat = RangeFloat(low=0.1, high=10.0, log=True)
    scale_pos_weight: RangeFloat = RangeFloat(low=1.0, high=15.0)


class CatConfigSearch(BaseModel):
    learning_rate: RangeFloat = RangeFloat(low=0.005, high=0.1, log=True)
    depth: RangeInt = RangeInt(low=4, high=10)
    l2_leaf_reg: RangeFloat = RangeFloat(low=1.0, high=10.0, log=True)
    random_strength: RangeFloat = RangeFloat(low=1e-9, high=10.0, log=True)
    scale_pos_weight: RangeFloat = RangeFloat(low=1.0, high=15.0)


class OptunaConfig(BaseModel):
    n_trials: int = 50  # default 50
    direction: str = "maximize"
    xgboost_space: XGBConfigSearch = XGBConfigSearch()
    lightgbm_space: LGBMConfigSearch = LGBMConfigSearch()
    catboost_space: CatConfigSearch = CatConfigSearch()


class Settings(BaseSettings):
    project_name: str = "IEEE_Fraud_Detection"
    seed: int = 42

    paths: PathsConfig = PathsConfig()
    mlflow: MLflowConfig = MLflowConfig()

    id_col: str = "TransactionID"
    drop_cols: list[str] = ["isFraud", "TransactionID"]
    target: str = "isFraud"
    optuna: OptunaConfig = OptunaConfig()

    xgboost: XGBParams = XGBParams()
    lightgbm: LGBMParams = LGBMParams()
    catboost: CatParams = CatParams()


settings = Settings()
