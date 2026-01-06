from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional


DatasetName = Literal["synthetic_gaussian", "digits_oneclass"]
ModelName = Literal["robust_zscore", "isolation_forest", "lof", "ocsvm"]
OutlierMode = Literal["clustered", "scattered"]


class DatasetConfig(BaseModel):
    name: DatasetName = "synthetic_gaussian"

    # synthetic
    n_samples: int = 8000
    n_features: int = 12
    contamination: float = Field(default=0.03, ge=0.0, le=0.5)
    outlier_mode: OutlierMode = "clustered"

    # digits_oneclass
    normal_digit: int = Field(default=1, ge=0, le=9)



class ModelConfig(BaseModel):
    name: ModelName = "robust_zscore"
    contamination_hint: Optional[float] = Field(default=None, ge=0.0, le=0.5)

    # isolation_forest
    n_estimators: int = Field(default=300, ge=10, le=5000)

    # lof
    n_neighbors: int = Field(default=35, ge=5, le=500)

    # ocsvm
    nu: Optional[float] = Field(default=None, ge=1e-4, le=0.5)
    kernel: str = "rbf"
    gamma: str | float = "scale"




class ReportConfig(BaseModel):
    path: str = "reports/leaderboard.csv"


class RunConfig(BaseModel):
    seed: int = 42
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    report: ReportConfig = ReportConfig()
