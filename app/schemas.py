from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    X1: int = Field(..., ge=1, le=5)
    # X2: int = Field(..., ge=1, le=5)
    # X3: int = Field(..., ge=1, le=5)
    # X4: int = Field(..., ge=1, le=5)
    X5: int = Field(..., ge=1, le=5)
    # X6: int = Field(..., ge=1, le=5)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    features_used: list[str]
