from typing import Optional
from pydantic import BaseModel, Field


class CustomerRequest(BaseModel):
    """Raw Telco fields accepted by /api/predict."""

    customerID: Optional[str] = None
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None
