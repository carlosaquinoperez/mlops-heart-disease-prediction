import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from helpers import to_dicts

# 1. Define Pydantic Models for Input/Output

class Customer(BaseModel):
    """
    Defines the input data schema for a single customer prediction.
    We use Pydantic's Field for validation (e.g., must be > 0).
    """
    age_years: int = Field(..., gt=0)
    gender: int = Field(..., ge=1, le=2) # 1: Male, 2: Female
    height_cm: int = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    systolic_bp: int
    diastolic_bp: int
    cholesterol: int = Field(..., ge=1, le=3) # 1, 2, or 3
    glucose: int = Field(..., ge=1, le=3)     # 1, 2, or 3
    is_smoker: int = Field(..., ge=0, le=1)   # 0 or 1
    is_alcoholic: int = Field(..., ge=0, le=1) # 0 or 1
    is_active: int = Field(..., ge=0, le=1)   # 0 or 1

class PredictResponse(BaseModel):
    """
    Defines the output data schema for the prediction.
    """
    has_cardiovascular_disease: bool
    probability: float


# 2. Load the Model Pipeline
model_file = 'model.bin'

print(f"Loading model from {model_file}...")
with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)
print("Model loaded successfully.")


# 3. Initialize the FastAPI App
app = FastAPI(title="Cardiovascular disease prediction API")


# 4. Define the Prediction Endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(customer: Customer):
    """
    Main prediction endpoint.
    Takes a Customer JSON object and returns a prediction.
    """
    # Convert the Pydantic Customer object back to a DataFrame
    # because our scikit-learn pipeline expects it.
    customer_df = pd.DataFrame([customer.model_dump()])
    
    # Use the pipeline to get the probability
    # The pipeline handles all scaling and encoding automatically.
    # [:, 1] gets the probability of the "positive" class (disease=1)
    probability = pipeline.predict_proba(customer_df)[0, 1]
    
    # Make a final decision (True/False)
    has_disease = bool(probability >= 0.5)
    
    # Return the response
    return PredictResponse(
        has_cardiovascular_disease=has_disease,
        probability=probability
    )

# 5. Health Check Endpoint
@app.get("/health")
def health_check():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok"}

# 6. Run the API (if script is run directly)
# This allows us to run the app with: python predict.py
if __name__ == "__main__":
    print("Starting API with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=9696)