"""
FastAPI application for sentiment analysis inference.
Loads the best production model from MLflow Model Registry and serves predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import os
import asyncio
import logging
from typing import Optional
from app.utils.preprocess import clean_text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="Sentiment analysis service using MLflow Model Registry",
    version="1.0.0"
)

# Environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "SentimentClassifier")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "production")

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables for model and preprocessor
sentiment_model = None
text_preprocessor = None
model_version = None


# ============================================================================
# Request and Response Models
# ============================================================================

class TextInput(BaseModel):
    """Input schema for prediction endpoint."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment"
    )


class PredictionOutput(BaseModel):
    """Output schema for prediction endpoint."""
    sentiment: str = Field(description="Predicted sentiment: 'positive' or 'negative'")
    confidence: float = Field(description="Confidence score between 0 and 1")
    model_version: str = Field(description="Version of the model used for prediction")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded")
    model_name: Optional[str] = Field(description="Name of the loaded model")
    model_version: Optional[str] = Field(description="Version of the loaded model")


# ============================================================================
# Model Loading
# ============================================================================

@app.on_event("startup")
async def load_model():
    """
    Load the production model and preprocessor from MLflow Model Registry.
    Implements retry logic for reliability when MLflow server might not be ready.
    """
    global sentiment_model, text_preprocessor, model_version
    
    max_retries = 15
    retry_delay = 3  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Attempt {attempt + 1}/{max_retries}: Loading model from MLflow...")
            
            # Get model metadata for version info
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            
            if versions:
                current_version = versions[0]
                model_version = f"v{current_version.version}"
                model_uri = f"models:/{MODEL_NAME}/{current_version.version}"
                vectorizer_uri = f"runs:/{current_version.run_id}/tfidf_vectorizer"

                logger.info(f"Loading model from URI: {model_uri}")
                sentiment_model = mlflow.sklearn.load_model(model_uri)
                text_preprocessor = mlflow.sklearn.load_model(vectorizer_uri)

                logger.info(f"✅ Model loaded successfully: {MODEL_NAME} {model_version}")
                
                # Log model details
                logger.info(f"   Model Registry URI: {model_uri}")
                logger.info(f"   Registered at: {current_version.creation_timestamp}")
                return
            else:
                logger.warning(f"No model found in {MODEL_STAGE} stage")
                
        except Exception as e:
            logger.warning(f"⚠️  Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ Failed to load model after {max_retries} attempts")
                logger.error("Model will not be available for predictions until it's loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server")


# ============================================================================
# Health and Info Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health of the API and model status."""
    return HealthResponse(
        status="healthy" if sentiment_model is not None and text_preprocessor is not None else "degraded",
        model_loaded=sentiment_model is not None and text_preprocessor is not None,
        model_name=MODEL_NAME if sentiment_model is not None and text_preprocessor is not None else None,
        model_version=model_version if sentiment_model is not None and text_preprocessor is not None else None
    )


@app.get("/info", tags=["Info"])
async def model_info():
    """Get information about the loaded model."""
    if sentiment_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    return {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "model_stage": MODEL_STAGE,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "status": "ready"
    }


# ============================================================================
# Prediction Endpoint
# ============================================================================

@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for the given text.
    
    Args:
        input_data: TextInput containing the text to analyze
        
    Returns:
        PredictionOutput with sentiment label and confidence score
        
    Raises:
        HTTPException: For invalid input, missing model, or server errors
    """
    # Validate input
    if not input_data.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty."
        )
    
    if len(input_data.text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Input text exceeds maximum length of 5000 characters."
        )
    
    # Check if model is loaded
    if sentiment_model is None or text_preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service temporarily unavailable. Please try again later."
        )
    
    try:
        cleaned_text = clean_text(input_data.text)
        features = text_preprocessor.transform([cleaned_text])
        prediction_result = sentiment_model.predict(features)
        
        # Extract prediction (0 or 1)
        prediction = prediction_result[0]
        
        # Get probability scores if available
        try:
            proba_result = sentiment_model.predict_proba(features)
            proba = proba_result[0]
            # confidence is the probability of the predicted class
            confidence = float(max(proba))
        except (AttributeError, TypeError, IndexError):
            # If predict_proba is not available, use prediction as confidence
            confidence = 1.0
        
        # Map prediction to sentiment label
        sentiment_label = "positive" if prediction == 1 else "negative"
        
        logger.info(f"Prediction: {sentiment_label} (confidence: {confidence:.4f})")
        
        return PredictionOutput(
            sentiment=sentiment_label,
            confidence=confidence,
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# Root and Documentation
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Welcome to Sentiment Analysis API",
        "documentation": "/docs",
        "health_check": "/health",
        "model_info": "/info",
        "prediction_endpoint": "/predict"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

