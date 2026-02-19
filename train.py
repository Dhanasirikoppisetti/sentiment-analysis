"""
Training script for sentiment analysis model with MLflow integration.
This script trains a Logistic Regression model on the IMDB dataset,
logs all metrics and parameters to MLflow, and registers the best model.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from app.utils.preprocess import clean_text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RANDOM_STATE = 42
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATASET_PATH = "IMDB Dataset.csv"
MODEL_NAME = "SentimentClassifier"

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Sentiment-Analysis")


def load_and_preprocess_data(data_path: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load and preprocess the IMDB dataset.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Tuple of (reviews, sentiments) as pandas Series
    """
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Clean reviews
    logger.info("Cleaning text data...")
    df["review"] = df["review"].apply(clean_text)
    
    # Encode sentiments
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    
    logger.info(f"Data preprocessing completed. Positive: {(df['sentiment'] == 1).sum()}, Negative: {(df['sentiment'] == 0).sum()}")
    
    return df["review"], df["sentiment"]


def train_and_evaluate_model(
    X_train: pd.Series,
    X_val: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
    vectorizer_params: dict,
    model_params: dict
) -> Tuple[TfidfVectorizer, LogisticRegression, dict]:
    """
    Train and evaluate the sentiment classification model.
    
    Args:
        X_train: Training reviews
        X_val: Validation reviews
        y_train: Training labels
        y_val: Validation labels
        vectorizer_params: Parameters for TfidfVectorizer
        model_params: Parameters for LogisticRegression
        
    Returns:
        Tuple of (fitted_vectorizer, fitted_model, metrics_dict)
    """
    logger.info("Vectorizing text data...")
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    logger.info(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    logger.info(f"Training data shape: {X_train_vec.shape}")
    
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(**model_params, random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    logger.info("Evaluating model on validation set...")
    preds = model.predict(X_val_vec)
    proba = model.predict_proba(X_val_vec)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_val, preds),
        "f1_score": f1_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
    }
    
    logger.info(f"Validation Metrics - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return vectorizer, model, metrics


def train(run_name: str = "LogisticRegression-TFIDF"):
    """
    Main training function with MLflow integration.
    
    Args:
        run_name: Name for the MLflow run
    """
    try:
        logger.info(f"Starting training run: {run_name}")
        
        # Load and preprocess data
        reviews, sentiments = load_and_preprocess_data(DATASET_PATH)
        
        # Split data: 70% train, 15% validation, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            reviews, sentiments,
            test_size=0.3,
            random_state=RANDOM_STATE,
            stratify=sentiments
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
        
        logger.info(f"Train set: {len(X_train)}, Val set: {len(X_val)}, Test set: {len(X_test)}")
        
        # Define hyperparameters
        vectorizer_params = {
            "max_features": 5000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95
        }
        
        model_params = {
            "C": 1.0,
            "solver": "lbfgs",
            "class_weight": "balanced"
        }
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            # Log parameters
            mlflow.log_params({
                **{f"vectorizer_{k}": v for k, v in vectorizer_params.items()},
                **{f"model_{k}": v for k, v in model_params.items()}
            })
            
            # Train and evaluate
            vectorizer, model, val_metrics = train_and_evaluate_model(
                X_train, X_val, y_train, y_val,
                vectorizer_params, model_params
            )
            
            # Evaluate on test set
            X_test_vec = vectorizer.transform(X_test)
            test_preds = model.predict(X_test_vec)
            
            test_metrics = {
                "test_accuracy": accuracy_score(y_test, test_preds),
                "test_f1_score": f1_score(y_test, test_preds),
                "test_precision": precision_score(y_test, test_preds),
                "test_recall": recall_score(y_test, test_preds),
            }
            
            logger.info(f"Test Metrics - Accuracy: {test_metrics['test_accuracy']:.4f}, F1: {test_metrics['test_f1_score']:.4f}")
            
            # Log all metrics
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)
            
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log models as artifacts
            logger.info("Logging model artifacts...")
            mlflow.sklearn.log_model(
                model,
                artifact_path="sentiment_model",
                registered_model_name=MODEL_NAME
            )
            
            # Save vectorizer separately and log it
            mlflow.sklearn.log_model(
                vectorizer,
                artifact_path="tfidf_vectorizer"
            )
            
            # Register/update model in MLflow Model Registry
            logger.info(f"Registering model: {MODEL_NAME}")
            
            # Get all versions of this model to find the latest
            client = mlflow.tracking.MlflowClient()
            try:
                latest_versions = client.get_latest_versions(MODEL_NAME)
                latest_version_num = max([int(v.version) for v in latest_versions if v.version])
            except:
                latest_version_num = 0
            
            # Transition new model to Production if it's the best (highest F1-score)
            all_runs = client.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name("Sentiment-Analysis").experiment_id]
            )
            
            best_run = max(all_runs, key=lambda x: x.data.metrics.get("val_f1_score", 0))
            
            if best_run.info.run_id == run.info.run_id:
                logger.info("✅ This is the best model! Transitioning to production...")
                
                # Find the registered model version
                model_versions = client.get_latest_versions(MODEL_NAME)
                if model_versions:
                    latest_model = model_versions[0]
                    client.transition_model_version_stage(
                        name=MODEL_NAME,
                        version=latest_model.version,
                        stage="production",
                        archive_existing_versions=True
                    )
                    logger.info(f"Model {MODEL_NAME} v{latest_model.version} transitioned to production")
            else:
                logger.info("⚠️ Not the best model. Keeping in Staging stage.")
            
            logger.info(f"✅ Training completed successfully! Run ID: {run.info.run_id}")
            
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    train()

