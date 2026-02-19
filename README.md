# Sentiment Analysis MLOps Service

A production-ready sentiment analysis inference service that demonstrates modern MLOps practices using MLflow for experiment tracking and model registry management, combined with FastAPI for building a scalable REST API.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Setup Instructions](#setup-instructions)
- [Training a Model](#training-a-model)
- [API Documentation](#api-documentation)
- [Example API Requests](#example-api-requests)
- [Running Tests](#running-tests)
- [MLflow Model Registry](#mlflow-model-registry)
- [Design Decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)

## 🎯 Overview

This project implements a complete MLOps workflow for sentiment analysis:

1. **Data Processing**: Load and preprocess IMDb movie reviews
2. **Model Training**: Train a Logistic Regression model with TF-IDF vectorization
3. **Experiment Tracking**: Log all experiments to MLflow for reproducibility
4. **Model Registry**: Manage model versions and lifecycle stages
5. **API Service**: Serve predictions via a FastAPI REST endpoint
6. **Containerization**: Deploy using Docker and Docker Compose

### Key Features

- ✅ **MLflow Integration**: Complete experiment tracking and model registry management
- ✅ **FastAPI Inference**: High-performance REST API with input validation
- ✅ **Docker Compose**: Orchestrate MLflow server and API service
- ✅ **Comprehensive Testing**: Unit and integration tests with pytest
- ✅ **Production Ready**: Error handling, logging, health checks
- ✅ **Reproducible**: Fixed random seeds and pinned dependencies

## 📁 Project Structure

```
sentiment-mlops-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   └── utils/
│       ├── __init__.py
│       └── preprocess.py        # Text preprocessing utilities
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py       # Unit tests for preprocessing
│   └── test_api.py              # Integration tests for API
├── train.py                     # Model training script with MLflow
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Multi-container orchestration
├── .env.example                 # Environment variable template
├── README.md                    # This file
└── IMDB Dataset.csv            # Dataset (download from Kaggle)
```

## 📦 Prerequisites

- **Docker** (version 20.10+)
- **Docker Compose** (version 1.29+)
- **Python 3.9+** (for local development)
- **Git**

### Dataset

Download the IMDb Movie Review dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews):

1. Download `IMDB Dataset.csv`
2. Place it in the root directory of the project

## 🚀 Quick Start

The easiest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd sentiment-mlops-service

# Build and start services
docker-compose up --build

# Check service health
curl http://localhost:8000/health

# Access MLflow UI
# Open browser: http://localhost:5000
```

## 🔧 Setup Instructions

### Option 1: Using Docker Compose (Recommended)

#### 1. Build and Start Services

```bash
docker-compose up --build
```

This starts:
- **MLflow Server**: http://localhost:5000 (Experiment tracking & model registry)
- **FastAPI Application**: http://localhost:8000 (Sentiment inference API)

#### 2. Verify Services Are Running

```bash
# Check MLflow server
curl http://localhost:5000/

# Check API health
curl http://localhost:8000/health
```

### Option 2: Local Development Setup

#### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Download and Setup Dataset

```bash
# Place IMDB Dataset.csv in the root directory
```

#### 4. Start MLflow Server (in one terminal)

```bash
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts
```

#### 5. Start FastAPI Server (in another terminal)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 Training a Model

### Automated Training with Docker Compose

```bash
# Start services
docker-compose up

# In another terminal, run training inside the container
docker-compose exec fastapi-app python train.py
```

### Local Training

```bash
# Ensure MLflow server is running
python train.py
```

### Training Workflow

The training script (`train.py`):

1. **Loads** the IMDb dataset
2. **Preprocesses** text (cleaning, tokenization)
3. **Splits** data into train (70%), validation (15%), and test (15%)
4. **Trains** a Logistic Regression model with TF-IDF vectorization
5. **Evaluates** on validation and test sets
6. **Logs** hyperparameters, metrics, and artifacts to MLflow
7. **Registers** the model in MLflow Model Registry
8. **Transitions** the best model to production stage

### Model Parameters

The training script uses these hyperparameters:

**Vectorizer:**
- `max_features`: 5000 (vocabulary size)
- `ngram_range`: (1, 2) (unigrams and bigrams)
- `min_df`: 2 (minimum document frequency)
- `max_df`: 0.95 (maximum document frequency)

**Model:**
- Algorithm: Logistic Regression
- `C`: 1.0 (inverse regularization strength)
- `solver`: lbfgs
- `class_weight`: balanced

### Expected Training Output

```
INFO - Loading dataset from IMDB Dataset.csv
INFO - Dataset shape: (50000, 2)
INFO - Cleaning text data...
INFO - Data preprocessing completed. Positive: 25000, Negative: 25000
INFO - Train set: 35000, Val set: 7500, Test set: 7500
INFO - MLflow Run ID: abc123def456
INFO - Validation Metrics - Accuracy: 0.8950, F1: 0.8945
INFO - Test Metrics - Accuracy: 0.8940, F1: 0.8935
INFO - This is the best model! Transitioning to production...
✅ Training completed successfully! Run ID: abc123def456
```

## 🔌 API Documentation

The API provides sentiment analysis predictions through a RESTful interface.

### Base URL

- **Local**: `http://localhost:8000`
- **Docker**: `http://localhost:8000`

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "SentimentClassifier",
  "model_version": "v1"
}
```

#### 2. Model Information

```http
GET /info
```

**Response** (200 OK):
```json
{
  "model_name": "SentimentClassifier",
  "model_version": "v1",
  "model_stage": "production",
  "mlflow_tracking_uri": "http://mlflow-server:5000",
  "status": "ready"
}
```

#### 3. Sentiment Prediction

```http
POST /predict
Content-Type: application/json

{
  "text": "This movie was absolutely amazing!"
}
```

**Response** (200 OK):
```json
{
  "sentiment": "positive",
  "confidence": 0.95,
  "model_version": "v1"
}
```

### Error Responses

**Input Validation Error** (400 Bad Request):
```json
{
  "detail": "Input text cannot be empty."
}
```

**Text Too Long** (400 Bad Request):
```json
{
  "detail": "Input text exceeds maximum length of 5000 characters."
}
```

**Model Not Loaded** (503 Service Unavailable):
```json
{
  "detail": "Model not loaded. Service temporarily unavailable. Please try again later."
}
```

**Server Error** (500 Internal Server Error):
```json
{
  "detail": "Prediction failed: [error details]"
}
```

### Input Constraints

- **Minimum length**: 1 character
- **Maximum length**: 5000 characters
- **Required**: text field must be provided

## 📝 Example API Requests

### Using cURL

#### Positive Sentiment

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is the best movie I have ever seen! Absolutely incredible."}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "model_version": "v1"
}
```

#### Negative Sentiment

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible movie. Waste of time. Disappointing and boring."}'
```

**Response:**
```json
{
  "sentiment": "negative",
  "confidence": 0.88,
  "model_version": "v1"
}
```

#### Check Health

```bash
curl http://localhost:8000/health
```

#### Get Model Info

```bash
curl http://localhost:8000/info
```

### Using Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Check health
health = requests.get(f"{BASE_URL}/health")
print(health.json())

# Make prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "This movie was fantastic!"}
)
print(response.json())
```

### Using HTTPie

```bash
# Prediction
http POST localhost:8000/predict text="Great movie, highly recommended!"

# Health check
http GET localhost:8000/health

# Model info
http GET localhost:8000/info
```

## 🧪 Running Tests

### Using Docker

```bash
# Run tests inside container
docker-compose exec fastapi-app pytest tests/ -v
```

### Local Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocess.py -v
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test class
pytest tests/test_api.py::TestPredictEndpoint -v

# Run with markers
pytest -m "not slow" tests/
```

### Test Structure

**Preprocessing Tests** (`tests/test_preprocess.py`):
- Text cleaning (lowercase, HTML removal, punctuation)
- Special character handling
- URL and email removal
- Stopword filtering
- Batch processing

**API Tests** (`tests/test_api.py`):
- Health check endpoint
- Model info endpoint
- Prediction with valid inputs
- Error handling (empty text, model not loaded, server errors)
- Input validation
- Edge cases (very long text, unicode characters)
- Model error scenarios

### Test Coverage

Current test suite covers:
- ✅ Unit tests for preprocessing functions
- ✅ Integration tests for API endpoints
- ✅ Error handling and validation
- ✅ Edge cases and special scenarios
- ✅ Mock model predictions

Run `pytest tests/ --cov=app` to see coverage report.

## 🏛️ MLflow Model Registry

### Accessing MLflow UI

Open your browser and navigate to: **http://localhost:5000**

### Key Features

1. **Experiments**: View all training runs with their parameters and metrics
2. **Model Registry**: Manage registered models and versions
3. **Artifacts**: Download trained models and preprocessors
4. **Comparison**: Compare metrics across different runs

### Model Management Workflow

#### 1. View All Experiments

MLflow UI → Experiments tab → Select "Sentiment-Analysis"

#### 2. View Best Run

The run with the highest F1-score is typically the best performer.

#### 3. Register a Model

If a model isn't automatically registered, you can:

```python
import mlflow

client = mlflow.tracking.MlflowClient()

# Register the model
mv = client.create_model_version(
    name="SentimentClassifier",
    source="runs:/abc123def456/sentiment_model",
    run_id="abc123def456"
)

# Transition to production
client.transition_model_version_stage(
    name="SentimentClassifier",
    version=mv.version,
    stage="production"
)
```

#### 4. Load Model from Registry

```python
import mlflow

# Load production model
model = mlflow.pyfunc.load_model("models:/SentimentClassifier/production")

# Make prediction
prediction = model.predict(["This is great!"])
```

### Model Versioning

The MLflow registry automatically manages model versions:
- Version 1, 2, 3... for each registration
- Stages: None, staging, production, archived
- Fast stage transitions for promoting models

## 💡 Design Decisions

### 1. Model Choice: Logistic Regression with TF-IDF

**Rationale:**
- Simple, interpretable, and fast
- TF-IDF is effective for text classification
- Minimal resource requirements
- Focus is on MLOps, not model complexity
- Easily trainable in seconds on standard hardware

**Alternative considered:**
- Neural networks (Keras/TensorFlow): More complex, slower to train
- Naive Bayes: Simpler but less performant
- Complex embeddings: Overkill for this use case

### 2. Text Preprocessing Pipeline

**Components:**
- Lowercasing: Normalize text case
- HTML tag removal: Clean web-scraped content
- URL removal: Ignore links
- Number removal: Not informative for sentiment
- Stopword removal: Focus on meaningful words
- Punctuation removal: Reduce noise

**Rationale:**
- Improves model generalization
- Reduces vocabulary size
- Consistent preprocessing in both training and inference
- NLTK stopwords are language-specific

### 3. MLflow for MLOps

**Why MLflow?**
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Centralized model management and versioning
- **Reproducibility**: Restore any past experiment
- **Production Integration**: Easy model serving

**Alternative considered:**
- Weights & Biases: More features but requires account
- Kubeflow: Overkill for single-model service
- Custom solution: Time-consuming and error-prone

### 4. FastAPI for Inference

**Why FastAPI?**
- **Modern**: Built on Pydantic and Starlette
- **Fast**: High-performance async support
- **Easy**: Automatic OpenAPI documentation
- **Validated**: Built-in request validation
- **Standard**: Industry-standard Python web framework

**Key features used:**
- Pydantic models for request/response validation
- Async endpoints for non-blocking operations
- Startup/shutdown events for model loading
- Proper HTTP status codes and error messages
- OpenAPI auto-documentation

### 5. Docker Compose Orchestration

**Services:**
1. **MLflow Server**: Provides experiment tracking and model registry
2. **FastAPI Application**: Serves predictions

**Why Docker Compose?**
- Simplifies local development
- Ensures reproducibility
- Easy service coordination
- Single command to start everything
- Environment isolation

**Alternative considered:**
- Kubernetes: Overkill for development
- Manual Docker: More complex to manage multiple services
- Cloud platforms: Adds cost and complexity

### 6. Model Loading Strategy

**Current approach:**
- Load model from MLflow Registry by stage (production)
- Retry logic (15 attempts, 3-second intervals)
- Global model instance for performance
- Error handling with fallback behavior

**Benefits:**
- Never works with stale local files
- Automatic updates when model is promoted
- Graceful degradation if model unavailable
- Production-ready reliability

### 7. API Design

**RESTful Principles:**
- POST for predictions (creates inference result)
- GET for health and info (safe operations)
- Proper HTTP status codes
- Descriptive error messages
- JSON request/response format

**Validation:**
- Pydantic models enforce schema
- Input length constraints
- Type checking
- Clear error feedback

**Additional endpoints:**
- `/health`: Service health check
- `/info`: Model metadata
- `/docs`: Auto-generated API docs (Swagger UI)

## 🔍 Troubleshooting

### Issue: "Model not loaded"

**Cause:** MLflow server not ready or credentials issue

**Solution:**
```bash
# Check MLflow server is running
curl http://localhost:5000/

# Check logs
docker-compose logs mlflow-server

# Restart services
docker-compose restart
```

### Issue: "Failed to connect to MLflow"

**Cause:** Networking or DNS issue in container

**Solution:**
```bash
# Ensure correct tracking URI in container
# Should be: http://mlflow-server:5000 (not localhost)

# Verify in logs
docker-compose logs fastapi-app
```

### Issue: API responds with 503

**Cause:** Model loading failed

**Solution:**
```bash
# Train a model first
docker-compose exec fastapi-app python train.py

# Check if model is registered
# MLflow UI: http://localhost:5000 → Model Registry

# Check service logs
docker-compose logs fastapi-app
```

### Issue: Dataset not found

**Cause:** IMDB Dataset.csv not in root directory

**Solution:**
```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Place in root directory
ls -la IMDB\ Dataset.csv
```

### Issue: Port already in use

**Cause:** Another service using port 5000 or 8000

**Solution:**
```bash
# Check what's using the port
lsof -i :5000   # MLflow
lsof -i :8000   # FastAPI

# Change port in docker-compose.yml
# Or stop conflicting service
```

### Issue: Tests fail with import errors

**Cause:** Dependencies not installed

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run tests with proper path
python -m pytest tests/
```

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Client / User                          │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP Requests
                       ▼
┌─────────────────────────────────────────────────────────┐
│          FastAPI Application (Port 8000)                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  /predict    - Sentiment prediction endpoint      │  │
│  │  /health     - Health check                       │  │
│  │  /info       - Model information                  │  │
│  │  /docs       - API documentation (Swagger UI)     │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Text Preprocessing                              │  │
│  │  (Cleaning, Tokenization, TF-IDF)                │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Sentiment Model (Logistic Regression)           │  │
│  │  Loaded from MLflow Model Registry               │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │ Query Model Version
                       ▼
┌─────────────────────────────────────────────────────────┐
│      MLflow Tracking Server (Port 5000)                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Experiment Tracking                              │  │
│  │  - Parameters, Metrics, Artifacts                 │  │
│  │  - Run History and Comparison                     │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Model Registry                                   │  │
│  │  - Model Versions                                 │  │
│  │  - Stage Management (Production)                  │  │
│  │  - Model Artifacts (sklearn models)               │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Backend Store (SQLite Database)                  │  │
│  │  Artifact Store (File System or Cloud)            │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                    Volumes
                       │
                       ▼
               ┌─────────────────┐
               │  mlruns/        │
               │  ├── artifacts/ │
               │  └── mlflow.db  │
               └─────────────────┘
```

### Data Flow

```
Training Pipeline:
┌──────────────────┐
│ Load IMDB Data   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Preprocess Text  │ (clean, tokenize, TF-IDF)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Train Model      │ (Logistic Regression)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Evaluate Metrics │ (accuracy, F1, precision, recall)
└────────┬─────────┘
         ▼
┌──────────────────────────────┐
│ Log to MLflow                │
│ - Parameters                 │
│ - Metrics                    │
│ - Model + Vectorizer         │
└────────┬─────────────────────┘
         ▼
┌──────────────────────────────┐
│ Register in Model Registry   │
│ - Version Management         │
│ - Stage Transition           │
└──────────────────────────────┘

Inference Pipeline:
┌──────────────────┐
│ API Request      │ (JSON text input)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Validate Input   │ (length, type, content)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Load Model       │ (from MLflow Registry)
│ (if not cached)  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Preprocess Text  │ (same as training)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Make Prediction  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Return Response  │ (JSON: sentiment, confidence)
└──────────────────┘
```

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.9+ |
| **API Framework** | FastAPI | 0.104.1 |
| **ASGI Server** | Uvicorn | 0.24.0 |
| **Validation** | Pydantic | 2.5.0 |
| **ML Tracker** | MLflow | 2.10.0 |
| **ML Model** | scikit-learn | 1.3.2 |
| **Vectorization** | scikit-learn TF-IDF | 1.3.2 |
| **Data Processing** | pandas, numpy | 2.1.3, 1.24.3 |
| **Testing** | pytest | 7.4.3 |
| **Containerization** | Docker | 20.10+ |
| **Orchestration** | Docker Compose | 1.29+ |

## 📊 Performance Metrics

Expected performance on IMDb dataset:

| Metric | Value |
|--------|-------|
| **Model Accuracy** | ~89-90% |
| **F1-Score** | ~89-90% |
| **Inference Time** | <100ms per request |
| **Model Size** | ~5-10 MB |
| **Training Time** | ~30-60 seconds |

## 🔒 Security Considerations

1. **Input Validation**: All inputs validated with Pydantic
2. **Text Length Limits**: Max 5000 characters per request
3. **Error Handling**: Informative but safe error messages
4. **CORS**: Configure as needed for frontend integration
5. **Logging**: No sensitive data in logs

## 📈 Scaling Considerations

For production deployment:

1. **Load Balancing**: Deploy multiple API instances
2. **Caching**: Cache model predictions if applicable
3. **Database**: Use PostgreSQL instead of SQLite for MLflow
4. **Model Optimization**: Quantize or prune model for faster inference
5. **Monitoring**: Add Prometheus/Grafana for metrics
6. **Logging**: Use centralized logging (ELK, Splunk)

## 🤝 Contributing

To extend or improve this project:

1. **Add New Models**: Implement different architectures in `train.py`
2. **Enhance Preprocessing**: Expand `app/utils/preprocess.py`
3. **Add Monitoring**: Integrate Prometheus metrics
4. **Optimize**: Implement model distillation or quantization
5. **Scale**: Add message queues for async processing

## 📄 License

This project is provided as-is for educational and evaluation purposes.

## 🙏 Acknowledgments

- Dataset: [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Frameworks: [MLflow](https://mlflow.org/), [FastAPI](https://fastapi.tiangolo.com/), [scikit-learn](https://scikit-learn.org/)
- Community: Open-source ML and DevOps communities

## 📞 Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Design Decisions](#design-decisions) for architecture details
3. Check Docker logs: `docker-compose logs <service>`
4. Review test files for usage examples

---

**Last Updated**: February 2026  
**Version**: 1.0.0
