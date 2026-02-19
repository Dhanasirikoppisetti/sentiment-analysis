# System Architecture

## Overview

This document provides detailed architecture information for the Sentiment Analysis MLOps Service.

## System Components

### 1. Data Processing Layer

**Location**: `app/utils/preprocess.py`

**Responsibilities**:
- Load raw text input
- Clean and normalize text
- Tokenize and vectorize text
- Handle edge cases and invalid input

**Key Functions**:
```python
clean_text(text: str) -> str
batch_clean_texts(texts: List[str]) -> List[str]
```

**Processing Pipeline**:
1. Convert to lowercase
2. Remove HTML tags
3. Remove URLs and emails
4. Remove numbers
5. Remove punctuation
6. Normalize whitespace
7. Remove stopwords
8. Filter short words (< 3 chars)

### 2. Model Training Pipeline

**Location**: `train.py`

**Workflow**:
1. Load IMDB dataset (50,000 reviews)
2. Apply text preprocessing to all reviews
3. Split data: train (70%), validation (15%), test (15%)
4. Vectorize text using TF-IDF:
   - Max features: 5000
   - N-grams: unigrams (1) and bigrams (2)
   - Min document frequency: 2
   - Max document frequency: 0.95
5. Train Logistic Regression classifier
6. Evaluate on validation and test sets
7. Log all metrics and artifacts to MLflow
8. Register model in MLflow Model Registry
9. Transition best model to production stage

**Metrics Tracked**:
- Accuracy
- F1-Score
- Precision
- Recall
- Confusion Matrix

### 3. ML Model

**Algorithm**: Logistic Regression (Linear Classification)

**Characteristics**:
- Binary classification (positive/negative)
- Trained on TF-IDF vectors
- Fast inference (<100ms per request)
- Interpretable predictions with probabilities

**Hyperparameters**:
```python
LogisticRegression(
    C=1.0,                    # Inverse regularization strength
    solver='lbfgs',           # Optimization algorithm
    class_weight='balanced',  # Handle class imbalance
    random_state=42,          # Reproducibility
    max_iter=1000             # Max iterations
)
```

### 4. Feature Engineering

**TF-IDF Vectorization**:
- Converts text to numerical features
- Measures word importance (frequency × inverse document frequency)
- Creates sparse matrix representation
- Reduces vocabulary to meaningful terms

**Example**:
```
Text: "This movie is great"
TF-IDF Vector: [0.2, 0.3, 0.1, 0.4, ...]  (5000 dimensions)
```

### 5. MLflow Integration

**Components**:

**A. Experiment Tracking**
- Logs all training runs
- Records parameters, metrics, and artifacts
- Enables run comparison and reproducibility
- Backend: SQLite database (`mlruns/mlflow.db`)

**B. Model Registry**
- Centralized model management
- Version control for deployed models
- Stage management (None, staging, production)
- Artifact storage (file system or cloud)

**C. Model Versioning**
- Automatic version numbering (v1, v2, v3...)
- Track model lineage and changes
- Easy rollback to previous versions

### 6. FastAPI Inference Service

**Architecture**:
```
HTTP Request
    ↓
Input Validation (Pydantic)
    ↓
Text Preprocessing
    ↓
Model Inference (Logistic Regression)
    ↓
Post-processing (label + confidence)
    ↓
HTTP Response (JSON)
```

**Endpoint Structure**:

1. **Health Endpoints**
   - `GET /health`: Service status
   - `GET /info`: Model metadata

2. **Prediction Endpoint**
   - `POST /predict`: Main inference endpoint
   - Input: JSON with text field
   - Output: JSON with sentiment and confidence

3. **Documentation**
   - `GET /docs`: Swagger UI
   - `GET /redoc`: ReDoc

**Request Validation**:
```python
class TextInput(BaseModel):
    text: str  # 1-5000 characters
```

**Response Validation**:
```python
class PredictionOutput(BaseModel):
    sentiment: str          # "positive" or "negative"
    confidence: float       # 0.0 to 1.0
    model_version: str      # "v1", "v2", etc.
```

### 7. Model Loading Strategy

**Initialization Flow**:
```
Application Startup
    ↓
Configure MLflow Tracking URI
    ↓
Attempt to Load Model from Registry
    ↓
Retry Logic (15 attempts, 3-second intervals)
    ↓
Load Model Successfully OR Degrade Gracefully
    ↓
Set Global Model Reference
```

**Why This Design**:
- Tolerates temporary MLflow unavailability
- Never uses stale local files
- Automatic model updates
- Production-ready reliability

**Automatic Preprocessing**:
- MLflow logs both model and vectorizer
- Vectorizer loaded automatically when model loaded
- Ensures consistency between training and inference

### 8. Docker Architecture

**Multi-container Setup**:

```
Docker Host
├── Container: mlflow-server
│   ├── Image: ghcr.io/mlflow/mlflow:v3.1.4
│   ├── Port: 5000
│   ├── Volumes: mlflow_data (persistent)
│   └── Health Check: Enabled
│
├── Container: sentiment-api
│   ├── Image: python:3.9-slim (custom build)
│   ├── Port: 8000
│   ├── Depends On: mlflow-server (healthy)
│   └── Health Check: Enabled
│
└── Named Volume: mlflow_data
    ├── Database: mlflow.db (SQLite)
    └── Artifacts: artifacts/ (models, vectorizers)
```

**Benefits**:
- Service isolation
- Easy scaling
- Reproducible environments
- Volume persistence across container restarts

### 9. Data Flow Diagrams

#### Training Flow

```
┌─────────────────┐
│ IMDB Dataset    │
│ (50,000 samples)│
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Text Preprocessing  │
│ (clean, normalize)  │
└────────┬────────────┘
         │
         ▼
┌──────────────────────────┐
│ Data Splitting           │
│ Train: 35,000 (70%)      │
│ Val:   7,500 (15%)       │
│ Test:  7,500 (15%)       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Feature Extraction       │
│ TF-IDF Vectorization     │
│ Vocabulary: 5,000 terms  │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Model Training           │
│ Logistic Regression      │
│ 1000 max iterations      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Model Evaluation         │
│ Val metrics: A, F1, P, R │
│ Test metrics              │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ MLflow Logging               │
│ - Parameters                 │
│ - Metrics                    │
│ - Model artifact             │
│ - Vectorizer artifact        │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Model Registry               │
│ - Register model             │
│ - Version management         │
│ - Transition to production   │
└──────────────────────────────┘
```

#### Inference Flow

```
┌──────────────────┐
│ User Request     │
│ JSON: {text: ...}│
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│ FastAPI Endpoint         │
│ /predict                 │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Input Validation         │
│ - Type check (str)       │
│ - Length check (1-5000)  │
│ - Content check (empty?) │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Text Preprocessing       │
│ (same pipeline as train) │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Load ML Model            │
│ (from memory if cached)  │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Model Inference          │
│ - Predict class (0/1)    │
│ - Get probability        │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Post-processing          │
│ - Map 0→negative, 1→pos  │
│ - Format confidence score│
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Return Response          │
│ JSON: {sentiment, conf}  │
└──────────────────────────┘
```

### 10. Error Handling Strategy

**Input Validation Errors (400)**:
- Empty text
- Text too long (>5000 chars)
- Wrong data type

**Service Unavailable Errors (503)**:
- Model not loaded
- MLflow connection failed
- Model registry inaccessible

**Server Errors (500)**:
- Model inference failed
- Unexpected exceptions
- Database connection issues

**Graceful Degradation**:
- Service starts even if model unavailable
- Returns appropriate status codes
- Informative error messages
- Logging for debugging

### 11. Testing Architecture

**Unit Tests** (`tests/test_preprocess.py`):
- Test individual preprocessing functions
- Cover edge cases
- Validate text cleaning
- Batch processing tests

**Integration Tests** (`tests/test_api.py`):
- Test API endpoints
- Mock models for isolation
- Validate request/response formats
- Error handling tests
- End-to-end scenarios

**Testing Pyramid**:
```
        ▲
       /│\
      / │ \  Integration Tests (20%)
     /  │  \
    ┌───┼───┐
    │   │   │ Unit Tests (60%)
    ├───┼───┤
    │       │ Contract Tests (20%)
    └───────┘
```

### 12. Deployment Considerations

**Production Ready Features**:
✓ Reproducible environment (pinned dependencies)
✓ Health checks (liveness + readiness)
✓ Comprehensive logging
✓ Error handling and validation
✓ Model versioning
✓ Graceful service degradation

**Scaling Options**:
1. **Horizontal Scaling**: Run multiple API containers with load balancer
2. **Model Optimization**: Quantize or prune model for faster inference
3. **Caching**: Cache predictions for identical inputs
4. **Async Processing**: Queue predictions for batch processing
5. **Monitoring**: Add Prometheus metrics for observability

**Security Best Practices**:
✓ Input validation
✓ Sanitized error messages
✓ No sensitive data in logs
✓ Container security (minimal base image)
✓ Resource limits (CPU, memory)

## Technology Stack Rationale

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| Container | Docker | Industry standard, reproducible |
| API Framework | FastAPI | Modern, fast, easy validation |
| ML Tracker | MLflow | Open-source, industry standard |
| ML Library | scikit-learn | Interpretable, maintains focus on MLOps |
| Vectorizer | TF-IDF | Effective, fast, proven baseline |
| Database | SQLite | Simple, no external dependencies |
| Testing | pytest | Industry standard, excellent plugins |
| Language | Python 3.9 | Stable, widely used in ML |

## Performance Characteristics

**Training Phase**:
- Time: ~30-60 seconds (on modern hardware)
- Memory: ~2-4 GB
- Disk: ~50 MB for dataset
- Output: ~5-10 MB for model + vectorizer

**Inference Phase**:
- Latency: p50: ~20ms, p99: ~50ms
- Throughput: ~100 requests/sec (single container)
- Memory per request: <1 MB
- CPU usage: <10% per request

**Storage**:
- Dataset: 50 MB (IMDB CSV)
- Model: 5-10 MB
- Vectorizer: 1-2 MB
- MLflow database: 10-20 MB
- MLflow artifacts: 20-50 MB

## Monitoring and Observability

**Built-in Metrics**:
- Model loaded status
- Prediction latency
- Error rates
- Model version info

**Health Checks**:
- API liveness: `GET /health`
- API readiness: `GET /predict` (with validation)
- MLflow connectivity: Startup health check

**Logging**:
- Training logs: parameters, metrics, timing
- API logs: requests, predictions, errors
- Container logs: startup, shutdown, issues

## Security Architecture

**Input Security**:
- Pydantic validation
- Length constraints
- Type checking
- Sanitization

**Service Security**:
- Environment variable isolation
- No hardcoded credentials
- Container user isolation
- Network isolation (docker network)

**Data Security**:
- No model data in logs
- No sensitive features exposed
- Read-only model artifacts
- Filesystem permissions

---

**Last Updated**: February 2026  
**Version**: 1.0.0
