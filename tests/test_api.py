"""
Integration tests for the FastAPI sentiment analysis application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)
    
    def test_health_check_structure(self, client):
        """Test health check response structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"status", "model_loaded", "model_name", "model_version"}


class TestInfoEndpoint:
    """Tests for the info endpoint."""
    
    def test_info_endpoint_without_model(self, client):
        """Test info endpoint when model is not loaded."""
        from app import main
        main.sentiment_model = None
        
        response = client.get("/info")
        # Should return 503 Service Unavailable
        assert response.status_code == 503
    
    @patch('app.main.sentiment_model', MagicMock())
    @patch('app.main.model_version', 'v1')
    def test_info_endpoint_with_model(self, client):
        """Test info endpoint when model is loaded."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "model_stage" in data
        assert "status" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "documentation" in data


class TestPredictEndpoint:
    """Tests for the predict endpoint."""
    
    def test_predict_without_model(self, client):
        """Test prediction when model is not loaded."""
        from app import main
        main.sentiment_model = None
        main.text_preprocessor = None
        
        payload = {"text": "This is a great movie!"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    def test_predict_empty_text(self, client):
        """Test prediction with empty text."""
        from app import main
        main.sentiment_model = None
        main.text_preprocessor = None
        
        payload = {"text": ""}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_predict_whitespace_only(self, client):
        """Test prediction with whitespace-only text."""
        from app import main
        main.sentiment_model = None
        main.text_preprocessor = None
        
        payload = {"text": "   \n\t  "}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
    
    def test_predict_text_too_long(self, client):
        """Test prediction with text exceeding max length."""
        from app import main
        main.sentiment_model = None
        main.text_preprocessor = None
        
        payload = {"text": "a" * 6000}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    @patch('app.main.sentiment_model')
    def test_predict_valid_input_positive(self, mock_model, client):
        """Test prediction with valid positive sentiment text."""
        # Mock the model to return positive prediction
        mock_model.predict.return_value = [1]  # 1 = positive
        mock_model.predict_proba.return_value = [[0.1, 0.9]]  # High confidence for positive
        
        from app import main
        main.sentiment_model = mock_model
        main.text_preprocessor = MagicMock()
        main.text_preprocessor.transform.return_value = [[0.1, 0.2]]
        main.model_version = "v1"
        
        payload = {"text": "This movie is absolutely fantastic!"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert 0 <= data["confidence"] <= 1
        assert "model_version" in data
    
    @patch('app.main.sentiment_model')
    def test_predict_valid_input_negative(self, mock_model, client):
        """Test prediction with valid negative sentiment text."""
        # Mock the model to return negative prediction
        mock_model.predict.return_value = [0]  # 0 = negative
        mock_model.predict_proba.return_value = [[0.95, 0.05]]  # High confidence for negative
        
        from app import main
        main.sentiment_model = mock_model
        main.text_preprocessor = MagicMock()
        main.text_preprocessor.transform.return_value = [[0.1, 0.2]]
        main.model_version = "v1"
        
        payload = {"text": "This movie was terrible and boring."}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "negative"
        assert 0 <= data["confidence"] <= 1
    
    @patch('app.main.sentiment_model')
    def test_predict_response_structure(self, mock_model, client):
        """Test prediction response has correct structure."""
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        
        from app import main
        main.sentiment_model = mock_model
        main.text_preprocessor = MagicMock()
        main.text_preprocessor.transform.return_value = [[0.1, 0.2]]
        main.model_version = "v1"
        
        payload = {"text": "Good movie"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"sentiment", "confidence", "model_version"}
    
    @patch('app.main.sentiment_model')
    def test_predict_without_predict_proba(self, mock_model, client):
        """Test prediction when model doesn't have predict_proba."""
        mock_model.predict.return_value = [1]
        # Remove predict_proba from mock
        del mock_model.predict_proba
        
        from app import main
        main.sentiment_model = mock_model
        main.text_preprocessor = MagicMock()
        main.text_preprocessor.transform.return_value = [[0.1, 0.2]]
        main.model_version = "v1"
        
        payload = {"text": "Good movie"}
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] == 1.0  # Should default to 1.0


class TestInvalidRequests:
    """Tests for invalid request handling."""
    
    def test_predict_missing_text_field(self, client):
        """Test prediction with missing text field."""
        payload = {}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Unprocessable Entity (validation error)
    
    def test_predict_wrong_content_type(self, client):
        """Test prediction with wrong content type."""
        response = client.post("/predict", data="not json")
        assert response.status_code in [415, 422]  # Unsupported Media Type or validation error
    
    def test_predict_text_as_wrong_type(self, client):
        """Test prediction when text field is wrong type."""
        payload = {"text": 123}
        response = client.post("/predict", json=payload)
        # Pydantic should handle this - either reject or coerce
        # Depending on Pydantic settings, might accept and convert to string


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    @patch('app.main.sentiment_model')
    def test_predict_very_short_text(self, mock_model, client):
        """Test prediction with very short text."""
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.4, 0.6]]
        
        from app import main
        main.sentiment_model = mock_model
        main.text_preprocessor = MagicMock()
        main.text_preprocessor.transform.return_value = [[0.1, 0.2]]
        main.model_version = "v1"
        
        payload = {"text": "Good"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    
    @patch('app.main.sentiment_model')
    def test_predict_very_long_valid_text(self, mock_model, client):
        """Test prediction with very long but valid text."""
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        
        from app import main
        main.sentiment_model = mock_model
        main.text_preprocessor = MagicMock()
        main.text_preprocessor.transform.return_value = [[0.1, 0.2]]
        main.model_version = "v1"
        
        long_text = "This is a great movie. " * 100  # Create long text within limit
        payload = {"text": long_text}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    
    @patch('app.main.sentiment_model')
    def test_predict_special_characters(self, mock_model, client):
        """Test prediction with special characters."""
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.4, 0.6]]
        
        from app import main
        main.sentiment_model = mock_model
        main.text_preprocessor = MagicMock()
        main.text_preprocessor.transform.return_value = [[0.1, 0.2]]
        main.model_version = "v1"
        
        payload = {"text": "Amazing!!! 😀 #awesome @movie"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    
    @patch('app.main.sentiment_model')
    def test_predict_unicode_characters(self, mock_model, client):
        """Test prediction with unicode characters."""
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        
        from app import main
        main.sentiment_model = mock_model
        main.text_preprocessor = MagicMock()
        main.text_preprocessor.transform.return_value = [[0.1, 0.2]]
        main.model_version = "v1"
        
        payload = {"text": "Très magnifique film! 中文文本"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


class TestModelErrorHandling:
    """Tests for error handling when model operations fail."""
    
    @patch('app.main.sentiment_model')
    def test_predict_model_raises_exception(self, mock_model, client):
        """Test handling when model.predict raises an exception."""
        mock_model.predict.side_effect = Exception("Model error")
        
        from app import main
        main.sentiment_model = mock_model
        main.text_preprocessor = MagicMock()
        main.text_preprocessor.transform.return_value = [[0.1, 0.2]]
        main.model_version = "v1"
        
        payload = {"text": "Test text"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]
