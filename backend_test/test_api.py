import sys
import os
import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Add the backend directory to the path so we can import the main app
sys.path.append(os.path.abspath("backend"))

# Load environment variables from .env file
load_dotenv("backend_test/.env")

# Import the main app
from main import app

# Create a test client
client = TestClient(app)

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_auth_token_endpoint():
    """Test the token endpoint with invalid credentials."""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "test@example.com", "password": "wrongpassword"}
    )
    assert response.status_code == 401

def test_users_me_unauthorized():
    """Test the users/me endpoint without authentication."""
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401

if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", "backend_test/test_api.py"]) 