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
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "message" in data
    assert "version" in data
    print(f"Root endpoint response: {data}")

# Add a health endpoint to the main app
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print(f"Health check response: {response.json()}")

if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", "backend_test/test_api_health.py"]) 