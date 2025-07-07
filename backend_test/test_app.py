import pytest
from fastapi.testclient import TestClient
from app import app

# Create a test client
client = TestClient(app)

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data
    assert "version" in data
    print(f"Root endpoint response: {data}")

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print(f"Health check response: {response.json()}")

def test_auth_token_endpoint_invalid():
    """Test the token endpoint with invalid credentials."""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "test@example.com", "password": "wrongpassword"}
    )
    assert response.status_code == 401
    print(f"Auth token invalid response: {response.status_code}")

def test_auth_token_endpoint_valid():
    """Test the token endpoint with valid credentials."""
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "user@example.com", "password": "secret"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    print(f"Auth token valid response: {data}")

def test_users_me_unauthorized():
    """Test the users/me endpoint without authentication."""
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401
    print(f"Users me unauthorized response: {response.status_code}")

def test_users_me_authorized():
    """Test the users/me endpoint with authentication."""
    # First get a token
    auth_response = client.post(
        "/api/v1/auth/token",
        data={"username": "user@example.com", "password": "secret"}
    )
    token = auth_response.json()["access_token"]
    
    # Then use the token to access the protected endpoint
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "user@example.com"
    print(f"Users me authorized response: {data}")

def test_digital_twins_authorized():
    """Test the digital-twins endpoint with authentication."""
    # First get a token
    auth_response = client.post(
        "/api/v1/auth/token",
        data={"username": "user@example.com", "password": "secret"}
    )
    token = auth_response.json()["access_token"]
    
    # Then use the token to access the protected endpoint
    response = client.get(
        "/api/v1/digital-twins",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "name" in data[0]
    print(f"Digital twins response: {data}")

if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__]) 