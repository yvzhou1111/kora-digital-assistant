import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_root():
    """Test the root endpoint."""
    response = requests.get(f"{BASE_URL}/")
    print(f"Root endpoint status code: {response.status_code}")
    print(f"Root endpoint response: {response.json()}")
    
def test_health():
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Health check status code: {response.status_code}")
    print(f"Health check response: {response.json()}")
    
def test_auth_invalid():
    """Test authentication with invalid credentials."""
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/token",
        data={"username": "wrong@example.com", "password": "wrongpassword"}
    )
    print(f"Auth invalid status code: {response.status_code}")
    
def test_auth_valid():
    """Test authentication with valid credentials."""
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/token",
        data={"username": "user@example.com", "password": "secret"}
    )
    print(f"Auth valid status code: {response.status_code}")
    print(f"Auth valid response: {response.json()}")
    return response.json()["access_token"]
    
def test_users_me(token):
    """Test the users/me endpoint."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/v1/users/me", headers=headers)
    print(f"Users me status code: {response.status_code}")
    print(f"Users me response: {response.json()}")
    
def test_digital_twins(token):
    """Test the digital-twins endpoint."""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/v1/digital-twins", headers=headers)
    print(f"Digital twins status code: {response.status_code}")
    print(f"Digital twins response: {response.json()}")
    
if __name__ == "__main__":
    print("Testing API endpoints...")
    test_root()
    test_health()
    test_auth_invalid()
    token = test_auth_valid()
    test_users_me(token)
    test_digital_twins(token) 