import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add the backend directory to the path so we can import the main app
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(backend_dir)

try:
    # Import the main app
    from main import app
    
    # Create a test client
    client = TestClient(app)
    
    def test_read_root():
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        print(f"Root endpoint response: {response.json()}")
        
    if __name__ == "__main__":
        # Run the tests
        pytest.main(["-xvs", __file__])
        
except Exception as e:
    print(f"Error importing app: {str(e)}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Backend directory: {backend_dir}")
    print(f"Python path: {sys.path}")
    sys.exit(1) 