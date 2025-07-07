from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Create FastAPI app
app = FastAPI(
    title="Digital Twin API",
    description="API for Personal Digital Twin Platform",
    version="0.1.0",
)

# Dummy user database
fake_users_db = {
    "user@example.com": {
        "email": "user@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "full_name": "John Doe",
        "disabled": False,
    }
}

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class User(BaseModel):
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class DigitalTwin(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    user_id: int

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Helper functions
def get_user(db, email: str):
    if email in db:
        user_dict = db[email]
        return UserInDB(**user_dict)

def fake_decode_token(token):
    # This is a dummy function, in a real app you would decode the token
    user = get_user(fake_users_db, token)
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Routes
@app.get("/")
async def root():
    return {"status": "ok", "message": "Welcome to Digital Twin API", "version": "0.1.0"}

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/v1/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(fake_users_db, form_data.username)
    if not user or form_data.password != "secret":  # In a real app, verify password properly
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In a real app, generate a JWT token
    access_token = form_data.username
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/v1/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/api/v1/digital-twins", response_model=List[DigitalTwin])
async def get_digital_twins(current_user: User = Depends(get_current_active_user)):
    # Dummy data
    return [
        DigitalTwin(id=1, name="Professional Twin", description="Work-related digital twin", user_id=1),
        DigitalTwin(id=2, name="Personal Twin", description="Personal life digital twin", user_id=1)
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 