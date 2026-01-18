from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import abc
import os
import hashlib
import bcrypt
from dotenv import load_dotenv
from google.oauth2 import id_token
from google.auth.transport import requests

# New Imports for Dependency
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import get_db
import models

load_dotenv()

# Config
SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_key_change_this") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 
GOOGLE_CLIENT_ID = "595917889227-8js5rpugt95nnrvqjlbodobea128f86i.apps.googleusercontent.com"

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def _pre_hash(password: str) -> bytes:
    # 1. SHA256 hash to ensure fixed length (64 chars) < 72 bytes limit of bcrypt
    # 2. Return as bytes for bcrypt
    return hashlib.sha256(password.encode("utf-8")).hexdigest().encode("utf-8")

def verify_password(plain_password, hashed_password):
    if not hashed_password:
        return False
    # Ensure hashed_password is bytes
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode("utf-8")
    return bcrypt.checkpw(_pre_hash(plain_password), hashed_password)

def get_password_hash(password):
    return bcrypt.hashpw(_pre_hash(password), bcrypt.gensalt()).decode("utf-8")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_google_token(token: str):
    try:
        id_info = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        return id_info
    except ValueError as e:
        print(f"Google Token Verification Failed: {e}")
        return None

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account pending approval. Please contact support."
        )
        
    return user
