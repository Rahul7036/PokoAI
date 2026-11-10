from sqlalchemy import Boolean, Column, Integer, String, DateTime
from database import Base

# Free Copilot time every new account gets (7 minutes). Existing accounts keep their current limit.
FREE_TIME_SECONDS = 420

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True) # Null if Google Auth
    google_id = Column(String, nullable=True, unique=True, index=True)
    is_active = Column(Boolean, default=False)
    full_name = Column(String, nullable=True)
    profession = Column(String, nullable=True)
    time_limit_seconds = Column(Integer, default=FREE_TIME_SECONDS)
    time_used_seconds = Column(Integer, default=0)
    
    # OTP Verification
    otp_code = Column(String, nullable=True)
    otp_expires_at = Column(DateTime, nullable=True)
