from sqlalchemy import Boolean, Column, Integer, String
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True) # Null if Google Auth
    google_id = Column(String, nullable=True, unique=True, index=True)
    is_active = Column(Boolean, default=False)
    full_name = Column(String, nullable=True)
    profession = Column(String, nullable=True)
    time_limit_seconds = Column(Integer, default=300) # 5 minutes free
    time_used_seconds = Column(Integer, default=0)
