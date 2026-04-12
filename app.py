"""
PokoAI — Real-Time AI Interview Assistant (Production Refactor)

Architecture:
    Audio → STT (streaming, auto-restart) → Utterance Builder (pause-based)
    → Speaker Filter → Intent Filter → Debounce → State Machine
    → Streaming LLM → WebSocket

Key changes from original:
    • Global USER_CONTEXT eliminated → per-user SessionStore
    • JWT removed from WebSocket URL → initial auth message
    • Streaming LLM responses (token-by-token) instead of full-response delay
    • Proper silence detection prevents mid-sentence triggers
    • Speaker diarization filters out candidate speech
    • Intent filter blocks fillers ("yeah", "okay", etc.)
    • Debounce prevents duplicate responses
    • State machine (LISTENING/PROCESSING/RESPONDING) prevents race conditions
    • Accurate usage tracking (wall-clock session duration)
    • STT auto-restart before 5-min limit with zero transcript loss
"""

import os
import asyncio
import json
import io
import logging
import random
import string
import smtplib
import bcrypt
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from sqlalchemy.orm import Session as DBSession
from google.oauth2 import service_account

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================================
# Google Cloud / Vertex AI Initialization
# ======================================================================

project_id = os.getenv("GCP_PROJECT_ID", "intrepid-honor-484608-e0")
location = "us-central1"

credentials_dict = {
    "type": "service_account",
    "project_id": os.getenv("GCP_PROJECT_ID"),
    "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
    "private_key": os.getenv("GCP_PRIVATE_KEY", "").replace("\\n", "\n"),
    "client_email": os.getenv("GCP_CLIENT_EMAIL"),
    "client_id": os.getenv("GCP_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": (
        f"https://www.googleapis.com/robot/v1/metadata/x509/"
        f"{os.getenv('GCP_CLIENT_EMAIL', '').replace('@', '%40')}"
    ),
    "universe_domain": "googleapis.com",
}

try:
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict
    )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=project_id, location=location, credentials=credentials)
    model = GenerativeModel("gemini-2.0-flash-001")
    logger.info("Vertex AI initialized successfully.")
except Exception as e:
    logger.error(f"Vertex AI Init Failed: {e}")
    model = None
    credentials = None

# ======================================================================
# Database & Auth Imports
# ======================================================================

import models
from database import engine, get_db, SessionLocal
from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    verify_google_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

# ======================================================================
# Core Pipeline Imports (the new modular architecture)
# ======================================================================

from core.session import SessionStore
from core.stt_manager import STTManager
from core.llm_stream import LLMStreamer
from core.pipeline import InterviewPipeline

# ======================================================================
# App Setup
# ======================================================================

models.Base.metadata.create_all(bind=engine)


def ensure_schema_updates():
    """Ensure existing DB has columns from recent updates."""
    from sqlalchemy import text

    with engine.connect() as conn:
        for col_def in [
            "ALTER TABLE users ADD COLUMN time_limit_seconds INTEGER DEFAULT 1200",
            "ALTER TABLE users ADD COLUMN time_used_seconds INTEGER DEFAULT 0",
            "ALTER TABLE users ADD COLUMN otp_code VARCHAR",
            "ALTER TABLE users ADD COLUMN otp_expires_at DATETIME",
        ]:
            try:
                conn.execute(text(col_def))
                logger.info(f"Schema update: {col_def.split('ADD COLUMN ')[1].split()[0]}")
            except Exception:
                pass
        conn.commit()


ensure_schema_updates()

app = FastAPI(title="PokoAI Interview Assistant", version="2.0.0")

# Global session store (replaces the old global USER_CONTEXT)
session_store = SessionStore()


# ======================================================================
# Auth Schemas
# ======================================================================


class UserLogin(BaseModel):
    email: str
    password: str


class UserRegister(BaseModel):
    email: str
    password: str
    fullName: str
    profession: str


class PasswordChange(BaseModel):
    oldPassword: str
    newPassword: str


class GoogleLogin(BaseModel):
    token: str


class OTPVerify(BaseModel):
    email: str
    otp: str


# ======================================================================
# Email Utility
# ======================================================================


def send_otp_email(to_email: str, otp: str):
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")

    if not sender_email or not sender_password:
        logger.error("EMAIL_USER or EMAIL_PASS not set")
        return False

    msg = MIMEMultipart()
    msg["From"] = f"PokoAI <{sender_email}>"
    msg["To"] = to_email
    msg["Subject"] = "Your OTP Code"

    html = f"""
      <h2>PokoAI Email Verification</h2>
      <p>Your OTP is:</p>
      <h1>{otp}</h1>
      <p>This code expires in 10 minutes.</p>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


# ======================================================================
# Auth Routes (kept from original — no changes needed)
# ======================================================================


@app.post("/auth/register")
def register(user: UserRegister, db: DBSession = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)

    otp = "".join(random.choices(string.digits, k=6))
    otp_hash = bcrypt.hashpw(otp.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    otp_expires = datetime.utcnow() + timedelta(minutes=10)

    new_user = models.User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.fullName,
        profession=user.profession,
        is_active=False,
        otp_code=otp_hash,
        otp_expires_at=otp_expires,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    send_otp_email(user.email, otp)

    access_token = create_access_token(
        data={"sub": new_user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "message": "OTP sent to email",
    }


@app.post("/auth/verify-otp")
def verify_otp(data: OTPVerify, db: DBSession = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if user.is_active:
        return {"message": "Account already active"}

    if not user.otp_code:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    try:
        if not bcrypt.checkpw(data.otp.encode("utf-8"), user.otp_code.encode("utf-8")):
            raise HTTPException(status_code=400, detail="Invalid OTP")
    except Exception:
        if user.otp_code != data.otp:
            raise HTTPException(status_code=400, detail="Invalid OTP")

    if not user.otp_expires_at or user.otp_expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="OTP Expired")

    user.is_active = True
    user.otp_code = None
    user.otp_expires_at = None
    db.commit()

    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {
        "message": "Account verified successfully",
        "access_token": access_token,
        "token_type": "bearer",
    }


@app.post("/auth/resend-otp")
def resend_otp(data: OTPVerify, db: DBSession = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if user.is_active:
        return {"message": "Account already active"}

    otp = "".join(random.choices(string.digits, k=6))
    otp_hash = bcrypt.hashpw(otp.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    otp_expires = datetime.utcnow() + timedelta(minutes=10)

    user.otp_code = otp_hash
    user.otp_expires_at = otp_expires
    db.commit()

    send_otp_email(user.email, otp)
    return {"message": "OTP resent successfully"}


@app.post("/auth/login")
def login(user: UserLogin, db: DBSession = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not db_user.hashed_password:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not db_user.is_active:
        raise HTTPException(
            status_code=403,
            detail="Account not verified. Please check your email for the OTP code.",
        )

    access_token = create_access_token(
        data={"sub": db_user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/google")
def google_auth(login: GoogleLogin, db: DBSession = Depends(get_db)):
    id_info = verify_google_token(login.token)
    if not id_info:
        raise HTTPException(status_code=400, detail="Invalid Google Token")

    email = id_info["email"]
    google_id = id_info["sub"]

    db_user = db.query(models.User).filter(models.User.email == email).first()

    if not db_user:
        full_name = id_info.get("name")
        db_user = models.User(
            email=email,
            google_id=google_id,
            full_name=full_name,
            is_active=True,
            time_limit_seconds=1200,
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    else:
        db_user.google_id = google_id
        db_user.is_active = True
        db.commit()

    access_token = create_access_token(
        data={"sub": db_user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/change-password")
def change_password(
    data: PasswordChange,
    current_user: models.User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    if not current_user.hashed_password:
        raise HTTPException(
            status_code=400, detail="Cannot change password for Google-only accounts"
        )

    if not verify_password(data.oldPassword, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password incorrect")

    current_user.hashed_password = get_password_hash(data.newPassword)
    db.commit()
    return {"status": "success", "message": "Password updated successfully"}


# ======================================================================
# Usage / Credits API
# ======================================================================


@app.get("/api/user/status")
def get_user_status(
    current_user: models.User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    return {
        "email": current_user.email,
        "full_name": current_user.full_name or "Candidate",
        "profession": current_user.profession or "Professional",
        "time_limit_seconds": current_user.time_limit_seconds,
        "time_used_seconds": current_user.time_used_seconds,
        "remaining_seconds": max(
            0, current_user.time_limit_seconds - current_user.time_used_seconds
        ),
    }


@app.post("/api/heartbeat")
def heartbeat(
    current_user: models.User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    if current_user.time_used_seconds >= current_user.time_limit_seconds:
        raise HTTPException(status_code=403, detail="Credit limit reached")

    current_user.time_used_seconds += 10
    db.commit()
    return {
        "status": "success",
        "remaining_seconds": max(
            0, current_user.time_limit_seconds - current_user.time_used_seconds
        ),
    }


# ======================================================================
# Context Management (now per-user via SessionStore)
# ======================================================================


@app.post("/update_context")
async def update_context(
    resume_file: UploadFile = File(None),
    jd: str = Form(...),
    company: str = Form(""),
    current_user: models.User = Depends(get_current_user),
):
    """
    Update the interview context for the authenticated user.
    
    Now stores context in the per-user session (no global state).
    """
    session = await session_store.create_or_restore(current_user.email)

    resume_text = ""
    if resume_file:
        try:
            content = await resume_file.read()
            pdf = PdfReader(io.BytesIO(content))
            for page in pdf.pages:
                resume_text += (page.extract_text() or "") + "\n"
            logger.info(f"Resume PDF processed ({len(resume_text)} chars)")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return {"status": "error", "message": str(e)}

    session.update_context(resume=resume_text, jd=jd, company=company)
    return {"status": "success", "message": "Context updated"}


# ======================================================================
# AI Feature Endpoints (now per-user context)
# ======================================================================


@app.post("/api/generate-briefing")
async def generate_briefing(
    current_user: models.User = Depends(get_current_user),
):
    """Generate interview prep cards — uses per-user session context."""
    try:
        session = await session_store.get_by_email(current_user.email)
        if not session or not session.context["resume"] or not session.context["jd"]:
            return {"status": "error", "message": "Resume and JD required"}

        ctx = session.context
        prompt = f"""
        You are a career coach. Based on this RESUME and JOB DESCRIPTION, generate 4 "Prep Cards" to help the candidate in the final 5 minutes before the interview.
        
        RESUME: {ctx['resume']}
        JD: {ctx['jd']}

        OUTPUT FORMAT (JSON ONLY):
        {{
            "cards": [
                {{
                    "title": "Elevator Pitch",
                    "content": "A 2-sentence intro tailored to this role.",
                    "icon": "🚀"
                }},
                {{
                    "title": "Must-Mention Project",
                    "content": "Which project fits this JD best and why.",
                    "icon": "⭐"
                }},
                {{
                    "title": "The Challenge",
                    "content": "A potential weakness/gap and how to defend it.",
                    "icon": "⚠️"
                }},
                {{
                    "title": "Top Skill",
                    "content": "The #1 technical skill this JD wants most.",
                    "icon": "🛠️"
                }}
            ]
        }}
        """
        response = await asyncio.to_thread(model.generate_content, prompt)
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        return json.loads(text)
    except Exception as e:
        logger.error(f"Briefing failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/analyze-resume")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    try:
        content = await resume.read()
        reader = PdfReader(io.BytesIO(content))
        resume_text = ""
        for page in reader.pages:
            resume_text += page.extract_text() or ""

        prompt = f"""
        You are an extremely strict, elite HR Recruiter and Technical Interviewer from a Top Tier Tech Company.
        Your job is to be BRUTALLY HONEST. Do not sugarcoat anything. If the candidate is bad, say it.
        
        Analyze the following Resume against the Job Description (JD).
        
        RESUME:
        {resume_text}
        
        JOB DESCRIPTION:
        {job_description}
        
        Your task:
        1. Compare every requirement in the JD with the experience in the Resume.
        2. Assign a strict Match Score (0 to 100). Be stingy. Only a perfect match gets 90+.
        3. Provide an 'Honest Verdict': A single, direct, blunt sentence about why they match or why they are failing miserably.
        4. List 'Missing Critical Skills': Specific technologies or experiences requested in JD that are nowhere to be found in the Resume.
        5. Provide 3-5 'Actionable Fixes': Direct instructions on what to add or change to stop being rejected.
        
        OUTPUT FORMAT (JSON ONLY):
        {{
            "score": <int>,
            "verdict": "<string>",
            "missing_skills": ["<string>", "<string>"],
            "suggestions": ["<string>", "<string>"]
        }}
        Do not output markdown code blocks. Just the raw JSON string.
        """

        if not model:
            raise HTTPException(status_code=500, detail="AI Model not initialized")

        response = model.generate_content(prompt)
        response_text = response.text.strip()

        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        return json.loads(response_text)

    except Exception as e:
        logger.error(f"Resume Analysis Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# Health & Static Routes
# ======================================================================

RATE = 16000
CHUNK = 1024
LANGUAGE_CODE = "en-US"


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


@app.get("/")
async def get():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Error: templates/index.html not found</h1>")


# ======================================================================
# WebSocket Endpoint — Complete Rewrite
# ======================================================================


def _authenticate_user(token: str):
    """
    Authenticate a user by JWT token.
    
    Creates its own DB session (not via Depends, since we're in a WS context
    where we need manual control).
    """
    db = SessionLocal()
    try:
        user = get_current_user(token, db)
        # Detach from the session so we can use the user object after closing
        email = user.email
        time_used = user.time_used_seconds
        time_limit = user.time_limit_seconds
        return {
            "email": email,
            "time_used": time_used,
            "time_limit": time_limit,
        }
    finally:
        db.close()


def _update_usage(email: str, duration_seconds: int):
    """Record actual session duration in the database."""
    db = SessionLocal()
    try:
        db_user = db.query(models.User).filter(models.User.email == email).first()
        if db_user:
            db_user.time_used_seconds += duration_seconds
            db.commit()
            logger.info(
                f"Usage updated for {email}: +{duration_seconds}s "
                f"(total: {db_user.time_used_seconds}s)"
            )
    except Exception as exc:
        logger.error(f"Usage update failed: {exc}")
    finally:
        db.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    """
    Production WebSocket endpoint for real-time interview assistance.

    Authentication flow:
        1. (Preferred) Connect to /ws, then send {"type": "auth", "token": "JWT"}
        2. (Legacy)    Connect to /ws?token=JWT — still works but logs a warning

    After auth, the client can:
        • Send binary audio frames → processed through the full pipeline
        • Send {"type": "context", "resume": "...", "jd": "...", "company": "..."}

    The server streams back:
        • {"type": "transcript", "transcript": "...", "is_final": bool}
        • {"type": "answer_chunk", "chunk": "...", "question": "..."}
        • {"type": "answer_complete", "question": "...", "answer": "..."}
        • {"type": "state", "state": "listening|processing|responding"}
        • {"type": "status", "state": "...", "message": "..."}
        • {"type": "error", "message": "..."}
    """
    await websocket.accept()

    # ------------------------------------------------------------------
    # Phase 1: Authentication
    # ------------------------------------------------------------------
    user_info = None

    if token:
        # Legacy: JWT in query string (insecure but backward-compatible)
        logger.warning(
            "DEPRECATION: JWT in URL query param is insecure. "
            "Migrate to initial auth message."
        )
        try:
            user_info = _authenticate_user(token)
        except Exception:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Invalid credentials"})
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    else:
        # Preferred: auth via initial WebSocket message
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            auth_data = json.loads(raw)
            if auth_data.get("type") != "auth" or not auth_data.get("token"):
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Auth message required"})
                )
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            user_info = _authenticate_user(auth_data["token"])

        except asyncio.TimeoutError:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Auth timeout"})
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        except Exception:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Invalid credentials"})
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    # ------------------------------------------------------------------
    # Phase 2: Credit check
    # ------------------------------------------------------------------
    if user_info["time_used"] >= user_info["time_limit"]:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "Credit limit reached"})
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    email = user_info["email"]
    logger.info(f"WebSocket authenticated: {email}")

    # ------------------------------------------------------------------
    # Phase 3: Session setup
    # ------------------------------------------------------------------
    session = await session_store.create_or_restore(email)
    session.mark_connected()

    await websocket.send_text(
        json.dumps(
            {
                "type": "auth_success",
                "session_id": session.session_id,
                "email": email,
            }
        )
    )

    # ------------------------------------------------------------------
    # Phase 4: Pipeline setup
    # ------------------------------------------------------------------
    stt_manager = STTManager(
        credentials=credentials,
        sample_rate=RATE,
        language_code=LANGUAGE_CODE,
        enable_diarization=True,
    )
    llm_streamer = LLMStreamer(model=model)
    pipeline = InterviewPipeline(
        session=session,
        websocket=websocket,
        stt_manager=stt_manager,
        llm_streamer=llm_streamer,
    )

    await pipeline.start()

    # ------------------------------------------------------------------
    # Phase 5: Main message loop
    # ------------------------------------------------------------------
    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                pipeline.feed_audio(message["bytes"])

            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "context":
                        # Live context update via WebSocket
                        session.update_context(
                            resume=data.get("resume", ""),
                            jd=data.get("jd", ""),
                            company=data.get("company", ""),
                        )
                        await websocket.send_text(
                            json.dumps(
                                {"type": "context_updated", "status": "success"}
                            )
                        )

                    elif msg_type == "ping":
                        await websocket.send_text(
                            json.dumps({"type": "pong"})
                        )

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {email}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # ------------------------------------------------------------------
        # Phase 6: Cleanup
        # ------------------------------------------------------------------
        await pipeline.stop()

        # Accurate usage tracking — real session duration
        duration = session.get_session_duration_seconds()
        if duration > 0:
            _update_usage(email, duration)

        logger.info(
            f"Session ended for {email} — duration: {duration}s"
        )
