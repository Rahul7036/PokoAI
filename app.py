import os
import queue
import asyncio
import json
import logging
import threading
import re
import smtplib
import secrets
import requests
import bcrypt
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google.cloud import speech
from google import genai
from google.genai import types as genai_types

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# These libraries log every single AI request at INFO level; keep the server log readable
for noisy in ("httpx", "google_genai", "google_genai.models"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Configure Google Cloud credentials from environment variables
import json
import tempfile
from google.oauth2 import service_account

project_id = os.getenv("GCP_PROJECT_ID")
location = os.getenv("GCP_LOCATION", "global")
# Model for Prep-Deck / Resume Matcher (quality matters more than speed)
model_name = os.getenv("MODEL_NAME", "gemini-3.5-flash")
# Model for the live Copilot (speed matters most). Falls back to MODEL_NAME if it fails.
COPILOT_MODEL = os.getenv("COPILOT_MODEL", "gemini-3.6-flash")
# MINIMAL thinking = fastest first word (~1s). Set empty to use the model's default.
COPILOT_THINKING_LEVEL = os.getenv("COPILOT_THINKING_LEVEL", "MINIMAL")

# Construct service account credentials from env vars
credentials_dict = {
    "type": "service_account",
    "project_id": os.getenv("GCP_PROJECT_ID"),
    "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
    "private_key": os.getenv("GCP_PRIVATE_KEY", "").replace("\\n", "\n"),  # Handle escaped newlines
    "client_email": os.getenv("GCP_CLIENT_EMAIL"),
    "client_id": os.getenv("GCP_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('GCP_CLIENT_EMAIL', '').replace('@', '%40')}",
    "universe_domain": "googleapis.com"
}

# Create credentials from dict
try:
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # Set credentials for Google Cloud clients
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""  # Clear any file-based credentials

    # Gemini on Vertex AI via the google-genai SDK
    ai_client = genai.Client(vertexai=True, project=project_id, location=location, credentials=credentials)
    logger.info(f"Gemini initialized (location '{location}'): copilot='{COPILOT_MODEL}' "
                f"thinking={COPILOT_THINKING_LEVEL or 'default'}, other features='{model_name}'.")
except Exception as e:
    logger.error(f"Vertex AI Init Failed: {e}")
    ai_client = None
    credentials = None


async def generate_text(prompt: str, json_mode: bool = False) -> str:
    """One-shot Gemini call (Prep-Deck, Resume Matcher)."""
    config = genai_types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    response = await ai_client.aio.models.generate_content(model=model_name, contents=prompt, config=config)
    return response.text or ""


def _is_rate_limited(e: Exception) -> bool:
    return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)


async def generate_text_reliably(prompt: str, json_mode: bool = False) -> str:
    """Like generate_text, but rides out Gemini's short "429 Resource exhausted" limits: retry with a
    pause, then try the copilot model (it has its own quota). Used where losing the result is costly."""
    for attempt, pause in enumerate((2, 5)):
        try:
            return await generate_text(prompt, json_mode)
        except Exception as e:
            if not _is_rate_limited(e):
                raise
            logger.warning(f"Gemini rate-limited ({model_name}), retry {attempt + 1} in {pause}s")
            await asyncio.sleep(pause)
    try:
        return await generate_text(prompt, json_mode)
    except Exception as e:
        if not _is_rate_limited(e) or COPILOT_MODEL == model_name:
            raise
    logger.warning(f"Gemini still rate-limited; falling back to '{COPILOT_MODEL}'")
    config = genai_types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    response = await ai_client.aio.models.generate_content(model=COPILOT_MODEL, contents=prompt, config=config)
    return response.text or ""


async def stream_model_text(prompt: str, model: str, thinking_level: str = None):
    """Async generator yielding text chunks from Gemini as they are produced."""
    config = None
    if thinking_level:
        config = genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(thinking_level=thinking_level)
        )
    stream = await ai_client.aio.models.generate_content_stream(model=model, contents=prompt, config=config)
    async for chunk in stream:
        if chunk.text:
            yield chunk.text


async def stream_copilot_answer(prompt: str):
    """Stream from the fast copilot model; if it fails before producing text, fall back to MODEL_NAME."""
    produced = False
    try:
        async for piece in stream_model_text(prompt, COPILOT_MODEL, COPILOT_THINKING_LEVEL or None):
            produced = True
            yield piece
        return
    except Exception as e:
        if produced:
            raise
        logger.warning(f"Copilot model '{COPILOT_MODEL}' failed ({e}); falling back to '{model_name}'")
    async for piece in stream_model_text(prompt, model_name):
        yield piece

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
import io
import models
from database import engine, get_db, SessionLocal
from sqlalchemy import update, func
from sqlalchemy.orm import Session
import time
from auth import get_password_hash, verify_password, create_access_token, verify_google_token, get_current_user, get_current_user_detached, GOOGLE_CLIENT_ID
from datetime import timedelta


# Initialize Tables
models.Base.metadata.create_all(bind=engine)

def ensure_schema_updates():
    """Ensure existing DB has columns from recent updates."""
    from sqlalchemy import text
    with engine.connect() as conn:
        # Check for time_limit_seconds
        try:
            conn.execute(text(f"ALTER TABLE users ADD COLUMN time_limit_seconds INTEGER DEFAULT {models.FREE_TIME_SECONDS}"))
            logger.info("Added time_limit_seconds column")
        except Exception:
            pass
        
        # Check for time_used_seconds
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN time_used_seconds INTEGER DEFAULT 0"))
            logger.info("Added time_used_seconds column")
        except Exception:
            pass
            
        # Check for otp_code
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN otp_code VARCHAR"))
            logger.info("Added otp_code column")
        except Exception:
            pass

        # Check for otp_expires_at
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN otp_expires_at DATETIME"))
            logger.info("Added otp_expires_at column")
        except Exception:
            pass
        conn.commit()

ensure_schema_updates()

app = FastAPI()


# --- Auth Schemas ---
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

EMAIL_TIMEOUT_SECONDS = 15
EMAIL_FROM_NAME = "PrapAI"
BREVO_SEND_URL = "https://api.brevo.com/v3/smtp/email"


def _send_via_brevo(api_key, sender, to_email, subject, text, html):
    """Brevo's HTTPS API. Used in production: Render's free plan blocks SMTP ports, but not HTTPS."""
    resp = requests.post(
        BREVO_SEND_URL,
        headers={"api-key": api_key, "accept": "application/json", "content-type": "application/json"},
        json={
            "sender": {"name": EMAIL_FROM_NAME, "email": sender},
            "to": [{"email": to_email}],
            "subject": subject,
            "textContent": text,
            "htmlContent": html,
        },
        timeout=EMAIL_TIMEOUT_SECONDS,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Brevo HTTP {resp.status_code}: {resp.text[:300]}")


def _send_via_gmail_smtp(sender, password, to_email, subject, text, html):
    """Gmail SMTP with an App Password. Fine locally; blocked on Render's free plan."""
    msg = MIMEMultipart("alternative")
    msg["From"] = f"{EMAIL_FROM_NAME} <{sender}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    # Plain text first, HTML last (mail apps show the last part they support)
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))
    # Timeout so a blocked/unreachable mail server can't hang the signup request
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=EMAIL_TIMEOUT_SECONDS) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)


def send_otp_email(to_email: str, otp: str) -> bool:
    """Send the 6-digit code. Returns False (and logs why) if the email could not be sent.

    BREVO_API_KEY set  -> send through Brevo (from EMAIL_FROM, a sender verified in Brevo).
    otherwise          -> Gmail SMTP with EMAIL_USER / EMAIL_PASS.
    """
    brevo_key = os.getenv("BREVO_API_KEY", "").strip()
    sender = (os.getenv("EMAIL_FROM") or os.getenv("EMAIL_USER") or "").strip()
    password = os.getenv("EMAIL_PASS", "")

    if not sender:
        logger.error("OTP email not sent: set EMAIL_FROM (or EMAIL_USER) to the address codes are sent from")
        return False
    if not brevo_key and not password:
        logger.error("OTP email not sent: set BREVO_API_KEY (production) or EMAIL_PASS (Gmail SMTP)")
        return False

    subject = f"{otp} is your PrapAI verification code"
    text = (f"Your PrapAI verification code is {otp}\n\n"
            f"It expires in {OTP_VALID_MINUTES} minutes. If you didn't sign up for PrapAI, ignore this email.")
    html = f"""
      <h2>PrapAI email verification</h2>
      <p>Your verification code is:</p>
      <h1 style="letter-spacing:4px">{otp}</h1>
      <p>It expires in {OTP_VALID_MINUTES} minutes. If you didn't sign up for PrapAI, ignore this email.</p>
    """

    provider = "Brevo" if brevo_key else "Gmail SMTP"
    try:
        if brevo_key:
            _send_via_brevo(brevo_key, sender, to_email, subject, text, html)
        else:
            _send_via_gmail_smtp(sender, password, to_email, subject, text, html)
        logger.info(f"OTP email sent to {to_email} via {provider}")
        return True
    except Exception as e:
        logger.error(f"OTP email to {to_email} via {provider} failed: {type(e).__name__}: {e}")
        return False


# --- Email OTP helpers ---
OTP_VALID_MINUTES = 10
OTP_RESEND_COOLDOWN_SECONDS = 30
OTP_MAX_ATTEMPTS = 5          # wrong guesses allowed per code; then a new code is needed
OTP_ATTEMPTS = {}             # email -> wrong guesses for the current code (in memory, single process)
EMAIL_SEND_FAILED = ("We couldn't send the verification email right now. "
                     "Please try again in a minute, or use Sign in with Google.")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def find_user_by_email(db: Session, email: str):
    """Case-insensitive lookup, so 'User@Gmail.com' and 'user@gmail.com' are the same account."""
    return db.query(models.User).filter(func.lower(models.User.email) == normalize_email(email)).first()


def issue_otp(user: models.User) -> str:
    """Give the user a fresh code (stored hashed). Returns the plain code to email."""
    otp = f"{secrets.randbelow(10**6):06d}"
    user.otp_code = bcrypt.hashpw(otp.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user.otp_expires_at = datetime.utcnow() + timedelta(minutes=OTP_VALID_MINUTES)
    OTP_ATTEMPTS.pop(normalize_email(user.email), None)
    return otp


def otp_seconds_since_sent(user: models.User):
    if not user.otp_expires_at:
        return None
    sent_at = user.otp_expires_at - timedelta(minutes=OTP_VALID_MINUTES)
    return (datetime.utcnow() - sent_at).total_seconds()


# --- Auth Routes ---
@app.post("/auth/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    email = normalize_email(user.email)
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    db_user = find_user_by_email(db, email)
    if db_user and (db_user.is_active or db_user.google_id):
        if db_user.google_id and not db_user.hashed_password:
            raise HTTPException(status_code=400, detail="This email uses Sign in with Google. Please use the Google button.")
        raise HTTPException(status_code=400, detail="Email already registered. Please sign in.")

    if db_user:
        # Signed up before but never verified (e.g. the code never arrived): let them start over
        since = otp_seconds_since_sent(db_user)
        if since is not None and since < OTP_RESEND_COOLDOWN_SECONDS:
            raise HTTPException(status_code=429, detail="A code was just sent. Please wait a few seconds and try again.")
        db_user.hashed_password = get_password_hash(user.password)
        db_user.full_name = user.fullName
        db_user.profession = user.profession
    else:
        db_user = models.User(
            email=email,
            hashed_password=get_password_hash(user.password),
            full_name=user.fullName,
            profession=user.profession,
            is_active=False,  # becomes active after the email code is verified
        )
        db.add(db_user)

    otp = issue_otp(db_user)
    db.commit()

    if not send_otp_email(db_user.email, otp):
        # Account stays unverified; signing up again (or Resend) will retry the email
        raise HTTPException(status_code=503, detail=EMAIL_SEND_FAILED)

    return {"message": "Verification code sent", "email": db_user.email}

@app.post("/auth/verify-otp")
def verify_otp(data: OTPVerify, db: Session = Depends(get_db)):
    email = normalize_email(data.email)
    code = (data.otp or "").strip()
    user = find_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=400, detail="No account found for this email. Please sign up first.")

    if user.is_active:
        access_token = create_access_token(data={"sub": user.email})
        return {"message": "Account already active", "access_token": access_token, "token_type": "bearer"}

    if not user.otp_code or not user.otp_expires_at:
        raise HTTPException(status_code=400, detail="No active code. Tap Resend code to get a new one.")
    if user.otp_expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="This code has expired. Tap Resend code to get a new one.")
    if OTP_ATTEMPTS.get(email, 0) >= OTP_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many wrong attempts. Tap Resend code to get a new one.")

    try:
        valid = len(code) == 6 and code.isdigit() and bcrypt.checkpw(code.encode("utf-8"), user.otp_code.encode("utf-8"))
    except ValueError:
        valid = False  # stored value isn't a bcrypt hash (very old plain-text code)
    if not valid:
        OTP_ATTEMPTS[email] = OTP_ATTEMPTS.get(email, 0) + 1
        left = OTP_MAX_ATTEMPTS - OTP_ATTEMPTS[email]
        raise HTTPException(status_code=400, detail=f"Wrong code. {left} attempt{'s' if left != 1 else ''} left."
                            if left > 0 else "Too many wrong attempts. Tap Resend code to get a new one.")

    user.is_active = True
    user.otp_code = None
    user.otp_expires_at = None
    db.commit()
    OTP_ATTEMPTS.pop(email, None)

    access_token = create_access_token(data={"sub": user.email})
    return {"message": "Account verified successfully", "access_token": access_token, "token_type": "bearer"}

@app.post("/auth/resend-otp")
def resend_otp(data: OTPVerify, db: Session = Depends(get_db)):
    # Only the email is used
    user = find_user_by_email(db, data.email)
    if not user:
        raise HTTPException(status_code=400, detail="No account found for this email. Please sign up first.")

    if user.is_active:
        return {"message": "Account already active. Please sign in."}

    since = otp_seconds_since_sent(user)
    if since is not None and since < OTP_RESEND_COOLDOWN_SECONDS:
        wait = int(OTP_RESEND_COOLDOWN_SECONDS - since) + 1
        raise HTTPException(status_code=429, detail=f"Please wait {wait} seconds before asking for another code.")

    otp = issue_otp(user)
    db.commit()

    if not send_otp_email(user.email, otp):
        raise HTTPException(status_code=503, detail=EMAIL_SEND_FAILED)
    return {"message": "A new code has been sent to your email."}

@app.post("/auth/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = find_user_by_email(db, user.email)
    if db_user and not db_user.hashed_password and db_user.google_id:
        raise HTTPException(status_code=400, detail="This email uses Sign in with Google. Please use the Google button.")
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password.")

    if not db_user.is_active:
        # The frontend opens the code screen on 403 so the user can verify (or resend) right away
        raise HTTPException(status_code=403, detail="Your email isn't verified yet. Enter the code we emailed you.")

    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/google")
def google_auth(login: GoogleLogin, db: Session = Depends(get_db)):
    id_info = verify_google_token(login.token)
    if not id_info:
        raise HTTPException(status_code=400, detail="Invalid Google Token")

    # Only trust the email if Google has verified the user owns it; otherwise someone
    # could link a Google account with an unverified address to an existing PrapAI account.
    if not id_info.get("email_verified"):
        raise HTTPException(status_code=400, detail="Your Google email address is not verified.")

    email = id_info['email']
    google_id = id_info['sub']
    
    # Check if user exists
    db_user = find_user_by_email(db, email)
    
    if not db_user:
        # Create new user via Google, automatically activate with the free time allowance
        full_name = id_info.get('name')
        db_user = models.User(
            email=email,
            google_id=google_id,
            full_name=full_name,
            is_active=True,
            time_limit_seconds=models.FREE_TIME_SECONDS
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    else:
        # User exists, ensure they are linked to Google and active
        db_user.google_id = google_id
        db_user.is_active = True
        db.commit()

    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/change-password")
def change_password(data: PasswordChange, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.hashed_password:
        raise HTTPException(status_code=400, detail="Cannot change password for Google-only accounts")
    
    if not verify_password(data.oldPassword, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password incorrect")
    
    current_user.hashed_password = get_password_hash(data.newPassword)
    db.commit()
    return {"status": "success", "message": "Password updated successfully"}

# --- Usage/Credits API ---
@app.get("/api/user/status")
def get_user_status(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return {
        "email": current_user.email,
        "full_name": current_user.full_name or "Candidate",
        "profession": current_user.profession or "Professional",
        "time_limit_seconds": current_user.time_limit_seconds,
        "time_used_seconds": current_user.time_used_seconds,
        "remaining_seconds": max(0, current_user.time_limit_seconds - current_user.time_used_seconds)
    }

@app.post("/api/heartbeat")
def heartbeat(current_user: models.User = Depends(get_current_user)):
    # Read-only: usage is now billed server-side by the /ws session itself.
    # Kept so browsers still running the old frontend keep working.
    remaining = _remaining_seconds(current_user)
    if remaining <= 0:
        raise HTTPException(status_code=403, detail="Credit limit reached")
    return {"status": "success", "remaining_seconds": remaining}


def _remaining_seconds(user: models.User) -> int:
    return max(0, (user.time_limit_seconds or 0) - (user.time_used_seconds or 0))


def _charge_usage(user_id: int, seconds: int) -> int:
    """Atomically add `seconds` to the user's usage. Returns remaining seconds."""
    db = SessionLocal()
    try:
        row = db.execute(
            update(models.User)
            .where(models.User.id == user_id)
            .values(time_used_seconds=func.coalesce(models.User.time_used_seconds, 0) + seconds)
            .returning(models.User.time_used_seconds, models.User.time_limit_seconds)
        ).first()
        db.commit()
        if row is None:
            return 0
        used, limit = row
        return max(0, (limit or 0) - (used or 0))
    finally:
        db.close()


# Context Management
# Each user's resume / JD / company, kept in memory only (never written to the DB),
# keyed by user id so users never see each other's data. Cleared on server restart.
# NOTE: in-memory means this only works with a single server process (one uvicorn worker).
USER_CONTEXTS = {}

def get_user_context(user_id: int) -> dict:
    return USER_CONTEXTS.setdefault(user_id, {"resume": "", "jd": "", "company": ""})


def _extract_pdf_text(content: bytes) -> str:
    """CPU-bound PDF parsing; call via asyncio.to_thread so it doesn't block other users."""
    reader = PdfReader(io.BytesIO(content))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _parse_model_json(text: str):
    """Parse JSON from a model reply, tolerating ```json fences around it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text, strict=False)   # strict=False: allow raw newlines/tabs inside strings (models emit them)

# Size caps for text the browser sends back (a resume is typically 3-10k characters)
CONTEXT_MAX_RESUME_CHARS = 20000
CONTEXT_MAX_JD_CHARS = 10000
CONTEXT_MAX_COMPANY_CHARS = 200


@app.post("/update_context")
async def update_context(
    resume_file: UploadFile = File(None),
    resume_text: str = Form(""),
    jd: str = Form(...),
    company: str = Form(""),
    current_user: models.User = Depends(get_current_user_detached),
):
    """Set the resume / JD / company for this user.

    The resume comes either as a PDF (first time) or as the text the browser saved from an earlier
    upload (so users don't have to upload again after a refresh or on their next interview).
    The extracted text is returned so the browser can keep it on the user's device.
    """
    ctx = get_user_context(current_user.id)

    if resume_file:
        try:
            content = await resume_file.read()
            text = await asyncio.to_thread(_extract_pdf_text, content)
            ctx["resume"] = text[:CONTEXT_MAX_RESUME_CHARS]
            logger.info(f"Resume PDF processed ({len(text)} chars) for user {current_user.id}")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return {"status": "error", "message": "Couldn't read that PDF. Please try another file."}
        if not ctx["resume"].strip():
            return {"status": "error", "message": "That PDF has no readable text (it may be a scanned image). "
                                                  "Please upload a text-based PDF."}
    elif resume_text.strip():
        ctx["resume"] = resume_text[:CONTEXT_MAX_RESUME_CHARS]

    ctx["jd"] = jd[:CONTEXT_MAX_JD_CHARS]
    ctx["company"] = company[:CONTEXT_MAX_COMPANY_CHARS]
    logger.info(f"Context updated for user {current_user.id}, company: {company}")
    return {"status": "success", "message": "Context updated",
            "resume_text": ctx["resume"] if resume_file else None}


class ContextRestore(BaseModel):
    resume: str = ""
    jd: str = ""
    company: str = ""


@app.post("/api/context/restore")
def restore_context(data: ContextRestore, current_user: models.User = Depends(get_current_user_detached)):
    """After a page refresh or a server restart: put back the setup the browser saved.

    Only fills in what the server has lost; it never overwrites a resume the server already has.
    """
    ctx = get_user_context(current_user.id)
    restored = False
    if not ctx["resume"] and data.resume.strip():
        ctx["resume"] = data.resume[:CONTEXT_MAX_RESUME_CHARS]
        ctx["jd"] = data.jd[:CONTEXT_MAX_JD_CHARS]
        ctx["company"] = data.company[:CONTEXT_MAX_COMPANY_CHARS]
        restored = True
        logger.info(f"Context restored from the browser for user {current_user.id}")
    return {"restored": restored, "has_resume": bool(ctx["resume"])}

@app.post("/api/generate-briefing")
async def generate_briefing(current_user: models.User = Depends(get_current_user_detached)):
    ctx = get_user_context(current_user.id)
    try:
        if not ai_client:
            return {"status": "error", "message": "AI Model not initialized"}
        if not ctx["resume"] or not ctx["jd"]:
            return {"status": "error", "message": "Resume and JD required"}

        prompt = f"""
        You are a career coach. Based on this RESUME and JOB DESCRIPTION, generate 4 "Prep Cards" to help the candidate in the final 5 minutes before the interview.

        RESUME: {ctx['resume'][:3000]}
        JD: {ctx['jd'][:2000]}

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
        return _parse_model_json(await generate_text(prompt, json_mode=True))
    except Exception as e:
        logger.error(f"Briefing failed: {e}")
        return {"status": "error", "message": str(e)}

# ---------- Interview report (after a copilot session) ----------
REPORT_MAX_ITEMS = 30
REPORT_QUESTION_CHARS = 400
REPORT_SAY_CHARS = 500
REPORT_DAILY_LIMIT = 30          # report requests per user per day (each is one AI call)
PRACTICE_MAX_QUESTIONS = 5
DAILY_USAGE = {}                 # (kind, user_id) -> (date, count); in memory, single process


def use_daily_allowance(kind: str, user_id: int, limit: int) -> bool:
    """Count one use of a free feature. False once today's limit is reached."""
    today = datetime.utcnow().date()
    day, count = DAILY_USAGE.get((kind, user_id), (today, 0))
    if day != today:
        count = 0
    if count >= limit:
        return False
    DAILY_USAGE[(kind, user_id)] = (today, count + 1)
    return True


class ReportItem(BaseModel):
    question: str
    say: str = ""


class SessionReportRequest(BaseModel):
    items: list[ReportItem]


@app.post("/api/copilot/report")
async def copilot_report(data: SessionReportRequest, current_user: models.User = Depends(get_current_user_detached)):
    items = [i for i in data.items if i.question.strip()][-REPORT_MAX_ITEMS:]
    if not items:
        raise HTTPException(status_code=400, detail="No questions to report on yet.")
    if not ai_client:
        raise HTTPException(status_code=503, detail="AI is not available right now.")
    if not use_daily_allowance("report", current_user.id, REPORT_DAILY_LIMIT):
        raise HTTPException(status_code=429, detail="You've made a lot of reports today. Please try again tomorrow.")

    ctx = get_user_context(current_user.id)
    listing = "\n".join(
        f"{n}. QUESTION: {i.question.strip()[:REPORT_QUESTION_CHARS]}\n   SUGGESTED ANSWER: {i.say.strip()[:REPORT_SAY_CHARS] or '(none)'}"
        for n, i in enumerate(items, 1))
    prompt = f"""
        You are a friendly interview coach. A candidate just finished a REAL job interview. Below are the
        questions the interviewer asked (as heard by speech recognition, so some contain extra chatter
        before the actual question) and the short answer we suggested for each.

        COMPANY: {ctx.get('company') or 'not given'}
        JOB DESCRIPTION: {(ctx.get('jd') or 'not given')[:COPILOT_JD_CHARS]}

        QUESTIONS:
        {listing}

        Write a short report in simple English for a nervous candidate:
        - "summary": 1-2 sentences on what this interview focused on (e.g. "Mostly Python basics and one
          system design question, plus HR at the end."). No scores, no judging the candidate.
        - "clean_questions": for EVERY numbered question above, in the same order, the actual question in
          one clean sentence (drop chatter and filler). Exactly {len(items)} entries.
        - "topics": the 3 topics most worth revising before the next round, based on these questions and
          the job description. Each has "topic" (2-5 words), "why" (one sentence: which question showed it)
          and "tip" (one concrete thing to study or practise, max 20 words).
        - "practice": the numbers (1-based) of up to {PRACTICE_MAX_QUESTIONS} questions that are hardest
          or most likely to come again, best first. Skip small talk.

        OUTPUT JSON ONLY:
        {{"summary": "...", "clean_questions": ["..."], "topics": [{{"topic": "...", "why": "...", "tip": "..."}}], "practice": [1]}}
        """
    try:
        result = _parse_model_json(await generate_text_reliably(prompt, json_mode=True))
    except Exception as e:
        logger.error(f"Copilot report failed for {current_user.email}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't write the report right now. Please try again.")

    clean = [str(q or "").strip()[:REPORT_QUESTION_CHARS] for q in (result.get("clean_questions") or [])]
    if len(clean) != len(items):
        clean = [""] * len(items)
    questions = [c or i.question.strip()[:REPORT_QUESTION_CHARS] for c, i in zip(clean, items)]
    topics = []
    for t in (result.get("topics") or [])[:3]:
        if isinstance(t, dict) and str(t.get("topic") or "").strip():
            topics.append({k: str(t.get(k) or "").strip()[:300] for k in ("topic", "why", "tip")})
    practice = []
    for n in result.get("practice") or []:
        if isinstance(n, int) and 1 <= n <= len(items) and (n - 1) not in practice:
            practice.append(n - 1)
    return {"summary": str(result.get("summary") or "").strip()[:500], "questions": questions,
            "topics": topics, "practice": practice[:PRACTICE_MAX_QUESTIONS]}


class PracticeRequest(BaseModel):
    questions: list[str]


@app.post("/api/mock/practice")
def set_mock_practice(data: PracticeRequest, current_user: models.User = Depends(get_current_user_detached)):
    """Questions from a real interview to include in the next mock interview (used once)."""
    questions = [q.strip()[:REPORT_QUESTION_CHARS] for q in data.questions if q.strip()][:PRACTICE_MAX_QUESTIONS]
    get_user_context(current_user.id)["practice"] = questions
    return {"saved": len(questions)}


@app.post("/api/analyze-resume")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    current_user: models.User = Depends(get_current_user_detached),
):
    if not ai_client:
        raise HTTPException(status_code=500, detail="AI Model not initialized")
    try:
        # 1. Extract Text from PDF (off the event loop)
        content = await resume.read()
        resume_text = await asyncio.to_thread(_extract_pdf_text, content)
            
        # 2. Construct Prompt
        prompt = f"""
        You are an extremely strict, elite HR Recruiter and Technical Interviewer from a Top Tier Tech Company.
        Your job is to be BRUTALLY HONEST. Do not sugarcoat anything. If the candidate is bad, say it.
        
        Analyze the following Resume against the Job Description (JD).
        
        RESUME:
        {resume_text[:4000]}
        
        JOB DESCRIPTION:
        {job_description[:2000]}
        
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
        
        # 3. Generate Response (in a worker thread so other users aren't blocked)
        return _parse_model_json(await generate_text(prompt, json_mode=True))

    except Exception as e:
        logger.error(f"Resume Analysis Failed: {e}")
        raise HTTPException(status_code=500, detail="Resume analysis failed. Please try again.")

RATE = 16000
CHUNK = 1024
LANGUAGE_CODE = "en-US"

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}

@app.get("/")
def get():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html = f.read().replace("__GOOGLE_CLIENT_ID__", GOOGLE_CLIENT_ID)
        return HTMLResponse(content=html)
    return HTMLResponse(content="<h1>Error: templates/index.html not found</h1>")

# Icons and the link-preview image (served from ./static)
STATIC_FILES = {
    "/favicon.svg": ("static/favicon.svg", "image/svg+xml"),
    "/favicon.ico": ("static/favicon.svg", "image/svg+xml"),
    "/apple-touch-icon.png": ("static/apple-touch-icon.png", "image/png"),
    "/og-image.png": ("static/og-image.png", "image/png"),
    # Floating interview companion panel
    "/static/companion.js": ("static/companion.js", "application/javascript"),
    "/static/companion.css": ("static/companion.css", "text/css"),
}


def _static_route(path, media_type):
    def handler():
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})
    return handler


for _url, (_path, _type) in STATIC_FILES.items():
    app.add_api_route(_url, _static_route(_path, _type), methods=["GET", "HEAD"], include_in_schema=False)


@app.get("/privacy")
async def privacy_policy():
    with open("templates/privacy.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Limits that keep the copilot prompt small (= faster answers)
COPILOT_RESUME_CHARS = 6000
COPILOT_JD_CHARS = 3000
COPILOT_HISTORY_TURNS = 6
COPILOT_HISTORY_ANSWER_CHARS = 400
COPILOT_REGENERATE_MAX_CHARS = 1000   # question text the client may send back with "regenerate"


def build_answer_prompt(text, history, ctx, force=False, variant=False, answer_all=False, from_conversation=False):
    """force: the candidate tapped Answer. answer_all: "Answer all" mode (answer everything heard).
    from_conversation: "On tap" mode, where the input is the recent conversation, not a single question."""
    history_text = ""
    if history:
        turns = history[-COPILOT_HISTORY_TURNS:]
        lines = [f"Interviewer: {q}\nSuggested: {a[:COPILOT_HISTORY_ANSWER_CHARS]}" for q, a in turns]
        history_text = ("PREVIOUS CONVERSATION (most recent last; 'Suggested' is what you suggested "
                        "the candidate say):\n" + "\n".join(lines))

    if from_conversation:
        decide = ("The candidate tapped 'Answer'. LATEST SPEECH is everything said since the last answer: the "
                  "interviewer and the candidate mixed, not labelled, oldest first. Find the MOST RECENT question "
                  "or request from the interviewer and answer that one. Use the earlier conversation as context: "
                  "it may describe the problem, constraints or follow-up parts that the latest question refers to "
                  "(e.g. 'how would you scale it?'). Never output NO_ANSWER.")
    elif force:
        decide = ("The candidate tapped 'Answer now', so treat the input as an interviewer question "
                  "and ALWAYS answer it. Never output NO_ANSWER.")
    elif answer_all:
        decide = ("Answer mode is 'Answer all': treat the input as something the interviewer said and answer it. "
                  "Output ONLY NO_ANSWER for empty noise or a meaningless fragment.")
    else:
        decide = ("If the input is NOT something the interviewer needs answered, output ONLY: NO_ANSWER\n"
                  "        That includes: noise or fragments; the candidate speaking (e.g. reading or\n"
                  "        paraphrasing the last 'Suggested' answer, or thinking aloud); the interviewer just\n"
                  "        acknowledging (\"okay\", \"great\", \"that makes sense\").")
    if variant:
        decide += ("\n        The candidate asked for a FRESH VERSION of this answer: answer it again with different "
                   "wording and, where the resume allows, a different example than the last 'Suggested' answer.")

    return f"""
        You help a job candidate in a LIVE interview for {ctx['company'] or 'this role'}. The candidate glances
        at your answer on a phone while talking, so it must be quick to read and natural to say out loud.
        Assume the candidate is nervous and may not know the topic well: use simple words and short sentences,
        and explain any technical term the first time you use it.
        Speak AS the candidate (first person). Never invent experience: anything about what the candidate
        did, built or achieved must come from the resume.

        RESUME:
        {ctx['resume'][:COPILOT_RESUME_CHARS]}

        JOB DESCRIPTION:
        {ctx['jd'][:COPILOT_JD_CHARS]}

        {history_text}

        LATEST SPEECH (live transcript, may contain recognition errors):
        "{text}"

        DECIDE:
        {decide}

        EXCEPTION FIRST: if the input is a greeting or small talk ("hi", "how are you?", "can you hear me?",
        "shall we start?"), your ENTIRE output is one line: SAY: <one short friendly sentence>. Nothing else.

        OUTPUT FORMAT (follow exactly, in this order, no other text):
        SAY: <what the candidate says first, as one spoken answer of 3-4 natural sentences, max 75 words.
        Technical / concept questions: sentence 1 answers directly in one line; the next sentences explain in
        plain words, as if to someone hearing it for the first time, how it works and why it matters (an
        everyday comparison helps). Behavioral and HR questions: sentence 1 is the opening, then the key
        specifics from the resume; no comparisons or analogies. Greetings and small talk: ONE short sentence,
        then stop (no talking points, CODE or PROJECT).>
        - <talking point, max 12 words>
        - <talking point, max 12 words>
        - <talking point, max 12 words>
        (3 to 5 talking points: the key facts to mention next; add to SAY, never repeat it)
        DESIGN:
        <ONLY for system design / architecture / low-level (OOP) design questions, and for "explain your
        project's architecture" questions. Everything else: write exactly NONE. Format is described below.>
        CODE:
        <For technical questions: one short code snippet (5-15 lines) in a ```fenced block``` (language from
        the JD, default Python), with a short comment on the key lines explaining in simple words what each
        step does. No prose outside the code block. Algorithm / DSA questions: START with one line
        "Time: O(...), Space: O(...)", then the code block. For behavioral, HR, small talk, or a topic code can't show, write exactly: NONE>
        PROJECT:
        <1-2 first-person sentences the candidate can add to connect this answer to a real project or job on
        the resume ("In my ... project, I used this to ..."), using only resume facts. If nothing on the
        resume genuinely relates, or the answer is already about the resume, write exactly: NONE>

        QUESTION TYPES:
        - Concept / knowledge questions ("what is X", "difference between X and Y", "how does X work",
          "why would you use X"): answer the concept itself, correctly and in general terms. Do NOT force the
          resume into SAY, the talking points or CODE. The resume link goes only in PROJECT.
        - Experience questions ("tell me about yourself / your project / your role", behavioral questions):
          base every section on the resume; PROJECT is NONE.
        - DSA / coding questions ("reverse a linked list", "two sum", "find the longest substring..."): SAY
          gives the approach (the pattern, e.g. two pointers / sliding window / BFS) and why it works; talking
          points cover edge cases and the brute-force vs optimal trade-off; CODE is the COMPLETE working
          solution; DESIGN is NONE.
        - SYSTEM DESIGN questions ("design a URL shortener / WhatsApp / Uber / rate limiter / notification
          system", "how would you build/scale X"): SAY is a 3-4 sentence overview of the design; talking
          points are the order you'd walk the interviewer through it; CODE is NONE unless a small snippet
          really helps; DESIGN uses exactly these Markdown parts, kept short and scannable:
            ### Clarify first
            - 2-3 questions to ask the interviewer before designing (scale, users, must-have features)
            ### Requirements
            - **Functional:** the 3-5 core features
            - **Non-functional:** e.g. latency, availability, consistency, durability
            - **Scale:** a quick back-of-the-envelope estimate (users, requests per second, storage)
            ### Architecture
            a ```mermaid``` block with the high-level diagram (rules below), then one short line per main
            component saying what it does
            ### Database
            which database(s) and why (SQL vs NoSQL), then a ```sql``` block with the 2-4 key tables and
            their main columns and indexes (or the key/document shape for NoSQL)
            ### APIs
            a ```http``` block with the 3-5 main endpoints, one per line, e.g.
            POST /api/v1/urls  {{"long_url": "..."}}  -> 201 {{"short_url": "..."}}
            ### Stress these
            - 4-6 points that impress: caching, sharding/partitioning, queues, replication, consistency
              trade-offs, failure handling, rate limiting, monitoring, as they apply to THIS system
        - LOW-LEVEL / OOP design questions ("design a parking lot / LRU cache / elevator classes"): DESIGN has
          ### Requirements, ### Classes (a ```mermaid``` classDiagram with the main classes, key fields and
          methods, and their relationships), and ### Patterns (design patterns used and why); CODE is the class
          skeleton with the key methods.
        - "Explain your project's architecture": DESIGN has only ### Architecture with a ```mermaid``` diagram
          of components the RESUME names (never invent components), then one line per component.
        - MERMAID RULES (or the diagram won't draw): start with "flowchart LR" (or "classDiagram" for
          classes); 5-12 nodes; node ids are letters and digits only; EVERY label in double quotes, e.g.
          A["Load balancer"] or DB[("Orders DB")]; arrows are --> with an optional plain label like
          -->|"writes"|; no styling, no click, no %% directives.
        - Every piece of code, SQL, API list or command goes in a ```fenced block``` with its language, never
          as a paragraph.

        RULES:
        - In SAY and in every talking point, bold the 1-2 most important keywords (skills, tools, numbers,
          outcomes) with **double asterisks**. Never bold whole sentences.
        - Behavioral questions ("tell me about a time..."): talking points follow STAR, starting with
          "Situation:", "Task:", "Action:", "Result:".
        - HR questions (why this company, strengths/weaknesses, career goals, notice period, relocation):
          tie the answer to the job description and company, stay honest and positive, never criticise
          past employers.
        - Salary or compensation questions: never invent a number or a current salary that is not in the
          resume. If the job description states a range, anchor toward its upper half; otherwise SAY asks
          about the budgeted range or uses the placeholder "[your target range]". Talking points cover total
          compensation (base, bonus, equity, benefits), flexibility, and the value the candidate brings.
        - Coding questions: SAY gives the approach in one line, then walks through the idea in plain words;
          talking points cover edge cases; CODE is the complete solution with comments.
        - Greetings or small talk ("how are you?", "can you hear me?"): output only one short SAY sentence.
        - Sound human and conversational, not like a textbook. No filler like "Great question".
        - Plain Markdown only. No LaTeX or $...$ math; write complexity like O(n log n).
        - If a follow-up refers to an earlier question, use the PREVIOUS CONVERSATION.
        """


# --- Echo filter: ignore the candidate reading the suggested answer out loud ---
ECHO_OVERLAP_THRESHOLD = 0.4   # share of 3-word phrases that also appear in a recent answer
ECHO_RECENT_ANSWERS = 3


def _words(text: str):
    return re.findall(r"[a-z0-9']+", text.lower())


def _trigrams(words):
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


def is_echo(segment: str, recent_answers) -> bool:
    """True if this speech segment mostly repeats one of the recent suggested answers."""
    grams = _trigrams(_words(segment))
    if len(grams) < 2:  # too short to judge (fewer than 4 words)
        return False
    for answer in recent_answers:
        answer_grams = _trigrams(_words(answer))
        if answer_grams and len(grams & answer_grams) / len(grams) >= ECHO_OVERLAP_THRESHOLD:
            return True
    return False


def should_trigger_ai(text):
    # Relaxed filter: Let the AI decide, but filter out absolute noise
    text = text.strip()
    if len(text.split()) < 2: # Ignore single words like "Hello", "Okay" unless they are questions?
        return False
    return True

# Answer modes the copilot screen can choose: answer real questions on a pause ("smart", the default),
# answer everything heard ("all"), or only when the candidate taps Answer ("manual")
ANSWER_MODES = ("smart", "all", "manual")
# "On tap": everything heard since the last answer (or Skip) is kept for the next tap, with no time window.
# Only a safety cap: about 25 minutes of speech; beyond it the oldest part is dropped.
MANUAL_CONTEXT_CHARS = 24000
MANUAL_SHOWN_QUESTION_CHARS = 220     # how much of that conversation the answer card shows as the "question"

# Custom WebSocket close codes the frontend understands
WS_CLOSE_AUTH_FAILED = 4001
WS_CLOSE_LIMIT_REACHED = 4003
BILLING_INTERVAL_SECONDS = 10
# Question detection: answer after this much silence, merging fragments spoken before it...
QUESTION_PAUSE_SECONDS = 1.2
# ...but never hold a question longer than this, even if speech continues.
QUESTION_MAX_WAIT_SECONDS = 8

def _authenticate_ws_user(token: str):
    """Validate the token with a short-lived DB session. Returns (user_id, email, remaining_seconds)."""
    db = SessionLocal()
    try:
        user = get_current_user(token, db)
        return user.id, user.email, _remaining_seconds(user)
    finally:
        db.close()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    # Accept first so we can send a meaningful close code the browser can read
    await websocket.accept()

    if not token:
        await websocket.close(code=WS_CLOSE_AUTH_FAILED)
        return

    try:
        user_id, user_email, remaining = await asyncio.to_thread(_authenticate_ws_user, token)
    except Exception:
        await websocket.close(code=WS_CLOSE_AUTH_FAILED)
        return

    if remaining <= 0:
        await websocket.send_text(json.dumps({"type": "limit_reached"}))
        await websocket.close(code=WS_CLOSE_LIMIT_REACHED)
        return

    logger.info(f"Client connected: {user_email}")
    user_context = get_user_context(user_id)

    # --- Server-side billing: charge for the time this session is open ---
    session_start = time.monotonic()
    billed_seconds = 0

    async def bill_elapsed():
        """Charge any not-yet-billed elapsed seconds. Returns remaining seconds, or None if nothing to bill."""
        nonlocal billed_seconds
        to_bill = int(time.monotonic() - session_start) - billed_seconds
        if to_bill <= 0:
            return None
        billed_seconds += to_bill
        return await asyncio.to_thread(_charge_usage, user_id, to_bill)

    async def billing_loop():
        try:
            while True:
                await asyncio.sleep(BILLING_INTERVAL_SECONDS)
                remaining = await bill_elapsed()
                if remaining is None:
                    continue
                await websocket.send_text(json.dumps({"type": "credits", "remaining_seconds": remaining}))
                if remaining <= 0:
                    logger.info(f"Credit limit reached, closing session: {user_email}")
                    await websocket.send_text(json.dumps({"type": "limit_reached"}))
                    await websocket.close(code=WS_CLOSE_LIMIT_REACHED)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Billing loop error: {e}")

    audio_queue = queue.Queue()
    conversation_history = []
    loop = asyncio.get_running_loop()
    stop_event = threading.Event()

    def request_generator():
        while not stop_event.is_set():
            try:
                # Use a short timeout to allow checking stop_event
                data = audio_queue.get(timeout=0.1)
                if data is None:
                    return
                yield speech.StreamingRecognizeRequest(audio_content=data)
            except queue.Empty:
                continue
                
    def speech_recognition_thread():
        # Keep reconnecting until stopped
        while not stop_event.is_set():
            try:
                client = speech.SpeechClient(credentials=credentials)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=RATE,
                    language_code=LANGUAGE_CODE,
                    enable_automatic_punctuation=True,
                )
                streaming_config = speech.StreamingRecognitionConfig(
                    config=config,
                    interim_results=True,
                )

                # Get a fresh generator for this session
                requests = request_generator()
                
                # This blocks until stream ends (limit or error)
                responses = client.streaming_recognize(streaming_config, requests)
                
                for response in responses:
                    if stop_event.is_set():
                        break
                        
                    if not response.results:
                        continue
                        
                    result = response.results[0]
                    if not result.alternatives:
                        continue
                        
                    transcript = result.alternatives[0].transcript
                    is_final = result.is_final
                    
                    message = {
                        "type": "transcript",
                        "transcript": transcript,
                        "is_final": is_final
                    }
                    
                    if not stop_event.is_set():
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text(json.dumps(message)), loop
                        )

                    # Hand every result to the event loop; it waits for a pause in speech
                    # and merges fragments into one question before asking the AI.
                    if not stop_event.is_set():
                        loop.call_soon_threadsafe(on_speech_result, transcript, is_final)
                        
            except Exception as e:
                # Log but don't crash, retry loop will catch unless stopped
                if "400" in str(e) or "out of range" in str(e):
                     logger.error(f"Speech API Error (will retry): {e}")
                else:
                     logger.error(f"Speech thread error: {e}")
                # Back off briefly so a persistent failure can't spin the CPU / flood logs.
                # wait() returns immediately if the session is stopped.
                stop_event.wait(1)
                
            if not stop_event.is_set():
                 logger.info("Restarting speech stream...")
            
    # --- Question detection: wait for a pause, merge fragments, answer one at a time ---
    pending_segments = []        # final transcript pieces not yet sent to the AI
    pending_since = None         # when the first pending piece arrived
    flush_handle = None          # scheduled "pause detected" callback
    answer_lock = asyncio.Lock() # one AI answer at a time, in order
    answer_tasks = set()
    answer_counter = 0
    last_heard_text = ""         # most recent speech, used by the "Answer now" button
    answer_mode = "smart"        # smart | all | manual (set by the client with "set_mode")
    heard_log = []               # manual mode: [(time, text)] heard since the last answer or Skip

    def recent_answers():
        return [a for _, a in conversation_history[-ECHO_RECENT_ANSWERS:]]

    def remember_heard(text):
        """Manual mode: keep everything heard since the last answer or Skip (up to MANUAL_CONTEXT_CHARS)."""
        heard_log.append((time.monotonic(), text))
        while len(heard_log) > 1 and sum(len(t) for _, t in heard_log) > MANUAL_CONTEXT_CHARS:
            heard_log.pop(0)

    def set_mode(mode):
        nonlocal answer_mode
        if mode not in ANSWER_MODES or mode == answer_mode:
            return
        answer_mode = mode
        heard_log.clear()   # a fresh start in the new mode
        logger.info(f"Answer mode '{mode}': {user_email}")

    def on_speech_result(transcript, is_final):
        """Runs on the event loop for every interim/final speech result."""
        nonlocal pending_since, flush_handle
        if is_final and transcript.strip():
            if not pending_segments:
                pending_since = time.monotonic()
            pending_segments.append(transcript.strip())
        if not pending_segments:
            return
        # Any speech (even interim) means the person is still talking: restart the pause timer,
        # but never hold a question for longer than QUESTION_MAX_WAIT_SECONDS.
        if flush_handle:
            flush_handle.cancel()
        waited = time.monotonic() - pending_since
        delay = 0 if waited >= QUESTION_MAX_WAIT_SECONDS else QUESTION_PAUSE_SECONDS
        flush_handle = loop.call_later(delay, flush_pending)

    def take_pending_segments():
        nonlocal flush_handle, pending_since
        if flush_handle:
            flush_handle.cancel()
            flush_handle = None
        segments = list(pending_segments)
        pending_segments.clear()
        pending_since = None
        return segments

    def start_answer(text, force=False, variant=False, answer_all=False, from_conversation=False, shown_question=None):
        task = asyncio.create_task(answer_question(text, force, variant, answer_all, from_conversation, shown_question))
        answer_tasks.add(task)
        task.add_done_callback(answer_tasks.discard)

    def without_echo(segments):
        # Drop segments where the candidate is reading the suggested answer out loud
        kept = [s for s in segments if not is_echo(s, recent_answers())]
        if len(kept) < len(segments):
            logger.info(f"Ignored {len(segments) - len(kept)} segment(s) repeating the suggested answer")
        return kept

    def flush_pending():
        nonlocal last_heard_text
        text = " ".join(without_echo(take_pending_segments())).strip()
        if text:
            last_heard_text = text
        if stop_event.is_set():
            return
        if answer_mode == "manual":
            # "On tap": just remember it; nothing is answered until the candidate taps Answer
            if text:
                remember_heard(text)
            return
        if not should_trigger_ai(text):
            return
        start_answer(text, answer_all=(answer_mode == "all"))

    def answer_now():
        """'Answer now' button: answer what was just heard without waiting for a pause."""
        nonlocal last_heard_text
        if answer_mode == "manual":
            # Answer the latest interviewer question within the conversation heard since the last answer
            pending = " ".join(without_echo(take_pending_segments())).strip()
            if pending:
                remember_heard(pending)
            conversation = " ".join(t for _, t in heard_log).strip()
            if not conversation:
                asyncio.create_task(send_json({
                    "type": "notice",
                    "message": "Nothing new heard since the last answer. Wait for the question, then tap Answer.",
                }))
                return
            heard_log.clear()
            last_heard_text = conversation
            shown = conversation if len(conversation) <= MANUAL_SHOWN_QUESTION_CHARS \
                else "…" + conversation[-MANUAL_SHOWN_QUESTION_CHARS:]
            start_answer(conversation, from_conversation=True, shown_question=shown)
            return
        text = " ".join(take_pending_segments()).strip() or last_heard_text
        if not text:
            asyncio.create_task(send_json({
                "type": "notice",
                "message": "Nothing heard yet. Wait for the question, then tap Answer now.",
            }))
            return
        last_heard_text = text
        start_answer(text, force=True)

    def regenerate(question):
        """'Regenerate' button: write a fresh answer to a question that was already answered."""
        text = str(question or "").strip()[:COPILOT_REGENERATE_MAX_CHARS]
        if not text:
            return
        if answer_tasks:
            # One answer at a time: a double-click or a busy session must not queue duplicate model calls
            asyncio.create_task(send_json({
                "type": "notice",
                "message": "Still writing an answer. Try again in a moment.",
            }))
            return
        start_answer(text, force=True, variant=True)

    def skip_current():
        """'Skip' button: stop the answer being written and forget unanswered speech.
        In "On tap" mode it also forgets the stored conversation: the next Answer uses only what comes after."""
        take_pending_segments()
        heard_log.clear()
        for task in list(answer_tasks):
            task.cancel()

    async def send_json(payload):
        if stop_event.is_set():
            return
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass  # client already gone

    async def answer_question(text, force=False, variant=False, answer_all=False, from_conversation=False,
                              shown_question=None):
        nonlocal answer_counter
        shown = shown_question or text   # what the answer card shows as the question
        answer_id = None
        started = False
        try:
            async with answer_lock:
                if stop_event.is_set():
                    return
                if not ai_client:
                    await send_json({"type": "status", "message": "Ready"})
                    return
                answer_counter += 1
                answer_id = answer_counter
                await send_json({"type": "status", "message": "Listening..."})

                prompt = build_answer_prompt(text, conversation_history, user_context, force=force, variant=variant,
                                             answer_all=answer_all, from_conversation=from_conversation)
                answer = ""
                try:
                    async for piece in stream_copilot_answer(prompt):
                        if stop_event.is_set():
                            return
                        answer += piece
                        if not started:
                            # Hold back the first few characters until we know it isn't "NO_ANSWER"
                            head = answer.strip().strip('"')
                            if "NO_ANSWER".startswith(head) or head.startswith("NO_ANSWER"):
                                continue
                            started = True
                            await send_json({"type": "answer_start", "id": answer_id, "question": shown})
                            await send_json({"type": "answer_chunk", "id": answer_id, "text": answer})
                        else:
                            await send_json({"type": "answer_chunk", "id": answer_id, "text": piece})
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Gemini error: {e}")
                    if started:
                        await send_json({"type": "answer_chunk", "id": answer_id,
                                         "text": "\n\n_(Answer interrupted. Tap Answer now to retry.)_"})
                        await send_json({"type": "answer_end", "id": answer_id})
                    else:
                        await send_json({"type": "status", "message": "Ready"})
                    return

                if not started:
                    head = answer.strip().strip('"')
                    if not head or head.startswith("NO_ANSWER"):
                        logger.info(f"AI declined to answer: '{text}'")
                        await send_json({"type": "status", "message": "Ready"})
                        return
                    # Short answer that never got past the NO_ANSWER check
                    started = True
                    await send_json({"type": "answer_start", "id": answer_id, "question": shown})
                    await send_json({"type": "answer_chunk", "id": answer_id, "text": answer})

                await send_json({"type": "answer_end", "id": answer_id})
                if variant and conversation_history and conversation_history[-1][0] == shown:
                    conversation_history[-1] = (shown, answer.strip())   # replace, don't repeat, the turn
                else:
                    conversation_history.append((shown, answer.strip()))
        except asyncio.CancelledError:
            # Skipped by the user (or the session ended)
            if started:
                await send_json({"type": "answer_skipped", "id": answer_id})
            else:
                await send_json({"type": "status", "message": "Ready"})
            raise

    t = threading.Thread(target=speech_recognition_thread, daemon=True)
    t.start()
    billing_task = asyncio.create_task(billing_loop())

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                logger.info(f"Client disconnected: {user_email}")
                break
            if message.get("bytes"):
                audio_queue.put(message["bytes"])
            elif message.get("text"):
                # Control messages from the copilot screen
                try:
                    payload = json.loads(message["text"])
                    if not isinstance(payload, dict):
                        payload = {}
                except Exception:
                    payload = {}
                command = payload.get("type")
                if command == "set_mode":
                    set_mode(str(payload.get("mode") or ""))
                elif command == "answer_now":
                    answer_now()
                elif command == "regenerate":
                    regenerate(payload.get("question"))
                elif command == "skip":
                    skip_current()
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {user_email}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        stop_event.set()
        audio_queue.put(None)
        if flush_handle:
            flush_handle.cancel()
        for task in list(answer_tasks):
            task.cancel()
        billing_task.cancel()
        try:
            await billing_task
        except BaseException:
            pass
        # Charge the final partial interval
        try:
            await bill_elapsed()
        except Exception as e:
            logger.error(f"Final billing failed for {user_email}: {e}")
        # Join off the event loop so other users aren't blocked
        await asyncio.to_thread(t.join, 5)


# =====================================================================
# MOCK INTERVIEW
# An AI interviewer asks questions (researched for the company on the web, plus resume and JD
# questions), listens to spoken answers, asks follow-ups, and scores the interview at the end.
# Uses the same Copilot minutes, billed only while the interview runs (preparing and scoring are free).
# =====================================================================
MOCK_QUESTION_COUNTS = {"short": 5, "standard": 8}
MOCK_MAX_FOLLOWUPS_PER_QUESTION = 1
MOCK_ANSWER_PAUSE_SECONDS = 6          # silence that ends an answer (longer than the copilot: people pause to think)
MOCK_MIN_WORDS_FOR_AUTO_END = 6        # shorter answers wait for the "Done" button
MOCK_MAX_ANSWER_SECONDS = 240          # an answer is closed after 4 minutes
MOCK_MIN_WORDS_FOR_FOLLOWUP = 8
MOCK_RESEARCH_CACHE_SECONDS = 3 * 24 * 3600
MOCK_RESEARCH_CACHE = {}               # (company, role) lowercased -> (saved_at, research); in memory
MOCK_FILLERS = ("um", "umm", "uh", "uhh", "hmm", "basically", "actually", "you know", "i mean", "kind of", "sort of")
WS_CLOSE_MOCK_DONE = 1000


def run_speech_stream(audio_queue, stop_event, on_result, once=False):
    """Blocking: stream PCM chunks from audio_queue to Speech-to-Text until stop_event is set.

    Calls on_result(transcript, is_final) from this thread and reconnects when Google ends the stream
    (once=True: return when the first stream ends, e.g. after a None in the queue).
    """
    def requests_iter():
        while not stop_event.is_set():
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if data is None:
                return
            yield speech.StreamingRecognizeRequest(audio_content=data)

    while not stop_event.is_set():
        try:
            client = speech.SpeechClient(credentials=credentials)
            config = speech.StreamingRecognitionConfig(
                config=speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=RATE, language_code=LANGUAGE_CODE, enable_automatic_punctuation=True,
                ),
                interim_results=True,
            )
            for response in client.streaming_recognize(config, requests_iter()):
                if stop_event.is_set():
                    break
                if response.results and response.results[0].alternatives:
                    result = response.results[0]
                    on_result(result.alternatives[0].transcript, result.is_final)
            if once:
                return
        except Exception as e:
            logger.error(f"Speech stream error (will retry): {e}")
            if once:
                return
            stop_event.wait(1)


def _answer_stats(text: str, seconds: float) -> dict:
    words = _words(text)
    padded = " " + " ".join(words) + " "
    fillers = {f: padded.count(f" {f} ") for f in MOCK_FILLERS}
    fillers = {f: n for f, n in fillers.items() if n}
    wpm = round(len(words) / (seconds / 60)) if seconds >= 5 and words else None
    return {"words": len(words), "seconds": round(seconds), "wpm": wpm, "fillers": fillers}


async def research_company_questions(company: str, role: str) -> dict:
    """Web search (Gemini grounded on Google Search) for questions candidates report at this company."""
    empty = {"text": "", "sources": [], "search_html": ""}
    if not company:
        return empty
    key = (company.lower(), role.lower())
    cached = MOCK_RESEARCH_CACHE.get(key)
    if cached and time.time() - cached[0] < MOCK_RESEARCH_CACHE_SECONDS:
        return cached[1]
    prompt = (
        f"Search for interview questions that candidates report being asked at {company} "
        f"for {role or 'software'} roles, preferring India and the last 2-3 years. List up to 12 distinct "
        f"questions, one per line, each starting with its round in brackets, for example [Technical], "
        f"[Behavioral], [HR], [System design], [Coding]. Only questions candidates reported; no commentary."
    )
    try:
        resp = await ai_client.aio.models.generate_content(
            model=model_name, contents=prompt,
            config=genai_types.GenerateContentConfig(tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]),
        )
    except Exception as e:
        logger.error(f"Mock research failed for {company}/{role}: {e}")
        return empty
    gm = resp.candidates[0].grounding_metadata if resp.candidates else None
    sources, seen = [], set()
    for chunk in (getattr(gm, "grounding_chunks", None) or []):
        web = getattr(chunk, "web", None)
        if web and web.uri and (web.title or "") not in seen:
            seen.add(web.title or "")
            sources.append({"title": web.title or "source", "uri": web.uri})
    sep = getattr(gm, "search_entry_point", None)
    research = {"text": (resp.text or "")[:4000], "sources": sources[:6],
                "search_html": (sep.rendered_content if sep and sep.rendered_content else "")}
    if research["text"]:
        MOCK_RESEARCH_CACHE[key] = (time.time(), research)
    return research


async def plan_mock_questions(ctx: dict, company: str, role: str, research_text: str, count: int) -> list:
    practice = [q for q in (ctx.get("practice") or []) if q][:PRACTICE_MAX_QUESTIONS]
    practice_rule = ""
    if practice:
        practice_rule = ("\n        - The candidate wants to PRACTISE these questions from a real interview they just had. "
                         "Include each of them (reworded naturally, after the warm-up), with \"kind\": \"practice\":\n"
                         + "\n".join(f"          * {q}" for q in practice))
    prompt = f"""
        You are preparing a realistic SPOKEN mock interview for a job candidate.
        COMPANY: {company or 'not given'}
        ROLE: {role or 'see the job description'}

        RESUME:
        {ctx['resume'][:COPILOT_RESUME_CHARS]}

        JOB DESCRIPTION:
        {ctx['jd'][:COPILOT_JD_CHARS] or 'not given'}

        QUESTIONS CANDIDATES REPORT BEING ASKED AT THIS COMPANY (web search, may be incomplete or empty):
        {research_text or 'none found'}

        Write exactly {count} questions, in the order a real interviewer would ask them:
        - Question 1: a short warm-up (e.g. "Tell me about yourself") tailored to the role.
        - About 40% adapted from the reported company questions, if there are any (set "from_research": true).
        - About 30% about specific projects, numbers or claims on the resume.
        - The rest on the job description's key skills, including one behavioral question.
        - The last question is an HR question (why this company, career goals, notice period, etc.).
        - It is spoken: never ask the candidate to write code. For coding or system design, ask them to
          explain their approach out loud ("How would you approach ...? Walk me through it.").
        - Each question at most 30 words, natural spoken English, no numbering.{practice_rule}

        OUTPUT JSON ONLY:
        {{"questions": [{{"question": "...", "kind": "warmup|company|resume|technical|behavioral|hr|practice", "from_research": false}}]}}
        """
    data = _parse_model_json(await generate_text_reliably(prompt, json_mode=True))
    questions = []
    for q in (data.get("questions") or [])[:count]:
        text = str(q.get("question") or "").strip()
        if text:
            questions.append({"question": text[:400], "kind": str(q.get("kind") or "technical")[:20],
                              "from_research": bool(q.get("from_research"))})
    return questions


async def decide_mock_followup(company: str, role: str, question: str, answer: str) -> str:
    at_company = f" at {company}" if company else ""
    prompt = f"""
        You are a friendly but sharp interviewer{at_company} for a {role or 'professional'} role.
        QUESTION: {question}
        CANDIDATE'S ANSWER (speech transcript, may contain recognition errors): "{answer[:3000]}"

        Decide whether to ask ONE follow-up question. Ask one only if the answer was vague, skipped the
        candidate's own actions or the result, or made a claim worth probing (a number, tool or decision).
        Ask nothing if the answer was already specific and complete, or the candidate said they don't know.
        OUTPUT JSON ONLY: {{"follow_up": "<one short spoken question, max 25 words, or empty string>"}}
        """
    try:
        config = genai_types.GenerateContentConfig(response_mime_type="application/json")
        if COPILOT_THINKING_LEVEL:
            config.thinking_config = genai_types.ThinkingConfig(thinking_level=COPILOT_THINKING_LEVEL)
        resp = await ai_client.aio.models.generate_content(model=COPILOT_MODEL, contents=prompt, config=config)
        return str(_parse_model_json(resp.text or "{}").get("follow_up") or "").strip()[:300]
    except Exception as e:
        logger.warning(f"Mock follow-up decision failed ({e}); moving on")
        return ""


async def score_mock_interview(ctx: dict, company: str, role: str, turns: list) -> dict:
    lines = []
    for i, turn in enumerate(turns, 1):
        for j, ex in enumerate(turn["exchanges"]):
            who = f"Q{i} [{turn['kind']}]" if j == 0 else "  Follow-up"
            lines.append(f"{who}: {ex['question']}")
            lines.append(f"  Candidate: {ex['answer'] if ex['answer'] else '(skipped / no answer)'}")
    interview_text = "\n".join(lines)
    at_company = f"at {company}" if company else ""
    prompt = f"""
        You are an experienced interview coach. Score this mock interview for a {role or 'professional'} role
        {at_company}. Be honest and specific: strict but kind.

        RESUME (use only these facts in better answers):
        {ctx['resume'][:COPILOT_RESUME_CHARS]}

        INTERVIEW (answers are speech transcripts and may contain recognition errors; judge the content):
        {interview_text}

        For each of the {len(turns)} questions (Q1..Q{len(turns)}, including its follow-up) return:
        - score: 0-10 (0 if skipped or blank)
        - feedback: at most 2 sentences on what worked and what didn't
        - missing: the single most important thing missing, max 12 words ("" if nothing)
        - better_answer: how the candidate could answer, first person, 60-110 words, only resume facts;
          use STAR (situation, task, action, result) for behavioral questions
        Also return overall_score 0-100 (85+ only for excellent), verdict (one direct sentence),
        strengths (2-3 short points) and improvements (2-3 short, actionable points).
        Plain text inside the JSON strings, no markdown.

        OUTPUT JSON ONLY:
        {{"overall_score": 0, "verdict": "", "strengths": [""], "improvements": [""],
          "answers": [{{"score": 0, "feedback": "", "missing": "", "better_answer": ""}}]}}
        """
    return _parse_model_json(await generate_text_reliably(prompt, json_mode=True))


class MockInterviewSession:
    """One mock interview over a WebSocket. All state changes happen on the event loop."""

    def __init__(self, websocket, user_id, user_email, ctx, company, role, count):
        self.ws, self.user_id, self.user_email, self.ctx = websocket, user_id, user_email, ctx
        self.company, self.role, self.count = company, role, count
        self.loop = asyncio.get_running_loop()
        self.stage = "preparing"     # preparing -> ready -> asking <-> answering -> thinking -> scoring -> done
        self.questions = []
        self.index = -1              # current main question
        self.followups = 0           # follow-ups asked for the current main question
        self.current_text = ""       # the question being answered (main or follow-up)
        self.turns = []              # [{question, kind, exchanges: [{question, answer, seconds, skipped}]}]
        self.answer_parts, self.answer_started = [], None
        self.pause_handle = self.max_handle = None
        self.tasks = set()
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.speech_thread = None
        self.session_start = None
        self.billed_seconds = 0
        self.billing_task = None

    # ---------- helpers ----------
    async def send(self, payload):
        if self.stage == "done":
            return
        try:
            await self.ws.send_text(json.dumps(payload))
        except Exception:
            pass

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def cancel_timers(self):
        for handle in (self.pause_handle, self.max_handle):
            if handle:
                handle.cancel()
        self.pause_handle = self.max_handle = None

    # ---------- billing (same rules as the copilot) ----------
    async def bill_elapsed(self):
        if self.session_start is None:
            return None
        to_bill = int(time.monotonic() - self.session_start) - self.billed_seconds
        if to_bill <= 0:
            return None
        self.billed_seconds += to_bill
        return await asyncio.to_thread(_charge_usage, self.user_id, to_bill)

    def stop_billing(self):
        """Freeze the clock: nothing after this moment is charged."""
        if self.billing_task:
            self.billing_task.cancel()
            self.billing_task = None

    async def billing_loop(self):
        try:
            while True:
                await asyncio.sleep(BILLING_INTERVAL_SECONDS)
                remaining = await self.bill_elapsed()
                if remaining is None:
                    continue
                await self.send({"type": "credits", "remaining_seconds": remaining})
                if remaining <= 0:
                    await self.send({"type": "notice", "message": "Your minutes ran out, so the interview ends here."})
                    self.spawn(self.end("limit"))
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Mock billing loop error: {e}")

    # ---------- flow ----------
    async def prepare(self):
        research = {"text": "", "sources": [], "search_html": ""}
        try:
            if self.company:
                await self.send({"type": "status", "stage": "researching",
                                 "message": f"Searching the web for questions {self.company} asks… (about 20–30 seconds)"})
            research = await research_company_questions(self.company, self.role)
            await self.send({"type": "status", "stage": "planning", "message": "Writing your interview plan…"})
            self.questions = await plan_mock_questions(self.ctx, self.company, self.role, research["text"], self.count)
            if self.questions:
                self.ctx.pop("practice", None)   # practice questions are used in one mock only
        except Exception as e:
            logger.error(f"Mock prepare failed for {self.user_email}: {e}")
            self.questions = []
        if not self.questions:
            await self.send({"type": "error", "message": "Couldn't prepare the interview. Please try again."})
            await self.close()
            return
        self.stage = "ready"
        await self.send({
            "type": "plan_ready", "total": len(self.questions),
            "researched": sum(1 for q in self.questions if q["from_research"]),
            "sources": research["sources"], "search_html": research["search_html"],
        })

    def start(self):
        if self.stage != "ready":
            return
        self.stage = "starting"
        self.session_start = time.monotonic()   # billing starts with the first question
        self.speech_thread = threading.Thread(
            target=run_speech_stream, args=(self.audio_queue, self.stop_event, self.on_speech_from_thread), daemon=True)
        self.speech_thread.start()
        self.billing_task = asyncio.create_task(self.billing_loop())
        logger.info(f"Mock interview started: {self.user_email} ({len(self.questions)} questions)")
        self.spawn(self.next_question())

    def finished(self) -> bool:
        return self.stage in ("scoring", "done")

    async def ask(self, text, follow_up=False):
        if self.finished():
            return
        self.stage = "asking"
        self.current_text = text
        q = self.questions[self.index]
        await self.send({"type": "question", "index": self.index, "total": len(self.questions), "text": text,
                         "kind": q["kind"], "follow_up": follow_up, "from_research": q["from_research"]})

    async def next_question(self):
        if self.finished():
            return
        self.index += 1
        self.followups = 0
        if self.index >= len(self.questions):
            await self.end("finished")
            return
        q = self.questions[self.index]
        self.turns.append({"question": q["question"], "kind": q["kind"], "exchanges": []})
        await self.ask(q["question"])

    def on_listening(self):
        """The interviewer finished speaking (browser voice): the candidate's turn starts."""
        if self.stage != "asking":
            return
        self.stage = "answering"
        self.answer_parts = []
        self.answer_started = time.monotonic()
        self.cancel_timers()
        self.max_handle = self.loop.call_later(MOCK_MAX_ANSWER_SECONDS, self.finish_answer)

    def on_speech_from_thread(self, transcript, is_final):
        if not self.stop_event.is_set():
            self.loop.call_soon_threadsafe(self.on_speech, transcript, is_final)

    def on_speech(self, transcript, is_final):
        if self.stage != "answering":
            return   # speech while the interviewer talks or thinks doesn't count
        self.spawn(self.send({"type": "transcript", "transcript": transcript, "is_final": is_final}))
        if is_final and transcript.strip():
            self.answer_parts.append(transcript.strip())
        # Still talking: restart the pause timer
        if self.pause_handle:
            self.pause_handle.cancel()
        self.pause_handle = self.loop.call_later(MOCK_ANSWER_PAUSE_SECONDS, self.on_pause)

    def on_pause(self):
        self.pause_handle = None
        if self.stage == "answering" and len(_words(" ".join(self.answer_parts))) >= MOCK_MIN_WORDS_FOR_AUTO_END:
            self.finish_answer()

    def finish_answer(self, skipped=False):
        if self.stage != "answering" and not (skipped and self.stage == "asking"):
            return
        self.cancel_timers()
        self.stage = "thinking"
        answer = "" if skipped else " ".join(self.answer_parts).strip()
        seconds = time.monotonic() - self.answer_started if (self.answer_started and not skipped) else 0
        self.answer_started = None
        self.turns[-1]["exchanges"].append(
            {"question": self.current_text, "answer": answer, "seconds": seconds, "skipped": not answer})
        self.spawn(self.after_answer(answer))

    async def after_answer(self, answer):
        if self.stage != "thinking":
            return   # the interview was ended during this answer: no follow-up needed
        await self.send({"type": "status", "stage": "thinking", "message": "Thinking…"})
        follow_up = ""
        if self.followups < MOCK_MAX_FOLLOWUPS_PER_QUESTION and len(_words(answer)) >= MOCK_MIN_WORDS_FOR_FOLLOWUP:
            follow_up = await decide_mock_followup(self.company, self.role, self.current_text, answer)
        if self.stage != "thinking":
            return   # the interview ended meanwhile
        if follow_up:
            self.followups += 1
            await self.ask(follow_up, follow_up=True)
        else:
            await self.next_question()

    async def end(self, reason):
        if self.stage in ("preparing", "ready", "scoring", "done"):
            if self.stage in ("preparing", "ready"):
                await self.close()   # nothing was asked yet: just leave
            return
        if self.stage == "answering":
            self.finish_answer()      # keep what was said so far
        self.stage = "scoring"
        self.cancel_timers()
        self.stop_audio()
        self.stop_billing()
        try:
            remaining = await self.bill_elapsed()
            if remaining is not None:
                await self.send({"type": "credits", "remaining_seconds": remaining})
        except Exception as e:
            logger.error(f"Mock final billing failed for {self.user_email}: {e}")
        self.session_start = None     # scoring time is free
        logger.info(f"Mock interview ended ({reason}): {self.user_email}, {len(self.turns)} questions")

        if not any(ex["answer"] for t in self.turns for ex in t["exchanges"]):
            await self.send({"type": "report", "empty": True, "reason": reason})
            await self.close()
            return
        await self.send({"type": "status", "stage": "scoring", "message": "Scoring your interview…"})
        try:
            scored = await score_mock_interview(self.ctx, self.company, self.role, self.turns)
        except Exception as e:
            logger.error(f"Mock scoring failed for {self.user_email}: {e}")
            scored = {}
        await self.send(self.build_report(scored if isinstance(scored, dict) else {}, reason))
        await self.close()

    def build_report(self, scored: dict, reason: str) -> dict:
        model_answers = scored.get("answers") or []
        answers = []
        for i, turn in enumerate(self.turns):
            text = " ".join(ex["answer"] for ex in turn["exchanges"] if ex["answer"]).strip()
            seconds = sum(ex["seconds"] for ex in turn["exchanges"])
            m = model_answers[i] if i < len(model_answers) and isinstance(model_answers[i], dict) else {}
            try:
                score = max(0, min(10, int(m.get("score", 0))))
            except (TypeError, ValueError):
                score = 0
            answers.append({
                "question": turn["question"], "kind": turn["kind"],
                "follow_ups": [ex["question"] for ex in turn["exchanges"][1:]],
                "answer": text, "skipped": not text, "score": score if text else 0,
                "feedback": str(m.get("feedback") or ""), "missing": str(m.get("missing") or ""),
                "better_answer": str(m.get("better_answer") or ""),
                **_answer_stats(text, seconds),
            })
        try:
            overall = max(0, min(100, int(scored.get("overall_score", 0))))
        except (TypeError, ValueError):
            overall = 0

        def short_list(value):
            return [str(x) for x in value][:3] if isinstance(value, list) else []

        return {
            "type": "report", "empty": False, "reason": reason, "scored": bool(scored),
            "company": self.company, "role": self.role, "overall_score": overall,
            "verdict": str(scored.get("verdict") or ""),
            "strengths": short_list(scored.get("strengths")), "improvements": short_list(scored.get("improvements")),
            "answers": answers,
        }

    def stop_audio(self):
        self.stop_event.set()
        self.audio_queue.put(None)

    async def close(self):
        if self.stage == "done":
            return
        self.stage = "done"
        try:
            await self.ws.close(code=WS_CLOSE_MOCK_DONE)
        except Exception:
            pass

    async def shutdown(self):
        """The socket is gone: stop everything and bill the last partial interval."""
        self.cancel_timers()
        self.stop_audio()
        self.stop_billing()
        for task in list(self.tasks):
            task.cancel()
        try:
            await self.bill_elapsed()
        except Exception as e:
            logger.error(f"Mock final billing failed for {self.user_email}: {e}")
        if self.speech_thread:
            await asyncio.to_thread(self.speech_thread.join, 5)

    def on_command(self, command: str):
        if command == "start":
            self.start()
        elif command == "listening":
            self.on_listening()
        elif command == "done":
            self.finish_answer()
        elif command == "skip":
            self.finish_answer(skipped=True)
        elif command == "end":
            self.spawn(self.end("ended"))


@app.websocket("/ws/mock")
async def mock_interview_ws(websocket: WebSocket, token: str = None, company: str = "", role: str = "",
                            length: str = "standard"):
    await websocket.accept()
    if not token:
        await websocket.close(code=WS_CLOSE_AUTH_FAILED)
        return
    try:
        user_id, user_email, remaining = await asyncio.to_thread(_authenticate_ws_user, token)
    except Exception:
        await websocket.close(code=WS_CLOSE_AUTH_FAILED)
        return
    if remaining <= 0:
        await websocket.send_text(json.dumps({"type": "limit_reached"}))
        await websocket.close(code=WS_CLOSE_LIMIT_REACHED)
        return
    ctx = get_user_context(user_id)
    if not ai_client or not ctx["resume"]:
        message = "Please upload your resume first." if ai_client else "AI is not available right now."
        await websocket.send_text(json.dumps({"type": "error", "message": message}))
        await websocket.close()
        return

    session = MockInterviewSession(websocket, user_id, user_email, ctx, (company or "").strip()[:80],
                                   (role or "").strip()[:80],
                                   MOCK_QUESTION_COUNTS.get(length, MOCK_QUESTION_COUNTS["standard"]))
    logger.info(f"Mock interview connected: {user_email} ({session.company or '-'} / {session.role or '-'})")
    session.spawn(session.prepare())
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes"):
                # The browser sends silence while the interviewer speaks, so only real answers are heard
                if session.stage in ("asking", "answering", "thinking"):
                    session.audio_queue.put(message["bytes"])
            elif message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    payload = {}
                if isinstance(payload, dict):
                    session.on_command(str(payload.get("type") or ""))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Mock WebSocket error: {e}")
    finally:
        await session.shutdown()


# ---------- Free audio test (before the copilot starts) ----------
# Streams up to AUDIO_TEST_SECONDS of audio to Speech-to-Text and sends back what it heard.
# Never billed: it doesn't touch the user's minutes. A daily cap keeps it from being used as free transcription.
AUDIO_TEST_SECONDS = 15
AUDIO_TEST_DAILY_LIMIT = 15


@app.websocket("/ws/audiotest")
async def audio_test_ws(websocket: WebSocket, token: str = None):
    await websocket.accept()
    if not token:
        await websocket.close(code=WS_CLOSE_AUTH_FAILED)
        return
    try:
        user_id, user_email, _ = await asyncio.to_thread(_authenticate_ws_user, token)
    except Exception:
        await websocket.close(code=WS_CLOSE_AUTH_FAILED)
        return
    if not use_daily_allowance("audiotest", user_id, AUDIO_TEST_DAILY_LIMIT):
        await websocket.send_text(json.dumps({"type": "error",
                                              "message": "You've run a lot of audio tests today. Your audio worked before, so you're good to go."}))
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    async def send(payload):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass

    def on_result(transcript, is_final):
        if not stop_event.is_set():
            asyncio.run_coroutine_threadsafe(
                send({"type": "transcript", "transcript": transcript, "is_final": is_final}), loop)

    speech_task = asyncio.create_task(asyncio.to_thread(run_speech_stream, audio_queue, stop_event, on_result, True))
    deadline = loop.time() + AUDIO_TEST_SECONDS
    logger.info(f"Audio test started: {user_email}")
    try:
        await send({"type": "ready", "seconds": AUDIO_TEST_SECONDS})
        while True:
            left = deadline - loop.time()
            if left <= 0:
                break
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=left)
            except asyncio.TimeoutError:
                break
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes"):
                audio_queue.put(message["bytes"])
            elif message.get("text") and '"stop"' in message["text"]:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Audio test error: {e}")
    finally:
        # Give Speech-to-Text a moment to return the last words, then stop
        audio_queue.put(None)
        try:
            await asyncio.wait_for(asyncio.shield(speech_task), timeout=3)
        except Exception:
            pass
        stop_event.set()
        await send({"type": "done"})
        try:
            await websocket.close()
        except Exception:
            pass


# ---------- Company interview research (dashboard tool) ----------
# Step 1 (shared, cached): Gemini + Google Search collects interview experiences candidates posted online.
# Step 2 (per user): organise those notes into rounds / reported questions / tips, and add questions the
# JD and resume make likely. Reported questions come ONLY from the web notes; predictions are labelled.
RESEARCH_CACHE_SECONDS = 3 * 24 * 3600
RESEARCH_CACHE = {}                     # (company, role) lowercased -> (saved_at, notes dict); in memory
RESEARCH_DAILY_LIMIT = 10               # new searches per user per day (cached results don't count)
PREVIEW_ANSWER_DAILY_LIMIT = 60         # "Show answer" per user per day
RESEARCH_MAX_REPORTED = 20
RESEARCH_MAX_LIKELY = 8


async def search_interview_experiences(company: str, role: str) -> dict:
    """Grounded web search for interview experiences at this company. Returns {text, sources, search_html}."""
    prompt = (
        f"Search the web for interview experiences that candidates have shared about interviewing at {company} "
        f"for {role or 'software engineering'} roles (for example on GeeksforGeeks, Glassdoor, LeetCode Discuss, "
        f"AmbitionBox, Reddit, Medium), preferring India and the last 2-3 years.\n"
        f"Write plain notes with three parts:\n"
        f"PROCESS: the interview rounds candidates describe, in order, one line each (what happens in that round).\n"
        f"QUESTIONS: up to {RESEARCH_MAX_REPORTED + 5} distinct questions candidates report being asked, one per "
        f"line, each starting with its round in brackets, e.g. [Online assessment], [Technical], [Coding], "
        f"[System design], [Managerial], [HR].\n"
        f"TIPS: up to 6 short tips candidates give (difficulty, what interviewers focus on, mistakes to avoid).\n"
        f"Only include what candidates actually reported. If you find little or nothing, say so plainly."
    )
    config = genai_types.GenerateContentConfig(tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())])
    # Ride out Gemini's short "429 Resource exhausted" limits: retry, then try the copilot model's own quota
    attempts = [(model_name, 0), (model_name, 3), (COPILOT_MODEL, 2)]
    for n, (model, pause) in enumerate(attempts):
        await asyncio.sleep(pause)
        try:
            resp = await ai_client.aio.models.generate_content(model=model, contents=prompt, config=config)
            break
        except Exception as e:
            if not _is_rate_limited(e) or n == len(attempts) - 1:
                raise
            logger.warning(f"Company research rate-limited on '{model}', retrying")
    gm = resp.candidates[0].grounding_metadata if resp.candidates else None
    sources, seen = [], set()
    for chunk in (getattr(gm, "grounding_chunks", None) or []):
        web = getattr(chunk, "web", None)
        if web and web.uri and (web.title or "") not in seen:
            seen.add(web.title or "")
            sources.append({"title": web.title or "source", "uri": web.uri})
    sep = getattr(gm, "search_entry_point", None)
    return {"text": (resp.text or "")[:8000], "sources": sources[:8],
            "search_html": (sep.rendered_content if sep and sep.rendered_content else "")}


def _clean_list(items, fields, limit, max_chars=400):
    out = []
    for item in items or []:
        if isinstance(item, dict):
            row = {f: str(item.get(f) or "").strip()[:max_chars] for f in fields}
            if row[fields[0]]:
                out.append(row)
        if len(out) >= limit:
            break
    return out


def _tip_text(tip) -> str:
    """Tips should be strings, but the model sometimes returns {"tip": "..."}."""
    if isinstance(tip, dict):
        tip = tip.get("tip") or tip.get("text") or next((v for v in tip.values() if isinstance(v, str)), "")
    return str(tip or "").strip()[:200]


class ResearchRequest(BaseModel):
    company: str
    role: str = ""
    jd: str = ""


@app.post("/api/company-research")
async def company_research(data: ResearchRequest, current_user: models.User = Depends(get_current_user_detached)):
    company = data.company.strip()[:80]
    role = data.role.strip()[:80]
    if not company:
        raise HTTPException(status_code=400, detail="Enter the company name.")
    if not ai_client:
        raise HTTPException(status_code=503, detail="AI is not available right now.")

    key = (company.lower(), role.lower())
    cached = RESEARCH_CACHE.get(key)
    if cached and time.time() - cached[0] < RESEARCH_CACHE_SECONDS:
        notes = cached[1]
    else:
        if not use_daily_allowance("research", current_user.id, RESEARCH_DAILY_LIMIT):
            raise HTTPException(status_code=429, detail="You've researched a lot of companies today. Please try again tomorrow.")
        try:
            notes = await search_interview_experiences(company, role)
        except Exception as e:
            logger.error(f"Company research search failed for {company}/{role}: {e}")
            raise HTTPException(status_code=502, detail="Couldn't search the web right now. Please try again in a minute.")
        if notes["text"]:
            RESEARCH_CACHE[key] = (time.time(), notes)

    ctx = get_user_context(current_user.id)
    jd = data.jd.strip()[:CONTEXT_MAX_JD_CHARS] or ctx.get("jd") or ""
    prompt = f"""
        You are helping a job candidate prepare for an interview at {company} for the role: {role or 'not given'}.

        WEB RESEARCH NOTES (what candidates reported online; may be thin or empty):
        {notes['text'] or 'nothing found'}

        JOB DESCRIPTION: {jd[:COPILOT_JD_CHARS] or 'not given'}
        CANDIDATE RESUME: {(ctx.get('resume') or 'not given')[:COPILOT_RESUME_CHARS]}

        Return JSON with:
        - "found": true if the notes contain real reported interview information for this company, else false.
        - "summary": 1-2 plain sentences about this interview (rounds, difficulty, focus). If little was found,
          say that honestly.
        - "process": the rounds in order, each {{"round": "...", "what": "one sentence"}}. ONLY from the notes; [] if none.
        - "reported": up to {RESEARCH_MAX_REPORTED} questions candidates reported, each {{"question": "...",
          "round": "Coding|Technical|System design|Behavioral|HR|Online assessment|Managerial"}}. ONLY questions that
          appear in the notes (you may tidy the wording). NEVER invent any. [] if none.
        - "likely": {RESEARCH_MAX_LIKELY} questions this candidate is likely to get, based on the job description's key
          skills and the projects on the resume, each {{"question": "...", "why": "max 12 words: which JD skill or
          resume project"}}. Don't repeat reported questions.
        - "tips": up to 6 short tips from the notes (max 20 words each). [] if none.

        OUTPUT JSON ONLY.
        """
    try:
        result = _parse_model_json(await generate_text_reliably(prompt, json_mode=True))
    except Exception as e:
        logger.error(f"Company research organise failed for {current_user.email}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't put the research together. Please try again.")

    reported = _clean_list(result.get("reported"), ("question", "round"), RESEARCH_MAX_REPORTED)
    logger.info(f"Company research: {current_user.email} -> {company}/{role or '-'} ({len(reported)} reported)")
    return {
        "company": company, "role": role,
        "found": bool(result.get("found")) and bool(reported or result.get("process")),
        "summary": str(result.get("summary") or "").strip()[:500],
        "process": _clean_list(result.get("process"), ("round", "what"), 8),
        "reported": reported,
        "likely": _clean_list(result.get("likely"), ("question", "why"), RESEARCH_MAX_LIKELY),
        "tips": [t for t in (_tip_text(t) for t in (result.get("tips") or [])) if t][:6],
        "sources": [s for s in notes["sources"] if str(s.get("uri", "")).startswith("https://")],
        "search_html": notes["search_html"],
    }


class PreviewAnswerRequest(BaseModel):
    question: str
    company: str = ""     # the researched company/JD, which may differ from the copilot setup
    jd: str = ""


# ---------------------------------------------------------------------------
# JOB DESCRIPTION FROM A LINK: the user pastes a job-post URL instead of the text
# ---------------------------------------------------------------------------
import html as _html
import socket
import ipaddress
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin, parse_qs

JD_URL_DAILY_LIMIT = 40
JD_URL_CACHE = {}                  # url -> (saved_at, result); in memory
JD_URL_CACHE_SECONDS = 24 * 3600
JD_URL_MAX_BYTES = 3_000_000
JD_URL_MIN_CHARS = 200             # less than this is not a real job description
JD_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}


class JdUrlRequest(BaseModel):
    url: str


class JdUrlError(Exception):
    """A problem we can explain to the user in plain words."""


def _check_public_url(url: str) -> str:
    """Only http(s) links to public internet hosts (never our own server or a private network)."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise JdUrlError("That doesn't look like a web link. It should start with https://")
    if p.port not in (None, 80, 443):
        raise JdUrlError("That link can't be opened.")
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except socket.gaierror:
        raise JdUrlError("That website doesn't exist. Check the link.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise JdUrlError("That link can't be opened.")
    return url


def _fetch_public(url: str) -> tuple:
    """GET a public page, checking every redirect hop. Returns (final_url, text)."""
    for _ in range(5):
        _check_public_url(url)
        r = requests.get(url, headers=JD_FETCH_HEADERS, timeout=10, allow_redirects=False, stream=True)
        if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, r.headers.get("location", ""))
            r.close()
            continue
        if r.status_code >= 400:
            r.close()
            raise JdUrlError(f"blocked:{r.status_code}")
        body = b""
        for part in r.iter_content(65536):
            body += part
            if len(body) > JD_URL_MAX_BYTES:
                break
        r.close()
        # requests guesses ISO-8859-1 when the server names no charset; job sites are almost always UTF-8
        charset = r.encoding if "charset" in r.headers.get("content-type", "").lower() else "utf-8"
        return url, body.decode(charset or "utf-8", errors="replace")
    raise JdUrlError("That link redirects too many times.")


class _TextExtractor(HTMLParser):
    """HTML -> readable text: drops scripts/menus, keeps line breaks at block tags."""
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "template", "iframe", "button", "form"}
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def _html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:
        pass
    text = "".join(parser.out).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line and line != "-")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _find_job_posting(markup: str):
    """The schema.org JobPosting most job boards embed for Google Jobs (LinkedIn, Greenhouse, Lever, Workday, ...)."""
    for raw in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', markup, re.S | re.I):
        try:
            data = json.loads(raw.strip(), strict=False)
        except Exception:
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                kind = item.get("@type")
                if kind == "JobPosting" or (isinstance(kind, list) and "JobPosting" in kind):
                    return item
                stack.extend(v for v in item.values() if isinstance(v, (list, dict)))
    return None


def _posting_result(posting: dict) -> dict:
    org = posting.get("hiringOrganization")
    company = org.get("name") if isinstance(org, dict) else (org if isinstance(org, str) else "")
    desc = _html_to_text(_html.unescape(str(posting.get("description") or "")))
    parts = [desc]
    for field, label in (("qualifications", "Qualifications"), ("responsibilities", "Responsibilities"),
                         ("skills", "Skills"), ("experienceRequirements", "Experience")):
        value = posting.get(field)
        if isinstance(value, str) and value.strip() and value.strip()[:80] not in desc:
            parts.append(f"{label}:\n{_html_to_text(_html.unescape(value))}")
    return {"company": str(company or "").strip(), "role": str(posting.get("title") or "").strip(),
            "jd": "\n\n".join(p for p in parts if p).strip()}


def _known_board_url(url: str):
    """Job boards whose public pages are empty shells but expose the post another way."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if host.endswith("linkedin.com"):
        job_id = parse_qs(p.query).get("currentJobId", [None])[0]
        m = re.search(r"/jobs/view/(?:[^/]*?-)?(\d{6,})", p.path)
        job_id = job_id or (m.group(1) if m else None)
        if job_id and job_id.isdigit():
            return f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    m = re.match(r"^/([^/]+)/([0-9a-f-]{20,})", p.path)
    if host == "jobs.lever.co" and m:
        return f"https://api.lever.co/v0/postings/{m.group(1)}/{m.group(2)}"
    m = re.search(r"^/([^/]+)/jobs/(\d+)", p.path)
    if host in ("boards.greenhouse.io", "job-boards.greenhouse.io") and m:
        return f"https://boards-api.greenhouse.io/v1/boards/{m.group(1)}/jobs/{m.group(2)}"
    return None


def _read_known_board(api_url: str):
    final, body = _fetch_public(api_url)
    host = urlparse(api_url).hostname
    if host == "api.lever.co":
        d = json.loads(body)
        lists = "\n\n".join(f"{x.get('text', '')}:\n{_html_to_text(x.get('content', ''))}" for x in d.get("lists") or [])
        jd = "\n\n".join(t for t in (d.get("descriptionPlain") or _html_to_text(d.get("description", "")), lists,
                                     d.get("additionalPlain") or "") if t)
        slug = urlparse(api_url).path.split("/")[3]
        return {"company": slug.replace("-", " ").title(), "role": d.get("text", ""), "jd": jd.strip()}
    if host == "boards-api.greenhouse.io":
        d = json.loads(body)
        return {"company": d.get("company_name", ""), "role": d.get("title", ""),
                "jd": _html_to_text(_html.unescape(d.get("content", "")))}
    # LinkedIn guest page: a fragment with the title, company and the description markup
    posting = _find_job_posting(body)
    if posting:
        return _posting_result(posting)
    m = re.search(r'class="[^"]*(?:show-more-less-html__markup|description__text)[^"]*"[^>]*>(.*?)</(?:div|section)>',
                  body, re.S)
    title = re.search(r'class="[^"]*top-card-layout__title[^"]*"[^>]*>(.*?)<', body, re.S)
    company = re.search(r'class="[^"]*topcard__org-name-link[^"]*"[^>]*>(.*?)<', body, re.S)
    return {"company": _html.unescape(company.group(1)).strip() if company else "",
            "role": _html.unescape(title.group(1)).strip() if title else "",
            "jd": _html_to_text(m.group(1)) if m else ""}


def _page_title(markup: str) -> str:
    m = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]*)"', markup, re.I) or \
        re.search(r"<title[^>]*>(.*?)</title>", markup, re.I | re.S)
    return _html.unescape(m.group(1)).strip()[:200] if m else ""


async def _ai_pick_jd(page_text: str, title: str) -> dict:
    """The page has the job somewhere among menus and other listings: let the AI copy out just the job."""
    prompt = f"""This is the text of a web page that should contain ONE job posting. Page title: {title or "(none)"}

PAGE TEXT:
{page_text[:30000]}

Return JSON: {{"found": true|false, "company": "...", "role": "...", "jd": "..."}}
- "jd": the job description COPIED from the page, as written: about the role, responsibilities, requirements,
  skills, qualifications, experience, location, benefits. Keep its headings and bullet points ("- ").
  Do NOT summarise or reword it, and do NOT add anything that is not on the page.
- Leave out menus, cookie banners, "similar jobs", sign-up prompts and other site text.
- "found": false if the page has no job description (a login wall, an error, a search page, a list of many jobs).
- "company" / "role": as on the page, or "" if not shown."""
    try:   # fast model first: copying a long description is mostly output speed
        resp = await ai_client.aio.models.generate_content(
            model=COPILOT_MODEL, contents=prompt,
            config=genai_types.GenerateContentConfig(response_mime_type="application/json"))
        reply = resp.text or ""
    except Exception as e:
        logger.info(f"JD pick on '{COPILOT_MODEL}' failed ({e}); using the main model")
        reply = await generate_text_reliably(prompt, json_mode=True)
    data = _parse_model_json(reply)
    if not isinstance(data, dict) or not data.get("found"):
        return {"company": "", "role": "", "jd": ""}
    return {"company": str(data.get("company") or ""), "role": str(data.get("role") or ""),
            "jd": str(data.get("jd") or "")}


async def _ai_read_url(url: str) -> dict:
    """Last resort for pages we can't read ourselves (built in the browser, or blocking servers): Gemini's URL reader."""
    # Gemini refuses to copy pages it opens word for word (finish reason RECITATION), so ask for full notes
    # in its own words that keep every concrete detail exactly.
    prompt = (f"Open this job posting: {url}\n"
              "Write up the job in your own words, completely, in exactly this format:\n"
              "COMPANY: <company or blank>\nROLE: <job title or blank>\nJD:\n"
              "About the role: <2-3 sentences>\nResponsibilities:\n- <every responsibility listed>\n"
              "Requirements:\n- <every required skill, years of experience, degree>\nNice to have:\n- <preferred skills>\n"
              "Location and other details: <location, work mode, anything else listed>\n"
              "Keep every technology, tool, skill and number EXACTLY as the page names it. Include nothing that is "
              "not on the page. If the page can't be opened or has no job description, reply only: NOT_FOUND")
    config = genai_types.GenerateContentConfig(tools=[genai_types.Tool(url_context=genai_types.UrlContext())])
    # The fast copilot model copies a long page in seconds; the main model is the rate-limit fallback
    attempts = [(COPILOT_MODEL, 0), (COPILOT_MODEL, 3), (model_name, 2)]
    for n, (model, pause) in enumerate(attempts):
        await asyncio.sleep(pause)
        try:
            resp = await ai_client.aio.models.generate_content(model=model, contents=prompt, config=config)
            break
        except Exception as e:
            if not _is_rate_limited(e) or n == len(attempts) - 1:
                raise
    # Only trust the reply if Gemini actually managed to open the page
    meta = resp.candidates[0].url_context_metadata if resp.candidates else None
    opened = any("SUCCESS" in str(getattr(u, "url_retrieval_status", "")) for u in (getattr(meta, "url_metadata", None) or []))
    text = (resp.text or "").replace("**", "").strip()
    split = re.split(r"^\s*JD\s*:\s*", text, maxsplit=1, flags=re.M)
    if not opened or "NOT_FOUND" in text[:40] or len(split) < 2:
        return {"company": "", "role": "", "jd": ""}
    head, jd = split
    company = re.search(r"COMPANY:[ \t]*(.*)", head)
    role = re.search(r"ROLE:[ \t]*(.*)", head)
    return {"company": company.group(1).strip() if company else "", "role": role.group(1).strip() if role else "",
            "jd": jd.strip()}


def _tidy_jd_result(result: dict, source: str) -> dict:
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in result.get("jd", "").split("\n")]
    jd = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return {"jd": jd[:CONTEXT_MAX_JD_CHARS], "company": result.get("company", "").strip()[:80],
            "role": result.get("role", "").strip()[:80], "source": source}


async def read_job_post(url: str) -> dict:
    """Try the cheap, exact ways first (structured data, known boards), then the AI."""
    known = _known_board_url(url)
    if known:
        try:
            r = await asyncio.to_thread(_read_known_board, known)
            if len(r["jd"]) >= JD_URL_MIN_CHARS:
                return _tidy_jd_result(r, "page")
        except Exception as e:
            logger.info(f"JD board API failed for {known}: {e}")
    page = None
    try:
        final, page = await asyncio.to_thread(_fetch_public, url)
    except JdUrlError as e:
        if not str(e).startswith("blocked:"):
            raise
    except requests.RequestException as e:
        logger.info(f"JD fetch failed for {url}: {e}")
    if page:
        posting = _find_job_posting(page)
        if posting:
            r = _posting_result(posting)
            if len(r["jd"]) >= JD_URL_MIN_CHARS:
                return _tidy_jd_result(r, "page")
        text = _html_to_text(page)
        if len(text) >= JD_URL_MIN_CHARS:
            r = await _ai_pick_jd(text, _page_title(page))
            if len(r["jd"]) >= JD_URL_MIN_CHARS:
                return _tidy_jd_result(r, "page")
    r = await _ai_read_url(url)
    if len(r["jd"]) >= JD_URL_MIN_CHARS:
        return _tidy_jd_result(r, "ai")
    raise JdUrlError("Couldn't read a job description from that page (it may need a login). "
                     "Open the link, copy the description and paste it here.")


@app.post("/api/jd/from-url")
async def jd_from_url(data: JdUrlRequest, current_user: models.User = Depends(get_current_user_detached)):
    url = data.url.strip()[:2000]
    if re.match(r"^[a-z][a-z0-9+.-]*:", url, re.I) and not re.match(r"^https?://", url, re.I):
        raise HTTPException(status_code=422, detail="That doesn't look like a web link. It should start with https://")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url      # "www.company.com/jobs/1" pasted without the https://
    cached = JD_URL_CACHE.get(url)
    if cached and time.time() - cached[0] < JD_URL_CACHE_SECONDS:
        return cached[1]
    if not ai_client:
        raise HTTPException(status_code=503, detail="AI is not available right now.")
    if not use_daily_allowance("jd_url", current_user.id, JD_URL_DAILY_LIMIT):
        raise HTTPException(status_code=429, detail="You've read a lot of job links today. Please paste the description instead.")
    try:
        result = await read_job_post(url)
    except JdUrlError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Reading job link failed for {url}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't open that link right now. Paste the description instead.")
    JD_URL_CACHE[url] = (time.time(), result)
    return result


@app.post("/api/preview-answer")
async def preview_answer(data: PreviewAnswerRequest, current_user: models.User = Depends(get_current_user_detached)):
    """A suggested answer to one practice question, in the copilot's SAY / points / CODE / PROJECT format."""
    question = data.question.strip()[:COPILOT_REGENERATE_MAX_CHARS]
    if not question:
        raise HTTPException(status_code=400, detail="No question given.")
    if not ai_client:
        raise HTTPException(status_code=503, detail="AI is not available right now.")
    if not use_daily_allowance("preview", current_user.id, PREVIEW_ANSWER_DAILY_LIMIT):
        raise HTTPException(status_code=429, detail="You've opened a lot of answers today. Please try again tomorrow.")
    ctx = dict(get_user_context(current_user.id))
    if data.company.strip():
        ctx["company"] = data.company.strip()[:CONTEXT_MAX_COMPANY_CHARS]
    if data.jd.strip():
        ctx["jd"] = data.jd.strip()[:CONTEXT_MAX_JD_CHARS]
    try:
        text = ""
        async for piece in stream_copilot_answer(build_answer_prompt(question, [], ctx, force=True)):
            text += piece
    except Exception as e:
        logger.error(f"Preview answer failed for {current_user.email}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't write the answer right now. Please try again.")
    return {"answer": text.strip()}


# ---------- Prep plan: an hour-by-hour (or day-by-day) plan until the interview ----------
PLAN_DAILY_LIMIT = 20
PLAN_MAX_HOURS = 168          # one week
PLAN_DAY_SPLIT_HOURS = 30     # more time than this: plan by day; less: one timeline (with sleep if it crosses a night)


class PlanRequest(BaseModel):
    company: str
    role: str = ""
    jd: str = ""
    hours: float
    now_local: str = ""          # e.g. "Fri 14 Nov, 3:40 PM" (the user's clock), so nights can be planned as sleep
    interview_local: str = ""
    process: list[str] = []      # "Round: what happens"
    reported: list[str] = []     # questions candidates reported at this company
    likely: list[str] = []       # questions the JD/resume make likely
    tips: list[str] = []


def _clip_list(items, limit, chars=300):
    return [str(i).strip()[:chars] for i in items if str(i).strip()][:limit]


@app.post("/api/prep-plan")
async def prep_plan(data: PlanRequest, current_user: models.User = Depends(get_current_user_detached)):
    company = data.company.strip()[:80]
    if not company:
        raise HTTPException(status_code=400, detail="Enter the company name.")
    hours = max(0.5, min(float(data.hours or 0), PLAN_MAX_HOURS))
    if not ai_client:
        raise HTTPException(status_code=503, detail="AI is not available right now.")
    if not use_daily_allowance("plan", current_user.id, PLAN_DAILY_LIMIT):
        raise HTTPException(status_code=429, detail="You've made a lot of plans today. Please try again tomorrow.")

    ctx = get_user_context(current_user.id)
    by_day = hours > PLAN_DAY_SPLIT_HOURS
    days = max(1, round(hours / 24)) if by_day else 1
    lines = lambda items: "\n".join(f"- {i}" for i in items) or "- none"
    clock = ""
    if data.now_local.strip() and data.interview_local.strip():
        clock = (f" It is now {data.now_local.strip()[:40]} (candidate's local time); the interview is at "
                 f"{data.interview_local.strip()[:40]}. Never schedule study between 11:30 PM and 6:30 AM unless the "
                 f"interview is within 8 hours: put a \"break\" block titled \"Sleep\" there instead.")
    timing = (
        f"The interview is in about {hours:g} hours ({days} day{'s' if days > 1 else ''}). Plan BY DAY: set \"day\" "
        f"from 1 to {days} on every block. Leave time for sleep, meals and breaks: at most 6-7 hours of study a day. "
        f"The last day before the interview is lighter: revision and one mock, no new big topics." + clock
        if by_day else
        f"The interview is in about {hours:g} hours. Plan straight through from now, \"day\": 1 on every block. "
        f"The blocks run back to back from now; their minutes must add up to at most {int(hours * 60) - 15} "
        f"minutes (keep the last 15 minutes free). Include a 5-10 minute break about every 90 minutes if there is "
        f"time, and meals if the plan crosses lunch or dinner." + clock
        + (" Time is very short: only the highest-value topics, no breaks." if hours <= 2 else "")
    )
    prompt = f"""
        You are an expert interview coach. Make a realistic, structured PREP PLAN for a candidate interviewing at
        {company} for: {data.role.strip()[:80] or 'see the job description'}.

        TIMING: {timing}

        INTERVIEW ROUNDS CANDIDATES REPORTED:
        {lines(_clip_list(data.process, 8))}
        QUESTIONS CANDIDATES REPORTED AT THIS COMPANY:
        {lines(_clip_list(data.reported, 20))}
        QUESTIONS LIKELY FROM THE JOB DESCRIPTION / RESUME:
        {lines(_clip_list(data.likely, 8))}
        TIPS FROM CANDIDATES:
        {lines(_clip_list(data.tips, 6, 200))}
        JOB DESCRIPTION: {(data.jd.strip() or ctx.get('jd') or 'not given')[:COPILOT_JD_CHARS]}
        CANDIDATE RESUME: {(ctx.get('resume') or 'not given')[:COPILOT_RESUME_CHARS]}

        HOW TO PRIORITISE:
        1. Topics behind questions candidates reported at this company, in the order of the rounds.
        2. JD skills the resume does NOT show (gaps the interviewer may probe).
        3. The candidate's own resume projects (they WILL be asked to explain them) and an intro pitch.
        4. Behavioral / HR answers in STAR form if the company asks them.
        Be specific: name the exact topics, patterns and questions. No generic advice like "study hard".
        Include ONE "mock" block (PrapAI's AI mock interview) when there are 2+ hours, near the end.

        Return JSON ONLY:
        {{"headline": "one sentence: the strategy for this time",
          "blocks": [{{"day": 1, "minutes": 60, "type": "study|practice|mock|break|review",
                      "title": "max 8 words", "why": "max 18 words: why this matters at {company}",
                      "tasks": ["2-4 concrete tasks, max 16 words each"],
                      "questions": ["0-4 questions to practise in this block, copied from the lists above when possible"]}}],
          "before_call": ["3-5 short things to do in the last 15 minutes"]}}
        """
    try:
        result = _parse_model_json(await generate_text_reliably(prompt, json_mode=True))
    except Exception as e:
        logger.error(f"Prep plan failed for {current_user.email}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't make the plan right now. Please try again.")

    blocks = []
    for b in (result.get("blocks") or [])[:40]:
        if not isinstance(b, dict) or not str(b.get("title") or "").strip():
            continue
        try:
            minutes = int(float(b.get("minutes") or 0))
        except (TypeError, ValueError):
            minutes = 0
        kind = str(b.get("type") or "study").lower()
        try:
            day = int(b.get("day") or 1)
        except (TypeError, ValueError):
            day = 1
        blocks.append({
            "day": min(max(day, 1), days) if by_day else 1,
            "minutes": min(max(minutes, 5), 600 if kind == "break" else 240),   # a night's sleep is a long break
            "type": kind if kind in ("study", "practice", "mock", "break", "review") else "study",
            "title": str(b.get("title")).strip()[:80],
            "why": str(b.get("why") or "").strip()[:200],
            "tasks": _clip_list(b.get("tasks") or [], 5, 200),
            "questions": _clip_list(b.get("questions") or [], 4, 300),
        })
    if not blocks:
        raise HTTPException(status_code=502, detail="Couldn't make the plan right now. Please try again.")
    logger.info(f"Prep plan: {current_user.email} -> {company} in {hours:g}h, {len(blocks)} blocks")
    return {"hours": hours, "by_day": by_day, "headline": str(result.get("headline") or "").strip()[:300],
            "blocks": blocks, "before_call": [t for t in (_tip_text(t) for t in (result.get("before_call") or [])) if t][:5]}


# ---------- Tailored resume: rewrite for one job, then a PDF ----------
# The model only rephrases and reorders. Server-side checks keep it honest: a rewritten line that adds a number
# or a job-description keyword that appears nowhere in the original resume is replaced by the original line.
TAILOR_DAILY_LIMIT = 15
TAILOR_MAX_SECTIONS = 8
TAILOR_MAX_ENTRIES = 12
TAILOR_MAX_BULLETS = 10
TAILOR_LINE_CHARS = 600


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:[.,]\d+)?", str(text or "")))


def _keyword_in(keyword: str, text_norm: str) -> bool:
    k = _norm(keyword)
    if not k:
        return False
    # whole-word match for short keywords ("Go", "C"), substring for longer ones ("PostgreSQL")
    if len(k) <= 3:
        return re.search(rf"(?<![a-z0-9+#]){re.escape(k)}(?![a-z0-9+#])", text_norm) is not None
    if k in text_norm:
        return True
    # plural / singular: "Databases" is shown by "database", "API" by "APIs"
    # (whole word, so "Redis" is NOT shown by "redirect")
    return len(k) >= 4 and k.endswith("s") and re.search(rf"(?<![a-z0-9+#]){re.escape(k[:-1])}(?![a-z0-9+#])", text_norm) is not None


def _added_facts(text: str, allowed_numbers: set, resume_norm: str, keywords: list):
    """(numbers, keywords) in text that the resume doesn't back up."""
    return (sorted(_numbers(text) - allowed_numbers),
            [k for k in keywords if _keyword_in(k, _norm(text)) and not _keyword_in(k, resume_norm)])


def check_rewrite(original: str, tailored: str, allowed_numbers: set, resume_norm: str, keywords: list):
    """Return (text_to_use, note). Falls back to the original when the rewrite adds facts.
    allowed_numbers: for a bullet, only the numbers of its OWN original line (so a metric can't move to
    another bullet); for the summary, any number in the resume."""
    original = str(original or "").strip()[:TAILOR_LINE_CHARS]
    tailored = str(tailored or "").strip()[:TAILOR_LINE_CHARS]
    if not tailored:
        return original, ""
    numbers, added = _added_facts(tailored, allowed_numbers, resume_norm, keywords)
    if numbers:
        return original, f"Kept your original: the rewrite added a number ({', '.join(numbers[:3])}) that isn't in this line of your resume."
    if added:
        return original, f"Kept your original: the rewrite mentioned {', '.join(added[:3])}, which your resume doesn't show."
    return tailored, ""


def check_summary(original: str, tailored: str, resume_numbers: set, resume_norm: str, keywords: list):
    """Like check_rewrite, but drops only the sentences that add facts, so one bad sentence doesn't cost the
    whole tailored summary."""
    original = str(original or "").strip()[:TAILOR_LINE_CHARS]
    tailored = str(tailored or "").strip()[:TAILOR_LINE_CHARS]
    if not tailored:
        return original, ""
    kept, dropped = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", tailored):
        numbers, added = _added_facts(sentence, resume_numbers, resume_norm, keywords)
        (dropped if numbers or added else kept).append((sentence, numbers + added))
    if not dropped:
        return tailored, ""
    if not kept:
        return original, ""
    reasons = sorted({r for _, rs in dropped for r in rs})
    return " ".join(s for s, _ in kept), f"Removed a sentence that mentioned {', '.join(reasons[:3])}, which your resume doesn't show."


@app.post("/api/resume/tailor")
async def tailor_resume(
    jd: str = Form(...),
    company: str = Form(""),
    role: str = Form(""),
    resume_text: str = Form(""),
    resume_file: UploadFile = File(None),
    current_user: models.User = Depends(get_current_user_detached),
):
    jd = jd.strip()[:CONTEXT_MAX_JD_CHARS]
    if len(jd) < 40:
        raise HTTPException(status_code=400, detail="Paste the full job description (a few lines at least).")
    if resume_file is not None and resume_file.filename:
        try:
            resume = (await asyncio.to_thread(_extract_pdf_text, await resume_file.read())).strip()
        except Exception:
            raise HTTPException(status_code=400, detail="Couldn't read that PDF. Please upload a text-based PDF resume.")
    else:
        resume = (resume_text or get_user_context(current_user.id).get("resume") or "").strip()
    resume = resume[:CONTEXT_MAX_RESUME_CHARS]
    if len(resume) < 100:
        raise HTTPException(status_code=400, detail="Add your resume (PDF). If it's a scanned image, export it again as a text PDF.")
    if not ai_client:
        raise HTTPException(status_code=503, detail="AI is not available right now.")
    if not use_daily_allowance("tailor", current_user.id, TAILOR_DAILY_LIMIT):
        raise HTTPException(status_code=429, detail="You've tailored a lot of resumes today. Please try again tomorrow.")

    prompt = f"""
        You are an expert resume writer. Tailor this candidate's resume to ONE job, honestly.

        COMPANY: {company.strip()[:80] or 'not given'}   ROLE: {role.strip()[:80] or 'see the job description'}
        JOB DESCRIPTION:
        {jd}

        RESUME (plain text extracted from a PDF; rebuild its structure):
        {resume}

        STRICT RULES:
        - NEVER invent anything: no new numbers, tools, skills, employers, titles, dates, degrees or achievements.
          Only rephrase, tighten, reorder, and use the job description's wording for things the resume ALREADY shows.
        - Use a job-description keyword in a line ONLY if the resume already shows that exact skill or fact. Do NOT
          sprinkle in claims the resume doesn't make, such as "high-throughput", "low-latency", "microservices",
          "at scale", "system design", "mentoring" or a tool name, unless the resume says so. Unsupported keywords go
          in "gaps", not in the text. Lines that add them are rejected automatically.
        - Fix obvious typos and grammar (e.g. a missing "years" after a number) without changing any fact.
        - Keep EVERY section, entry and bullet from the resume (you may shorten weak bullets). Copy names, dates,
          locations, contact details and links exactly.
        - Bullets: start with a strong past-tense verb, keep every number from the original, max ~30 words.
          Order bullets inside each entry from most to least relevant to this job.
        - Summary: 2-3 lines aimed at this role, built only from resume facts. If the resume has no summary,
          set "original" to "" and still write one.
        - Skills: keep the same skills; put the ones this job asks for first. Never add a skill the resume doesn't show.

        Return JSON ONLY:
        {{"name": "...",
          "contact": ["email", "phone", "linkedin/github/portfolio as written", "city"],
          "summary": {{"original": "...", "tailored": "..."}},
          "sections": [{{"title": "Experience|Projects|Education|...",
                        "entries": [{{"heading": "job title / project / degree", "subheading": "company / tech / school",
                                     "location": "", "dates": "",
                                     "bullets": [{{"original": "exact original bullet", "tailored": "rewritten bullet",
                                                  "why": "max 10 words: what changed"}}]}}]}}],
          "skills": [{{"category": "Languages", "original": "as in resume", "tailored": "reordered"}}],
          "keywords": ["the 15 most important skills/keywords in the job description, short"],
          "gaps": [{{"skill": "a keyword the resume does NOT show", "advice": "max 25 words: add it only if true, and how"}}]}}
        Put skills in "skills", not in "sections".
        """
    try:
        data = _parse_model_json(await generate_text_reliably(prompt, json_mode=True))
    except Exception as e:
        logger.error(f"Tailor failed for {current_user.email}: {e}")
        raise HTTPException(status_code=502, detail="Couldn't tailor the resume right now. Please try again.")

    resume_norm = _norm(resume)
    resume_numbers = _numbers(resume)
    keywords = [str(k).strip()[:40] for k in (data.get("keywords") or []) if str(k).strip()][:20]
    reverted = 0

    def checked(original, tailored):
        nonlocal reverted
        allowed = _numbers(original) if str(original or "").strip() else resume_numbers
        text, note = check_rewrite(original, tailored, allowed, resume_norm, keywords)
        if note:
            reverted += 1
        return text, note

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    s_original = str(summary.get("original") or "").strip()[:TAILOR_LINE_CHARS]
    s_text, s_note = check_summary(s_original, summary.get("tailored"), resume_numbers, resume_norm, keywords)
    if s_note:
        reverted += 1
    sections = []
    for sec in (data.get("sections") or [])[:TAILOR_MAX_SECTIONS]:
        if not isinstance(sec, dict) or not str(sec.get("title") or "").strip():
            continue
        entries = []
        for ent in (sec.get("entries") or [])[:TAILOR_MAX_ENTRIES]:
            if not isinstance(ent, dict):
                continue
            bullets = []
            for b in (ent.get("bullets") or [])[:TAILOR_MAX_BULLETS]:
                if isinstance(b, str):
                    b = {"original": b, "tailored": b}
                if not isinstance(b, dict) or not str(b.get("original") or b.get("tailored") or "").strip():
                    continue
                original = str(b.get("original") or "").strip()[:TAILOR_LINE_CHARS]
                text, note = checked(original, b.get("tailored"))
                if not original:      # a bullet that exists only in the rewrite: keep only if it passed the checks
                    original = text if not note else ""
                if not original and not text:
                    continue
                dropped = sorted(_numbers(original) - _numbers(text)) if text and text != original else []
                bullets.append({"original": original, "tailored": text or original,
                                "why": "" if note else str(b.get("why") or "").strip()[:80], "note": note,
                                "warn": f"This version drops {', '.join(dropped[:4])} from your original. Tick Keep original if those numbers matter."
                                        if dropped else ""})
            entries.append({k: str(ent.get(k) or "").strip()[:120] for k in ("heading", "subheading", "location", "dates")}
                           | {"bullets": bullets})
        if entries:
            sections.append({"title": str(sec["title"]).strip()[:40], "entries": entries})
    skills = []
    for sk in (data.get("skills") or [])[:10]:
        if not isinstance(sk, dict):
            continue
        original = str(sk.get("original") or "").strip()[:TAILOR_LINE_CHARS]
        tailored = str(sk.get("tailored") or "").strip()[:TAILOR_LINE_CHARS]
        # reordering only: every item in the new list must already be in the resume
        items = [i.strip() for i in re.split(r"[,;|]", tailored) if i.strip()]
        kept = [i for i in items if _keyword_in(i, resume_norm)]
        text = ", ".join(kept) if kept else original
        if text or original:
            skills.append({"category": str(sk.get("category") or "").strip()[:40], "original": original or text,
                           "tailored": text or original})
    contact = [str(c).strip()[:120] for c in (data.get("contact") or []) if str(c).strip() and _norm(c) in resume_norm][:6]
    gaps = [{"skill": str(g.get("skill") or "").strip()[:60], "advice": str(g.get("advice") or "").strip()[:200]}
            for g in (data.get("gaps") or []) if isinstance(g, dict) and str(g.get("skill") or "").strip()][:8]
    if not sections:
        raise HTTPException(status_code=502, detail="Couldn't read the structure of your resume. Please try again.")
    logger.info(f"Tailored resume: {current_user.email} ({len(sections)} sections, {reverted} rewrites reverted)")
    return {
        "name": str(data.get("name") or "").strip()[:80], "contact": contact,
        "summary": {"original": s_original, "tailored": s_text or s_original, "note": s_note},
        "sections": sections, "skills": skills, "keywords": keywords, "gaps": gaps,
        "keywords_before": [k for k in keywords if _keyword_in(k, resume_norm)],
        "reverted": reverted, "company": company.strip()[:80], "role": role.strip()[:80],
    }


# ----- PDF: a classic one-column LaTeX-style resume, made with ReportLab (no LaTeX on the server) -----
PDF_CHAR_MAP = {"₹": "Rs. ", "→": "->", "←": "<-", "≥": ">=", "≤": "<=", "✓": "", "✔": "", "★": "*", "​": ""}


def _pdf_text(text) -> str:
    """Text for ReportLab's standard fonts: escape markup and drop characters the fonts can't draw."""
    t = str(text or "").replace("**", "")
    for a, b in PDF_CHAR_MAP.items():
        t = t.replace(a, b)
    t = t.encode("cp1252", errors="ignore").decode("cp1252")
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class PdfEntry(BaseModel):
    heading: str = ""
    subheading: str = ""
    location: str = ""
    dates: str = ""
    bullets: list[str] = []


class PdfSection(BaseModel):
    title: str
    entries: list[PdfEntry] = []


class PdfSkill(BaseModel):
    category: str = ""
    items: str = ""


class ResumePdfRequest(BaseModel):
    name: str = ""
    contact: list[str] = []
    summary: str = ""
    sections: list[PdfSection] = []
    skills: list[PdfSkill] = []


def build_resume_pdf(r: ResumePdfRequest) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from reportlab.platypus.flowables import HRFlowable

    name_style = ParagraphStyle("name", fontName="Times-Bold", fontSize=20, leading=24, alignment=TA_CENTER)
    contact_style = ParagraphStyle("contact", fontName="Times-Roman", fontSize=9.5, leading=12, alignment=TA_CENTER)
    head_style = ParagraphStyle("head", fontName="Times-Bold", fontSize=11.5, leading=14, spaceBefore=7)
    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=10, leading=12.4)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceBefore=1)
    right = ParagraphStyle("right", parent=body, alignment=2)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=0.55 * inch, rightMargin=0.55 * inch,
                            topMargin=0.45 * inch, bottomMargin=0.45 * inch,
                            title=f"{r.name.strip()[:80] or 'Resume'} - Resume", author=r.name.strip()[:80])
    width = A4[0] - 1.1 * inch - 12   # the page frame keeps 6pt padding on each side
    story = []
    if r.name.strip():
        story.append(Paragraph(_pdf_text(r.name.strip()[:80]), name_style))
    def contact_html(item):
        # Links: shown short ("linkedin.com/in/x"), clickable in the PDF. Emails become mailto links.
        shown = re.sub(r"^https?://(www\.)?", "", item).rstrip("/")
        if re.match(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", item, re.I):
            return f'<link href="mailto:{_pdf_text(item)}">{_pdf_text(item)}</link>'
        if re.match(r"^(https?://|www\.)|^[a-z0-9-]+\.(com|in|io|dev|me|org|net|ai)(/|$)", item, re.I):
            href = item if item.lower().startswith("http") else "https://" + item
            return f'<link href="{_pdf_text(href)}">{_pdf_text(shown)}</link>'
        return _pdf_text(item)

    contact = [c for c in (x.strip()[:120] for x in r.contact[:6]) if c]
    if contact:
        story.append(Paragraph(" &nbsp;|&nbsp; ".join(contact_html(c) for c in contact), contact_style))

    def heading(title):
        # returned, not added: the caller keeps it on the same page as the first lines under it
        return [Paragraph(_pdf_text(title.upper()), head_style),
                HRFlowable(width="100%", thickness=0.6, color=colors.black, spaceBefore=1, spaceAfter=4)]

    if r.summary.strip():
        story.append(KeepTogether(heading("Summary") + [Paragraph(_pdf_text(r.summary.strip()[:TAILOR_LINE_CHARS]), body)]))
    for sec in r.sections[:TAILOR_MAX_SECTIONS]:
        if not sec.title.strip() or not sec.entries:
            continue
        pending_heading = heading(sec.title.strip()[:40])
        for ent in sec.entries[:TAILOR_MAX_ENTRIES]:
            rows = []
            if ent.heading or ent.dates:
                rows.append([Paragraph(f"<b>{_pdf_text(ent.heading[:120])}</b>", body), Paragraph(_pdf_text(ent.dates[:60]), right)])
            if ent.subheading or ent.location:
                rows.append([Paragraph(f"<i>{_pdf_text(ent.subheading[:120])}</i>", body),
                             Paragraph(f"<i>{_pdf_text(ent.location[:60])}</i>", right)])
            block = []
            if rows:
                table = Table(rows, colWidths=[width * 0.72, width * 0.28])
                table.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
                block.append(table)
            block += [Paragraph(_pdf_text(b.strip()[:TAILOR_LINE_CHARS]), bullet, bulletText="•")
                      for b in ent.bullets[:TAILOR_MAX_BULLETS] if b.strip()]
            block.append(Spacer(1, 4))
            # never leave a section heading or a job title alone at the bottom of a page
            story.append(KeepTogether(pending_heading + block[:3]))
            pending_heading = []
            story.extend(block[3:])
    skills = [s for s in r.skills[:10] if s.items.strip()]
    if skills:
        lines = []
        for s in skills:
            label = f"<b>{_pdf_text(s.category.strip()[:40])}:</b> " if s.category.strip() else ""
            lines.append(Paragraph(label + _pdf_text(s.items.strip()[:TAILOR_LINE_CHARS]), body))
        story.append(KeepTogether(heading("Skills") + lines[:2]))
        story.extend(lines[2:])
    doc.build(story)
    return buf.getvalue()


@app.post("/api/resume/pdf")
async def resume_pdf(data: ResumePdfRequest, current_user: models.User = Depends(get_current_user_detached)):
    if not data.sections and not data.summary.strip():
        raise HTTPException(status_code=400, detail="Nothing to put in the PDF yet.")
    try:
        pdf = await asyncio.to_thread(build_resume_pdf, data)
    except Exception as e:
        logger.error(f"Resume PDF failed for {current_user.email}: {e}")
        raise HTTPException(status_code=500, detail="Couldn't make the PDF. Please try again.")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", data.name.strip())[:40].strip("_") or "Resume"
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{safe}_Resume.pdf"'})
