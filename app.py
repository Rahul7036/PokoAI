import os
import queue
import asyncio
import json
import logging
import threading
import re
import smtplib
import random
import string
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
from fastapi.responses import HTMLResponse, FileResponse
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
            conn.execute(text("ALTER TABLE users ADD COLUMN time_limit_seconds INTEGER DEFAULT 1200"))
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

def send_otp_email(to_email: str, otp: str):
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")
    
    if not sender_email or not sender_password:
        logger.error("EMAIL_USER or EMAIL_PASS not set")
        return False

    msg = MIMEMultipart()
    msg["From"] = f"PrapAI <{sender_email}>"
    msg['To'] = to_email
    msg['Subject'] = "Your OTP Code"

    html = f"""
      <h2>PrapAI Email Verification</h2>
      <p>Your OTP is:</p>
      <h1>{otp}</h1>
      <p>This code expires in 10 minutes.</p>
    """
    msg.attach(MIMEText(html, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

# --- Auth Routes ---
BETA_INVITE_CODE = "PRAPAI2026"

@app.post("/auth/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    
    # Generate OTP
    otp = ''.join(random.choices(string.digits, k=6))
    otp_hash = bcrypt.hashpw(otp.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    otp_expires = datetime.utcnow() + timedelta(minutes=10)

    new_user = models.User(
        email=user.email, 
        hashed_password=hashed_password,
        full_name=user.fullName,
        profession=user.profession,
        is_active=False, # Accout pending approval
        otp_code=otp_hash,
        otp_expires_at=otp_expires
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Send OTP Email
    email_sent = send_otp_email(user.email, otp)
    if not email_sent:
        logger.error("Failed to send OTP email")
        # Ensure we don't rollback user creation, but maybe warn?
        # Ideally, we might want to delete user if email fails, but for now let's keep it.

    # We still return a token so they can see the "pending" message if we want,
    # but the get_current_user dependency or the login route will block them.
    access_token = create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer", "message": "OTP sent to email"}

@app.post("/auth/verify-otp")
def verify_otp(data: OTPVerify, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
        
    if user.is_active:
        return {"message": "Account already active"}
        
    if not user.otp_code:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    # Verify OTP Hash
    try:
        if not bcrypt.checkpw(data.otp.encode('utf-8'), user.otp_code.encode('utf-8')):
             raise HTTPException(status_code=400, detail="Invalid OTP")
    except Exception:
        # Fallback for old plain-text OTPs if any exist (optional, but safer during transition)
        if user.otp_code != data.otp:
             raise HTTPException(status_code=400, detail="Invalid OTP")
        
    if not user.otp_expires_at or user.otp_expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="OTP Expired")
        
    user.is_active = True
    user.otp_code = None
    user.otp_expires_at = None
    db.commit()
    
    access_token = create_access_token(data={"sub": user.email})
    return {"message": "Account verified successfully", "access_token": access_token, "token_type": "bearer"}

@app.post("/auth/resend-otp")
def resend_otp(data: OTPVerify, db: Session = Depends(get_db)):
    # We only need email from data
    user = db.query(models.User).filter(models.User.email == data.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found")
        
    if user.is_active:
        return {"message": "Account already active"}
        
    otp = ''.join(random.choices(string.digits, k=6))
    otp_hash = bcrypt.hashpw(otp.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    otp_expires = datetime.utcnow() + timedelta(minutes=10)
    
    user.otp_code = otp_hash
    user.otp_expires_at = otp_expires
    db.commit()
    
    send_otp_email(user.email, otp)
    return {"message": "OTP resent successfully"}

@app.post("/auth/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not db_user.hashed_password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
        
    if not db_user.is_active:
        raise HTTPException(status_code=403, detail="Account not verified. Please check your email for the OTP code.")

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
    db_user = db.query(models.User).filter(models.User.email == email).first()
    
    if not db_user:
        # Create new user via Google, automatically activate and set 20 min limit (1200s)
        full_name = id_info.get('name')
        db_user = models.User(
            email=email, 
            google_id=google_id, 
            full_name=full_name,
            is_active=True, 
            time_limit_seconds=1200
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
    return json.loads(text)

@app.post("/update_context")
async def update_context(
    resume_file: UploadFile = File(None),
    jd: str = Form(...),
    company: str = Form(""),
    current_user: models.User = Depends(get_current_user_detached),
):
    ctx = get_user_context(current_user.id)

    # Process PDF Resume if provided
    if resume_file:
        try:
            content = await resume_file.read()
            text = await asyncio.to_thread(_extract_pdf_text, content)

            ctx["resume"] = text
            logger.info(f"Resume PDF processed ({len(text)} chars) for user {current_user.id}")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return {"status": "error", "message": str(e)}

    ctx["jd"] = jd
    ctx["company"] = company
    logger.info(f"Context updated for user {current_user.id}, company: {company}")
    return {"status": "success", "message": "Context updated"}

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


def build_answer_prompt(text, history, ctx, force=False, variant=False):
    history_text = ""
    if history:
        turns = history[-COPILOT_HISTORY_TURNS:]
        lines = [f"Interviewer: {q}\nSuggested: {a[:COPILOT_HISTORY_ANSWER_CHARS]}" for q, a in turns]
        history_text = ("PREVIOUS CONVERSATION (most recent last; 'Suggested' is what you suggested "
                        "the candidate say):\n" + "\n".join(lines))

    if force:
        decide = ("The candidate tapped 'Answer now', so treat the input as an interviewer question "
                  "and ALWAYS answer it. Never output NO_ANSWER.")
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
        Speak AS the candidate (first person), using only facts from the resume.

        RESUME:
        {ctx['resume'][:COPILOT_RESUME_CHARS]}

        JOB DESCRIPTION:
        {ctx['jd'][:COPILOT_JD_CHARS]}

        {history_text}

        LATEST SPEECH (live transcript, may contain recognition errors):
        "{text}"

        DECIDE:
        {decide}

        OUTPUT FORMAT (follow exactly, no other text):
        SAY: <one natural opening sentence the candidate can say immediately, max 25 words>
        - <talking point, max 12 words>
        - <talking point, max 12 words>
        - <talking point, max 12 words>
        (3 to 5 talking points: 3 for simple questions, up to 5 for broad ones)
        DETAIL:
        <2-4 short sentences of extra detail, specific numbers or examples from the resume>

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
        - Coding questions: SAY explains the approach in one line; talking points cover approach and edge cases;
          DETAIL contains a short code block (language from the JD, default Python) and a line
          "Time: O(...), Space: O(...)".
        - Greetings or small talk ("how are you?", "can you hear me?"): output only a short SAY line.
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

    def recent_answers():
        return [a for _, a in conversation_history[-ECHO_RECENT_ANSWERS:]]

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

    def start_answer(text, force=False, variant=False):
        task = asyncio.create_task(answer_question(text, force, variant))
        answer_tasks.add(task)
        task.add_done_callback(answer_tasks.discard)

    def flush_pending():
        nonlocal last_heard_text
        segments = take_pending_segments()
        # Drop segments where the candidate is reading the suggested answer out loud
        kept = [s for s in segments if not is_echo(s, recent_answers())]
        if len(kept) < len(segments):
            logger.info(f"Ignored {len(segments) - len(kept)} segment(s) repeating the suggested answer")
        text = " ".join(kept).strip()
        if text:
            last_heard_text = text
        if stop_event.is_set() or not should_trigger_ai(text):
            return
        start_answer(text)

    def answer_now():
        """'Answer now' button: answer what was just heard without waiting for a pause."""
        nonlocal last_heard_text
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
        """'Skip' button: stop the answer being written and forget unanswered speech."""
        take_pending_segments()
        for task in list(answer_tasks):
            task.cancel()

    async def send_json(payload):
        if stop_event.is_set():
            return
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass  # client already gone

    async def answer_question(text, force=False, variant=False):
        nonlocal answer_counter
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

                prompt = build_answer_prompt(text, conversation_history, user_context, force=force, variant=variant)
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
                            await send_json({"type": "answer_start", "id": answer_id, "question": text})
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
                    await send_json({"type": "answer_start", "id": answer_id, "question": text})
                    await send_json({"type": "answer_chunk", "id": answer_id, "text": answer})

                await send_json({"type": "answer_end", "id": answer_id})
                if variant and conversation_history and conversation_history[-1][0] == text:
                    conversation_history[-1] = (text, answer.strip())   # replace, don't repeat, the turn
                else:
                    conversation_history.append((text, answer.strip()))
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
                if command == "answer_now":
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
