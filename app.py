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
import vertexai
from vertexai.generative_models import GenerativeModel

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Cloud credentials from environment variables
import json
import tempfile
from google.oauth2 import service_account

project_id = os.getenv("GCP_PROJECT_ID", "intrepid-honor-484608-e0")
location = os.getenv("GCP_LOCATION", "us-central1")
model_name = os.getenv("MODEL_NAME", "gemini-3.5-flash")

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
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    
    # Set credentials for Google Cloud clients
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""  # Clear any file-based credentials
    
    # Initialize Vertex AI with credentials
    vertexai.init(project=project_id, location=location, credentials=credentials)
    model = GenerativeModel(model_name)
    logger.info(f"Vertex AI initialized successfully using model '{model_name}' from environment variables.")
except Exception as e:
    logger.error(f"Vertex AI Init Failed: {e}")
    model = None
    credentials = None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
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
    msg['From'] = f"PokoAI <{sender_email}>"
    msg['To'] = to_email
    msg['Subject'] = "Your OTP Code"

    html = f"""
      <h2>PokoAI Email Verification</h2>
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
        if not model:
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
        response = await asyncio.to_thread(model.generate_content, prompt)
        return _parse_model_json(response.text)
    except Exception as e:
        logger.error(f"Briefing failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/analyze-resume")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    current_user: models.User = Depends(get_current_user_detached),
):
    if not model:
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
        response = await asyncio.to_thread(model.generate_content, prompt)
        return _parse_model_json(response.text)

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

@app.get("/privacy")
async def privacy_policy():
    with open("templates/privacy.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

def build_answer_prompt(text, history, ctx):
    # Format history
    history_text = ""
    if history:
        history_text = "PREVIOUS CONVERSATION:\n" + "\n".join([f"Interviewer: {q}\nYou: {a}" for q, a in history[-10:]])

    # Smart Prompt with Resume & JD
    return f"""
        You are the candidate in a job interview for {ctx['company']}.
        Identify yourself using the name and details provided in YOUR RESUME below.
        You are listening to the INTERVIEWER.

        YOUR RESUME (The "Truth"):
        {ctx['resume']}

        JOB DESCRIPTION (Target Role):
        {ctx['jd']}
        
        {history_text}
        
        CURRENT INPUT FROM INTERVIEWER:
        "{text}"
        
        INSTRUCTIONS:
        1. ANALYZE the input internally. DO NOT OUTPUT YOUR ANALYSIS.
        2. IF IT IS NOT A QUESTION/COMMAND (e.g. noise, mumbles, small talk, or the candidate
           speaking their own answer), output ONLY: "NO_ANSWER"
        3. IF IT IS A QUESTION/COMMAND, output ONLY the response:
           - Answer in the FIRST PERSON.
           - Stick to the facts in your RESUME.
           - Align with JD requirements.
           - Be friendly, professional, and conversational.
        4. IF CODING PROBLEM:
           - Detect language from JD/context (default Python).
           - Brief explanation.
           - Code block.
           - Complexity.
        5. FORMATTING: Use plain Markdown only. Do NOT use LaTeX or $...$ math notation;
           write complexity as plain text, e.g. O(n) or O(n log n).
        """


async def stream_model_text(prompt):
    """Async generator yielding text chunks from Gemini as they are produced.

    The Vertex SDK call is blocking, so it runs in a worker thread and hands chunks
    back to the event loop through an asyncio.Queue.
    """
    loop = asyncio.get_running_loop()
    chunks = asyncio.Queue()
    done = object()

    def worker():
        try:
            for chunk in model.generate_content(prompt, stream=True):
                try:
                    piece = chunk.text
                except Exception:
                    piece = ""  # chunks without text (e.g. final metadata) raise on .text
                if piece:
                    loop.call_soon_threadsafe(chunks.put_nowait, piece)
        except Exception as e:
            loop.call_soon_threadsafe(chunks.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(chunks.put_nowait, done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await chunks.get()
        if item is done:
            return
        if isinstance(item, Exception):
            raise item
        yield item

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

    def flush_pending():
        nonlocal flush_handle, pending_since
        flush_handle = None
        text = " ".join(pending_segments).strip()
        pending_segments.clear()
        pending_since = None
        if stop_event.is_set() or not should_trigger_ai(text):
            return
        task = asyncio.create_task(answer_question(text))
        answer_tasks.add(task)
        task.add_done_callback(answer_tasks.discard)

    async def send_json(payload):
        if stop_event.is_set():
            return
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass  # client already gone

    async def answer_question(text):
        nonlocal answer_counter
        async with answer_lock:
            if stop_event.is_set():
                return
            if not model:
                await send_json({"type": "status", "message": "Ready"})
                return
            answer_counter += 1
            answer_id = answer_counter
            await send_json({"type": "status", "message": "Listening..."})

            prompt = build_answer_prompt(text, conversation_history, user_context)
            answer = ""
            started = False
            try:
                async for piece in stream_model_text(prompt):
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
            except Exception as e:
                logger.error(f"Vertex AI Error: {e}")
                if started:
                    await send_json({"type": "answer_chunk", "id": answer_id,
                                     "text": "\n\n_(Answer interrupted — please ask again.)_"})
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
                await send_json({"type": "answer_start", "id": answer_id, "question": text})
                await send_json({"type": "answer_chunk", "id": answer_id, "text": answer})

            await send_json({"type": "answer_end", "id": answer_id})
            conversation_history.append((text, answer.strip()))

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
