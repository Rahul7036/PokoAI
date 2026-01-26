import os
import queue
import asyncio
import json
import logging
import threading
import re
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
location = "us-central1"

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
    model = GenerativeModel("gemini-2.0-flash-001")
    logger.info("Vertex AI initialized successfully from environment variables.")
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
from database import engine, get_db
from sqlalchemy.orm import Session
from auth import get_password_hash, verify_password, create_access_token, verify_google_token, get_current_user
from datetime import timedelta


# Initialize Tables
models.Base.metadata.create_all(bind=engine)

def ensure_schema_updates():
    """Ensure existing DB has columns from recent updates."""
    from sqlalchemy import text
    with engine.connect() as conn:
        # Check for time_limit_seconds
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN time_limit_seconds INTEGER DEFAULT 300"))
            logger.info("Added time_limit_seconds column")
        except Exception:
            pass
        
        # Check for time_used_seconds
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN time_used_seconds INTEGER DEFAULT 0"))
            logger.info("Added time_used_seconds column")
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

# --- Auth Routes ---
BETA_INVITE_CODE = "PREPAI2026"

@app.post("/auth/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = models.User(
        email=user.email, 
        hashed_password=hashed_password,
        full_name=user.fullName,
        profession=user.profession,
        is_active=False # Accout pending approval
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # We still return a token so they can see the "pending" message if we want,
    # but the get_current_user dependency or the login route will block them.
    access_token = create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not db_user.hashed_password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
        
    if not db_user.is_active:
        raise HTTPException(status_code=403, detail="Account pending approval. Please contact support.")

    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/google")
def google_auth(login: GoogleLogin, db: Session = Depends(get_db)):
    id_info = verify_google_token(login.token)
    if not id_info:
        raise HTTPException(status_code=400, detail="Invalid Google Token")
    
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
def heartbeat(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.time_used_seconds >= current_user.time_limit_seconds:
        raise HTTPException(status_code=403, detail="Credit limit reached")
    
    # Increment by 10 seconds (heartbeat interval)
    current_user.time_used_seconds += 10
    db.commit()
    return {"status": "success", "remaining_seconds": max(0, current_user.time_limit_seconds - current_user.time_used_seconds)}


# Context Management
USER_CONTEXT = {
    "resume": "",
    "jd": "",
    "company": ""
}

@app.post("/update_context")
async def update_context(
    resume_file: UploadFile = File(None),
    jd: str = Form(...),
    company: str = Form("")
):
    # Process PDF Resume if provided
    if resume_file:
        try:
            content = await resume_file.read()
            pdf = PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            
            USER_CONTEXT["resume"] = text
            logger.info(f"Resume PDF processed ({len(text)} chars)")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return {"status": "error", "message": str(e)}
            
    USER_CONTEXT["jd"] = jd
    USER_CONTEXT["company"] = company
    logger.info(f"Context updated via API for company: {company}")
    return {"status": "success", "message": "Context updated"}

@app.post("/api/generate-briefing")
async def generate_briefing():
    try:
        if not USER_CONTEXT["resume"] or not USER_CONTEXT["jd"]:
            return {"status": "error", "message": "Resume and JD required"}

        prompt = f"""
        You are a career coach. Based on this RESUME and JOB DESCRIPTION, generate 4 "Prep Cards" to help the candidate in the final 5 minutes before the interview.
        
        RESUME: {USER_CONTEXT['resume'][:3000]}
        JD: {USER_CONTEXT['jd'][:2000]}

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
    job_description: str = Form(...)
):
    try:
        # 1. Extract Text from PDF
        content = await resume.read()
        reader = PdfReader(io.BytesIO(content))
        resume_text = ""
        for page in reader.pages:
            resume_text += page.extract_text() or ""
            
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
        
        if not model:
             raise HTTPException(status_code=500, detail="AI Model not initialized")

        # 3. Generate Response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean potential markdown
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        return json.loads(response_text)
        
    except Exception as e:
        logger.error(f"Resume Analysis Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

RATE = 16000
CHUNK = 1024
LANGUAGE_CODE = "en-US"

@app.get("/")
async def get():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Error: templates/index.html not found</h1>")

async def get_vertex_response(text, history):
    if not model:
        return "Error: Vertex AI not initialized."
    try:
        # Format history
        history_text = ""
        if history:
            history_text = "PREVIOUS CONVERSATION:\n" + "\n".join([f"Interviewer: {q}\nYou: {a}" for q, a in history[-10:]])
        
        # Smart Prompt with Resume & JD
        prompt = f"""
        You are the candidate in a job interview for {USER_CONTEXT['company']}.
        Identify yourself using the name and details provided in YOUR RESUME below.
        You are listening to the INTERVIEWER.
        
        YOUR RESUME (The "Truth"):
        {USER_CONTEXT['resume']}
        
        JOB DESCRIPTION (Target Role):
        {USER_CONTEXT['jd']}
        
        {history_text}
        
        CURRENT INPUT FROM INTERVIEWER:
        "{text}"
        
        INSTRUCTIONS:
        1. ANALYZE the input internally. DO NOT OUTPUT YOUR ANALYSIS.
        2. IF IT IS NOT A QUESTION/COMMAND (e.g. noise, mumbles), output ONLY: "NO_ANSWER"
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
        """
        
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Vertex AI Error: {e}")
        return "Error generating answer from Vertex AI."

def should_trigger_ai(text):
    # Relaxed filter: Let the AI decide, but filter out absolute noise
    text = text.strip()
    if len(text.split()) < 2: # Ignore single words like "Hello", "Okay" unless they are questions?
        return False
    return True

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = None, db: Session = Depends(get_db)):
    # Authenticate user via token query param
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
        
    try:
        user = get_current_user(token, db)
    except Exception:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if user.time_used_seconds >= user.time_limit_seconds:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    logger.info(f"Client connected: {user.email}")
    
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

                    # TRIGGER AI LOGIC - "AI Decides" Strategy
                    if is_final and should_trigger_ai(transcript):
                        asyncio.run_coroutine_threadsafe(
                            trigger_ai_response(transcript, websocket), loop
                        )
                        
            except Exception as e:
                # Log but don't crash, retry loop will catch unless stopped
                if "400" in str(e) or "out of range" in str(e): 
                     logger.error(f"Speech API Error (will retry): {e}")
                else:
                     logger.error(f"Speech thread error: {e}")
                
            if not stop_event.is_set():
                 logger.info("Restarting speech stream...")
            
    async def trigger_ai_response(text, ws):
        # Notify UI we are thinking (optional, maybe too noisy if we do it for everything?)
        # Let's send a subtle status
        await ws.send_text(json.dumps({"type": "status", "message": "Listening..."}))
        
        answer = await get_vertex_response(text, conversation_history)
        
        if answer == "NO_ANSWER":
            # AI decided this wasn't worth answering
            logger.info(f"AI declined to answer: '{text}'")
            await ws.send_text(json.dumps({"type": "status", "message": "Ready"}))
            return

        # Update History
        conversation_history.append((text, answer))
        
        await ws.send_text(json.dumps({
            "type": "answer",
            "question": text,
            "answer": answer
        }))

    t = threading.Thread(target=speech_recognition_thread)
    t.start()
    
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                audio_queue.put(message["bytes"])
            elif "text" in message:
                pass
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        stop_event.set()
        audio_queue.put(None)
        t.join()
