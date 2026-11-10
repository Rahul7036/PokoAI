# PrapAI — Real-Time Interview Copilot

Live at **https://prapai.com**

PrapAI listens to a live interview, transcribes it in real time, and streams suggested
first-person answers based on the candidate's resume and the job description.

## Features

- **Live Copilot** — streams audio to Google Speech-to-Text; when the interviewer pauses, the
  question is sent to Gemini and the answer streams onto the screen (~1 s to the first words).
  - Focus-card screen built for a phone: one answer at a time — a big **Say first** line,
    3 short talking points, and collapsible detail/code. Swipe for earlier answers.
  - **Answer now** (answer immediately / force an answer) and **Skip** buttons.
  - Ignores the candidate reading the suggested answer aloud (echo filter + prompt).
  - Keeps the phone screen awake and vibrates when a new answer arrives.
  - *Listen to: Mic only* — microphone only.
  - *Listen to: Mic + meeting audio* — also captures the Zoom/Meet/Teams tab (or screen) audio,
    so the copilot hears the interviewer even when the candidate wears headphones.
- **Prep-Deck** — four quick briefing cards (pitch, project to mention, weak spot, top skill).
- **Resume Matcher** — strict resume-vs-JD score, verdict, missing skills and fixes.
- **Accounts** — email + password with a 6-digit email OTP, or Sign in with Google.
- **Credits** — each new account gets 7 free minutes of Copilot time
  (`models.FREE_TIME_SECONDS = 420`), metered by the server while a Copilot session is open.
- **Plans (INR)** — Starter ₹299 / 45 min, Pro ₹699 / 120 min. Payment is manual for now: "Buy"
  opens an email to the team, the user pays by UPI, and an admin adds the minutes (see Administration).

## Tech stack

| Layer | What |
|---|---|
| Backend | FastAPI (single file: `app.py`), WebSocket `/ws` for audio |
| Auth | JWT (HS256, 24 h), bcrypt passwords, Google Identity Services |
| Database | PostgreSQL via SQLAlchemy (one `users` table) |
| AI | Gemini on Vertex AI via `google-genai`, Google Cloud Speech-to-Text |
| Frontend | One page: `templates/index.html` (vanilla JS, `marked` + `DOMPurify`) |

## Project structure

```
app.py                  FastAPI app: auth routes, credits, AI endpoints, /ws copilot
auth.py                 Password hashing, JWT, Google token check, current-user dependencies
database.py             SQLAlchemy engine/session (reads DB_STRING)
models.py               User model
templates/index.html    The whole frontend (landing, dashboard, copilot, resume matcher)
templates/privacy.html  Privacy policy served at /privacy
requirements.txt        Python dependencies
run_server.bat          Windows shortcut to start the dev server
handler.py              Dev utility: list available Gemini models (not used by the app)
realtime_transcribe.py  Dev utility: local mic -> Speech-to-Text test (not used by the app)
```

## Setup

### 1. Python environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Google Cloud

In your Google Cloud project:

1. Link a billing account.
2. Enable **Vertex AI API** (`aiplatform.googleapis.com`, now shown as "Agent Platform API")
   and **Cloud Speech-to-Text API** (`speech.googleapis.com`).
3. Create a service account with the roles **Vertex AI User** and **Cloud Speech Client**,
   then create a **JSON key** for it. If key creation is blocked, override the org policy
   `iam.disableServiceAccountKeyCreation` for the project.
4. **Google Auth Platform**: set app name, home page (`https://prapai.com`), privacy policy
   (`https://prapai.com/privacy`) and authorized domain; create a **Web application** OAuth
   client with JavaScript origins `http://localhost`, `http://localhost:8000` and
   `https://prapai.com`; then **Publish app**.

### 3. `.env`

Create `.env` in the project root (never commit it):

```env
# Google Cloud service account (copy from the JSON key)
GCP_PROJECT_ID=your-project-id
GCP_PRIVATE_KEY_ID=...
GCP_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
GCP_CLIENT_EMAIL=your-sa@your-project-id.iam.gserviceaccount.com
GCP_CLIENT_ID=...

# Gemini
GCP_LOCATION=global
MODEL_NAME=gemini-3.5-flash          # Prep-Deck + Resume Matcher (quality first)
COPILOT_MODEL=gemini-3.6-flash       # Live Copilot (speed first); falls back to MODEL_NAME on error
COPILOT_THINKING_LEVEL=MINIMAL       # MINIMAL = ~1s to first word; leave empty for model default

# Sign in with Google (OAuth Client ID, not secret)
GOOGLE_CLIENT_ID=xxxx.apps.googleusercontent.com

# Database
DB_STRING=postgresql://user:password@host:5432/dbname

# JWT signing key: a long random string (keep it the same across deploys)
SECRET_KEY=change-me

# Gmail account used to send OTP emails (use a Gmail App Password)
EMAIL_USER=you@gmail.com
EMAIL_PASS=your-app-password
```

`GCP_PRIVATE_KEY` must be one line, in double quotes, with the `\n` sequences kept as-is.

### 4. Run

```bash
python -m uvicorn app:app --reload
```

Open http://localhost:8000. Tables are created automatically on startup.

## Deployment notes

- **Run a single server process** (one uvicorn worker). Each user's resume/JD is kept in that
  process's memory (never in the database), so multiple workers would not share it.
- Serve over **HTTPS** — browsers only allow microphone and meeting-audio capture on secure pages
  (and on `localhost`).
- Add the production URL to the OAuth client's **Authorized JavaScript origins**.
- `/health` returns `{"status": "ok"}` for uptime checks.

## Administration

New email accounts become active after OTP verification; Google accounts are active
immediately. After a user pays, add their minutes:

```sql
-- Starter (₹299): 45 min = 2700 s     Pro (₹699): 120 min = 7200 s
UPDATE users SET time_limit_seconds = time_limit_seconds + 2700 WHERE email = 'user@example.com';
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `404 Publisher model ... not found` | The model isn't available in that location for your project; these models need `GCP_LOCATION=global` |
| Copilot answers feel slow | Check the startup log line `Gemini initialized ... thinking=MINIMAL`; `COPILOT_THINKING_LEVEL` must be `MINIMAL` |
| `Could not deserialize key data` | `GCP_PRIVATE_KEY` formatting: one line, double quotes, keep `\n` |
| Google button: `invalid_client` / origin error | Add the site's origin to the OAuth client; wait ~5 minutes |
| Google login: "Access blocked" | OAuth app is still in Testing — publish it or add the user as a test user |
| Copilot doesn't hear the interviewer | Use **Listen to: Mic + meeting audio** and enable "Also share tab audio" in the share dialog |

## Contact

pokoai0206@gmail.com

## License

Proprietary — all rights reserved.
