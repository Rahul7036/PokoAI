# 🎯 PrapAI - Real-Time Interview Copilot

A production-ready AI-powered interview assistant that listens to live interviews, understands questions, and provides intelligent real-time suggestions to help candidates perform their best.

## ✨ Features

### Core Functionality
- **🎧 Live Interview Copilot**: Real-time audio transcription and AI-powered response suggestions during interviews
- **📄 Resume Matcher**: Analyzes resume-JD alignment with actionable improvement suggestions
- **🔐 Secure Authentication**: Google OAuth + Manual signup with admin approval workflow
- **⏱️ Usage Limits**: 5-minute free tier with visual credit tracking
- **👤 User Profiles**: Personalized dashboard with name and profession data

### Security & Access Control
- Manual account approval system (all new accounts start as inactive)
- JWT-based authentication
- Protected secrets via `.env` and `.gitignore`
- Role-based access control ready

### UI/UX Highlights
- **Responsive Design**: Fully mobile-compatible (768px+ breakpoints)
- **Premium Aesthetics**: Glassmorphism, neon gradients, and smooth animations
- **Real-time Feedback**: Pulsating status indicators and live AI suggestion cards
- **Mission Control Theme**: Radar-pulse backgrounds and voice waveform visualizations

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **PostgreSQL** (or compatible database)
- **Google Cloud Project** (for Vertex AI & OAuth)
- **Node.js** (optional, for frontend tooling)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd PrapAIMVP
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # Google Cloud
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your-service-account-key.json
   API_KEY=your_google_api_key
   
   # Database
   DB_STRING=postgresql://username:password@localhost:5432/prapai
   
   # Optional
   HUG_KEY=your_huggingface_key
   ```

5. **Set up the database**
   
   Create a PostgreSQL database and run the migration:
   ```bash
   python migrate_db.py  # Adds profile and approval columns
   ```

6. **Configure Google OAuth**
   
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create OAuth 2.0 credentials
   - Add authorized origins: `http://localhost:8000`
   - Add authorized redirect URIs: `http://localhost:8000`
   - Update the `client_id` in `templates/index.html` (line ~819)

7. **Run the application**
   ```bash
   uvicorn app:app --reload
   ```
   
   Visit `http://localhost:8000`

---

## 🐳 Docker Deployment (Recommended)

### Quick Start with Docker

1. **Prerequisites**
   - Docker Desktop installed
   - Docker Compose installed
   - Google Cloud service account JSON file

2. **Setup**
   ```bash
   # Clone the repository
   git clone <your-repo-url>
   cd PrapAI

   # Copy your Google service account JSON to credentials folder
   mkdir credentials
   cp /path/to/your-service-account.json credentials/service-account.json

   # Copy and configure environment file
   cp .env.docker .env
   # Edit .env and add your API_KEY and other credentials
   ```

3. **Run with Docker Compose**
   ```bash
   # Build and start all services (app + PostgreSQL)
   docker-compose up -d

   # View logs
   docker-compose logs -f web

   # Stop services
   docker-compose down
   ```

4. **Access the application**
   - App: `http://localhost:8000`
   - Database: `localhost:5432` (credentials in docker-compose.yml)

5. **Run database migrations**
   ```bash
   # Execute migration inside the container
   docker-compose exec web python migrate_db.py
   ```

### Manual Docker Build

```bash
# Build the image
docker build -t prapai:latest .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e DB_STRING="your_db_string" \
  -e API_KEY="your_api_key" \
  -v $(pwd)/credentials:/app/credentials \
  --name prapai \
  prapai:latest
```

### Production Deployment

For cloud deployment (GCP, AWS, Azure):

```bash
# Tag and push to container registry
docker tag prapai:latest gcr.io/your-project/prapai:latest
docker push gcr.io/your-project/prapai:latest

# Deploy to Cloud Run (GCP example)
gcloud run deploy prapai \
  --image gcr.io/your-project/prapai:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 📂 Project Structure

```
PrapAIMVP/
├── app.py                  # FastAPI application & routes
├── auth.py                 # JWT authentication logic
├── models.py               # SQLAlchemy User model
├── database.py             # Database connection setup
├── ai_utils.py             # Vertex AI integration
├── templates/
│   └── index.html          # Single-page application UI
├── .env                    # Environment variables (not tracked)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔐 User Management & Approval

### New User Registration Flow
1. User signs up (manual or Google)
2. Account is created with `is_active = False`
3. User sees "Pending Approval" message
4. Admin manually approves in the database

### Approving a User (Admin)
```sql
-- Connect to your PostgreSQL database
psql -U username -d prapai

-- Activate a user
UPDATE users SET is_active = true WHERE email = 'user@example.com';
```

---

## 📱 Mobile Compatibility

The application is **fully responsive** and optimized for:
- ✅ **Desktop** (1024px+)
- ✅ **Tablet** (768px - 1024px)
- ✅ **Mobile** (320px - 768px)

### Mobile Optimizations
- Stacked layouts for dashboard and copilot views
- Touch-friendly buttons and navigation
- Collapsible sidebar on mobile
- Adaptive font sizes and spacing

---

## 🛠️ Development

### Running in Development Mode
```bash
# With auto-reload
uvicorn app:app --reload

# Custom host/port
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Database Schema Updates
If you modify `models.py`, create a migration script or use Alembic:
```bash
# Example manual migration
python migrate_db.py
```

---

## 🚨 Production Deployment Checklist

- [ ] Update `GOOGLE_APPLICATION_CREDENTIALS` to production service account
- [ ] Set strong `SECRET_KEY` for JWT in `auth.py`
- [ ] Use production PostgreSQL instance
- [ ] Enable HTTPS and update OAuth redirect URIs
- [ ] Set `--workers` for uvicorn/gunicorn
- [ ] Configure CORS for your production domain
- [ ] Set up monitoring (e.g., Sentry, Google Cloud Monitoring)
- [ ] Review and remove `migrate_db.py` after initial deployment

---

## 📦 Dependencies

### Backend
- **FastAPI**: Web framework
- **SQLAlchemy**: ORM
- **psycopg2**: PostgreSQL adapter
- **python-jose**: JWT handling
- **passlib**: Password hashing
- **google-cloud-aiplatform**: Vertex AI
- **google-cloud-speech**: Speech-to-Text
- **google-cloud-texttospeech**: Text-to-Speech
- **pypdf**: PDF parsing

### Frontend
- **Vanilla JS**: No framework required
- **Marked.js**: Markdown rendering
- **Google Identity Services**: OAuth integration

---

## 🎨 Design Philosophy

- **Glassmorphism**: Frosted glass effects for depth
- **Neon Accents**: Cyan and purple gradients for vibrancy
- **Micro-animations**: Pulse effects, floating elements, and smooth transitions
- **Mission Control**: Radar grids and tactical HUD aesthetics

---

## 🐛 Troubleshooting

### `ModuleNotFoundError: No module named 'vertexai'`
**Solution**: Ensure you're running from the virtual environment:
```bash
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### Database connection errors
**Solution**: Verify `DB_STRING` in `.env` and ensure PostgreSQL is running:
```bash
# Test connection
psql -U username -d prapai
```

### Google OAuth `invalid_client` error
**Solution**: 
1. Check `client_id` matches Google Cloud Console
2. Whitelist `http://localhost:8000` in authorized origins
3. Add your email to test users if project is in "Testing" mode

---

## 📄 License

Proprietary - All rights reserved.

---

## 👨‍💻 Author

**PrapAI Team**  
For support or inquiries: prepai0206@gmail.com

---

## 🙏 Acknowledgments

- Google Cloud Platform (Vertex AI, Speech API)
- FastAPI community
- The open-source community

---

**Ready to ace your next interview? Let's go! 🚀**
