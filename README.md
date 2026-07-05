# 🏪 Street Vendor Digitalization Agent

> **AI-powered platform to digitalize India's street vendors and micro-entrepreneurs**
> Built with **IBM watsonx.ai Studio**, **IBM Granite LLM**, **LangChain RAG**, and **FAISS**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [IBM watsonx.ai Configuration](#ibm-watsonxai-configuration)
- [Running the App](#running-the-app)
- [API Endpoints](#api-endpoints)
- [RAG Pipeline](#rag-pipeline)
- [Deployment on Render (Free)](#deployment-on-render-free)
- [GitHub Upload Guide](#github-upload-guide)
- [AGENT_INSTRUCTIONS](#agent-instructions)

---

## 🎯 Project Overview

India has over **10 million street vendors** who lack digital tools, formal credit access, and online visibility.
This agent bridges the gap by providing:

- Instant AI-generated **digital business profiles**
- Step-by-step **UPI/QR payment setup** guides
- **Google Business Profile** onboarding
- **Government scheme recommendations** (PM SVANidhi, MUDRA, PMEGP, MSME)
- **Multilingual support** — English, Hindi, Telugu
- **RAG** over uploaded government PDF documents

---

## ✨ Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | 🤖 AI Chatbot | 24/7 IBM Granite-powered chatbot for all vendor queries |
| 2 | 📄 Business Profile | Auto-generate professional digital business profile |
| 3 | 🗺️ Google Business Guide | Step-by-step Google Maps listing setup |
| 4 | 📱 UPI QR Setup | Generate QR code + setup guide for digital payments |
| 5 | 🔍 Local SEO Tips | Free local SEO strategies for street businesses |
| 6 | 👥 Customer Engagement | Tips to build repeat customers |
| 7 | 💬 WhatsApp Marketing | Ready-to-send promotional messages |
| 8 | 🎉 Festival Promotions | Campaign ideas for Diwali, Eid, Holi, etc. |
| 9 | 💰 Pricing Suggestions | Competitive pricing recommendations |
| 10 | 🏛️ Govt Schemes | PM SVANidhi, MUDRA, PMEGP, MSME eligibility |
| 11 | 🌐 Multilingual | English, Hindi (हिंदी), Telugu (తెలుగు) |
| 12 | 📤 RAG Documents | Upload PDFs and query them via LangChain + FAISS |

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|-------------|
| **LLM** | IBM watsonx.ai Studio · IBM Granite 13B Instruct |
| **Backend** | Python 3.11+ · Flask 3.0 |
| **RAG** | LangChain · FAISS · HuggingFace Embeddings (all-MiniLM-L6-v2) |
| **PDF Processing** | PyPDF · pdfplumber |
| **Frontend** | HTML5 · CSS3 · Bootstrap 5.3 · Bootstrap Icons · Vanilla JS |
| **QR Code** | qrcode[pil] |
| **Deployment** | Render (free) · Gunicorn |

---

## 📁 Project Structure

```
street_vendor_agent/
├── app.py                  # Flask application — routes & API endpoints
├── watsonx_client.py       # IBM Granite LLM integration (AGENT_INSTRUCTIONS here)
├── rag_pipeline.py         # LangChain + FAISS RAG pipeline
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn start command for Render
├── .env.example            # Environment variable template
├── .gitignore              # Files to exclude from git
├── README.md               # This file
│
├── templates/              # Jinja2 HTML templates
│   ├── base.html           # Shared navbar, footer, Bootstrap
│   ├── home.html           # Landing page
│   ├── chatbot.html        # AI chatbot interface
│   ├── register.html       # Vendor registration form
│   ├── dashboard.html      # Vendor dashboard + AI tools
│   ├── about.html          # About page
│   ├── contact.html        # Contact page
│   └── 404.html            # 404 error page
│
├── static/
│   ├── css/
│   │   └── style.css       # Custom design system
│   └── js/
│       └── main.js         # Global JS utilities
│
├── uploads/
│   └── pdfs/               # Uploaded PDF documents (auto-created)
│
└── faiss_index/            # FAISS vector index (auto-created)
```

---

## 🔧 Local Setup

### Prerequisites

- Python 3.11 or higher
- pip
- IBM Cloud account (free tier available)
- IBM watsonx.ai project

### Step 1 — Clone / Download

```bash
git clone https://github.com/YOUR_USERNAME/street-vendor-agent.git
cd street-vendor-agent
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Activate (Windows):
venv\Scripts\activate

# Activate (macOS/Linux):
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** First run downloads the HuggingFace embedding model (~90 MB). Ensure internet access.

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your IBM credentials (see next section).

---

## 🔑 IBM watsonx.ai Configuration

### Getting Your Credentials

1. **IBM Cloud API Key:**
   - Go to [cloud.ibm.com](https://cloud.ibm.com)
   - Click your profile → **Manage** → **Access (IAM)** → **API Keys**
   - Click **Create an IBM Cloud API key** → copy it

2. **watsonx.ai Project ID:**
   - Go to [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)
   - Open your project → **Manage** tab → copy the **Project ID**

3. **Endpoint URL:**
   - Default: `https://us-south.ml.cloud.ibm.com`
   - Adjust to your region (eu-de, jp-tok, etc.)

### .env File

```env
WATSONX_API_KEY=your_actual_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-13b-instruct-v2
FLASK_SECRET_KEY=change-this-to-a-random-string
FLASK_DEBUG=False
```

---

## 🚀 Running the App

```bash
# Development
python app.py

# Production (Gunicorn)
gunicorn app:app --workers 1 --timeout 120
```

Open your browser: **http://localhost:5000**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Main chatbot (RAG + Granite) |
| `POST` | `/api/clear-chat` | Clear session chat history |
| `POST` | `/api/upi-guide` | Generate UPI setup guide |
| `POST` | `/api/seo-tips` | Local SEO recommendations |
| `POST` | `/api/govt-schemes` | Government scheme info |
| `POST` | `/api/festival-promo` | Festival promotion ideas |
| `POST` | `/api/pricing` | Pricing recommendations |
| `POST` | `/api/google-business-guide` | Google Business Profile guide |
| `POST` | `/api/whatsapp-messages` | WhatsApp marketing messages |
| `POST` | `/api/customer-tips` | Customer engagement tips |
| `POST` | `/api/upload-pdf` | Upload & index PDF document |
| `GET`  | `/api/rag-stats` | FAISS index statistics |
| `POST` | `/api/generate-qr` | Generate UPI QR code (base64 PNG) |

### Chat API Example

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "PM SVANidhi loan ke liye kaise apply karein?", "language": "hi"}'
```

### Upload PDF Example

```bash
curl -X POST http://localhost:5000/api/upload-pdf \
  -F "file=@pm_svanidhi_guide.pdf"
```

---

## 📚 RAG Pipeline

The RAG pipeline works as follows:

```
User Question
      │
      ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
      │
      ▼
FAISS Similarity Search (top-4 chunks)
      │
      ▼
Context injected into prompt
      │
      ▼
IBM Granite LLM generates grounded answer
```

**Seed knowledge base** (pre-loaded, no PDFs required to start):
- PM SVANidhi Scheme details
- MSME Udyam Registration
- UPI Payment Setup Guide
- Google Business Profile Guide
- Digital Marketing Guide
- PMEGP & MUDRA Yojana

**Upload your own PDFs** via the chatbot sidebar or `/api/upload-pdf`.

---

## 🌐 Deployment on Render (Free)

### Step 1 — Push to GitHub
See [GitHub Upload Guide](#github-upload-guide) below.

### Step 2 — Create Render Account
Go to [render.com](https://render.com) → Sign up with GitHub.

### Step 3 — New Web Service
1. Click **New** → **Web Service**
2. Connect your GitHub repository
3. Configure:
   - **Name:** `street-vendor-agent`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --timeout 120`
   - **Instance Type:** `Free`

### Step 4 — Add Environment Variables
In Render dashboard → **Environment** tab, add:

```
WATSONX_API_KEY     = your_api_key
WATSONX_PROJECT_ID  = your_project_id
WATSONX_URL         = https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID    = ibm/granite-13b-instruct-v2
FLASK_SECRET_KEY    = your-random-secret-key
FLASK_DEBUG         = False
```

### Step 5 — Deploy
Click **Create Web Service**. Render will build and deploy automatically.

> 💡 **Free Tier Notes:**
> - App sleeps after 15 minutes of inactivity (first request takes ~30s to wake).
> - 512 MB RAM — the HuggingFace model downloads on first boot.
> - FAISS index is **ephemeral** on free tier (resets on restart). Use Render's Persistent Disk ($7/mo) for permanent storage.

---

## 📤 GitHub Upload Guide

```bash
# 1. Initialize git in project folder
cd street_vendor_agent
git init

# 2. Add all files
git add .

# 3. First commit
git commit -m "feat: initial Street Vendor Digitalization Agent"

# 4. Create repo on GitHub (github.com → New repository)
#    Name: street-vendor-agent
#    Visibility: Public or Private
#    Do NOT initialize with README (we have one)

# 5. Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/street-vendor-agent.git
git branch -M main
git push -u origin main
```

> ⚠️ **Security:** Make sure `.env` is in `.gitignore` (it is). Never commit real API keys.

---

## 🤖 AGENT_INSTRUCTIONS

The assistant's behaviour is fully customizable in [`watsonx_client.py`](watsonx_client.py) via the `AGENT_INSTRUCTIONS` constant at the top of the file.

**Current configuration includes:**
- Friendly, practical persona targeting street vendors in India
- Language detection (English / Hindi / Telugu)
- Response length cap (400 words unless asked for more)
- Focus areas: UPI, Google Business, Govt Schemes, Marketing, Pricing
- No legal/financial advice beyond general guidance

**To customize:** Edit the `AGENT_INSTRUCTIONS` string in `watsonx_client.py`:

```python
AGENT_INSTRUCTIONS = """
You are the Street Vendor Digitalization Agent...
[Your custom instructions here]
"""
```

---

## 📜 License

MIT License — Free for personal, educational, and commercial use.

---

## 🙏 Acknowledgements

- **IBM watsonx.ai** — Granite LLM platform
- **LangChain** — RAG framework
- **FAISS** — Vector similarity search (Meta AI)
- **HuggingFace** — Sentence transformers
- **Bootstrap** — UI components
- **Government of India** — PM SVANidhi, MSME, Digital India initiatives

---

*Made with ❤️ for Bharat's street vendors | Powered by IBM Granite AI*
