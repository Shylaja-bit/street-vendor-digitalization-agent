"""
======================================================================
Street Vendor Digitalization Agent — app.py
Flask Backend | IBM watsonx.ai (Granite) | LangChain RAG | FAISS
======================================================================
"""

import os
import io
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, flash
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# --- Load environment before importing our modules ---
load_dotenv()

import watsonx_client as wx
import rag_pipeline as rag

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

PDF_UPLOAD_FOLDER = os.getenv("PDF_UPLOAD_FOLDER", "uploads/pdfs")
ALLOWED_EXTENSIONS = {"pdf"}
Path(PDF_UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Helper: session chat history
# ---------------------------------------------------------------------------
def get_chat_history() -> list:
    if "chat_history" not in session:
        session["chat_history"] = []
    return session["chat_history"]


def append_chat(role: str, content: str) -> None:
    history = get_chat_history()
    history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M"),
    })
    session["chat_history"] = history[-50:]  # Keep last 50 messages


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html", chat_history=get_chat_history())


# ---------------------------------------------------------------------------
# Multi-profile store (max 10) — persists across restarts
# ---------------------------------------------------------------------------
PROFILE_STORE = Path("vendor_profiles.json")
MAX_PROFILES  = 10


def load_all_profiles() -> list:
    """Load all saved profiles from disk."""
    try:
        if PROFILE_STORE.exists():
            return json.loads(PROFILE_STORE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not load profiles: %s", e)
    return []


def save_all_profiles(profiles: list) -> None:
    """Save all profiles to disk."""
    try:
        PROFILE_STORE.write_text(
            json.dumps(profiles, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        logger.warning("Could not save profiles: %s", e)


def add_profile(vendor_data: dict, profile_text: str) -> tuple[list, int]:
    """
    Add a new profile to the store (max 10).
    Returns (all_profiles, new_index).
    """
    profiles = load_all_profiles()
    entry = {
        "id": str(uuid.uuid4()),
        "vendor": vendor_data,
        "profile": profile_text,
        "created_at": datetime.now().strftime("%d %b %Y %H:%M"),
    }
    profiles.append(entry)
    # Keep only the latest MAX_PROFILES
    if len(profiles) > MAX_PROFILES:
        profiles = profiles[-MAX_PROFILES:]
    save_all_profiles(profiles)
    return profiles, len(profiles) - 1


@app.route("/register", methods=["GET", "POST"])
def register():
    all_profiles = load_all_profiles()
    if request.method == "POST":
        # Check limit
        if len(all_profiles) >= MAX_PROFILES:
            flash(f"⚠️ Maximum {MAX_PROFILES} profiles reached. Please delete one first.", "warning")
            return redirect(url_for("dashboard"))

        vendor_data = {
            "name": request.form.get("vendor_name", ""),
            "business_name": request.form.get("business_name", ""),
            "category": request.form.get("category", ""),
            "location": request.form.get("location", ""),
            "city": request.form.get("city", ""),
            "products": request.form.get("products", ""),
            "mobile": request.form.get("mobile", ""),
            "language": request.form.get("language", "en"),
            "years_in_business": request.form.get("years", ""),
        }
        profile_text = wx.generate_business_profile(vendor_data)
        profiles, new_idx = add_profile(vendor_data, profile_text)

        # Set active profile in session
        session["vendor_profile"]   = vendor_data
        session["generated_profile"] = profile_text
        session["active_profile_id"] = profiles[new_idx]["id"]

        flash(f"✅ Profile for '{vendor_data['business_name']}' generated! ({len(profiles)}/{MAX_PROFILES} profiles used)", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html",
                           profiles_count=len(all_profiles),
                           max_profiles=MAX_PROFILES)


@app.route("/dashboard")
def dashboard():
    all_profiles = load_all_profiles()
    vendor  = session.get("vendor_profile", {})
    profile = session.get("generated_profile", "")

    # If session empty (after restart) load the last saved profile
    if not profile and all_profiles:
        last = all_profiles[-1]
        vendor  = last["vendor"]
        profile = last["profile"]
        session["vendor_profile"]    = vendor
        session["generated_profile"] = profile
        session["active_profile_id"] = last["id"]

    active_id = session.get("active_profile_id", "")
    stats = rag.get_index_stats()
    return render_template("dashboard.html",
                           vendor=vendor,
                           profile=profile,
                           stats=stats,
                           all_profiles=all_profiles,
                           active_profile_id=active_id,
                           max_profiles=MAX_PROFILES)


@app.route("/api/switch-profile", methods=["POST"])
def api_switch_profile():
    """Switch the active vendor profile."""
    data = request.get_json(force=True, silent=True) or {}
    profile_id = data.get("profile_id", "")
    all_profiles = load_all_profiles()
    entry = next((p for p in all_profiles if p["id"] == profile_id), None)
    if not entry:
        return jsonify({"error": "Profile not found."}), 404
    session["vendor_profile"]    = entry["vendor"]
    session["generated_profile"] = entry["profile"]
    session["active_profile_id"] = profile_id
    return jsonify({"status": "ok", "business_name": entry["vendor"].get("business_name", "")})


@app.route("/api/delete-profile", methods=["POST"])
def api_delete_profile():
    """Delete a vendor profile by id."""
    data = request.get_json(force=True, silent=True) or {}
    profile_id = data.get("profile_id", "")
    all_profiles = load_all_profiles()
    new_profiles = [p for p in all_profiles if p["id"] != profile_id]
    if len(new_profiles) == len(all_profiles):
        return jsonify({"error": "Profile not found."}), 404
    save_all_profiles(new_profiles)

    # If deleted profile was active, switch to last remaining
    if session.get("active_profile_id") == profile_id:
        if new_profiles:
            last = new_profiles[-1]
            session["vendor_profile"]    = last["vendor"]
            session["generated_profile"] = last["profile"]
            session["active_profile_id"] = last["id"]
        else:
            session.pop("vendor_profile", None)
            session.pop("generated_profile", None)
            session.pop("active_profile_id", None)

    return jsonify({"status": "ok", "remaining": len(new_profiles)})


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# ---------------------------------------------------------------------------
# API Routes — Chatbot
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Main chat endpoint — RAG + Granite response."""
    data = request.get_json(force=True, silent=True) or {}
    user_message = (data.get("message") or "").strip()
    language = data.get("language", "en")
    pdf_only = data.get("pdf_only", False)  # If True, answer only from uploaded PDFs

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Retrieve relevant context via RAG
    context = rag.retrieve_context(user_message, k=4, pdf_only=pdf_only)

    # If pdf_only mode but no context found, tell the user clearly
    if pdf_only and not context:
        response_text = (
            "⚠️ I couldn't find relevant information in the uploaded PDF. "
            "Please make sure the PDF was uploaded successfully and ask a question related to its content."
        )
        append_chat("user", user_message)
        append_chat("assistant", response_text)
        return jsonify({
            "response": response_text,
            "sources_used": False,
            "timestamp": datetime.now().strftime("%H:%M"),
        })

    # Generate response via IBM Granite
    if pdf_only and context:
        # Strict instruction: answer only from the provided context
        strict_message = (
            f"Answer the following question ONLY using the provided document context. "
            f"Do not use any outside knowledge. If the answer is not in the document, say so.\n\n"
            f"Question: {user_message}"
        )
        response_text = wx.generate_response(strict_message, context=context, language=language)
    else:
        response_text = wx.generate_response(user_message, context=context, language=language)

    # Save to session history
    append_chat("user", user_message)
    append_chat("assistant", response_text)

    return jsonify({
        "response": response_text,
        "sources_used": bool(context),
        "timestamp": datetime.now().strftime("%H:%M"),
    })


@app.route("/api/clear-chat", methods=["POST"])
def api_clear_chat():
    session.pop("chat_history", None)
    return jsonify({"status": "cleared"})


# ---------------------------------------------------------------------------
# API Routes — Feature Generators
# ---------------------------------------------------------------------------

@app.route("/api/upi-guide", methods=["POST"])
def api_upi_guide():
    data = request.get_json(force=True, silent=True) or {}
    vendor_name = data.get("vendor_name", "Vendor")
    language = data.get("language", "en")
    result = wx.generate_upi_guide(vendor_name, language=language)
    return jsonify({"result": result})


@app.route("/api/seo-tips", methods=["POST"])
def api_seo_tips():
    data = request.get_json(force=True, silent=True) or {}
    category = data.get("category", "general vendor")
    location = data.get("location", "India")
    language = data.get("language", "en")
    result = wx.generate_seo_tips(category, location, language=language)
    return jsonify({"result": result})


@app.route("/api/govt-schemes", methods=["POST"])
def api_govt_schemes():
    data = request.get_json(force=True, silent=True) or {}
    category = data.get("category", "street vendor")
    language = data.get("language", "en")
    # Use RAG to enrich with document context
    context = rag.retrieve_context(f"government schemes for {category}", k=5)
    result = wx.generate_govt_schemes(category, language=language)
    return jsonify({"result": result})


@app.route("/api/festival-promo", methods=["POST"])
def api_festival_promo():
    data = request.get_json(force=True, silent=True) or {}
    festival = data.get("festival", "Diwali")
    category = data.get("category", "street vendor")
    language = data.get("language", "en")
    result = wx.generate_festival_promotion(festival, category, language=language)
    return jsonify({"result": result})


@app.route("/api/pricing", methods=["POST"])
def api_pricing():
    data = request.get_json(force=True, silent=True) or {}
    category = data.get("category", "street vendor")
    items = data.get("items", "")
    location = data.get("location", "India")
    language = data.get("language", "en")
    result = wx.generate_pricing_advice(category, items, location, language=language)
    return jsonify({"result": result})


@app.route("/api/google-business-guide", methods=["POST"])
def api_google_business():
    data = request.get_json(force=True, silent=True) or {}
    vendor_name = data.get("vendor_name", "Vendor")
    category = data.get("category", "street vendor")
    language = data.get("language", "en")
    context = rag.retrieve_context("Google Business Profile setup guide", k=3)
    msg = (
        f"Provide a detailed Google Business Profile setup guide for "
        f"'{vendor_name}', a '{category}' street vendor. "
        "Include all steps, tips for adding photos, getting reviews, "
        "and appearing in local search results."
    )
    result = wx.generate_response(msg, context=context, language=language)
    return jsonify({"result": result})


@app.route("/api/whatsapp-messages", methods=["POST"])
def api_whatsapp():
    data = request.get_json(force=True, silent=True) or {}
    business_name = data.get("business_name", "My Business")
    category = data.get("category", "street vendor")
    occasion = data.get("occasion", "general promotion")
    language = data.get("language", "en")
    msg = (
        f"Write 5 professional WhatsApp marketing messages for "
        f"'{business_name}', a '{category}' business, for: {occasion}. "
        "Include emojis, a call-to-action, and keep each under 100 words."
    )
    result = wx.generate_response(msg, language=language)
    return jsonify({"result": result})


@app.route("/api/customer-tips", methods=["POST"])
def api_customer_tips():
    data = request.get_json(force=True, silent=True) or {}
    category = data.get("category", "street vendor")
    language = data.get("language", "en")
    msg = (
        f"Give 10 practical customer engagement and retention tips for "
        f"a '{category}' street vendor. Focus on zero-cost or low-cost strategies "
        "for building loyal customers in Indian markets."
    )
    result = wx.generate_response(msg, language=language)
    return jsonify({"result": result})


# ---------------------------------------------------------------------------
# API Routes — PDF Upload & RAG
# ---------------------------------------------------------------------------

@app.route("/api/upload-pdf", methods=["POST"])
def api_upload_pdf():
    """Upload a PDF document and ingest it into the FAISS vector store."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(PDF_UPLOAD_FOLDER, unique_name)
    file.save(save_path)
    logger.info("PDF saved: %s", save_path)

    try:
        chunks = rag.ingest_pdf(save_path)
        return jsonify({
            "status": "success",
            "filename": filename,
            "chunks_indexed": chunks,
            "message": f"✅ '{filename}' indexed successfully with {chunks} chunks.",
        })
    except Exception as exc:
        logger.error("PDF ingestion error: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to process PDF: {str(exc)}"}), 500


@app.route("/api/rag-stats", methods=["GET"])
def api_rag_stats():
    stats = rag.get_index_stats()
    return jsonify(stats)


@app.route("/api/clear-index", methods=["POST"])
def api_clear_index():
    """Delete all uploaded PDF chunks from the FAISS index, keeping only seed docs."""
    try:
        rag.reset_to_seed_documents()
        return jsonify({"status": "success", "message": "Index reset to seed documents."})
    except Exception as exc:
        logger.error("Index reset error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# QR Code Generator
# ---------------------------------------------------------------------------

@app.route("/api/generate-qr", methods=["POST"])
def api_generate_qr():
    """Generate a QR code PNG (base64) for a UPI payment link."""
    try:
        import qrcode
        import base64

        data = request.get_json(force=True, silent=True) or {}
        upi_id = data.get("upi_id", "")
        vendor_name = data.get("vendor_name", "Vendor")
        amount = data.get("amount", "")

        if not upi_id:
            return jsonify({"error": "UPI ID is required."}), 400

        upi_url = f"upi://pay?pa={upi_id}&pn={vendor_name}"
        if amount:
            upi_url += f"&am={amount}"

        qr = qrcode.QRCode(version=2, box_size=10, border=4)
        qr.add_data(upi_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        return jsonify({
            "qr_image": f"data:image/png;base64,{img_b64}",
            "upi_url": upi_url,
        })
    except ImportError:
        return jsonify({"error": "qrcode library not installed."}), 500
    except Exception as exc:
        logger.error("QR generation error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


@app.errorhandler(500)
def server_error(e):
    logger.error("Internal server error: %s", e)
    return jsonify({"error": "Internal server error. Please try again."}), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    logger.info("Starting Street Vendor Digitalization Agent on port %d", port)
    # Pre-load vector store on startup
    try:
        rag.get_vector_store()
        logger.info("Vector store ready.")
    except Exception as e:
        logger.warning("Vector store pre-load failed (will retry on first request): %s", e)
    app.run(host="0.0.0.0", port=port, debug=debug)
