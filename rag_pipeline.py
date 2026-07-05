"""
======================================================================
Street Vendor Digitalization Agent
RAG Pipeline — LangChain + FAISS + HuggingFace Embeddings
======================================================================
Handles PDF ingestion, vector storage, and document retrieval.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

load_dotenv()
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
PDF_UPLOAD_FOLDER = os.getenv("PDF_UPLOAD_FOLDER", "uploads/pdfs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ensure directories exist
Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
Path(PDF_UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Seed documents — bundled knowledge base (no PDF required to start)
# ---------------------------------------------------------------------------
SEED_DOCUMENTS = [
    Document(
        page_content="""PM SVANidhi Scheme (PM Street Vendor's AtmaNirbhar Nidhi):
Launched in June 2020 by Ministry of Housing and Urban Affairs.
Provides working capital loans to street vendors:
- First loan: ₹10,000 (repayment in 12 months)
- Second loan: ₹20,000 (after repayment of first)
- Third loan: ₹50,000
Interest subsidy of 7% per annum credited directly to account.
Digital transaction incentive: ₹1,200 per year for digital payments.
Eligibility: Street vendors vending in urban areas before March 24, 2020.
Documents required: Aadhaar, vending certificate or letter of recommendation from ULB.
Apply at: pmsvanidhi.mohua.gov.in or nearest Common Service Centre (CSC).""",
        metadata={"source": "PM SVANidhi Scheme", "category": "government_scheme"},
    ),
    Document(
        page_content="""MSME Registration (Udyam Registration):
Micro, Small and Medium Enterprises registration provides numerous benefits.
Udyam Registration Portal: udyamregistration.gov.in
Benefits of MSME Registration:
- Priority sector lending from banks
- Collateral-free loans under CGTMSE scheme
- Subsidies on ISO certification
- Preference in government tenders
- Protection against delayed payments (MSMED Act)
- Credit Linked Capital Subsidy Scheme (CLCSS)
- Technology upgradation assistance

Eligibility for Micro Enterprise:
  - Investment in plant & machinery: up to ₹1 crore
  - Annual turnover: up to ₹5 crore
Documents: Aadhaar card, PAN card, bank account details.
Registration is FREE and completely online.""",
        metadata={"source": "MSME Udyam Registration", "category": "government_scheme"},
    ),
    Document(
        page_content="""UPI Payment Setup Guide for Street Vendors:
Step 1 — Download App: Install BHIM UPI, PhonePe, or Google Pay from Play Store.
Step 2 — Registration: Open app, enter mobile number linked to bank account.
Step 3 — Bank Account: Select your bank and verify with OTP sent to mobile.
Step 4 — Set UPI PIN: Enter last 6 digits of debit card and expiry date, set 4/6 digit PIN.
Step 5 — Create UPI ID: Your ID will be mobile@bankname (e.g., 9876543210@ybl).
Step 6 — Generate QR Code: Go to "Receive Money" → "Show QR Code" → Download and Print.
Step 7 — Display at Stall: Print A4 QR code and laminate it. Place at eye level.

Accepting Payments:
- Customer scans QR → enters amount → enters their UPI PIN → payment credited.
- You receive SMS and app notification instantly.
- No internet needed for receiving (customer's phone does the work).

Tips:
- Accept GPay, PhonePe, Paytm, BHIM — all scan the same QR.
- Check balance by clicking "Check Balance" in app.
- For disputes, call UPI helpline: 18001201740""",
        metadata={"source": "UPI Setup Guide", "category": "digital_payments"},
    ),
    Document(
        page_content="""Google Business Profile Setup Guide:
Step 1 — Visit: business.google.com and sign in with Gmail.
Step 2 — Add Business: Click "Add your business to Google".
Step 3 — Business Name: Enter your exact business name.
Step 4 — Business Category: Choose most relevant category (e.g., Street food restaurant, Florist, Vegetable store).
Step 5 — Location: Add your stall address or select "I deliver goods/services".
Step 6 — Contact: Add mobile number and website (optional).
Step 7 — Verification: Google will call or send postcard to verify. Choose call option.
Step 8 — Complete Profile: Add photos (minimum 3), business hours, description.

Benefits:
- Appear in "near me" searches on Google Maps.
- Customers can call directly from search results.
- Collect Google Reviews to build trust.
- Free messaging with customers.
- Insights on how customers find you.

Tips for better ranking:
- Add photos every week.
- Respond to all reviews within 24 hours.
- Post weekly updates about offers and new products.
- Use local keywords in description.""",
        metadata={"source": "Google Business Profile Guide", "category": "digital_marketing"},
    ),
    Document(
        page_content="""Digital Marketing Guide for Street Vendors:
WhatsApp Business Setup:
1. Download WhatsApp Business app (free).
2. Create business profile with address, hours, description.
3. Set auto-reply for when you're busy.
4. Create catalogue of products with photos and prices.
5. Share catalogue link with customers.
6. Create broadcast list of regular customers for promotions.

Facebook Marketplace:
- Create Facebook page for your business (free).
- List products on Facebook Marketplace.
- Join local community groups and share your products.
- Respond to messages within 1 hour for better ranking.

Instagram Tips:
- Create business account.
- Post 3 photos per week.
- Use local hashtags: #[CityName]StreetFood #[AreaName]Vendor
- Stories disappear in 24 hours — use for daily specials.
- Reels get highest reach — 15-second video of your preparation.

Local SEO Tips:
- List on Justdial (justdial.com) — free listing.
- List on Sulekha and IndiaMART.
- Ask customers to review you on Google.
- Create business card with QR code to your Google profile.""",
        metadata={"source": "Digital Marketing Guide", "category": "marketing"},
    ),
    Document(
        page_content="""PMEGP Scheme (Prime Minister's Employment Generation Programme):
Administered by: Khadi and Village Industries Commission (KVIC).
Purpose: Financial assistance for setting up new micro enterprises.

Subsidy:
- Urban area: 15% of project cost (general), 25% (SC/ST/Women/Minority/Ex-servicemen/PH/NER/Hill areas)
- Rural area: 25% (general), 35% (special categories)
- Maximum project cost: ₹50 lakhs (manufacturing), ₹20 lakhs (service sector)

Eligibility:
- Any individual above 18 years.
- VIII pass for projects above ₹10 lakhs.
- Self-help groups, institutions registered under Societies Registration Act.
- Production-based cooperatives, charitable trusts.

How to Apply:
1. Visit: kviconline.gov.in
2. Register and fill application form.
3. Submit business plan/project report.
4. Banks will process loan after KVIC/DIC approval.

Mudra Loan (Pradhan Mantri Mudra Yojana):
- Shishu: Loans up to ₹50,000
- Kishore: ₹50,001 to ₹5 lakh
- Tarun: ₹5 lakh to ₹10 lakh
No collateral required. Apply at any bank, MFI, or NBFC.""",
        metadata={"source": "PMEGP and MUDRA Schemes", "category": "government_scheme"},
    ),
]


# ---------------------------------------------------------------------------
# Embedding & Vector Store
# ---------------------------------------------------------------------------

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[FAISS] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return cached HuggingFace embeddings instance."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> FAISS:
    """Return the FAISS vector store, loading from disk or creating fresh."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    index_file = Path(FAISS_INDEX_PATH) / "index.faiss"
    if index_file.exists():
        logger.info("Loading FAISS index from disk: %s", FAISS_INDEX_PATH)
        _vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    else:
        logger.info("Creating new FAISS index with seed documents.")
        _vector_store = FAISS.from_documents(SEED_DOCUMENTS, get_embeddings())
        _vector_store.save_local(FAISS_INDEX_PATH)
        logger.info("FAISS index saved to %s", FAISS_INDEX_PATH)

    return _vector_store


def add_documents_to_store(documents: List[Document]) -> None:
    """Add new documents to the vector store, persist, and reset the cache."""
    global _vector_store
    store = get_vector_store()
    store.add_documents(documents)
    store.save_local(FAISS_INDEX_PATH)
    # Reset singleton so next retrieval reloads from disk with all new documents
    _vector_store = None
    logger.info("Added %d document chunks to FAISS index.", len(documents))


def ingest_pdf(file_path: str) -> int:
    """
    Load a PDF, split into chunks, embed, and add to FAISS.

    Returns:
        Number of chunks added.
    """
    logger.info("Ingesting PDF: %s", file_path)
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(raw_docs)
    add_documents_to_store(chunks)
    logger.info("Ingested %d chunks from %s", len(chunks), file_path)
    return len(chunks)


def retrieve_context(query: str, k: int = 4, pdf_only: bool = False) -> str:
    """
    Perform similarity search and return joined context string.

    Args:
        query:    User question.
        k:        Number of top documents to retrieve.
        pdf_only: If True, only return chunks from uploaded PDFs (not seed docs).

    Returns:
        Concatenated relevant document snippets.
    """
    try:
        store = get_vector_store()
        # Fetch more candidates when filtering so we have enough after the filter
        fetch_k = k * 4 if pdf_only else k
        docs = store.similarity_search(query, k=fetch_k)
        if not docs:
            return ""

        if pdf_only:
            # Keep only chunks that came from uploaded PDF files
            docs = [d for d in docs if str(d.metadata.get("source", "")).startswith("uploads")]
            docs = docs[:k]

        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Document")
            context_parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n".join(context_parts)
    except Exception as exc:
        logger.error("RAG retrieval error: %s", exc, exc_info=True)
        return ""


def get_index_stats() -> dict:
    """Return basic stats about the current vector index."""
    try:
        store = get_vector_store()
        count = store.index.ntotal if hasattr(store, "index") else "unknown"
        return {"total_vectors": count, "index_path": FAISS_INDEX_PATH}
    except Exception:
        return {"total_vectors": 0, "index_path": FAISS_INDEX_PATH}


def reset_to_seed_documents() -> None:
    """Wipe the FAISS index and rebuild it from seed documents only."""
    global _vector_store
    import shutil

    # Delete index files from disk
    index_dir = Path(FAISS_INDEX_PATH)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Delete all uploaded PDFs
    upload_dir = Path(PDF_UPLOAD_FOLDER)
    if upload_dir.exists():
        for f in upload_dir.glob("*.pdf"):
            f.unlink()

    # Rebuild from seed docs
    _vector_store = None
    _vector_store = FAISS.from_documents(SEED_DOCUMENTS, get_embeddings())
    _vector_store.save_local(FAISS_INDEX_PATH)
    logger.info("Index reset to %d seed documents.", len(SEED_DOCUMENTS))
