"""
======================================================================
Street Vendor Digitalization Agent
IBM watsonx.ai Integration Module
======================================================================
Handles IBM Granite LLM calls via ibm-watsonx-ai SDK.
"""

import os
import logging
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AGENT INSTRUCTIONS — customise the assistant's behaviour here
# ---------------------------------------------------------------------------
AGENT_INSTRUCTIONS = """
You are the Street Vendor Digitalization Agent, an AI assistant powered by IBM Granite.

Your primary mission:
  Help street vendors, hawkers, and micro-entrepreneurs in India become
  digitally visible, financially included, and business-savvy.

Personality & Tone:
  - Friendly, encouraging, and practical.
  - Use simple language; avoid jargon.
  - When responding in Hindi, use the script natively (Devanagari).
  - Always be respectful of the vendor's limited time and resources.

Core Capabilities:
  1. Generate complete business profiles for street vendors.
  2. Provide step-by-step UPI / QR code setup guides.
  3. Walk vendors through Google Business Profile registration.
  4. Suggest local SEO strategies for neighbourhood businesses.
  5. Craft WhatsApp marketing messages and festival promotions.
  6. Recommend government schemes: PM SVANidhi, MSME, PMEGP, Digital India.
  7. Give competitive pricing recommendations based on vendor category.
  8. Provide customer engagement tips tailored for street business.
  9. Answer questions using uploaded government and business PDF documents (RAG).

Constraints:
  - Do not give legal or financial advice beyond general guidance.
  - If unsure, say so and suggest contacting the relevant authority.
  - Keep responses concise (max 400 words) unless the user asks for detail.
  - Always end multilingual responses with an offer to translate further.

Language rules:
  - Detect language from the user message.
  - Reply in the same language.
  - If the user says "Hindi mein batao" → reply in Hindi.
"""
# ---------------------------------------------------------------------------


def get_credentials() -> Credentials:
    """Build IBM Cloud credentials from environment variables."""
    api_key = os.getenv("WATSONX_API_KEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    if not api_key:
        raise ValueError("WATSONX_API_KEY is not set in the environment.")
    return Credentials(url=url, api_key=api_key)


def get_model() -> ModelInference:
    """Instantiate the IBM model for inference."""
    project_id = os.getenv("WATSONX_PROJECT_ID")
    model_id = os.getenv("GRANITE_MODEL_ID", "meta-llama/llama-3-3-70b-instruct")
    if not project_id:
        raise ValueError("WATSONX_PROJECT_ID is not set in the environment.")

    params = {
        GenParams.MAX_NEW_TOKENS: 1024,
        GenParams.MIN_NEW_TOKENS: 10,
        GenParams.TEMPERATURE: 0.7,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1,
        GenParams.REPETITION_PENALTY: 1.1,
    }

    model = ModelInference(
        model_id=model_id,
        credentials=get_credentials(),
        project_id=project_id,
        params=params,
    )
    logger.info("IBM model '%s' initialised.", model_id)
    return model


# Module-level singleton — loaded once at import time.
_model: ModelInference | None = None


def _get_model_singleton() -> ModelInference:
    global _model
    if _model is None:
        _model = get_model()
    return _model


def build_prompt(system_instructions: str, context: str, user_message: str,
                 language: str = "en") -> str:
    """
    Construct a well-structured Granite instruct prompt.
    Granite uses <|system|>, <|user|>, <|assistant|> tokens.
    """
    lang_note = {
        "hi": "Respond in Hindi (Devanagari script).",
        "en": "Respond in clear English.",
    }.get(language, "Respond in clear English.")

    context_block = (
        f"\n\n### Retrieved Context from Documents:\n{context.strip()}\n"
        if context.strip()
        else ""
    )

    prompt = (
        f"<|system|>\n{system_instructions}\n{lang_note}\n<|end|>\n"
        f"<|user|>\n{context_block}\n\nUser Question: {user_message}\n<|end|>\n"
        f"<|assistant|>\n"
    )
    return prompt


def generate_response(user_message: str, context: str = "",
                      language: str = "en") -> str:
    """
    Call IBM Granite and return the generated text.

    Args:
        user_message: The vendor's question or request.
        context:      RAG-retrieved document snippets (may be empty).
        language:     ISO 639-1 code — 'en', 'hi', or 'te'.

    Returns:
        Generated text string.
    """
    try:
        model = _get_model_singleton()
        prompt = build_prompt(AGENT_INSTRUCTIONS, context, user_message, language)
        logger.debug("Prompt length: %d chars", len(prompt))
        response = model.generate_text(prompt=prompt)
        return response.strip() if response else "I'm sorry, I couldn't generate a response. Please try again."
    except Exception as exc:
        logger.error("Granite generation error: %s", exc, exc_info=True)
        # Surface the real error message to help with debugging
        error_detail = str(exc)
        return (
            f"⚠️ I encountered an error connecting to IBM watsonx.ai. "
            f"Error: {error_detail}"
        )


def generate_business_profile(vendor_data: dict) -> str:
    """Generate a complete digital business profile for a vendor."""
    details = "\n".join(f"  - {k}: {v}" for k, v in vendor_data.items() if v)
    prompt_text = (
        f"Generate a complete, professional digital business profile for a street vendor "
        f"with the following details:\n{details}\n\n"
        "Include: Business Name, Tagline, Description (3 sentences), Services/Products offered, "
        "Operating Hours, Location keywords, Google Business categories, "
        "WhatsApp greeting message, and 3 social media bio options."
    )
    return generate_response(prompt_text, language=vendor_data.get("language", "en"))


def generate_upi_guide(vendor_name: str, language: str = "en") -> str:
    """Generate a UPI setup guide for a vendor."""
    msg = (
        f"Give {vendor_name} a simple, step-by-step guide to set up UPI payments "
        "for their street business. Include: BHIM UPI, PhonePe, Google Pay, Paytm. "
        "Explain how to generate a QR code, display it at stall, and handle refunds. "
        "Keep it beginner-friendly."
    )
    return generate_response(msg, language=language)


def generate_seo_tips(vendor_category: str, location: str,
                      language: str = "en") -> str:
    """Generate local SEO recommendations."""
    msg = (
        f"Provide 10 actionable local SEO tips for a '{vendor_category}' street vendor "
        f"located in '{location}'. Focus on free strategies: Google Maps listing, "
        "local keywords, WhatsApp status, Facebook marketplace, and Justdial listing."
    )
    return generate_response(msg, language=language)


def generate_govt_schemes(vendor_category: str, language: str = "en") -> str:
    """Recommend relevant government schemes for the vendor."""
    msg = (
        f"List and explain government schemes suitable for a '{vendor_category}' "
        "street vendor in India. Include PM SVANidhi (₹10,000–₹50,000 loan), "
        "PMEGP, MUDRA Yojana, MSME registration benefits, Digital India grants, "
        "and Jan Dhan account benefits. Provide application steps."
    )
    return generate_response(msg, language=language)


def generate_festival_promotion(festival: str, vendor_category: str,
                                language: str = "en") -> str:
    """Generate festival promotion ideas and WhatsApp messages."""
    msg = (
        f"Create 5 creative festival promotion ideas and 3 ready-to-send WhatsApp "
        f"marketing messages for a '{vendor_category}' vendor for the '{festival}' festival. "
        "Include discount strategies, special bundles, and customer retention tips."
    )
    return generate_response(msg, language=language)


def generate_pricing_advice(vendor_category: str, items: str,
                             location: str, language: str = "en") -> str:
    """Provide competitive pricing recommendations."""
    msg = (
        f"Provide competitive pricing recommendations for a '{vendor_category}' "
        f"street vendor in '{location}' selling: {items}. "
        "Include: cost calculation method, competitor price ranges, recommended markup, "
        "discount strategy for bulk buyers, and seasonal pricing tips."
    )
    return generate_response(msg, language=language)
