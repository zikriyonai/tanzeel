"""
Tanzeel Intelligence — Multi-Model Proxy Server.

ONE URL, MANY MODELS.
Frontend always calls the same endpoint; server routes to the right
Hugging Face model based on the "model" field in the request.

Architecture:
    [Frontend] → [This server] → [HF Model: tanzeel-alpha]
                              → [HF Model: tanzeel-beta]
                              → [HF Model: tanzeel-gamma]
                              → [HF Model: any-future-model]

Environment variables (Render dashboard):
    HF_TOKEN          = hf_xxxxxxxxxx
    TANZEEL_API_KEY   = tz_xxxxxxxxxxxxx
    ALLOWED_ORIGINS   = https://tanzeelai.web.app
    MODELS            = JSON map of model_name → HF repo id
                        (see default below)

TWO FIXES vs. the earlier draft of this file:

1. SYSTEM PROMPTS ARE HONEST. The base model isn't advertised anywhere
   in the UI or led with in casual conversation — normal, standard
   product branding. But the prompts no longer instruct the model to
   deny/deflect if a user directly and sincerely asks what the
   underlying model is. Apache 2.0 (Qwen's license) makes not
   advertising the base model legally fine; it doesn't make instructing
   active concealment when asked ethically fine — those are different
   things. If you change these prompts, please keep that distinction.

2. PROPER CHAT TEMPLATING. The earlier version hand-built a prompt
   string using `<|system|>`/`<|user|>` tags and hit HF's raw
   text-generation endpoint. Those tags aren't what Qwen was actually
   trained on (Qwen uses ChatML: `<|im_start|>role ... <|im_end|>`) —
   sending the wrong format can silently degrade output quality. This
   version sends a proper `messages` list to HF's OpenAI-compatible
   `/v1/chat/completions` endpoint instead, which applies the correct
   chat template for whichever model is being called automatically —
   correct regardless of what chat format the underlying model actually
   uses, and one less thing to keep in sync by hand.
"""

import os
import json
import uuid
import time
import requests
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
TANZEEL_API_KEY = os.getenv("TANZEEL_API_KEY", "").strip()
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "https://tanzeelai.web.app"
).split(",")

# ── Model registry ───────────────────────────────────────────────────────
# Map of PUBLIC name (what frontend sends) → HF repo id (where it's hosted).
# Add new models here — no other code change needed.
#
# NOTE: "tanzeel-beta" below points at "zeaipc/tanzeel-beta" — the earlier
# draft had this pointing at "zeaipc/tanzeel-delta", which looked like a
# copy-paste typo (the public name and repo name didn't match any other
# entry's pattern). Fixed here; change it back if that was intentional.
#
# You can override this on Render by setting the MODELS env var to a JSON
# string with the same shape.
DEFAULT_MODELS = {
    "tanzeel-alpha": "zeaipc/tanzeel-alpha",
    "tanzeel-beta":  "zeaipc/tanzeel-beta",
    "tanzeel-gamma": "zeaipc/tanzeel-gamma",
}

def load_models() -> dict:
    raw = os.getenv("MODELS", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except json.JSONDecodeError:
            print("[warn] MODELS env var is not valid JSON — using defaults.")
    return DEFAULT_MODELS

MODELS: dict[str, str] = load_models()
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "").strip() or next(iter(MODELS))

# Per-model system prompts — persona differs per model, but the identity/
# honesty rules are shared (see SHARED_IDENTITY_RULES below).
SHARED_IDENTITY_RULES = (
    "You are part of the Tanzeel Intelligence family of models, created by "
    "ZEAIPC (Zikr-e-Ameen Innovations & Programming Corporation), founded by "
    "Arman Ansari. If asked who made you: \"ZEAIPC, founded by Arman Ansari.\" "
    "If asked directly and sincerely what your underlying/base model is, answer "
    "honestly rather than denying or deflecting — only avoid bringing it up "
    "unprompted, the same way most products built on a base model don't lead "
    "with that detail in casual use. Your fine-tuning, training data, and "
    "adaptation recipe are ZEAIPC's own proprietary work."
)

SYSTEM_PROMPTS = {
    "tanzeel-alpha": (
        f"{SHARED_IDENTITY_RULES}\n\n"
        "You are Tanzeel Alpha: a friendly, conversational Hinglish assistant. "
        "Reply in the user's language (Hindi, English, or Hinglish), matching their tone."
    ),
    "tanzeel-beta": (
        f"{SHARED_IDENTITY_RULES}\n\n"
        "You are Tanzeel Beta: analytical and precise. Be structured and technical "
        "where it helps, while staying clear and easy to follow."
    ),
    "tanzeel-gamma": (
        f"{SHARED_IDENTITY_RULES}\n\n"
        "You are Tanzeel Gamma: creative and expressive. Be imaginative and engaging "
        "while staying genuinely helpful."
    ),
}
DEFAULT_SYSTEM_PROMPT = SHARED_IDENTITY_RULES

HF_CHAT_API_TEMPLATE = "https://router.huggingface.co/hf-inference/models/{repo}/v1/chat/completions"
# ────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Tanzeel Multi-Model API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

SESSIONS: dict[str, list[dict]] = {}
MAX_SESSION_TURNS = 10


# ─────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    model: Optional[str] = None          # ← which Tanzeel variant?
    session_id: Optional[str] = None
    max_new_tokens: int = Field(default=500, ge=1, le=2048)
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    allow_web_fallback: bool = True
    api_key: Optional[str] = None        # legacy fallback (header preferred)


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model: str                # resolved public name
    source: str               # "tanzeel" or "web_search"


class ResetRequest(BaseModel):
    session_id: str


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def resolve_model(requested: Optional[str]) -> tuple[str, str]:
    """Return (public_name, hf_repo). Fall back to default if unknown."""
    if not requested:
        requested = DEFAULT_MODEL
    if requested not in MODELS:
        # Unknown model — fall back to default (don't 400; be forgiving)
        return DEFAULT_MODEL, MODELS[DEFAULT_MODEL]
    return requested, MODELS[requested]


def web_search_fallback(query: str) -> Optional[str]:
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=5,
        )
        data = resp.json()
        text = data.get("AbstractText") or ""
        if not text and data.get("RelatedTopics"):
            first = data["RelatedTopics"][0]
            text = first.get("Text", "") if isinstance(first, dict) else ""
        return text.strip() or None
    except Exception:
        return None


def looks_low_quality(reply: str) -> bool:
    if not reply or len(reply.split()) < 3:
        return True
    words = reply.split()
    if len(words) >= 6 and len(set(words)) <= max(2, len(words) // 4):
        return True
    return False


def call_hf_model(
    repo: str,
    public_name: str,
    user_message: str,
    history: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    if not HF_TOKEN:
        raise HTTPException(500, "HF_TOKEN is not configured.")

    system_prompt = SYSTEM_PROMPTS.get(public_name, DEFAULT_SYSTEM_PROMPT)

    # Proper messages list — HF applies the correct chat template for
    # whichever model this repo actually is (ChatML for Qwen, etc.)
    # server-side. No manual prompt-string building, no risk of using the
    # wrong special tokens for a given base model.
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": repo,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    url = HF_CHAT_API_TEMPLATE.format(repo=repo)
    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if resp.status_code == 503:
        time.sleep(20)
        resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"HF error for '{repo}' ({resp.status_code}): {resp.text[:300]}",
        )

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return ""


# ─────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Tanzeel Multi-Model API",
        "status": "online",
        "default_model": DEFAULT_MODEL,
        "models": list(MODELS.keys()),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_count": len(MODELS),
        "default_model": DEFAULT_MODEL,
        "hf_configured": bool(HF_TOKEN),
        "api_key_configured": bool(TANZEEL_API_KEY),
    }


@app.get("/models")
def list_models():
    """
    Frontend can call this to discover available models.
    Returns public names + HF repos (repo hidden in production if you prefer).
    """
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {"name": name, "repo": repo}
            for name, repo in MODELS.items()
        ],
    }


@app.post("/reset")
def reset(req: ResetRequest):
    SESSIONS.pop(req.session_id, None)
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    # ── Auth ─────────────────────────────────────────────────────────────
    provided_key = x_api_key or req.api_key
    if TANZEEL_API_KEY and provided_key != TANZEEL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    # ── Model routing ────────────────────────────────────────────────────
    public_name, repo = resolve_model(req.model)

    # ── Session ──────────────────────────────────────────────────────────
    session_id = req.session_id or str(uuid.uuid4())
    # Sessions are namespaced per model so switching models doesn't mix context
    session_key = f"{public_name}:{session_id}"
    history = SESSIONS.get(session_key, [])

    # ── Inference ────────────────────────────────────────────────────────
    try:
        reply_text = call_hf_model(
            repo=repo,
            public_name=public_name,
            user_message=req.message,
            history=history,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    source = "tanzeel"

    if req.allow_web_fallback and looks_low_quality(reply_text):
        web_answer = web_search_fallback(req.message)
        if web_answer:
            reply_text = web_answer
            source = "web_search"

    if not reply_text:
        reply_text = "(no response — try rephrasing)"

    # ── Save session ─────────────────────────────────────────────────────
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply_text})
    if len(history) > MAX_SESSION_TURNS * 2:
        history = history[-MAX_SESSION_TURNS * 2:]
    SESSIONS[session_key] = history

    return ChatResponse(
        reply=reply_text,
        session_id=session_id,
        model=public_name,
        source=source,
    )


# ─────────────────────────────────────────────────────────────────────────
# LOCAL DEV
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Models loaded: {list(MODELS.keys())}")
    print(f"Default model: {DEFAULT_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=port)
