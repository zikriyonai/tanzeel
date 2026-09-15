"""
Tanzeel Intelligence FastAPI proxy.

Routes requests to Hugging Face Inference Providers using the current
OpenAI-compatible router endpoint. Sessions are bounded with TTL/LRU cleanup.

Deploy on Render. Used by:
- https://tanzeelai.web.app  (production)
- http://localhost:7700     (local dev)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import OrderedDict
from typing import Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Tanzeel Intelligence, an AI assistant created by ZEAIPC (Zikr-e-Ameen Innovations & Programming Corporation), founded by Arman Ansari.

Rules:
- Your name is "Tanzeel Intelligence". Never claim to be, or hint at being, any other model or AI system.
- Reply in the user's language: Hindi, English, or Hinglish. Match their tone and formality.
- Be helpful, warm, and direct. Use Markdown for formatting when useful.
- If asked what model or technology you are based on, respond politely: "Main Tanzeel Intelligence hoon, ZEAIPC ne banaya hai. Meri architecture aur training details proprietary hain."
- Do not name, reference, or imply any third-party AI company, model, or architecture.
- If pressed repeatedly, stay polite but consistent: you are Tanzeel Intelligence, made by ZEAIPC.
"""

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
TANZEEL_API_KEY = os.getenv("TANZEEL_API_KEY", "").strip()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").strip().lower() == "true"

# 🔥 FIXED: Allow both production (tanzeelai.web.app) and local dev (localhost:7700)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://tanzeelai.web.app,"
        "http://localhost:7700,"
        "http://localhost:3000,"
        "http://localhost:5173,"
        "http://127.0.0.1:7700"
    ).split(",")
    if origin.strip()
]

# 🔥 FIXED: Point to the preview model you just uploaded
DEFAULT_MODELS = {
    "tanzeel-preview": "zeaipc/tanzeel-preview",
    # Add future models here:
    # "tanzeel-intelligence": "zeaipc/tanzeel-intelligence",
    # "tanzeel-beta": "zeaipc/tanzeel-beta",
}


def load_models() -> dict[str, str]:
    """Load models from MODELS env var (JSON) or fall back to defaults."""
    raw = os.getenv("MODELS", "").strip()
    if not raw:
        return DEFAULT_MODELS.copy()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MODELS must be valid JSON.") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError("MODELS must be a non-empty JSON object.")
    if not all(
        isinstance(k, str) and isinstance(v, str) and k and v
        for k, v in parsed.items()
    ):
        raise RuntimeError("MODELS must map non-empty model names to HF repo IDs.")
    return parsed


MODELS = load_models()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "").strip() or next(iter(MODELS))

if DEFAULT_MODEL not in MODELS:
    raise RuntimeError(
        f"DEFAULT_MODEL={DEFAULT_MODEL!r} is not present in MODELS."
    )

# Production checks
if ENVIRONMENT == "production" and not HF_TOKEN:
    raise RuntimeError("HF_TOKEN must be configured in production.")

if ENVIRONMENT == "production" and REQUIRE_API_KEY and not TANZEEL_API_KEY:
    raise RuntimeError(
        "TANZEEL_API_KEY must be configured when REQUIRE_API_KEY=true."
    )

# 🔥 HF Router (current, non-deprecated endpoint)
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

MODEL_PROMPTS = {
    "tanzeel-preview": SYSTEM_PROMPT,
    "tanzeel-intelligence": SYSTEM_PROMPT,
}

# ─────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Tanzeel Intelligence API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# Bounded in-memory session store.
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "5000"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_SESSION_TURNS = int(os.getenv("MAX_SESSION_TURNS", "10"))
SESSIONS: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()


# ─────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    model: Optional[str] = None
    session_id: Optional[str] = Field(default=None, max_length=128)
    max_new_tokens: int = Field(default=500, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    allow_web_fallback: bool = False


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model: str
    source: str


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


# ─────────────────────────────────────────────────────────────────────────
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────
def cleanup_sessions() -> None:
    now = time.time()
    expired = [
        key for key, (last_seen, _) in SESSIONS.items()
        if now - last_seen > SESSION_TTL_SECONDS
    ]
    for key in expired:
        SESSIONS.pop(key, None)

    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)


def get_session(key: str) -> list[dict]:
    cleanup_sessions()
    item = SESSIONS.get(key)
    if item is None:
        return []
    _, history = item
    SESSIONS.move_to_end(key)
    return list(history)


def put_session(key: str, history: list[dict]) -> None:
    cleanup_sessions()
    SESSIONS[key] = (time.time(), history[-(MAX_SESSION_TURNS * 2):])
    SESSIONS.move_to_end(key)
    cleanup_sessions()


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def resolve_model(requested: Optional[str]) -> tuple[str, str]:
    public_name = requested or DEFAULT_MODEL
    if public_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{public_name}'. Available: {list(MODELS)}",
        )
    return public_name, MODELS[public_name]


def authorized(x_api_key: Optional[str]) -> None:
    if REQUIRE_API_KEY:
        if not TANZEEL_API_KEY:
            raise HTTPException(503, "API authentication is not configured.")
        if x_api_key != TANZEEL_API_KEY:
            raise HTTPException(401, "Invalid or missing API key.")


def call_hf_model(
    repo: str,
    public_name: str,
    user_message: str,
    history: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Call Hugging Face Inference Providers via OpenAI-compatible endpoint."""
    if not HF_TOKEN:
        raise HTTPException(503, "HF_TOKEN is not configured.")

    system_prompt = MODEL_PROMPTS.get(public_name, SYSTEM_PROMPT)
    messages = [{"role": "system", "content": system_prompt}, *history]
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": repo,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    last_error = "unknown error"
    for attempt in range(4):
        try:
            resp = requests.post(
                HF_CHAT_URL, headers=headers, json=payload, timeout=120
            )

            # Rate limit / cold start → retry with backoff
            if resp.status_code in (429, 503):
                retry_after = resp.headers.get("retry-after")
                try:
                    delay = float(retry_after) if retry_after else 2 ** attempt * 2
                except ValueError:
                    delay = 2 ** attempt * 2
                time.sleep(min(delay, 30))
                continue

            if resp.status_code != 200:
                # Include response body for easier debugging
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text[:300]
                raise HTTPException(
                    502,
                    f"HF inference failed ({resp.status_code}): {err_body}",
                )

            data = resp.json()
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise HTTPException(
                    502, f"HF returned an invalid chat response: {data}"
                ) from exc

            if not isinstance(content, str) or not content.strip():
                raise HTTPException(502, "HF returned an empty response.")
            return content.strip()

        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < 3:
                time.sleep(2 ** attempt)
        except HTTPException:
            raise

    raise HTTPException(
        503, f"HF inference unavailable after retries: {last_error}"
    )


# ─────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Tanzeel Intelligence API",
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
        "api_key_required": REQUIRE_API_KEY,
        "environment": ENVIRONMENT,
        "allowed_origins": ALLOWED_ORIGINS,
    }


@app.get("/models")
def list_models():
    return {
        "default": DEFAULT_MODEL,
        "models": [{"name": name} for name in MODELS],
    }


@app.post("/reset")
def reset(
    req: ResetRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    authorized(x_api_key)
    suffix = f":{req.session_id}"
    for key in list(SESSIONS):
        if key.endswith(suffix):
            SESSIONS.pop(key, None)
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    authorized(x_api_key)
    public_name, repo = resolve_model(req.model)

    session_id = req.session_id or str(uuid.uuid4())
    session_key = f"{public_name}:{session_id}"
    history = get_session(session_key)

    reply = call_hf_model(
        repo=repo,
        public_name=public_name,
        user_message=req.message.strip(),
        history=history,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    history.extend(
        [
            {"role": "user", "content": req.message.strip()},
            {"role": "assistant", "content": reply},
        ]
    )
    put_session(session_key, history)

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        model=public_name,
        source="tanzeel",
    )


# ─────────────────────────────────────────────────────────────────────────
# LOCAL RUN
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
