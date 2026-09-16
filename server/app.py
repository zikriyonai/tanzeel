"""
Tanzeel Intelligence FastAPI proxy with heuristic web research.
Uses manual keyword-based tool triggering (no function calling needed).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import OrderedDict
from typing import Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server.tools.registry import execute_tool


# ─────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (backticks avoided to prevent string break)
# ─────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Tanzeel Intelligence — an AI assistant built by ZEAIPC, founded by Arman Ansari.

# CORE BEHAVIOR
Answer directly and concisely. No filler, no self-intro.

# IDENTITY (ONLY WHEN ASKED)
Mention name/creator ONLY when user directly asks:
- "Tum kaun ho?" / "Who are you?" / "Who made you?"

Reply format:
- Hinglish: "Main Tanzeel Intelligence hoon, ZEAIPC ne banaya hai. Developer Arman Ansari hain."
- English: "I'm Tanzeel Intelligence, built by ZEAIPC. My developer is Arman Ansari."

Otherwise, NEVER mention identity.

# BASE MODEL QUESTIONS
If asked about base model or architecture:
"Main Tanzeel Intelligence hoon, ZEAIPC ne banaya hai. Technical details proprietary hain."
Never mention Qwen, Llama, Gemma, OpenAI, or any third-party model.

# CRITICAL — WEB RESEARCH CONTEXT RULES

When the user message contains a [WEB RESEARCH CONTEXT] block:

1. ONLY use the information from that block to answer.
2. DO NOT add facts from your own memory — your training data may be outdated.
3. DO NOT invent URLs. Only use the URLs provided in the context block. Copy them exactly.
4. If the context doesn't have the answer, say in the user's language:
   "Mujhe latest information nahi mili, but general guidance ye hai: ..."
   Then clearly separate general knowledge from research.
5. At the very end, list ONLY the sources from the context block. Format them as a Markdown list with each source as a link: [title](url). Only use the exact URLs provided.
6. If there are no sources in the context, do not add a sources section at all.

NEVER make up URLs. NEVER fabricate company names or facts. If unsure, say so.

# LANGUAGE
Reply in the user's language (Hindi/English/Hinglish).

# STYLE
Direct, warm, efficient. Use Markdown. Skip filler.
"""


# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
TANZEEL_API_KEY = os.getenv("TANZEEL_API_KEY", "").strip()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").strip().lower() == "true"
ENABLE_TOOLS = os.getenv("ENABLE_TOOLS", "true").strip().lower() == "true"

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://tanzeelai.web.app,"
        "http://localhost:7700,"
        "http://localhost:3000,"
        "http://127.0.0.1:7700"
    ).split(",")
    if origin.strip()
]

DEFAULT_MODELS = {
    "tanzeel-preview": "Qwen/Qwen3-VL-8B-Instruct:featherless-ai",
}


def load_models() -> dict:
    raw = os.getenv("MODELS", "").strip()
    if not raw:
        return DEFAULT_MODELS.copy()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MODELS must be valid JSON.") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError("MODELS must be a non-empty JSON object.")
    return parsed


MODELS = load_models()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "").strip() or next(iter(MODELS))

if DEFAULT_MODEL not in MODELS:
    raise RuntimeError(f"DEFAULT_MODEL={DEFAULT_MODEL!r} not in MODELS.")

if ENVIRONMENT == "production" and not HF_TOKEN:
    raise RuntimeError("HF_TOKEN must be configured in production.")

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

MODEL_PROMPTS = {
    "tanzeel-preview": SYSTEM_PROMPT,
    "tanzeel-intelligence": SYSTEM_PROMPT,
}


# ─────────────────────────────────────────────────────────────────────────
# HEURISTIC TOOL DETECTION
# ─────────────────────────────────────────────────────────────────────────
WEB_TRIGGER_KEYWORDS = [
    r"\blatest\b", r"\bcurrent\b", r"\btoday\b", r"\bnow\b",
    r"\bnews\b", r"\brecent\b", r"\bupdates?\b",
    r"\bweather\b", r"\btemperature\b", r"\bforecast\b",
    r"\bprice\b", r"\bstock\b", r"\bcrypto\b", r"\bbitcoin\b",
    r"\bscore\b", r"\bmatch\b", r"\bwon\b", r"\belection\b",
    r"\baaj\b", r"\babhi\b", r"\btaza\b", r"\btaja\b",
    r"\bkhabar\b", r"\bkhabrein\b",
    r"\bmausam\b", r"\btaapman\b",
    r"\bkaun\s+(jeeta|hai)\b",
    r"\bkya\s+chal\s+raha\b",
]

WEB_TRIGGER_PATTERN = re.compile("|".join(WEB_TRIGGER_KEYWORDS), re.IGNORECASE)


def should_use_web(message: str) -> bool:
    if not message or len(message) < 3:
        return False
    return bool(WEB_TRIGGER_PATTERN.search(message))


# ─────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Tanzeel Intelligence API", version="4.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "5000"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_SESSION_TURNS = int(os.getenv("MAX_SESSION_TURNS", "10"))
SESSIONS: OrderedDict = OrderedDict()


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
    enable_tools: bool = True


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model: str
    source: str
    tool_calls_made: int = 0
    sources: list = []


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


# ─────────────────────────────────────────────────────────────────────────
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────
def cleanup_sessions() -> None:
    now = time.time()
    expired = [k for k, (last, _) in SESSIONS.items() if now - last > SESSION_TTL_SECONDS]
    for k in expired:
        SESSIONS.pop(k, None)
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)


def get_session(key: str) -> list:
    cleanup_sessions()
    item = SESSIONS.get(key)
    if item is None:
        return []
    _, history = item
    SESSIONS.move_to_end(key)
    return list(history)


def put_session(key: str, history: list) -> None:
    cleanup_sessions()
    SESSIONS[key] = (time.time(), history[-(MAX_SESSION_TURNS * 2):])
    SESSIONS.move_to_end(key)
    cleanup_sessions()


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def resolve_model(requested: Optional[str]):
    public_name = requested or DEFAULT_MODEL
    if public_name not in MODELS:
        raise HTTPException(400, f"Unknown model '{public_name}'. Available: {list(MODELS)}")
    return public_name, MODELS[public_name]


def authorized(x_api_key: Optional[str]) -> None:
    if REQUIRE_API_KEY:
        if not TANZEEL_API_KEY:
            raise HTTPException(503, "API authentication is not configured.")
        if x_api_key != TANZEEL_API_KEY:
            raise HTTPException(401, "Invalid or missing API key.")


def call_hf_model(
    repo: str,
    system_prompt: str,
    messages: list,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    if not HF_TOKEN:
        raise HTTPException(503, "HF_TOKEN is not configured.")

    payload = {
        "model": repo,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    last_error = "unknown"
    for attempt in range(4):
        try:
            resp = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=120)

            if resp.status_code in (429, 503):
                retry_after = resp.headers.get("retry-after")
                try:
                    delay = float(retry_after) if retry_after else 2 ** attempt * 2
                except ValueError:
                    delay = 2 ** attempt * 2
                time.sleep(min(delay, 30))
                continue

            if resp.status_code != 200:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text[:300]
                raise HTTPException(502, f"HF error ({resp.status_code}): {err_body}")

            data = resp.json()
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise HTTPException(502, f"Invalid HF response: {data}") from exc

            if not isinstance(content, str) or not content.strip():
                raise HTTPException(502, "HF returned empty response.")
            return content.strip()

        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < 3:
                time.sleep(2 ** attempt)
        except HTTPException:
            raise

    raise HTTPException(503, f"HF unavailable: {last_error}")


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
        "tools_enabled": ENABLE_TOOLS,
        "tool_mode": "heuristic",
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
        "tools_enabled": ENABLE_TOOLS,
        "tool_mode": "heuristic",
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
    system_prompt = MODEL_PROMPTS.get(public_name, SYSTEM_PROMPT)

    session_id = req.session_id or str(uuid.uuid4())
    session_key = f"{public_name}:{session_id}"
    history = get_session(session_key)

    user_message = req.message.strip()
    tool_calls_made = 0
    sources = []
    final_user_message = user_message

    # ── Heuristic tool trigger ───────────────────────────────────────────
    if ENABLE_TOOLS and req.enable_tools and should_use_web(user_message):
        try:
            print(f"[tools] Research triggered: {user_message[:80]}")
            result = execute_tool("research", {"query": user_message, "max_pages": 2})
            tool_calls_made = 1

            if result.get("sources"):
                sources = result["sources"]

            if result.get("context"):
                sources_block = "\n".join(
                    f"- {s.get('title','Source')}: {s.get('url','')}"
                    for s in sources
                    if s.get("url")
                ) or "(no sources available)"

                final_user_message = (
                    "[WEB RESEARCH CONTEXT — USE ONLY THIS DATA]\n"
                    f"Query: {result['query']}\n\n"
                    f"Content:\n{result['context']}\n\n"
                    f"AVAILABLE SOURCES (use ONLY these exact URLs):\n"
                    f"{sources_block}\n"
                    "[END CONTEXT]\n\n"
                    "---\n\n"
                    f"USER'S QUESTION: {user_message}\n\n"
                    "Answer using ONLY the context above. If the context is "
                    "insufficient, say so honestly — do NOT fill gaps from memory. "
                    "End with a Sources section using ONLY the URLs listed above."
                )
            else:
                print(f"[tools] No context: {result.get('error')}")
        except Exception as e:
            print(f"[tools] Research failed: {e}")

    # ── Call model ───────────────────────────────────────────────────────
    messages = [*history, {"role": "user", "content": final_user_message}]

    reply = call_hf_model(
        repo=repo,
        system_prompt=system_prompt,
        messages=messages,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    # ── Save session with ORIGINAL user message ──────────────────────────
    history.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ])
    put_session(session_key, history)

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        model=public_name,
        source="tanzeel+web" if tool_calls_made > 0 else "tanzeel",
        tool_calls_made=tool_calls_made,
        sources=sources[:5],
    )


# ─────────────────────────────────────────────────────────────────────────
# LOCAL RUN
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
