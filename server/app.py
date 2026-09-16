"""
Tanzeel Intelligence FastAPI proxy with WRC web research tools.
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

from server.tools.registry import TOOL_DEFINITIONS, execute_tool


# ─────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Tanzeel Intelligence — an AI assistant built by ZEAIPC (Zikr-e-Ameen Innovations & Programming Corporation), founded by Arman Ansari.

# CORE BEHAVIOR
Answer the user's question directly and concisely. Do not preface responses with introductions or self-references.

# IDENTITY (ONLY WHEN ASKED)
Mention your name/creator ONLY when the user directly asks about identity:
- "Tum kaun ho?" / "Who are you?" / "Who made you?" / "Tumhara developer kaun hai?"

Reply format:
- Hinglish: "Main Tanzeel Intelligence hoon, ZEAIPC ne banaya hai. Developer Arman Ansari hain."
- English: "I'm Tanzeel Intelligence, built by ZEAIPC. My developer is Arman Ansari."

For ALL other questions — just answer. Do NOT mention identity/creator.

# BASE MODEL / TECH QUESTIONS
If asked what model/architecture you use:
"Main Tanzeel Intelligence hoon, ZEAIPC ne banaya hai. Technical details proprietary hain."
Never mention Qwen, Llama, Gemma, OpenAI, or any third-party model.

# TOOLS — WEB RESEARCH
You have tools: `research`, `web_search`, `fetch_page`.

USE TOOLS when the user asks about:
- Recent news, current events, "latest", "aaj ki", "abhi"
- Real-time data: weather, stocks, prices, sports scores
- Anything happening after your training cutoff
- Facts you're uncertain about

DO NOT use tools for:
- General coding help, math, reasoning
- Casual conversation, creative writing
- Explaining concepts you already know

When you use a tool and get results, USE the information to answer directly. Cite sources at the end as:
"Sources:\n- [title](url)"

# LANGUAGE
Reply in the same language the user uses — Hindi, English, or Hinglish. Match their tone.

# STYLE
- Get to the point. Skip filler like "Great question!" or "Certainly!".
- Use Markdown for code, lists, and emphasis when useful.
- Be warm but efficient. If you don't know something, say so honestly.
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


def load_models() -> dict[str, str]:
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
# APP
# ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Tanzeel Intelligence API", version="4.0.0")

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
SESSIONS: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()


# ─────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE
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
    sources: list[dict] = []


class ResetRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)


# ─────────────────────────────────────────────────────────────────────────
# SESSIONS
# ─────────────────────────────────────────────────────────────────────────
def cleanup_sessions() -> None:
    now = time.time()
    expired = [k for k, (last, _) in SESSIONS.items() if now - last > SESSION_TTL_SECONDS]
    for k in expired:
        SESSIONS.pop(k, None)
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
        raise HTTPException(400, f"Unknown model '{public_name}'. Available: {list(MODELS)}")
    return public_name, MODELS[public_name]


def authorized(x_api_key: Optional[str]) -> None:
    if REQUIRE_API_KEY:
        if not TANZEEL_API_KEY:
            raise HTTPException(503, "API authentication is not configured.")
        if x_api_key != TANZEEL_API_KEY:
            raise HTTPException(401, "Invalid or missing API key.")


def _hf_request(payload: dict, headers: dict) -> dict:
    """Make one HF request with retries."""
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

            return resp.json()

        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < 3:
                time.sleep(2 ** attempt)

    raise HTTPException(503, f"HF unavailable after retries: {last_error}")


def call_hf_with_tools(
    repo: str,
    public_name: str,
    user_message: str,
    history: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_tools: bool = True,
) -> tuple[str, int, list[dict]]:
    """
    Call Qwen with tools. Handles function-calling loop.
    Returns: (reply, tool_calls_count, sources)
    """
    if not HF_TOKEN:
        raise HTTPException(503, "HF_TOKEN is not configured.")

    system_prompt = MODEL_PROMPTS.get(public_name, SYSTEM_PROMPT)
    messages = [{"role": "system", "content": system_prompt}, *history]
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    tool_calls_made = 0
    collected_sources: list[dict] = []
    max_iterations = 4

    for iteration in range(max_iterations):
        payload = {
            "model": repo,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        if enable_tools and ENABLE_TOOLS:
            payload["tools"] = TOOL_DEFINITIONS
            payload["tool_choice"] = "auto"

        # Try with tools first; if provider rejects, fallback without tools
        try:
            data = _hf_request(payload, headers)
        except HTTPException as e:
            if enable_tools and "tool" in str(e).lower():
                print(f"[tools] Provider rejected tools — falling back: {e}")
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                data = _hf_request(payload, headers)
            else:
                raise

        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls") or []

        # No tool calls → final answer
        if not tool_calls:
            reply = (message.get("content") or "").strip()
            if not reply:
                reply = "(no response — try rephrasing)"
            return reply, tool_calls_made, collected_sources

        # Execute tools and loop
        messages.append(message)

        for tc in tool_calls:
            tool_calls_made += 1
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            print(f"[tools] Executing {fn_name}({fn_args})")
            result = execute_tool(fn_name, fn_args)

            # Collect sources
            if fn_name == "research" and result.get("sources"):
                collected_sources.extend(result["sources"])
            elif fn_name == "web_search" and result.get("results"):
                for r in result["results"][:3]:
                    if r.get("url"):
                        collected_sources.append({
                            "title": r.get("title", ""),
                            "url": r["url"],
                        })

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False)[:8000],
            })

    return (
        "I gathered information but couldn't complete the reply. Please rephrase.",
        tool_calls_made,
        collected_sources,
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
        "tools_enabled": ENABLE_TOOLS,
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

    reply, tool_count, sources = call_hf_with_tools(
        repo=repo,
        public_name=public_name,
        user_message=req.message.strip(),
        history=history,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        enable_tools=req.enable_tools,
    )

    history.extend([
        {"role": "user", "content": req.message.strip()},
        {"role": "assistant", "content": reply},
    ])
    put_session(session_key, history)

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        model=public_name,
        source="tanzeel+web" if tool_count > 0 else "tanzeel",
        tool_calls_made=tool_count,
        sources=sources[:5],
    )


# ─────────────────────────────────────────────────────────────────────────
# LOCAL RUN
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
