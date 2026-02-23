# ================================
# server_session.py  (FULL, STABLE, CLEAN)
#
# ✅ Keep: Realtime + Sora create/remix workflow
# ✅ Add: Chat streaming + server-side incremental TTS (no client-side TTS text submit)
# ✅ Add: /tts_prepare + /tts/live/{tts_id}.mp3 (true "download-while-play" for AVPlayer)
# ✅ FIX/ALIGN: Home Automation (Pi light) — bind + Realtime home-only assistant + dispatch
#
# 핵심:
# - GPT 文本在云端 stream 回来时，后端立刻按“句子块”送入 OpenAI TTS
# - iOS 不再把文本发到 /tts（避免前端再跑一遍 TTS），只需要播放后端给的 tts_url
# - /chat/prepare -> 返回 chat_id + events_url(SSE) + tts_url(HTTP MP3 live stream)
# - /chat/events/{chat_id} -> SSE: meta + delta + done
# - /tts/live/{tts_id}.mp3 -> MP3 chunked stream（边下边播）
#
# Home Automation alignment:
# - iOS binds gateway/profile to POST /home/gateway_profile
# - /session will automatically switch to "home-only assistant" if a binding exists
# - Assistant MUST emit: HOME_CMD: {"cmd":"light_on|light_off|light_toggle"}
# - iOS parses HOME_CMD and calls POST /home/dispatch
#
# ================================

from __future__ import annotations

import os
import io
import re
import json
import uuid
import time
import hashlib
import hmac
import secrets
import base64
import asyncio
from urllib.parse import urlparse
import logging
import threading
import subprocess
import queue
import zipfile
import sqlite3
import math
from array import array

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Iterator, List
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, UploadFile, File, Form, Request, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ✅✅✅ 必须最先加载 .env（否则 home_chat/billing/auth import 会读不到 OPENAI_API_KEY）
load_dotenv()

# ✅ routers（都放在 load_dotenv 之后）
from routers.home_chat import router as home_chat_router
from routers.media_upload import router as media_upload_router
from routers.home_automation import (
    router as home_automation_router,
    home_instructions_for_request as home_instructions_for_request_router,
    home_has_binding,
)

# -----------------------------
# ✅ Home voice control instructions aligned with iOS parser
# -----------------------------

# Debug/lab switch: export HOME_FORCE_ON=1 to always force home mode on /session.
HOME_FORCE_ON = (os.getenv("HOME_FORCE_ON") or "").strip().lower() in ("1", "true", "yes", "on")


def _home_instructions_ios_protocol(req: Request, client_id: Optional[str]) -> str:
    """Build a Realtime 'instructions' prompt aligned with iOS HOME_CMD parser.

    iOS expects ONE LINE:
      HOME_CMD: {"cmd":"...","device_id":"200","confidence":0.0-1.0,"reason":"..."}
    """
    device_id = "200"
    device_display = "家居设备"

    # Best-effort: reuse bound display name if the old builder is available.
    try:
        old = home_instructions_for_request_router(req, client_id=client_id)
        if old:
            m = re.search(r"【已绑定设备】([^\n\r]+)", old)
            if m:
                device_display = (m.group(1) or "").strip() or device_display
    except Exception:
        pass

    return (
        "你是“阿杜”，【家居控制专用语音助手】。你的唯一任务：把用户语音意图转换为可执行命令。\n"
        f"默认控制设备：{device_display}（device_id={device_id}）。\n"
        "\n"
        "【输出协议（必须严格遵守）】\n"
        "1) 你每次回复只能输出一行，且必须以 HOME_CMD: 开头，后面跟 JSON。\n"
        "2) JSON 固定结构：\n"
        f'{{"cmd":"light_on|light_off|light_toggle|ask_clarify","device_id":"{device_id}","confidence":0.0-1.0,"reason":"一句话原因"}}\n'
        "3) 除这一行外，禁止输出任何其他文字（不聊天、不解释、不寒暄）。\n"
        f"4) 默认 device_id=\"{device_id}\"，除非用户明确指定其他设备编号（否则不要改）。\n"
        "5) 若不确定用户意图，输出 cmd=\"ask_clarify\"，reason 写你要问的一句话。\n"
        "\n"
        "【语义映射】\n"
        "- 开灯/打开灯/亮灯/灯亮 => light_on\n"
        "- 关灯/关闭灯/灭灯/灯灭 => light_off\n"
        "- 切换/开关一下/翻转 => light_toggle\n"
    )

# ✅ billing / auth routers (keep your existing business logic)
from billing import router as billing_router
from auth import router as auth_router

# -----------------------------
# Remix profile (single source of truth)
# -----------------------------
try:
    from remix_profile import (
        REMIX_BEGIN,
        REMIX_END,
        REMIX_REALTIME_INSTRUCTIONS,
    )
except Exception:
    # Safe fallback (keeps server bootable even if remix_profile is missing)
    REMIX_BEGIN = "[REMIX_REQUEST]"
    REMIX_END = "[/REMIX_REQUEST]"
    REMIX_REALTIME_INSTRUCTIONS = "You are ChatAGI-阿杜 Remix Assistant. Output a stable English remix prompt."

# ============== ENV ==============
# Either OPENAI_API_KEY or SORA_API_KEY is acceptable; we normalize for both pipelines.
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("SORA_API_KEY") or "").strip()
SORA_API_KEY = (os.getenv("SORA_API_KEY") or OPENAI_API_KEY).strip()
OPENAI_ORG_ID = (os.getenv("OPENAI_ORG_ID") or "").strip()

REALTIME_MODEL_DEFAULT = (os.getenv("REALTIME_MODEL") or "gpt-realtime-mini").strip()
REALTIME_VOICE_DEFAULT = (os.getenv("REALTIME_VOICE") or "alloy").strip()

# ✅ Backend no longer hardcodes a long "work definition" prompt.
#    If you still want server-side defaults, set REALTIME_DEFAULT_INSTRUCTIONS in env.
REALTIME_DEFAULT_INSTRUCTIONS = (os.getenv("REALTIME_DEFAULT_INSTRUCTIONS") or "").strip() or None

# ---- Sora ----
SORA_MODEL_DEFAULT = (os.getenv("SORA_MODEL") or "sora-2").strip()
SORA_SECONDS_DEFAULT = int(os.getenv("SORA_SECONDS") or "8")         # default 8s
SORA_SIZE_DEFAULT = (os.getenv("SORA_SIZE") or "720x1280").strip()   # vertical

SORA_CONCURRENCY = int(os.getenv("SORA_CONCURRENCY", "1"))
SORA_SEM = threading.BoundedSemaphore(max(1, SORA_CONCURRENCY))

# ---- Video Provider Routing (Sora / MiniMax) ----
# VIDEO_PROVIDER_FORCE: if set to "sora" or "minimax", overrides all routing.
# VIDEO_PROVIDER_DEFAULT: default provider when force is not set.
VIDEO_PROVIDER_FORCE = (os.getenv("VIDEO_PROVIDER_FORCE") or "").strip().lower()
VIDEO_PROVIDER_DEFAULT = (os.getenv("VIDEO_PROVIDER_DEFAULT") or "sora").strip().lower()

# ---- MiniMax Video Generation ----
MINIMAX_API_KEY = (os.getenv("MINIMAX_API_KEY") or "").strip()
MINIMAX_BASE_URL = (os.getenv("MINIMAX_BASE_URL") or "https://api.minimax.io").strip().rstrip("/")
MINIMAX_MODEL_T2V_DEFAULT = (os.getenv("MINIMAX_VIDEO_MODEL_T2V") or os.getenv("MINIMAX_VIDEO_MODEL") or "MiniMax-Hailuo-2.3").strip()
MINIMAX_MODEL_I2V_DEFAULT = (os.getenv("MINIMAX_VIDEO_MODEL_I2V") or os.getenv("MINIMAX_VIDEO_MODEL") or "MiniMax-Hailuo-2.3").strip()
MINIMAX_MODEL_S2V_DEFAULT = (os.getenv("MINIMAX_VIDEO_MODEL_S2V") or "S2V-01").strip()
try:
    MINIMAX_DURATION_DEFAULT = int(os.getenv("MINIMAX_VIDEO_DURATION") or os.getenv("MINIMAX_DURATION") or "6")
except Exception:
    MINIMAX_DURATION_DEFAULT = 6
MINIMAX_RESOLUTION_DEFAULT = (os.getenv("MINIMAX_VIDEO_RESOLUTION") or os.getenv("MINIMAX_RESOLUTION") or "1080P").strip().upper() or "1080P"
try:
    MINIMAX_POLL_INTERVAL_SEC = float(os.getenv("MINIMAX_POLL_INTERVAL_SEC") or "2")
except Exception:
    MINIMAX_POLL_INTERVAL_SEC = 2.0
try:
    MINIMAX_CREATE_TIMEOUT_SEC = float(os.getenv("MINIMAX_CREATE_TIMEOUT_SEC") or "60")
except Exception:
    MINIMAX_CREATE_TIMEOUT_SEC = 60.0
try:
    MINIMAX_QUERY_TIMEOUT_SEC = float(os.getenv("MINIMAX_QUERY_TIMEOUT_SEC") or "60")
except Exception:
    MINIMAX_QUERY_TIMEOUT_SEC = 60.0
try:
    MINIMAX_DOWNLOAD_TIMEOUT_SEC = float(os.getenv("MINIMAX_DOWNLOAD_TIMEOUT_SEC") or "600")
except Exception:
    MINIMAX_DOWNLOAD_TIMEOUT_SEC = 600.0

# ---- Output Aspect / Portrait enforcement ----
# When enabled, all *persisted* videos will be normalized to a portrait MP4 (e.g. 1080x1920),
# so iOS always plays them in vertical fullscreen (no sideways/horizontal output).
VIDEO_FORCE_PORTRAIT = (os.getenv("VIDEO_FORCE_PORTRAIT") or os.getenv("MINIMAX_FORCE_PORTRAIT") or os.getenv("FORCE_PORTRAIT") or "").strip().lower() in ("1", "true", "yes", "on")
VIDEO_PORTRAIT_SIZE = (os.getenv("VIDEO_PORTRAIT_SIZE") or "1080x1920").strip().lower()
VIDEO_PORTRAIT_MODE = (os.getenv("VIDEO_PORTRAIT_MODE") or "crop").strip().lower()  # crop|pad
VIDEO_PORTRAIT_KEEP_ORIGINAL = (os.getenv("VIDEO_PORTRAIT_KEEP_ORIGINAL") or "0").strip().lower() in ("1", "true", "yes", "on")

# ---- Video request de-dup (same content => single job/asset) ----
# If enabled, repeated create/remix requests with the same semantic content will return the
# existing job/asset instead of creating a second provider request.
VIDEO_DEDUP_ENABLED = (os.getenv("VIDEO_DEDUP_ENABLED") or os.getenv("VIDEO_REQUEST_DEDUP") or "0").strip().lower() in ("1", "true", "yes", "on")
try:
    VIDEO_DEDUP_TTL_SEC = int(os.getenv("VIDEO_DEDUP_TTL_SEC") or os.getenv("VIDEO_REQUEST_DEDUP_TTL_SEC") or "3600")
except Exception:
    VIDEO_DEDUP_TTL_SEC = 3600

# ---- Chat + TTS streaming ----
CHAT_MODEL_DEFAULT = (os.getenv("CHAT_MODEL") or "gpt-5").strip()
CHAT_STREAM_TIMEOUT_SEC = int(os.getenv("CHAT_STREAM_TIMEOUT_SEC") or "180")
CHAT_STREAM_CHUNK_TIMEOUT_SEC = float(os.getenv("CHAT_STREAM_CHUNK_TIMEOUT_SEC") or "25")
CHAT_JOB_TTL_SEC = int(os.getenv("CHAT_JOB_TTL_SEC") or "1800")  # 30min

# ---- Smart Router: DeepSeek for default text, OpenAI for web search + Realtime ----
# Route modes:
#   A: allow_web=true -> OpenAI ; allow_web=false -> DeepSeek (default)
#   OPENAI_ONLY: always OpenAI
#   DEEPSEEK_ONLY: always DeepSeek (except image/video attachments which force OpenAI)
CHAT_ROUTE_MODE = (os.getenv("CHAT_ROUTE_MODE") or "OPENAI_ONLY").strip().upper()
DEEPSEEK_FALLBACK_TO_OPENAI = (os.getenv("DEEPSEEK_FALLBACK_TO_OPENAI") or "1").strip().lower() not in ("0","false","no")

# OpenAI text model used when provider=openai (allow_web=true or fallback/attachments)
OPENAI_TEXT_MODEL = (os.getenv("OPENAI_TEXT_MODEL") or CHAT_MODEL_DEFAULT).strip() or CHAT_MODEL_DEFAULT

# If true, allow the client payload to override the chat model.
CHAT_ALLOW_CLIENT_MODEL = (os.getenv("CHAT_ALLOW_CLIENT_MODEL") or "0").strip().lower() in ("1","true","yes","on")

# Server-side streaming TTS (sentence-by-sentence while text streams). Default OFF (manual speaker playback).
CHAT_ENABLE_TTS_STREAMING = (os.getenv("CHAT_ENABLE_TTS_STREAMING") or "0").strip().lower() in ("1","true","yes","on")

# DeepSeek config used when provider=deepseek (allow_web=false)
DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_BASE_URL = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/")
DEEPSEEK_MODEL_DEFAULT = (os.getenv("DEEPSEEK_MODEL") or "deepseek-reasoner").strip()
try:
    DEEPSEEK_TIMEOUT_SEC = float(os.getenv("DEEPSEEK_TIMEOUT_SEC") or "120")
except Exception:
    DEEPSEEK_TIMEOUT_SEC = 120.0



# ✅ UX / Rendering alignment (iOS code highlight depends on fenced blocks)
CHAT_SYSTEM_STYLE_PROMPT = (
    "You are ChatAGI-阿杜, a helpful assistant.\n"
    "Always respond in Simplified Chinese unless the user explicitly asks for another language.\n"
    "Keep replies clean and readable. Use Markdown only when presenting code blocks.\n"
    "- Whenever you output source code, ALWAYS wrap it in fenced code blocks using triple backticks and specify the language.\n"
    "- Use LaTeX for math where appropriate.\n"
    "- Do not output internal tool reference IDs (e.g., turn2search12) or raw system errors to the user.\n"
    "- IMPORTANT: In this app you CAN speak and will be played via TTS. Never claim you cannot speak/voice/play audio. If asked, confidently answer and continue.\n"
)
def _prepend_style_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return messages
    # Avoid duplicating if caller already provided the same prompt.
    first = (messages[0].get("content") or "").strip() if isinstance(messages[0], dict) else ""
    if first.startswith("You are ChatAGI-阿杜") or "Respond in Markdown" in first:
        return messages
    return [{"role": "system", "content": CHAT_SYSTEM_STYLE_PROMPT}] + messages

# ✅ Long-output safeguards (avoid mid-output truncation)
# - Responses API uses `max_output_tokens` as an upper bound for *all* generated tokens
#   (visible output + reasoning). If it hits the limit, the response ends with status=incomplete.
#   We default to 25000 to match official guidance for reasoning-capable models.
CHAT_MAX_OUTPUT_TOKENS_DEFAULT = int(os.getenv("CHAT_MAX_OUTPUT_TOKENS") or "25000")
# - Auto continuation loops when upstream ends as incomplete due to token cap.
CHAT_MAX_CONTINUATIONS_DEFAULT = int(os.getenv("CHAT_MAX_CONTINUATIONS") or "3")
# - requests timeout tuning for long SSE streams
CHAT_STREAM_CONNECT_TIMEOUT_SEC = float(os.getenv("CHAT_STREAM_CONNECT_TIMEOUT_SEC") or "20")
CHAT_STREAM_READ_TIMEOUT_SEC = float(os.getenv("CHAT_STREAM_READ_TIMEOUT_SEC") or "600")

# Enable OpenAI built-in web search tool (Responses API)
CHAT_ENABLE_WEB_SEARCH_DEFAULT = (os.getenv("CHAT_ENABLE_WEB_SEARCH_DEFAULT") or os.getenv("CHAT_ENABLE_WEB_SEARCH") or "1").strip().lower() not in ("0","false","no")

# TTS for chat streaming (sentence-by-sentence)
TTS_SPEECH_URL = "https://api.openai.com/v1/audio/speech"
TTS_MODEL_DEFAULT = (os.getenv("TTS_MODEL") or "gpt-5-mini-tts").strip()
TTS_VOICE_DEFAULT = (os.getenv("TTS_VOICE") or REALTIME_VOICE_DEFAULT).strip()
TTS_INSTRUCTIONS_DEFAULT = (os.getenv("TTS_INSTRUCTIONS") or "Use a natural, energetic, brisk, clear voice. Keep pauses short. Prefer Mandarin Chinese when the text is Chinese. Avoid reading code symbols or long punctuation verbatim.").strip()
TTS_SPEED_DEFAULT = float(os.getenv("TTS_SPEED") or "1.15")
TTS_LIVE_TTL_SEC = int(os.getenv("TTS_LIVE_TTL_SEC") or "1800")  # 30min

# Low-latency streaming knobs (reduce "thinking..." time before first audio/text)
CHAT_TTS_MIN_CHARS_DEFAULT = int(os.getenv("CHAT_TTS_MIN_CHARS") or "6")
TTS_STREAM_CHUNK_SIZE_DEFAULT = int(os.getenv("TTS_STREAM_CHUNK_SIZE") or "8192")
TTS_STREAM_CHUNK_SIZE_DEFAULT = max(1024, min(TTS_STREAM_CHUNK_SIZE_DEFAULT, 65536))

# ---- Audio transcription (voice notes) ----
# NOTE: Responses API input does NOT accept `input_audio` blocks.
# We transcribe user audio to text on the server and send as `input_text`.
TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
TRANSCRIBE_MODEL_DEFAULT = (os.getenv("TRANSCRIBE_MODEL") or "whisper-1").strip()
TRANSCRIBE_LANGUAGE_DEFAULT = (os.getenv("TRANSCRIBE_LANGUAGE") or "").strip() or None  # e.g. "zh"
TRANSCRIBE_TIMEOUT_SEC = float(os.getenv("TRANSCRIBE_TIMEOUT_SEC") or "60")
TRANSCRIBE_CACHE_TTL_SEC = int(os.getenv("TRANSCRIBE_CACHE_TTL_SEC") or "3600")  # 1h
_TRANSCRIBE_CACHE: Dict[str, Tuple[float, str]] = {}  # key -> (ts, text)
_TRANSCRIBE_LOCK = threading.Lock()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("solara-backend")

# ================================
# ✅ Confirm protocol (anti-early-trigger)
# ================================
CONFIRM_PHRASE = (os.getenv("REMIX_CONFIRM_PHRASE") or "客户已确定").strip() or "客户已确定"
TRIGGER_PREFIX = "REMIX:"
CONFIRM_MARK = f"##CONFIRM={CONFIRM_PHRASE}"

_CONFIRM_RE = re.compile(rf"({re.escape(CONFIRM_PHRASE)})\s*([0-9]{{4}})?", re.I)

def _user_has_confirm(user_confirm_text: str, expected_code: Optional[str] = None) -> bool:
    s = (user_confirm_text or "").strip()
    if not s:
        return False
    m = _CONFIRM_RE.search(s)
    if not m:
        return False
    if expected_code:
        code = m.group(2) or ""
        return code == expected_code
    return True

def _assistant_line_is_valid(assistant_line: str, expected_code: Optional[str] = None) -> bool:
    s = (assistant_line or "").strip()
    if not s.startswith(TRIGGER_PREFIX):
        return False
    if CONFIRM_MARK not in s:
        return False
    if expected_code:
        if f"##CODE={expected_code}" in s:
            return True
        if expected_code not in s:
            return False
    return True

def _extract_instruction_from_assistant_line(assistant_line: str) -> str:
    s = (assistant_line or "").strip()
    if not s.startswith(TRIGGER_PREFIX):
        return ""
    core = s[len(TRIGGER_PREFIX):].strip()
    if "##CONFIRM=" in core:
        core = core.split("##CONFIRM=", 1)[0].strip()
    if "##CODE=" in core:
        core = core.split("##CODE=", 1)[0].strip()
    core = re.sub(r"\s+", " ", core).strip()
    return core

# ============== PATHS ==============
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
# ================================
# ✅ Conversations + Vector Memory (SQLite)
# - Conversation history list (ChatGPT-style hamburger list)
# - Vector memory retrieval (semantic search via embeddings)
# ================================

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CONV_DB_PATH = str(DATA_DIR / "conversations.sqlite3")
MEM_DB_PATH = str(DATA_DIR / "memory.sqlite3")

MEMORY_ENABLED_DEFAULT = (os.getenv("SOLARA_MEMORY_ENABLED") or "1").strip().lower() not in ("0", "false", "no")
MEMORY_TOP_K_DEFAULT = int(os.getenv("SOLARA_MEMORY_TOP_K") or "6")
MEMORY_MIN_SCORE_DEFAULT = float(os.getenv("SOLARA_MEMORY_MIN_SCORE") or "0.25")
MEMORY_CONTEXT_MAX_CHARS = int(os.getenv("SOLARA_MEMORY_CONTEXT_MAX_CHARS") or "1600")
MEMORY_ITEM_MAX_CHARS = int(os.getenv("SOLARA_MEMORY_ITEM_MAX_CHARS") or "240")
MEMORY_MAX_ITEMS_PER_USER = int(os.getenv("SOLARA_MEMORY_MAX_ITEMS_PER_USER") or "2000")
MEMORY_RECENT_FALLBACK_N = int(os.getenv("SOLARA_MEMORY_RECENT_FALLBACK_N") or "6")
MEMORY_EMBED_MODEL = (os.getenv("SOLARA_MEMORY_EMBED_MODEL") or "text-embedding-3-small").strip()
VISION_CAPTION_MODEL = (os.getenv("SOLARA_VISION_CAPTION_MODEL") or os.getenv("VISION_MODEL") or "gpt-4o-mini").strip()
VISION_CAPTION_MAX_CHARS = int(os.getenv("SOLARA_VISION_CAPTION_MAX_CHARS") or "180")
VISION_CAPTION_MAX_IMAGES_PER_TURN = int(os.getenv("SOLARA_VISION_CAPTION_MAX_IMAGES_PER_TURN") or "2")


def _sanitize_user_key(key: str) -> str:
    k = (key or "").strip()
    if not k:
        return "default"
    if len(k) > 128:
        return hashlib.sha256(k.encode("utf-8")).hexdigest()
    return re.sub(r"[^a-zA-Z0-9_\-:.]", "_", k)

def _derive_user_key(req: Request, body: Dict[str, Any]) -> str:
    # Prefer explicit client id; fallback to IP
    cid = (body.get("client_id") or body.get("clientId") or body.get("user_key") or "").strip()
    if not cid:
        cid = (req.headers.get("x-client-id") or req.headers.get("x-user-key") or "").strip()
    if cid:
        return _sanitize_user_key(cid)
    ip = req.client.host if req.client else "unknown"
    return _sanitize_user_key(ip)

def _client_id(req: Request) -> str:
    """Best-effort stable client identifier for anonymous clients.

    iOS/clients should send `x-client-id` header (or `client_id` in JSON bodies).
    Falls back to IP.
    """
    try:
        return _derive_user_key(req, {})
    except Exception:
        ip = req.client.host if req.client else "unknown"
        return _sanitize_user_key(ip)

def _conv_conn() -> sqlite3.Connection:
    con = sqlite3.connect(CONV_DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def _mem_conn() -> sqlite3.Connection:
    con = sqlite3.connect(MEM_DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def _init_conv_db() -> None:
    with _conv_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
          id TEXT PRIMARY KEY,
          user_key TEXT NOT NULL,
          title TEXT NOT NULL,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          last_preview TEXT
        );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_conv_user_updated ON conversations(user_key, updated_at DESC);")
        con.execute("""
        CREATE TABLE IF NOT EXISTS messages (
          id TEXT PRIMARY KEY,
          conversation_id TEXT NOT NULL,
          user_key TEXT NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv_time ON messages(conversation_id, created_at ASC);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_msg_user_time ON messages(user_key, created_at DESC);")

def _init_mem_db() -> None:
    with _mem_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS memory_items (
          id TEXT PRIMARY KEY,
          user_key TEXT NOT NULL,
          text TEXT NOT NULL,
          text_sha1 TEXT NOT NULL,
          embedding BLOB NOT NULL,
          dim INTEGER NOT NULL,
          created_at REAL NOT NULL,
          last_used_at REAL NOT NULL
        );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_used ON memory_items(user_key, last_used_at DESC);")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_mem_user_sha1 ON memory_items(user_key, text_sha1);")

# --- Multimodal memory: images (store a short caption + embedding) ---
con.execute("""
CREATE TABLE IF NOT EXISTS memory_media (
  id TEXT PRIMARY KEY,
  user_key TEXT NOT NULL,
  media_type TEXT NOT NULL,      -- image|video|other
  source TEXT DEFAULT '',        -- url or note
  caption TEXT NOT NULL,
  sha1 TEXT NOT NULL,
  embedding BLOB NOT NULL,
  dim INTEGER NOT NULL,
  created_at REAL NOT NULL,
  last_used_at REAL NOT NULL
);
""")
con.execute("CREATE INDEX IF NOT EXISTS idx_mem_media_user_used ON memory_media(user_key, last_used_at DESC);")
con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_mem_media_user_sha1 ON memory_media(user_key, sha1);")

_init_conv_db()
_init_mem_db()


# =========================
# Video Plaza (Douyin-like feed) – local storage + SQLite
# =========================
VIDEO_DB_PATH = DATA_DIR / "video_plaza.sqlite3"
VIDEOS_DIR = DATA_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
SOLARA_MAX_VIDEO_MB = int(os.getenv("SOLARA_MAX_VIDEO_MB", "200"))


def _video_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(VIDEO_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_video_db() -> None:
    conn = _video_conn()
    try:
        # Better concurrency on SQLite for multi-request workloads.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass

        conn.execute(
            """CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                owner_key TEXT NOT NULL,
                owner_user_id TEXT,
                owner_display_name TEXT DEFAULT '',
                caption TEXT DEFAULT '',
                status TEXT NOT NULL,
                visibility TEXT DEFAULT 'public',
                tags TEXT DEFAULT '',
                file_path TEXT NOT NULL,
                mime TEXT DEFAULT '',
                size_bytes INTEGER DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER DEFAULT 0,
                published_at INTEGER,
                views INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                comments INTEGER DEFAULT 0,
                shares INTEGER DEFAULT 0
            )"""
        )

        def _ensure_column(table: str, col: str, ddl: str) -> None:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

        # Schema upgrades for older installs
        _ensure_column("videos", "owner_user_id", "owner_user_id TEXT")
        _ensure_column("videos", "owner_display_name", "owner_display_name TEXT DEFAULT ''")
        _ensure_column("videos", "visibility", "visibility TEXT DEFAULT 'public'")
        _ensure_column("videos", "tags", "tags TEXT DEFAULT ''")
        _ensure_column("videos", "updated_at", "updated_at INTEGER DEFAULT 0")
        _ensure_column("videos", "comments", "comments INTEGER DEFAULT 0")
        _ensure_column("videos", "shares", "shares INTEGER DEFAULT 0")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_owner ON videos(owner_key, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_owner_user ON videos(owner_user_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_status_published ON videos(status, published_at DESC)")

        
        # --- Durable generated assets (video won't be lost even if client exits) ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,

                -- source/provider metadata
                source TEXT DEFAULT '',
                source_video_id TEXT DEFAULT '',
                origin_url TEXT DEFAULT '',

                -- integrity & de-dup
                sha256 TEXT,
                phash TEXT,

                -- lifecycle
                status TEXT NOT NULL,           -- ready|transcoding|error|pending
                storage_key TEXT NOT NULL,      -- local path or s3 key
                hls_key TEXT DEFAULT '',        -- optional HLS manifest key/path

                -- media metadata
                width INTEGER,
                height INTEGER,
                duration_ms INTEGER,

                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_user_created ON assets(user_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_status_updated ON assets(status, updated_at DESC)")
        try:
            # Partial unique index (SQLite>=3.8) to dedup on sha256 when present
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_assets_user_sha256 ON assets(user_id, sha256) WHERE sha256 IS NOT NULL")
        except Exception:
            pass

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS asset_tags (
                asset_id TEXT NOT NULL,
                k TEXT NOT NULL,
                v TEXT NOT NULL,
                PRIMARY KEY (asset_id, k)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_tags_kv ON asset_tags(k, v)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS client_tasks (
                client_task_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                provider TEXT DEFAULT '',
                provider_job_id TEXT DEFAULT '',
                provider_ref_id TEXT DEFAULT '',

                latest_asset_id TEXT NULL,
                status TEXT NOT NULL,  -- queued|working|ready|error

                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_client_tasks_user_updated ON client_tasks(user_id, updated_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_client_tasks_status_updated ON client_tasks(status, updated_at DESC)")

        # --- Video request de-dup locks (same content => single job/asset) ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_request_locks (
                user_id TEXT NOT NULL,
                content_key TEXT NOT NULL,
                client_task_id TEXT DEFAULT '',
                provider TEXT DEFAULT '',
                provider_job_id TEXT DEFAULT '',
                latest_asset_id TEXT DEFAULT '',
                status TEXT NOT NULL,  -- queued|working|ready|error
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                PRIMARY KEY (user_id, content_key)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vrl_user_updated ON video_request_locks(user_id, updated_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vrl_expires ON video_request_locks(expires_at)")

        # --- Commercial social layer (accounts, follow graph, engagement, DM) ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                avatar_url TEXT DEFAULT '',
                created_at INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                issued_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                revoked INTEGER DEFAULT 0
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, revoked, expires_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS follows (
                follower_id TEXT NOT NULL,
                following_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (follower_id, following_id)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_follows_following ON follows(following_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_follows_follower ON follows(follower_id, created_at DESC)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_likes (
                user_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (user_id, video_id)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_video_likes_video ON video_likes(video_id, created_at DESC)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_comments (
                comment_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_video_comments_video ON video_comments(video_id, created_at DESC)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_shares (
                share_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                channel TEXT DEFAULT '',
                created_at INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_video_shares_video ON video_shares(video_id, created_at DESC)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reco_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                video_id TEXT,
                event_type TEXT NOT NULL,
                meta_json TEXT DEFAULT '',
                created_at INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reco_events_user ON reco_events(user_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reco_events_video ON reco_events(video_id, created_at DESC)")

        # Direct messages (simple, REST-first; you can add realtime WS later)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dm_threads (
                thread_id TEXT PRIMARY KEY,
                user_a TEXT NOT NULL,
                user_b TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(user_a, user_b)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dm_threads_a ON dm_threads(user_a, updated_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dm_threads_b ON dm_threads(user_b, updated_at DESC)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dm_messages (
                message_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dm_messages_thread ON dm_messages(thread_id, created_at ASC)")

        conn.commit()
    finally:
        conn.close()


_init_video_db()

# =========================
# Durable Video Assets (anti-loss, restore by client_task_id)
# =========================

ASSET_ROOT_DIR = DATA_DIR / "assets"
ASSET_VIDEOS_DIR = ASSET_ROOT_DIR / "videos"
ASSET_ROOT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

def _none_if_blank(s: Any) -> Optional[str]:
    try:
        ss = str(s or "").strip()
    except Exception:
        ss = ""
    return ss if ss else None

def uuid7_str() -> str:
    """Generate a UUIDv7 string (time-ordered)."""
    # UUIDv7: 48-bit unix epoch ms + version 7 + random
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    uuid_int = (ts_ms << 80)
    uuid_int |= (0x7 << 76)
    uuid_int |= (rand_a << 64)
    uuid_int |= (0x2 << 62)  # variant 10xx...
    uuid_int |= rand_b

    return str(uuid.UUID(int=uuid_int))

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _ffprobe_json(path: str | Path) -> Optional[Dict[str, Any]]:
    """Best-effort ffprobe; returns parsed json or None."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        return json.loads(p.stdout)
    except Exception:
        return None

def probe_video_meta(path: str | Path) -> Dict[str, Any]:
    """Return basic video metadata (best-effort).

    Returns keys:
      - width/height: coded dimensions
      - rotate: rotation metadata (if present)
      - display_width/display_height: dimensions after applying rotate
      - duration_ms
    """
    info = _ffprobe_json(path)
    width = height = duration_ms = None
    rotate = None
    display_width = display_height = None

    try:
        if info:
            # pick first video stream
            streams = info.get("streams") or []
            v0 = None
            for s in streams:
                if str(s.get("codec_type") or "") == "video":
                    v0 = s
                    break
            if v0:
                width = int(v0.get("width") or 0) or None
                height = int(v0.get("height") or 0) or None

                # rotation can be in tags or side_data_list depending on container
                rot = None
                tags = v0.get("tags") or {}
                if isinstance(tags, dict):
                    rot = tags.get("rotate") or tags.get("rotation")
                if rot is None:
                    for sd in (v0.get("side_data_list") or []):
                        if isinstance(sd, dict) and (sd.get("rotation") is not None):
                            rot = sd.get("rotation")
                            break
                try:
                    rotate = int(float(rot)) if rot is not None else None
                except Exception:
                    rotate = None

            fmt = info.get("format") or {}
            dur_s = fmt.get("duration")
            if dur_s is None and v0:
                dur_s = v0.get("duration")
            try:
                dur_f = float(dur_s) if dur_s is not None else None
            except Exception:
                dur_f = None
            if dur_f is not None and dur_f > 0:
                duration_ms = int(dur_f * 1000)

    except Exception:
        pass

    # display dimensions (after rotation)
    display_width = width
    display_height = height
    if rotate in (90, -90, 270, -270):
        if width and height:
            display_width, display_height = height, width

    return {
        "width": width,
        "height": height,
        "rotate": rotate,
        "display_width": display_width,
        "display_height": display_height,
        "duration_ms": duration_ms,
    }



def _parse_wh_str(size_str: str, default: tuple[int, int] = (1080, 1920)) -> tuple[int, int]:
    s = (size_str or "").strip().lower()
    if not s:
        return default
    # allow forms like 1080x1920 / 1080*1920 / 1080,1920
    for sep in ("x", "*", ",", " "):
        if sep in s:
            parts = [p for p in s.replace("*", "x").replace(",", "x").split("x") if p.strip()]
            if len(parts) >= 2:
                try:
                    w = int(float(parts[0]))
                    h = int(float(parts[1]))
                    if w > 0 and h > 0:
                        return (w, h)
                except Exception:
                    break
    return default


def _ffmpeg_force_portrait(
    src_path: Path,
    dst_path: Path,
    target_wh: tuple[int, int],
    mode: str = "crop",
) -> bool:
    """Transcode a video to portrait with fixed target_wh.

    mode:
      - crop: fill and center-crop
      - pad:  fit and pad
    """
    if not src_path.exists():
        return False

    tw, th = int(target_wh[0]), int(target_wh[1])
    if tw <= 0 or th <= 0:
        return False

    m = probe_video_meta(src_path)
    rot = m.get("rotate")

    rotate_f = ""
    try:
        r = int(rot) if rot is not None else 0
    except Exception:
        r = 0

    # IMPORTANT: we run ffmpeg with -noautorotate, so we need to apply rotation ourselves.
    # rotate tag is degrees clockwise.
    if r in (90, -270):
        rotate_f = "transpose=1"  # 90 clockwise
    elif r in (270, -90):
        rotate_f = "transpose=2"  # 90 counter-clockwise
    elif r in (180, -180):
        rotate_f = "transpose=2,transpose=2"  # 180

    mode = (mode or "crop").strip().lower()
    if mode not in ("crop", "pad"):
        mode = "crop"

    if mode == "pad":
        fit = f"scale={tw}:{th}:force_original_aspect_ratio=decrease"
        pad = f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2"
        core = f"{fit},{pad},setsar=1"
    else:
        fit = f"scale={tw}:{th}:force_original_aspect_ratio=increase"
        crop = f"crop={tw}:{th}"
        core = f"{fit},{crop},setsar=1"

    vf = core if not rotate_f else f"{rotate_f},{core}"

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".tmp.mp4")
    try:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-noautorotate",
        "-i",
        str(src_path),
        "-vf",
        vf,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-metadata:s:v:0",
        "rotate=0",
        str(tmp_path),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0 or (not tmp_path.exists()) or tmp_path.stat().st_size <= 0:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False

    try:
        tmp_path.replace(dst_path)
    except Exception:
        try:
            # fallback to copy
            shutil.copyfile(tmp_path, dst_path)
            tmp_path.unlink(missing_ok=True)
        except Exception:
            return False
    return True


def _maybe_force_portrait_video(path: Path) -> Path:
    """If VIDEO_FORCE_PORTRAIT enabled, normalize `path` into portrait mp4 and return new path."""
    if not VIDEO_FORCE_PORTRAIT:
        return path

    target_wh = _parse_wh_str(VIDEO_PORTRAIT_SIZE, default=(1080, 1920))
    mode = (VIDEO_PORTRAIT_MODE or "crop").strip().lower()

    # Skip if already matches target and has no rotate tag.
    try:
        meta = probe_video_meta(path)
        dw = meta.get("display_width")
        dh = meta.get("display_height")
        rot = meta.get("rotate")
        if dw == target_wh[0] and dh == target_wh[1] and (rot in (None, 0, "0")):
            return path
    except Exception:
        pass

    out_path = path.parent / ("portrait.mp4" if path.name.lower().endswith(".mp4") else "portrait.mp4")
    ok = _ffmpeg_force_portrait(path, out_path, target_wh=target_wh, mode=mode)
    if not ok:
        return path

    if not VIDEO_PORTRAIT_KEEP_ORIGINAL:
        try:
            if path.exists() and path.resolve() != out_path.resolve():
                path.unlink(missing_ok=True)
        except Exception:
            pass

    return out_path

def video_dhash16(path: str | Path) -> Optional[str]:
    """Best-effort perceptual hash (dHash) of the first frame; returns 16-hex chars or None."""
    try:
        # Extract 1 frame to memory (JPEG) via ffmpeg, then compute dHash with PIL.
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(path),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "pipe:1",
        ]
        p = subprocess.run(cmd, capture_output=True, check=False)
        if p.returncode != 0 or not p.stdout:
            return None

        from PIL import Image  # type: ignore
        im = Image.open(io.BytesIO(p.stdout)).convert("L").resize((9, 8))
        px = list(im.getdata())
        # 8 rows, 9 cols
        bits = 0
        for y in range(8):
            row = px[y * 9 : (y + 1) * 9]
            for x in range(8):
                bits = (bits << 1) | (1 if row[x] > row[x + 1] else 0)
        return f"{bits:016x}"
    except Exception:
        return None

def _asset_get_tag(asset_id: str, k: str) -> Optional[str]:
    conn = _video_conn()
    try:
        row = conn.execute(
            "SELECT v FROM asset_tags WHERE asset_id=? AND k=?",
            (asset_id, k),
        ).fetchone()
        return (row[0] if row else None)
    finally:
        conn.close()

def _asset_set_tag(asset_id: str, k: str, v: str) -> None:
    now = int(time.time())
    conn = _video_conn()
    try:
        conn.execute(
            "INSERT INTO asset_tags(asset_id, k, v) VALUES (?, ?, ?) "
            "ON CONFLICT(asset_id, k) DO UPDATE SET v=excluded.v",
            (asset_id, k, str(v)),
        )
        # touch asset updated_at if exists
        conn.execute(
            "UPDATE assets SET updated_at=? WHERE asset_id=?",
            (now, asset_id),
        )
        conn.commit()
    finally:
        conn.close()

def _task_get(user_id: str, client_task_id: str) -> Optional[Dict[str, Any]]:
    conn = _video_conn()
    try:
        row = conn.execute(
            "SELECT * FROM client_tasks WHERE client_task_id=? AND user_id=?",
            (client_task_id, user_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def _task_upsert(
    *,
    user_id: str,
    client_task_id: str,
    status: str,
    provider: str = "",
    provider_job_id: Optional[str] = None,
    provider_ref_id: Optional[str] = None,
    latest_asset_id: Optional[str] = None,
) -> None:
    now = int(time.time())
    conn = _video_conn()
    try:
        conn.execute(
            """
            INSERT INTO client_tasks(
                client_task_id, user_id, provider, provider_job_id, provider_ref_id,
                latest_asset_id, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_task_id) DO UPDATE SET
                user_id=excluded.user_id,
                provider=CASE WHEN excluded.provider!='' THEN excluded.provider ELSE client_tasks.provider END,
                provider_job_id=COALESCE(excluded.provider_job_id, client_tasks.provider_job_id),
                provider_ref_id=COALESCE(excluded.provider_ref_id, client_tasks.provider_ref_id),
                latest_asset_id=COALESCE(excluded.latest_asset_id, client_tasks.latest_asset_id),
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (
                client_task_id,
                user_id,
                str(provider or ""),
                _none_if_blank(provider_job_id),
                _none_if_blank(provider_ref_id),
                _none_if_blank(latest_asset_id),
                str(status),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# -------------------------
# Video request de-dup locks
# -------------------------

def _video_content_key(*, user_key: str, provider: str, mode: str, prompt_idem: str, img_h: str = "", vid_h: str = "") -> str:
    """Build a stable content key so the same request maps to one job/asset.

    We intentionally do NOT include client_task_id or IP here.
    """
    raw = f"{(user_key or '').strip()}|{(provider or '').strip().lower()}|{(mode or '').strip().lower()}|{(prompt_idem or '').strip()}|img:{img_h}|vid:{vid_h}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _vlock_get(user_id: str, content_key: str) -> Optional[Dict[str, Any]]:
    conn = _video_conn()
    try:
        row = conn.execute(
            "SELECT * FROM video_request_locks WHERE user_id=? AND content_key=?",
            (user_id, content_key),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _vlock_delete(user_id: str, content_key: str) -> None:
    conn = _video_conn()
    try:
        conn.execute(
            "DELETE FROM video_request_locks WHERE user_id=? AND content_key=?",
            (user_id, content_key),
        )
        conn.commit()
    finally:
        conn.close()


def _vlock_get_active(user_id: str, content_key: str) -> Optional[Dict[str, Any]]:
    """Return lock row if not expired; if expired, delete and return None."""
    row = _vlock_get(user_id, content_key)
    if not row:
        return None
    now = int(time.time())
    try:
        exp = int(row.get("expires_at") or 0)
    except Exception:
        exp = 0
    if exp and exp < now:
        try:
            _vlock_delete(user_id, content_key)
        except Exception:
            pass
        return None
    return row


def _vlock_upsert(
    *,
    user_id: str,
    content_key: str,
    status: str,
    provider: str = "",
    provider_job_id: str = "",
    client_task_id: str = "",
    latest_asset_id: str = "",
    ttl_sec: Optional[int] = None,
) -> None:
    now = int(time.time())
    ttl = int(ttl_sec if ttl_sec is not None else (VIDEO_DEDUP_TTL_SEC or 3600))
    exp = now + max(60, ttl)
    conn = _video_conn()
    try:
        conn.execute(
            """
            INSERT INTO video_request_locks(
                user_id, content_key, client_task_id, provider, provider_job_id,
                latest_asset_id, status, created_at, updated_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, content_key) DO UPDATE SET
                client_task_id=CASE WHEN excluded.client_task_id!='' THEN excluded.client_task_id ELSE video_request_locks.client_task_id END,
                provider=CASE WHEN excluded.provider!='' THEN excluded.provider ELSE video_request_locks.provider END,
                provider_job_id=CASE WHEN excluded.provider_job_id!='' THEN excluded.provider_job_id ELSE video_request_locks.provider_job_id END,
                latest_asset_id=CASE WHEN excluded.latest_asset_id!='' THEN excluded.latest_asset_id ELSE video_request_locks.latest_asset_id END,
                status=excluded.status,
                updated_at=excluded.updated_at,
                expires_at=excluded.expires_at
            """,
            (
                user_id,
                content_key,
                str(client_task_id or ""),
                str(provider or ""),
                str(provider_job_id or ""),
                str(latest_asset_id or ""),
                str(status or ""),
                now,
                now,
                exp,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _vlock_touch_ready(*, user_id: str, content_key: str, asset_id: str) -> None:
    """Mark a content lock as ready (asset persisted)."""
    _vlock_upsert(
        user_id=user_id,
        content_key=content_key,
        status="ready",
        latest_asset_id=str(asset_id or ""),
    )


def _vlock_touch_error(*, user_id: str, content_key: str) -> None:
    _vlock_upsert(user_id=user_id, content_key=content_key, status="error")


def _asset_get(user_id: str, asset_id: str) -> Optional[Dict[str, Any]]:
    conn = _video_conn()
    try:
        row = conn.execute(
            "SELECT * FROM assets WHERE asset_id=? AND user_id=?",
            (asset_id, user_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def _resolve_openai_video_id(user_id: str, maybe_asset_or_openai_id: str) -> Optional[str]:
    """If caller passes an internal asset_id, map it to source_video_id (OpenAI)."""
    sid = (maybe_asset_or_openai_id or "").strip()
    if not sid:
        return None
    # Already looks like OpenAI video id
    if re.search(r"\bvideo_[A-Za-z0-9]+\b", sid):
        return _normalize_video_id(sid)
    # Try resolve from assets table
    conn = _video_conn()
    try:
        row = conn.execute(
            "SELECT source_video_id FROM assets WHERE asset_id=? AND user_id=?",
            (sid, user_id),
        ).fetchone()
        if row and (row[0] or "").strip():
            return _normalize_video_id(row[0])
    finally:
        conn.close()
    return None

def _persist_sora_video_as_asset(
    *,
    openai_video_id: str,
    user_key: str,
    caption: str,
    kind: str,
    base_openai_video_id: Optional[str] = None,
    client_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Download completed Sora/OpenAI video, persist it, write assets+videos DB, set tags, update task."""
    user_key = _sanitize_user_key(user_key or "default")
    openai_vid = _normalize_video_id(openai_video_id)

    # Idempotency: if we already persisted an asset for this openai vid, reuse it.
    conn = _video_conn()
    try:
        row = conn.execute(
            "SELECT asset_id, storage_key FROM assets WHERE user_id=? AND source='sora' AND source_video_id=? ORDER BY created_at DESC LIMIT 1",
            (user_key, openai_vid),
        ).fetchone()
        if row:
            asset_id = row[0]
            storage_key = row[1]
            if client_task_id:
                _task_upsert(user_id=user_key, client_task_id=client_task_id, status="ready", provider="sora", provider_ref_id=openai_vid, latest_asset_id=asset_id)
            return {"asset_id": asset_id, "file_path": storage_key, "openai_video_id": openai_vid}
    finally:
        conn.close()

    asset_id = uuid7_str()
    dest_dir = (ASSET_VIDEOS_DIR / user_key / asset_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = (dest_dir / "original.mp4").resolve()

    size_bytes = _download_sora_video_to_file(openai_vid, dest_path)

    # Force portrait (stable iOS fullscreen) if enabled
    try:
        dest_path = _maybe_force_portrait_video(Path(dest_path))
        size_bytes = int(Path(dest_path).stat().st_size or 0)
    except Exception as e:
        try:
            log.warning("[VIDEO] portrait normalize failed (sora): %s", e)
        except Exception:
            pass


    sha = None
    ph = None
    meta = {}
    try:
        sha = sha256_file(dest_path)
        meta = probe_video_meta(dest_path)
        ph = video_dhash16(dest_path)
    except Exception:
        pass

    now = int(time.time())
    safe_caption = (caption or "").strip()
    if len(safe_caption) > 2000:
        safe_caption = safe_caption[:2000]

    # De-dup by sha256 when possible
    if sha:
        conn = _video_conn()
        try:
            row = conn.execute(
                "SELECT asset_id, storage_key FROM assets WHERE user_id=? AND sha256=? LIMIT 1",
                (user_key, sha),
            ).fetchone()
            if row:
                # remove duplicate file
                try:
                    if dest_path.exists():
                        dest_path.unlink(missing_ok=True)
                    try:
                        dest_dir.rmdir()
                    except Exception:
                        pass
                except Exception:
                    pass
                existing_asset_id = row[0]
                existing_path = row[1]
                if client_task_id:
                    _task_upsert(user_id=user_key, client_task_id=client_task_id, status="ready", provider="sora", provider_ref_id=openai_vid, latest_asset_id=existing_asset_id)
                return {"asset_id": existing_asset_id, "file_path": existing_path, "openai_video_id": openai_vid}
        finally:
            conn.close()

    # Write assets table + default tags + also insert into `videos` table so existing /v1/video/stream works.
    conn = _video_conn()
    try:
        conn.execute(
            """
            INSERT INTO assets(
                asset_id, user_id, source, source_video_id, origin_url,
                sha256, phash, status, storage_key, hls_key,
                width, height, duration_ms,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                user_key,
                "sora",
                openai_vid,
                "",
                sha,
                ph,
                "ready",
                str(dest_path),
                "",
                meta.get("width"),
                meta.get("height"),
                meta.get("duration_ms"),
                now,
                now,
            ),
        )

        # Default tags
        conn.execute(
            "INSERT OR REPLACE INTO asset_tags(asset_id, k, v) VALUES (?, 'view_state', 'unwatched')",
            (asset_id,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO asset_tags(asset_id, k, v) VALUES (?, 'lock', 'none')",
            (asset_id,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO asset_tags(asset_id, k, v) VALUES (?, 'ttl_hint', 'long')",
            (asset_id,),
        )

        # Mirror into videos table (draft/private) for compatibility with existing iOS lists/stream.
        tags = ["sora", str(kind or "create")]
        if base_openai_video_id:
            tags.append(f"base:{base_openai_video_id}")
        tags.append(f"source_video_id:{openai_vid}")
        conn.execute(
            """
            INSERT INTO videos (
                video_id, owner_key, owner_user_id, owner_display_name,
                caption, status, visibility, tags,
                file_path, mime, size_bytes,
                created_at, updated_at, published_at,
                views, likes, comments, shares
            ) VALUES (
                ?, ?, NULL, NULL,
                ?, 'draft', 'private', ?,
                ?, 'video/mp4', ?,
                ?, ?, NULL,
                0, 0, 0, 0
            )
            ON CONFLICT(video_id) DO UPDATE SET
                owner_key=excluded.owner_key,
                caption=excluded.caption,
                tags=excluded.tags,
                file_path=excluded.file_path,
                mime=excluded.mime,
                size_bytes=excluded.size_bytes,
                updated_at=excluded.updated_at
            """,
            (
                asset_id,
                user_key,
                safe_caption,
                json.dumps(tags, ensure_ascii=False),
                str(dest_path),
                int(size_bytes or 0),
                now,
                now,
            ),
        )

        conn.commit()
    finally:
        conn.close()

    if client_task_id:
        _task_upsert(
            user_id=user_key,
            client_task_id=client_task_id,
            status="ready",
            provider="sora",
            provider_ref_id=openai_vid,
            latest_asset_id=asset_id,
        )

    return {"asset_id": asset_id, "file_path": str(dest_path), "openai_video_id": openai_vid}

def _persist_minimax_video_as_asset(
    *,
    minimax_file_id: str,
    minimax_task_id: Optional[str],
    user_key: str,
    caption: str,
    kind: str,
    client_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Download completed MiniMax video, persist it, write assets+videos DB, set tags, update task."""
    user_key = _sanitize_user_key(user_key or "default")
    file_id = str(minimax_file_id or "").strip()
    if not file_id:
        raise RuntimeError("missing minimax_file_id")

    # Idempotency: if we already persisted an asset for this file_id, reuse it.
    conn = _video_conn()
    try:
        row = conn.execute(
            "SELECT asset_id, storage_key FROM assets WHERE user_id=? AND source='minimax' AND source_video_id=? ORDER BY created_at DESC LIMIT 1",
            (user_key, file_id),
        ).fetchone()
        if row:
            asset_id = row[0]
            storage_key = row[1]
            if client_task_id:
                _task_upsert(
                    user_id=user_key,
                    client_task_id=client_task_id,
                    status="ready",
                    provider="minimax",
                    provider_ref_id=file_id,
                    latest_asset_id=asset_id,
                )
            return {
                "asset_id": asset_id,
                "file_path": storage_key,
                "minimax_file_id": file_id,
                "minimax_task_id": minimax_task_id,
            }
    finally:
        conn.close()

    asset_id = uuid7_str()
    dest_dir = (ASSET_VIDEOS_DIR / user_key / asset_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = (dest_dir / "original.mp4").resolve()

    # 1) fetch download url
    dl_url = minimax_get_download_url(file_id)

    # 2) download to disk (atomic)
    size_bytes = 0
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    resp: Optional[requests.Response] = None
    try:
        resp = requests.get(dl_url, stream=True, timeout=MINIMAX_DOWNLOAD_TIMEOUT_SEC, allow_redirects=True)
        if resp.status_code >= 400:
            raise RuntimeError(f"minimax download failed {resp.status_code}: {resp.text[:400]}")
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(128 * 1024):
                if chunk:
                    f.write(chunk)
                    size_bytes += len(chunk)
        os.replace(tmp_path, dest_path)
    finally:
        try:
            if resp is not None:
                resp.close()
        except Exception:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Force portrait (stable iOS fullscreen) if enabled
    try:
        dest_path = _maybe_force_portrait_video(Path(dest_path))
        size_bytes = int(Path(dest_path).stat().st_size or size_bytes or 0)
    except Exception as e:
        try:
            log.warning("[VIDEO] portrait normalize failed (minimax): %s", e)
        except Exception:
            pass

    sha = None
    ph = None
    meta: Dict[str, Any] = {}
    try:
        sha = sha256_file(dest_path)
        meta = probe_video_meta(dest_path)
        ph = video_dhash16(dest_path)
    except Exception:
        pass

    now = int(time.time())
    safe_caption = (caption or "").strip()
    if len(safe_caption) > 2000:
        safe_caption = safe_caption[:2000]

    # De-dup by sha256 when possible
    if sha:
        conn = _video_conn()
        try:
            row = conn.execute(
                "SELECT asset_id, storage_key FROM assets WHERE user_id=? AND sha256=? LIMIT 1",
                (user_key, sha),
            ).fetchone()
            if row:
                # remove duplicate file
                try:
                    if dest_path.exists():
                        dest_path.unlink(missing_ok=True)
                    try:
                        dest_dir.rmdir()
                    except Exception:
                        pass
                except Exception:
                    pass
                existing_asset_id = row[0]
                existing_path = row[1]
                if client_task_id:
                    _task_upsert(
                        user_id=user_key,
                        client_task_id=client_task_id,
                        status="ready",
                        provider="minimax",
                        provider_ref_id=file_id,
                        latest_asset_id=existing_asset_id,
                    )
                return {
                    "asset_id": existing_asset_id,
                    "file_path": existing_path,
                    "minimax_file_id": file_id,
                    "minimax_task_id": minimax_task_id,
                }
        finally:
            conn.close()

    # Write assets table + default tags + also insert into `videos` table so existing /v1/video/stream works.
    conn = _video_conn()
    try:
        conn.execute(
            """
            INSERT INTO assets(
                asset_id, user_id, source, source_video_id, origin_url,
                sha256, phash, status, storage_key, hls_key,
                width, height, duration_ms,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                user_key,
                "minimax",
                file_id,
                dl_url or "",
                sha,
                ph,
                "ready",
                str(dest_path),
                "",
                meta.get("width"),
                meta.get("height"),
                meta.get("duration_ms"),
                now,
                now,
            ),
        )

        # Default tags
        conn.execute(
            "INSERT OR REPLACE INTO asset_tags(asset_id, k, v) VALUES (?, 'view_state', 'unwatched')",
            (asset_id,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO asset_tags(asset_id, k, v) VALUES (?, 'lock', 'none')",
            (asset_id,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO asset_tags(asset_id, k, v) VALUES (?, 'ttl_hint', 'long')",
            (asset_id,),
        )
        if minimax_task_id:
            conn.execute(
                "INSERT OR REPLACE INTO asset_tags(asset_id, k, v) VALUES (?, 'minimax_task_id', ?)",
                (asset_id, str(minimax_task_id)),
            )

        # Mirror into videos table (draft/private) for compatibility with existing iOS lists/stream.
        tags = ["minimax", str(kind or "create"), f"source_video_id:{file_id}"]
        if minimax_task_id:
            tags.append(f"task_id:{minimax_task_id}")
        conn.execute(
            """
            INSERT INTO videos (
                video_id, owner_key, owner_user_id, owner_display_name,
                caption, status, visibility, tags,
                file_path, mime, size_bytes,
                created_at, updated_at, published_at,
                views, likes, comments, shares
            ) VALUES (
                ?, ?, NULL, NULL,
                ?, 'draft', 'private', ?,
                ?, 'video/mp4', ?,
                ?, ?, NULL,
                0, 0, 0, 0
            )
            ON CONFLICT(video_id) DO UPDATE SET
                owner_key=excluded.owner_key,
                caption=excluded.caption,
                tags=excluded.tags,
                file_path=excluded.file_path,
                mime=excluded.mime,
                size_bytes=excluded.size_bytes,
                updated_at=excluded.updated_at
            """,
            (
                asset_id,
                user_key,
                safe_caption,
                json.dumps(tags, ensure_ascii=False),
                str(dest_path),
                int(size_bytes or 0),
                now,
                now,
            ),
        )

        conn.commit()
    finally:
        conn.close()

    if client_task_id:
        _task_upsert(
            user_id=user_key,
            client_task_id=client_task_id,
            status="ready",
            provider="minimax",
            provider_ref_id=file_id,
            latest_asset_id=asset_id,
        )

    return {
        "asset_id": asset_id,
        "file_path": str(dest_path),
        "minimax_file_id": file_id,
        "minimax_task_id": minimax_task_id,
    }

def _asset_compose_ready_payload(request: Request, asset: Dict[str, Any], task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = str(request.base_url).rstrip("/")
    aid = asset.get("asset_id")
    openai_vid = asset.get("source_video_id") or ""
    view_state = _asset_get_tag(aid, "view_state") if aid else None
    return {
        "ok": True,
        "status": "ready",
        "asset_id": aid,
        "openai_video_id": openai_vid,
        "play_url": f"{base}/v1/video/stream/{aid}" if aid else None,
        "view_state": view_state or "unwatched",
        "task": {
            "client_task_id": (task or {}).get("client_task_id"),
            "provider": (task or {}).get("provider"),
            "provider_job_id": (task or {}).get("provider_job_id"),
            "status": (task or {}).get("status"),
        } if task else None,
    }


# -------------------------
# Commercial Auth (v2)
# -------------------------

JWT_SECRET = os.getenv("CHATAGI_JWT_SECRET") or secrets.token_hex(32)
JWT_TTL_SECONDS = int(os.getenv("CHATAGI_JWT_TTL_SECONDS", "2592000"))  # 30 days
_PBKDF2_ITERS = int(os.getenv("CHATAGI_PBKDF2_ITERS", "200000"))

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))

def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return "pbkdf2_sha256$%d$%s$%s" % (
        _PBKDF2_ITERS,
        base64.b64encode(salt).decode("utf-8"),
        base64.b64encode(dk).decode("utf-8"),
    )

def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_b64, dk_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = base64.b64decode(salt_b64.encode("utf-8"))
        expected = base64.b64decode(dk_b64.encode("utf-8"))
        got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(got, expected)
    except Exception:
        return False

def _issue_access_token(user_id: str) -> str:
    now = int(time.time())
    token_id = uuid.uuid4().hex
    exp = now + JWT_TTL_SECONDS
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"iss": "chatagi", "sub": user_id, "iat": now, "exp": exp, "jti": token_id}

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    msg = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig = _b64url_encode(hmac.new(JWT_SECRET.encode("utf-8"), msg, hashlib.sha256).digest())
    token = f"{header_b64}.{payload_b64}.{sig}"

    conn = _video_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO sessions(token_id, user_id, issued_at, expires_at, revoked) VALUES (?, ?, ?, ?, 0)",
            (token_id, user_id, now, exp),
        )
        conn.commit()
    finally:
        conn.close()

    return token

def _decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".", 2)
        msg = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = _b64url_encode(hmac.new(JWT_SECRET.encode("utf-8"), msg, hashlib.sha256).digest())
        if not hmac.compare_digest(expected_sig, sig_b64):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None

def _get_bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None

def _auth_optional_user(request: Request) -> Optional[Dict[str, Any]]:
    token = _get_bearer_token(request)
    if not token:
        return None

    payload = _decode_access_token(token)
    if not payload:
        return None

    token_id = payload.get("jti")
    user_id = payload.get("sub")
    if not token_id or not user_id:
        return None

    conn = _video_conn()
    try:
        srow = conn.execute(
            "SELECT revoked, expires_at FROM sessions WHERE token_id=? AND user_id=?",
            (token_id, user_id),
        ).fetchone()
        if not srow:
            return None
        if int(srow["revoked"] or 0) == 1:
            return None
        if int(srow["expires_at"] or 0) < int(time.time()):
            return None

        urow = conn.execute(
            "SELECT user_id, username, display_name, avatar_url, created_at FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if not urow:
            return None

        return dict(urow)
    finally:
        conn.close()

def _auth_required_user(request: Request) -> Dict[str, Any]:
    user = _auth_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="unauthorized")
    return user

def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)

TURN_URLS = [u.strip() for u in (os.getenv("CHATAGI_TURN_URLS", "")).split(",") if u.strip()]
TURN_USERNAME = os.getenv("CHATAGI_TURN_USERNAME", "")
TURN_CREDENTIAL = os.getenv("CHATAGI_TURN_CREDENTIAL", "")

def _pack_f32(vec: List[float]) -> bytes:
    return array("f", [float(x) for x in vec]).tobytes()

def _unpack_f32(blob: bytes) -> array:
    a = array("f")
    a.frombytes(blob)
    return a

def _cosine(a: array, b: array) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        x = float(a[i]); y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb) + 1e-9
    return dot / denom

LOCAL_EMBED_DIM = int(os.getenv("SOLARA_LOCAL_EMBED_DIM") or "384")

def _local_embed(text: str, dim: int = LOCAL_EMBED_DIM) -> List[float]:
    """Local, no-network embedding (feature hashing + L2 norm).

    This is a fallback so long-term memory works even without paid embeddings.
    It is NOT as strong as real embedding models, but good enough for recall.
    """
    t = (text or "").strip().lower()
    if not t:
        return [0.0] * dim

    # Tokenize: keep alphanumerics as words; keep each CJK char; add CJK bigrams.
    toks = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", t)
    cjk = re.findall(r"[\u4e00-\u9fff]", t)
    if len(cjk) >= 2:
        toks.extend([cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)])

    vec = [0.0] * dim
    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 31) & 1) == 0 else -1.0
        vec[idx] += sign

    # L2 normalize
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]

def _openai_embed(text: str) -> Optional[List[float]]:
    text = (text or "").strip()
    if not text:
        return None

    # 1) Prefer real embeddings when key is configured
    if OPENAI_API_KEY:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MEMORY_EMBED_MODEL, "input": text}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code < 400:
                obj = r.json()
                emb = obj.get("data", [{}])[0].get("embedding")
                if isinstance(emb, list) and emb:
                    return emb
        except Exception:
            pass

    # 2) Fallback: local embedding (no network, no cost)
    try:
        return _local_embed(text)
    except Exception:
        return None

def memory_add(user_key: str, text: str) -> None:
    user_key = _sanitize_user_key(user_key)
    t = (text or "").strip()
    if not t:
        return
    emb = _openai_embed(t)
    if not emb:
        return
    sha = hashlib.sha1(t.encode("utf-8")).hexdigest()
    now = time.time()
    with _mem_conn() as con:
        con.execute("""
        INSERT INTO memory_items(id,user_key,text,text_sha1,embedding,dim,created_at,last_used_at)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(user_key,text_sha1)
        DO UPDATE SET text=excluded.text, embedding=excluded.embedding, dim=excluded.dim, last_used_at=excluded.last_used_at;
        """, (uuid.uuid4().hex, user_key, t, sha, _pack_f32(emb), len(emb), now, now))
        # prune
        con.execute("""
        DELETE FROM memory_items
        WHERE id IN (
          SELECT id FROM memory_items WHERE user_key=? ORDER BY last_used_at DESC LIMIT -1 OFFSET ?
        );
        """, (user_key, MEMORY_MAX_ITEMS_PER_USER))

def _openai_caption_image_data_url(data_url: str) -> str:
    """Return a short Chinese caption for an image (for multimodal memory)."""
    du = (data_url or "").strip()
    if not du:
        return ""
    # If no key, skip (we still can chat with images because /chat uses base64 input_image,
    # but memory captioning needs a model call).
    if not OPENAI_API_KEY:
        return ""
    prompt = (
        "请用中文为这张图片写一条【短caption】，用于长期记忆检索。\n"
        "要求：\n"
        "- 只描述客观可见内容（人物/物体/场景/文字）\n"
        "- 不要猜测身份、不要编造\n"
        "- 不要超过 1 句话，尽量精炼\n"
    )
    try:
        inp = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": du},
            ],
        }]
        obj = _openai_responses_create_nonstream(
            model=VISION_CAPTION_MODEL,
            inp=inp,
            max_output_tokens=120,
            truncation="auto",
        )
        # Extract output_text
        out = []
        for item in (obj.get("output") or []):
            for c in (item.get("content") or []):
                if c.get("type") == "output_text":
                    out.append(c.get("text") or "")
        cap = ("".join(out)).strip()
        cap = re.sub(r"\s+", " ", cap).strip()
        if len(cap) > VISION_CAPTION_MAX_CHARS:
            cap = cap[:VISION_CAPTION_MAX_CHARS].rstrip() + "…"
        return cap
    except Exception as e:
        log.warning("[mem.media] caption failed: %s", e)
        return ""


def memory_media_add_image(user_key: str, *, source: str, caption: str, image_bytes: bytes) -> None:
    user_key = _sanitize_user_key(user_key)
    cap = (caption or "").strip()
    if not cap:
        return
    if not image_bytes:
        return
    emb = _openai_embed(cap)
    if not emb:
        return
    sha = hashlib.sha1(image_bytes).hexdigest()
    now = time.time()
    with _mem_conn() as con:
        con.execute("""
        INSERT INTO memory_media(id,user_key,media_type,source,caption,sha1,embedding,dim,created_at,last_used_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_key,sha1)
        DO UPDATE SET caption=excluded.caption, embedding=excluded.embedding, dim=excluded.dim, last_used_at=excluded.last_used_at;
        """, (uuid.uuid4().hex, user_key, "image", (source or "")[:500], cap, sha, _pack_f32(emb), len(emb), now, now))


def memory_media_search(user_key: str, query: str, k: int, min_score: float) -> List[Dict[str, Any]]:
    user_key = _sanitize_user_key(user_key)
    q = (query or "").strip()
    if not q:
        return []
    qv_list = _openai_embed(q)
    if not qv_list:
        return []
    qv = array("f", [float(x) for x in qv_list])
    rows = []
    with _mem_conn() as con:
        cur = con.execute("SELECT id,media_type,source,caption,embedding,dim FROM memory_media WHERE user_key=? ORDER BY last_used_at DESC LIMIT 800", (user_key,))
        rows = cur.fetchall()
    scored = []
    for rid, mtype, src, cap, blob, dim in rows:
        try:
            v = _unpack_f32(blob)
            if len(v) != int(dim):
                continue
            s = _cosine(qv, v)
            if s >= float(min_score):
                scored.append((s, rid, mtype, src, cap))
        except Exception:
            continue
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:max(1, int(k))]
    now = time.time()
    if top:
        with _mem_conn() as con:
            for s, rid, *_ in top:
                con.execute("UPDATE memory_media SET last_used_at=? WHERE id=? AND user_key=?", (now, rid, user_key))
    return [
        {"id": rid, "media_type": mtype, "source": src, "caption": cap, "score": float(s)}
        for (s, rid, mtype, src, cap) in top
    ]

def memory_search(user_key: str, query: str, k: int, min_score: float) -> List[Dict[str, Any]]:
    user_key = _sanitize_user_key(user_key)
    q = (query or "").strip()
    if not q:
        return []
    qv_list = _openai_embed(q)
    if not qv_list:
        # fallback: recent
        with _mem_conn() as con:
            cur = con.execute("SELECT id,text,last_used_at FROM memory_items WHERE user_key=? ORDER BY last_used_at DESC LIMIT ?",
                              (user_key, MEMORY_RECENT_FALLBACK_N))
            return [{"id": rid, "text": t, "score": 0.0} for (rid,t,_) in cur.fetchall()]
    qv = array("f", [float(x) for x in qv_list])
    rows = []
    with _mem_conn() as con:
        cur = con.execute("SELECT id,text,embedding,dim FROM memory_items WHERE user_key=? ORDER BY last_used_at DESC LIMIT 1200", (user_key,))
        rows = cur.fetchall()
    scored = []
    for rid, txt, blob, dim in rows:
        try:
            v = _unpack_f32(blob)
            if len(v) != int(dim):
                continue
            s = _cosine(qv, v)
            if s >= float(min_score):
                scored.append((s, rid, txt))
        except Exception:
            continue
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:max(1,int(k))]
    now = time.time()
    if top:
        with _mem_conn() as con:
            for s,rid,_ in top:
                con.execute("UPDATE memory_items SET last_used_at=? WHERE id=? AND user_key=?", (now, rid, user_key))
    return [{"id": rid, "text": txt, "score": float(s)} for (s,rid,txt) in top]


def memory_build_context(user_key: str, query: str, k: int = MEMORY_TOP_K_DEFAULT, min_score: float = MEMORY_MIN_SCORE_DEFAULT) -> str:
    hits = memory_search(user_key, query, k=k, min_score=min_score)
    media_hits: List[Dict[str, Any]] = []
    try:
        media_hits = memory_media_search(user_key, query, k=max(1, int(k)//2), min_score=min_score)
    except Exception:
        media_hits = []

    if not hits and not media_hits:
        return ""

    lines = ["以下是【长期记忆】（仅供参考；若有冲突/不确定请向用户确认）："]
    total = 0
    for h in hits:
        t = (h.get("text") or "").strip()
        if not t:
            continue
        if len(t) > MEMORY_ITEM_MAX_CHARS:
            t = t[:MEMORY_ITEM_MAX_CHARS].rstrip() + "…"
        add = f"- {t}"
        if total + len(add) > MEMORY_CONTEXT_MAX_CHARS:
            break
        lines.append(add)
        total += len(add)

# Multimodal memory (images): add a compact section
if media_hits:
    lines.append("")
    lines.append("以下是【长期记忆-图片摘要】：")
    for h in media_hits:
        cap = (h.get("caption") or "").strip()
        if not cap:
            continue
        if len(cap) > MEMORY_ITEM_MAX_CHARS:
            cap = cap[:MEMORY_ITEM_MAX_CHARS].rstrip() + "…"
        add = f"- [image] {cap}"
        if total + len(add) > MEMORY_CONTEXT_MAX_CHARS:
            break
        lines.append(add)
        total += len(add)
    return "\n".join(lines).strip()


MEMORY_MIN_CHARS = int(os.getenv("MEMORY_MIN_CHARS") or "12")

def _looks_like_code(t: str) -> bool:
    s = (t or "").strip()
    if "```" in s:
        return True
    # Many braces/semicolons often indicates code/config
    sym = sum(1 for ch in s if ch in "{}[]();<>=")
    if len(s) >= 80 and sym / max(1, len(s)) > 0.08:
        return True
    return False

def _should_memory_add(text: str) -> bool:
    """Heuristic: store only useful long-term memories (avoid code/noise)."""
    t = (text or "").strip()
    if len(t) < MEMORY_MIN_CHARS:
        return False
    if _looks_like_code(t):
        return False

    # Explicit user intent to remember
    if re.search(r"(记住|记下|备忘|长期记忆|偏好|我的名字|我叫|我是|我住|我喜欢|我不喜欢|remember( that)?|my name is|i like|i prefer)", t, re.I):
        return True

    # Keep shorter conversational items by default
    return len(t) <= 600


# -----------------------------
# ✅ Memory store heuristics
# - Store user facts/preferences so the assistant can recall later.
# - Avoid storing code blocks or huge dumps.
# -----------------------------
_MEMORY_WORTHY_PATTERNS = [
    r"\b我叫\b", r"\b我的名字\b", r"\b叫我\b", r"\b我是\b",
    r"\b记住\b", r"\b以后\b", r"\b偏好\b", r"\b喜欢\b", r"\b不喜欢\b",
    r"\b生日\b", r"\b住在\b", r"\b来自\b", r"\b公司\b", r"\b工作\b",
    r"\bmy name is\b", r"\bcall me\b", r"\bremember\b", r"\bi like\b", r"\bi don't like\b",
]
_memory_worthy_re = re.compile("|".join(_MEMORY_WORTHY_PATTERNS), re.IGNORECASE)

def _should_memory_add(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # Avoid storing code blocks / stack traces / huge outputs
    if "```" in t:
        return False
    if len(t) < 8:
        return False
    # Trivial acks / greetings
    if t.lower() in ("hi", "hello", "ok", "okay") or t in ("你好", "在吗", "嗯", "好的", "收到"):
        return False
    # Strong signal patterns
    if _memory_worthy_re.search(t):
        return True
    # Soft signal: short first-person statements often encode preferences/facts
    if len(t) <= 220 and ("我" in t or "my " in t.lower() or t.lower().startswith("i ")):
        return True
    return False


def conv_create(user_key: str, title: str) -> str:
    cid = uuid.uuid4().hex
    now = time.time()
    t = (title or "").strip() or "新对话"
    t = t[:64]
    with _conv_conn() as con:
        con.execute("INSERT INTO conversations(id,user_key,title,created_at,updated_at,last_preview) VALUES(?,?,?,?,?,?)",
                    (cid, user_key, t, now, now, ""))
    return cid

def conv_touch(user_key: str, cid: str, preview: str = "") -> None:
    now = time.time()
    pv = (preview or "").strip()
    if len(pv) > 160:
        pv = pv[:160] + "…"
    with _conv_conn() as con:
        con.execute("UPDATE conversations SET updated_at=?, last_preview=? WHERE id=? AND user_key=?",
                    (now, pv, cid, user_key))

def conv_add_message(user_key: str, cid: str, role: str, content: str) -> None:
    mid = uuid.uuid4().hex
    now = time.time()
    with _conv_conn() as con:
        con.execute("INSERT INTO messages(id,conversation_id,user_key,role,content,created_at) VALUES(?,?,?,?,?,?)",
                    (mid, cid, user_key, role, content, now))
    conv_touch(user_key, cid, content)



# ============== JOB TABLES ==============
SORA_JOBS: Dict[str, Dict[str, Any]] = {}
RECENT_KEYS: Dict[str, Dict[str, Any]] = {}
RECENT_TTL = 120  # seconds

# Session media memory (voice -> intent uses last uploaded photo as reference)
SOLARA_LAST_MEDIA: Dict[str, Dict[str, Any]] = {}

# ================================
# Chat + Live TTS tables (in-memory)
# ================================

class LiveMP3Stream:
    """
    MP3 bytes pipe:
      - producer pushes mp3 bytes chunks
      - consumer (StreamingResponse) yields as soon as available
    """
    def __init__(self) -> None:
        self.q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self.done = False
        self.created = time.time()
        self.last_touch = self.created
        self._lock = threading.Lock()

        # For nicer concatenation: keep only first ID3 tag
        self._first_chunk = True

    def touch(self) -> None:
        self.last_touch = time.time()

    def close(self) -> None:
        with self._lock:
            if self.done:
                return
            self.done = True
            try:
                self.q.put_nowait(None)
            except Exception:
                pass

    @staticmethod
    def _strip_id3_if_present(data: bytes) -> bytes:
        # MP3 may start with ID3 header. Subsequent segments' ID3 can confuse players.
        if len(data) < 10:
            return data
        if data[0:3] != b"ID3":
            return data
        # ID3 header: bytes 6-9 are "syncsafe" size
        size_bytes = data[6:10]
        size = 0
        for b in size_bytes:
            size = (size << 7) | (b & 0x7F)
        total = 10 + size
        if total >= len(data):
            return b""
        return data[total:]

    def push(self, data: bytes) -> None:
        if not data:
            return
        self.touch()
        if not self._first_chunk:
            data = self._strip_id3_if_present(data)
        if not data:
            return
        self._first_chunk = False
        self.q.put(data)

    def generator(self) -> Iterator[bytes]:
        self.touch()
        while True:
            self.touch()
            item = self.q.get()
            if item is None:
                break
            yield item


LIVE_TTS: Dict[str, LiveMP3Stream] = {}
LIVE_TTS_LOCK = threading.Lock()

def _cleanup_live_tts() -> None:
    now = time.time()
    with LIVE_TTS_LOCK:
        for k in list(LIVE_TTS.keys()):
            s = LIVE_TTS.get(k)
            if not s:
                LIVE_TTS.pop(k, None)
                continue
            if (now - s.last_touch) > TTS_LIVE_TTL_SEC:
                try:
                    s.close()
                except Exception:
                    pass
                LIVE_TTS.pop(k, None)

def _create_live_tts_session() -> str:
    _cleanup_live_tts()
    tid = uuid.uuid4().hex
    with LIVE_TTS_LOCK:
        LIVE_TTS[tid] = LiveMP3Stream()
    return tid

def _get_live_tts(tid: str) -> Optional[LiveMP3Stream]:
    _cleanup_live_tts()
    with LIVE_TTS_LOCK:
        s = LIVE_TTS.get(tid)
    if s:
        s.touch()
    return s


# ================================
# ✅✅ True "text -> TTS" live feed (WS) —— 字声同步
# ================================
TTS_FEED_MIN_CHARS_DEFAULT = int(os.getenv("TTS_FEED_MIN_CHARS") or "18")
TTS_FEED_MAX_CHARS_DEFAULT = int(os.getenv("TTS_FEED_MAX_CHARS") or "80")

_TTS_FEED_PUNCT = set("。！？!?；;，,、：:\n")

def _tts_feed_pop_segments(buf: str, min_chars: int, max_chars: int, force_flush: bool = False) -> Tuple[List[str], str]:
    """Pop speakable segments from buffer.
    - Prefer punctuation cut after min_chars.
    - If too long, hard cut at max_chars.
    - If force_flush, also flush remainder (even if < min_chars).
    """
    if not buf:
        return [], ""

    # Normalize whitespace but keep Chinese
    buf = buf.replace("\r", "")
    out: List[str] = []

    while True:
        b = buf.strip()
        if not b:
            buf = ""
            break

        if not force_flush:
            # Not enough text yet, wait for more
            if len(b) < min_chars and not any((ch in _TTS_FEED_PUNCT) for ch in b):
                break

        # Work window
        window = b[: max_chars]
        cut = -1
        # Find last punctuation boundary within window, but after min_chars
        for i, ch in enumerate(window):
            if ch in _TTS_FEED_PUNCT and (i + 1) >= min_chars:
                cut = i + 1  # include punctuation

        if cut == -1:
            if len(b) >= max_chars:
                cut = max_chars
            else:
                if force_flush:
                    cut = len(b)
                else:
                    break

        seg = b[:cut].strip()
        if seg:
            out.append(seg)
        buf = b[cut:]

        if not buf:
            break

        # Keep looping to pop more segments if possible
        if not force_flush and len(buf.strip()) < min_chars and not any((ch in _TTS_FEED_PUNCT) for ch in buf):
            break

    return out, buf


class LiveTTSFeedWorker:
    """Background worker: consumes delta text and pushes MP3 bytes into LiveMP3Stream."""

    def __init__(self, tts_id: str) -> None:
        self.tts_id = tts_id
        self.q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()

        self.voice: str = "alloy"
        self.fmt: str = "mp3"
        self.instructions: str = (TTS_INSTRUCTIONS_DEFAULT or "").strip()

        try:
            self.speed: Optional[float] = float(TTS_SPEED_DEFAULT)
        except Exception:
            self.speed = None

        self.min_chars: int = max(4, int(TTS_FEED_MIN_CHARS_DEFAULT))
        self.max_chars: int = max(self.min_chars + 4, int(TTS_FEED_MAX_CHARS_DEFAULT))

        self.buf: str = ""
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def update_start(self, voice: Optional[str], fmt: Optional[str], speed: Any, instructions: Optional[str] = None,
                     min_chars: Any = None, max_chars: Any = None) -> None:
        with self.lock:
            if voice:
                self.voice = _normalize_tts_voice(str(voice))
            if fmt:
                self.fmt = str(fmt).strip().lower() or "mp3"
            if instructions is not None:
                self.instructions = (str(instructions) or "").strip()
            try:
                if speed is not None:
                    sp = float(speed)
                    # allow a bit faster than before
                    sp = max(0.5, min(sp, 2.0))
                    self.speed = sp
            except Exception:
                pass
            try:
                if min_chars is not None:
                    self.min_chars = max(4, int(min_chars))
            except Exception:
                pass
            try:
                if max_chars is not None:
                    self.max_chars = max(self.min_chars + 4, int(max_chars))
            except Exception:
                pass

    def push_delta(self, text: str) -> None:
        if not text:
            return
        self.q.put(("delta", text))

    def flush(self) -> None:
        self.q.put(("flush", ""))

    def end(self) -> None:
        self.q.put(("end", ""))

    def close(self) -> None:
        self.stop_evt.set()
        try:
            self.q.put_nowait(("end", ""))
        except Exception:
            pass

    def _tts_post(self, payload: Dict[str, Any]) -> requests.Response:
        # NOTE: model fallback for safety
        r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=payload, stream=True, timeout=120)

        if r.status_code >= 400 and payload.get("model") != "gpt-4o-mini-tts":
            try:
                r.close()
            except Exception:
                pass
            p2 = dict(payload)
            p2["model"] = "gpt-4o-mini-tts"
            r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p2, stream=True, timeout=120)

        if r.status_code >= 400 and "speed" in payload:
            try:
                r.close()
            except Exception:
                pass
            p3 = dict(payload)
            p3.pop("speed", None)
            r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p3, stream=True, timeout=120)

        if r.status_code >= 400 and "instructions" in payload:
            try:
                r.close()
            except Exception:
                pass
            p4 = dict(payload)
            p4.pop("instructions", None)
            p4.pop("speed", None)
            r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p4, stream=True, timeout=120)

        return r

    def _tts_push_segment(self, s: LiveMP3Stream, seg: str) -> None:
        seg = _tts_sanitize_text(seg)
        if not seg:
            return

        with self.lock:
            voice = self.voice
            fmt = self.fmt
            inst = self.instructions
            spd = self.speed

        payload: Dict[str, Any] = {"model": TTS_MODEL_DEFAULT, "voice": voice, "input": seg}
        if inst:
            payload["instructions"] = inst
        if spd is not None:
            payload["speed"] = spd
        if fmt and fmt != "mp3":
            payload["response_format"] = fmt

        r = self._tts_post(payload)
        rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
        log.info("[TTS.feed] voice=%s fmt=%s speed=%s len=%s -> %s rid=%s",
                 voice, fmt, spd, len(seg), r.status_code, rid)

        if r.status_code >= 400:
            try:
                r.close()
            except Exception:
                pass
            return

        for chunk in r.iter_content(chunk_size=TTS_STREAM_CHUNK_SIZE_DEFAULT):
            if chunk:
                s.push(chunk)

        try:
            r.close()
        except Exception:
            pass

    def _run(self) -> None:
        s = _get_live_tts(self.tts_id)
        if not s:
            return

        try:
            while not self.stop_evt.is_set():
                typ, payload = self.q.get()
                if typ == "delta":
                    self.buf += str(payload)
                    segs, self.buf = _tts_feed_pop_segments(self.buf, self.min_chars, self.max_chars, force_flush=False)
                    for seg in segs:
                        self._tts_push_segment(s, seg)

                elif typ == "flush":
                    segs, self.buf = _tts_feed_pop_segments(self.buf, self.min_chars, self.max_chars, force_flush=True)
                    for seg in segs:
                        self._tts_push_segment(s, seg)

                elif typ == "end":
                    # flush remainder
                    segs, self.buf = _tts_feed_pop_segments(self.buf, self.min_chars, self.max_chars, force_flush=True)
                    for seg in segs:
                        self._tts_push_segment(s, seg)
                    break

                else:
                    continue

        except Exception as e:
            log.warning("[TTS.feed] worker failed: %s", e)
        finally:
            try:
                s.close()
            except Exception:
                pass



class ChatJob:
    def __init__(self, chat_id: str, tts_id: str = "", tts_stream_enabled: bool = False) -> None:
        self.chat_id = chat_id
        self.tts_id = tts_id or ""
        self.tts_stream_enabled = bool(tts_stream_enabled) and bool(self.tts_id)
        self.created = time.time()
        self.last_touch = self.created

        self.events: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self.done_evt = threading.Event()
        self.full_text = ""
        self.error: Optional[str] = None

    def touch(self) -> None:
        self.last_touch = time.time()

    def push_event(self, event_obj: Dict[str, Any]) -> None:
        """Push a single OpenAI-style streaming event object (must include 'type')."""
        self.touch()
        try:
            self.events.put(event_obj)
        except Exception:
            pass

    def close_events(self) -> None:
        self.done_evt.set()
        try:
            self.events.put_nowait(None)
        except Exception:
            pass

CHAT_JOBS: Dict[str, ChatJob] = {}
CHAT_JOBS_LOCK = threading.Lock()

def _cleanup_chat_jobs() -> None:
    now = time.time()
    with CHAT_JOBS_LOCK:
        for k in list(CHAT_JOBS.keys()):
            j = CHAT_JOBS.get(k)
            if not j:
                CHAT_JOBS.pop(k, None)
                continue
            if (now - j.last_touch) > CHAT_JOB_TTL_SEC:
                CHAT_JOBS.pop(k, None)

def _create_chat_job(enable_tts_streaming: bool = False) -> ChatJob:
    _cleanup_chat_jobs()
    chat_id = uuid.uuid4().hex
    _tts_enabled = bool(enable_tts_streaming) and bool(CHAT_ENABLE_TTS_STREAMING)
    tts_id = _create_live_tts_session() if _tts_enabled else ""
    job = ChatJob(chat_id=chat_id, tts_id=tts_id, tts_stream_enabled=_tts_enabled)
    with CHAT_JOBS_LOCK:
        CHAT_JOBS[chat_id] = job
    return job

def _get_chat_job(chat_id: str) -> Optional[ChatJob]:
    _cleanup_chat_jobs()
    with CHAT_JOBS_LOCK:
        j = CHAT_JOBS.get(chat_id)
    if j:
        j.touch()
    return j

# ================================
# Utils
# ================================

def _cleanup_recent() -> None:
    now = time.time()
    for k in list(RECENT_KEYS.keys()):
        if now - RECENT_KEYS[k]["ts"] > RECENT_TTL:
            RECENT_KEYS.pop(k, None)

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _idem_key(ip: str, prompt: str, img_h: str, vid_h: str) -> str:
    raw = f"{ip}|{(prompt or '').strip()}|img:{img_h}|vid:{vid_h}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _short(s: str, n: int = 600) -> str:
    return (s or "")[:n]

def _log_http(r: requests.Response, tag: str) -> None:
    # NOTE: never call this on audio responses; r.text consumes body
    try:
        rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
        log.info("[%s] %s %s -> %s rid=%s body=%s",
                 tag, r.request.method, r.request.url, r.status_code, rid, _short(r.text))
    except Exception:
        pass

def _guess_mime_from_ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    if ext == ".mp4":
        return "video/mp4"
    if ext == ".mov":
        return "video/quicktime"
    if ext == ".webm":
        return "video/webm"
    return "application/octet-stream"

def _normalize_video_id(vid: str) -> str:
    s = (vid or "").strip()
    m = re.search(r"(video_[A-Za-z0-9]+)", s)
    return m.group(1) if m else s

def _parse_size(size_str: str) -> Optional[Tuple[int, int]]:
    try:
        w, h = size_str.lower().split("x")
        return int(w), int(h)
    except Exception:
        return None

def _ensure_portrait_size(size_str: str) -> str:
    wh = _parse_size(size_str)
    if not wh:
        return "720x1280"
    w, h = wh
    if w > h:
        w, h = h, w
    allowed = [(720, 1280), (1024, 1792)]
    if (w, h) in allowed:
        return f"{w}x{h}"
    return "720x1280" if h <= 1500 else "1024x1792"

SORA_SIZE_DEFAULT = _ensure_portrait_size(SORA_SIZE_DEFAULT)

async def _save_upload_to_file_and_sha1(upload: UploadFile, dst_path: Path) -> str:
    h = hashlib.sha1()
    with open(dst_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
    return h.hexdigest()

def _resize_image_bytes(raw: bytes, target_wh: Optional[Tuple[int, int]]) -> bytes:
    if not target_wh:
        return raw
    try:
        from PIL import Image
    except Exception:
        return raw
    try:
        w, h = target_wh
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im = im.resize((w, h))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        return raw

def _resize_video_file(src_path: str, dst_path: str, target_wh: Optional[Tuple[int, int]]) -> str:
    if not target_wh:
        return src_path
    try:
        w, h = target_wh
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-vf",
            f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "veryfast", "-an", dst_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path
    except Exception:
        return src_path

def _video_to_thumb_image(src_path: str, dst_path: str) -> Optional[str]:
    try:
        cmd = ["ffmpeg", "-y", "-i", src_path, "-frames:v", "1", dst_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path
    except Exception:
        return None

def _job_is_active(job: Optional[Dict[str, Any]]) -> bool:
    if not job:
        return False
    st = str(job.get("status") or "").lower().strip()
    return st not in ("done", "failed")

# ================================
# Remix parsing (fallback compatibility)
# ================================

def _extract_remix_block(raw_prompt: str) -> str:
    s = raw_prompt or ""
    if REMIX_BEGIN not in s:
        return ""
    if REMIX_END in s:
        try:
            return s.split(REMIX_BEGIN, 1)[1].split(REMIX_END, 1)[0]
        except Exception:
            return ""
    try:
        return s.split(REMIX_BEGIN, 1)[1]
    except Exception:
        return ""

def _strip_remix_block(raw_prompt: str) -> str:
    s = raw_prompt or ""
    if REMIX_BEGIN not in s:
        return s
    if REMIX_END in s:
        pre, rest = s.split(REMIX_BEGIN, 1)
        _, post = rest.split(REMIX_END, 1)
        return (pre + " " + post).strip()
    return s.split(REMIX_BEGIN, 1)[0].strip()

def parse_remix_request(raw_prompt: str) -> Optional[Dict[str, str]]:
    s = (raw_prompt or "").strip()
    if not s or REMIX_BEGIN not in s:
        return None

    block = _extract_remix_block(s) or s

    base_id = ""
    m = re.search(r"base_video_id\s*:\s*(video_[A-Za-z0-9]+)", block, flags=re.I)
    if m:
        base_id = m.group(1)
    else:
        m2 = re.search(r"(video_[A-Za-z0-9]+)", block)
        if m2:
            base_id = m2.group(1)

    base_id = _normalize_video_id(base_id)
    if not base_id:
        return None

    instr = ""
    m3 = re.search(r"(user_instruction|instruction|prompt)\s*:\s*([\s\S]*?)\Z", block, flags=re.I)
    if m3:
        instr = (m3.group(2) or "").strip()
    else:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        lines2 = []
        for ln in lines:
            if re.search(r"base_video_id\s*:", ln, flags=re.I):
                continue
            if ln.strip().lower().startswith("user_instruction:"):
                lines2.append(ln.split(":", 1)[1].strip())
            else:
                lines2.append(ln)
        instr = " ".join(lines2).strip()

    if not instr:
        instr = _strip_remix_block(s).strip()

    return {"base_video_id": base_id, "instruction": instr}

# ================================
# Sora prompt builders
# ================================

def build_sora_create_prompt(user_prompt: str) -> str:
    up = (user_prompt or "").strip()
    if not up:
        up = "Generate a clean vertical mobile video."
    guard = (
        "Constraints: no on-screen text, no subtitles, no watermarks, no logos, no UI elements. "
        "If reference media contains UI overlays, ignore them."
    )
    return f"{up}\n\n{guard}"

def build_sora_remix_prompt(base_video_id: str, user_instruction: str) -> str:
    base_video_id = _normalize_video_id(base_video_id)
    instr = (user_instruction or "").strip() or "Make subtle improvements. Keep everything else the same."

    return (
        f"REMIX TASK (base video: {base_video_id}).\n"
        "STRICT RULES:\n"
        "- Preserve the same main subject identity (same person/character/object), same environment, same composition/framing, and same camera motion.\n"
        "- Do NOT introduce new subjects, new backgrounds, or a different setting.\n"
        "- Do NOT remove the main subject.\n"
        "- Keep style/lighting consistent unless explicitly requested.\n"
        "- Apply ONLY the requested change(s). If ambiguous, make the smallest change possible.\n"
        "\n"
        "Requested changes:\n"
        f"{instr}\n"
    )

# ================================
# Video provider routing (Sora / MiniMax)
# ================================

def _normalize_video_provider(p: str) -> str:
    s = (p or "").strip().lower()
    if s in ("minimax", "hailuo", "hailuoai", "mm", "minimax_hailuo", "hailuo-2.3", "hailuo-02"):
        return "minimax"
    if s in ("sora", "openai", "oai"):
        return "sora"
    return ""

def _select_video_provider(mode: str = "create") -> str:
    """Select provider for current request/job. Remix is always handled by Sora."""
    m = (mode or "create").strip().lower()
    if m == "remix":
        return "sora"

    force = _normalize_video_provider(VIDEO_PROVIDER_FORCE)
    if force:
        if force == "minimax" and not MINIMAX_API_KEY:
            raise RuntimeError("VIDEO_PROVIDER_FORCE=minimax but MINIMAX_API_KEY is missing")
        return force

    default = _normalize_video_provider(VIDEO_PROVIDER_DEFAULT) or "sora"
    if default == "minimax" and not MINIMAX_API_KEY:
        # Safety: don't break existing Sora path if minimax is not configured.
        if SORA_API_KEY:
            log.warning("[VIDEO] VIDEO_PROVIDER_DEFAULT=minimax but MINIMAX_API_KEY missing; falling back to sora")
            return "sora"
        raise RuntimeError("VIDEO_PROVIDER_DEFAULT=minimax but MINIMAX_API_KEY is missing (and SORA_API_KEY not set)")

    return default

def _minimax_norm_resolution(res: str) -> str:
    r = (res or "").strip().upper() or "1080P"
    # Allowed values (docs): 512P/720P/768P/1080P
    if r not in ("512P", "720P", "768P", "1080P"):
        # If given like "1080" -> "1080P"
        if r.isdigit():
            r = r + "P"
        if r not in ("512P", "720P", "768P", "1080P"):
            r = "1080P"
    return r

def _minimax_norm_duration(sec: int) -> int:
    try:
        s = int(sec)
    except Exception:
        s = 6
    # For Hailuo models: typically 6 or 10s. Default to 6.
    if s <= 6:
        return 6
    if s >= 10:
        return 10
    # 7~9 -> 6 (avoid API error)
    return 6

def _file_to_data_url(path: str, mime_hint: Optional[str] = None) -> str:
    """Encode local file to data URL (for MiniMax first_frame_image)."""
    p = (path or "").strip()
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"file not found: {p}")
    mime = (mime_hint or _guess_mime_from_ext(p) or "application/octet-stream").strip()
    raw = Path(p).read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# ================================
# OpenAI headers
# ================================

def _json_headers() -> Dict[str, str]:
    if not SORA_API_KEY:
        raise RuntimeError("missing SORA_API_KEY / OPENAI_API_KEY")
    h = {
        "Authorization": f"Bearer {SORA_API_KEY}",
        "OpenAI-Beta": "video-generation=v1",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if OPENAI_ORG_ID:
        h["OpenAI-Organization"] = OPENAI_ORG_ID
    return h

def _auth_headers() -> Dict[str, str]:
    if not SORA_API_KEY:
        raise RuntimeError("missing SORA_API_KEY / OPENAI_API_KEY")
    h = {
        "Authorization": f"Bearer {SORA_API_KEY}",
        "OpenAI-Beta": "video-generation=v1",
    }
    if OPENAI_ORG_ID:
        h["OpenAI-Organization"] = OPENAI_ORG_ID
    return h

def _openai_headers(stream: bool = False) -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise RuntimeError("missing OPENAI_API_KEY / SORA_API_KEY")
    h = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if stream:
        h["Accept"] = "text/event-stream"
    if OPENAI_ORG_ID:
        h["OpenAI-Organization"] = OPENAI_ORG_ID
    return h


def _chat_headers(stream: bool = True) -> Dict[str, str]:
    """Headers for OpenAI Responses/Chat APIs (streaming-safe)."""
    return _openai_headers(stream=stream)


# ================================
# Smart Router helpers (DeepSeek text / OpenAI web+realtime)
# ================================

def _deepseek_headers(stream: bool = False) -> Dict[str, str]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("missing DEEPSEEK_API_KEY")
    h = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    if stream:
        h["Accept"] = "text/event-stream"
    return h


def _deepseek_chat_url() -> str:
    base = (DEEPSEEK_BASE_URL or "https://api.deepseek.com").strip().rstrip("/")
    return base + "/chat/completions"


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _extract_allow_web(req: Request, body: Dict[str, Any]) -> bool:
    # body flags
    for k in ("allow_web", "allowWeb", "enable_web_search", "enableWebSearch", "web_search", "webSearch", "use_web", "useWeb"):
        if k in body:
            return _boolish(body.get(k))
    # headers fallback
    for hk in ("x-allow-web", "x-solara-allow-web", "x-web-search"):
        hv = req.headers.get(hk)
        if hv is not None:
            return _boolish(hv)
    return True


def _attachments_require_openai(attachments: List[Dict[str, Any]]) -> bool:
    # DeepSeek text models cannot consume images/videos; force OpenAI for those.
    for a in (attachments or []):
        if not isinstance(a, dict):
            continue
        t = str(a.get("type") or a.get("kind") or "").strip().lower()
        if t in ("image", "video", "input_image", "input_video"):
            return True
        mime = str(a.get("mime") or "").strip().lower()
        if mime.startswith("image/") or mime.startswith("video/"):
            return True
    return False


def _route_provider(*, allow_web: bool, attachments: List[Dict[str, Any]]) -> Tuple[str, str]:
    # attachments (image/video) -> OpenAI
    if _attachments_require_openai(attachments):
        return "openai", "attachments"

    mode = (CHAT_ROUTE_MODE or "A").strip().upper()
    if mode == "A":
        return ("openai", "allow_web") if allow_web else ("deepseek", "allow_web=false")
    if mode in ("OPENAI", "OPENAI_ONLY", "GPT", "GPT_ONLY"):
        return "openai", f"mode={mode}"
    if mode in ("DEEPSEEK", "DEEPSEEK_ONLY"):
        return "deepseek", f"mode={mode}"
    # unknown -> safe fallback
    return ("openai", f"mode={mode}")


def _select_routed_model(provider: str, requested_model: str = "") -> str:
    req = (requested_model or "").strip()
    if provider == "deepseek":
        # server-controlled default; allow overriding only with explicit deepseek model name
        if req.lower().startswith("deepseek"):
            return req
        return DEEPSEEK_MODEL_DEFAULT or "deepseek-reasoner"
    # openai
    if req and not req.lower().startswith("deepseek"):
        return req
    return OPENAI_TEXT_MODEL or CHAT_MODEL_DEFAULT


def _ensure_provider_available(provider: str, reason: str) -> Tuple[str, str]:
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        if DEEPSEEK_FALLBACK_TO_OPENAI:
            return "openai", "deepseek_missing_fallback"
        raise RuntimeError("missing DEEPSEEK_API_KEY")
    return provider, reason

# ================================
# Sora REST
# ================================

def sora_create(prompt: str, ref_path: Optional[str] = None, ref_mime: Optional[str] = None) -> str:
    url = "https://api.openai.com/v1/videos"
    model = (os.getenv("SORA_MODEL") or SORA_MODEL_DEFAULT).strip()

    sec = SORA_SECONDS_DEFAULT if SORA_SECONDS_DEFAULT in (4, 8, 12) else 8
    size = SORA_SIZE_DEFAULT
    prompt_final = build_sora_create_prompt(prompt)

    if ref_path:
        files = None
        try:
            mime = ref_mime or _guess_mime_from_ext(ref_path)
            headers = _auth_headers()
            headers["Accept"] = "application/json"  # multipart

            fh = open(ref_path, "rb")
            files = {"input_reference": (os.path.basename(ref_path), fh, mime)}
            data = {"model": model, "prompt": prompt_final, "seconds": str(sec), "size": size}
            r = requests.post(url, headers=headers, data=data, files=files, timeout=60)
            _log_http(r, f"SORA.CREATE[ref:{mime}]")
        except Exception as e:
            log.warning("[SORA] create with ref failed -> text-only fallback: %s", e)
            body = {"model": model, "prompt": prompt_final, "seconds": sec, "size": size}
            r = requests.post(url, headers=_json_headers(), json=body, timeout=60)
            _log_http(r, f"SORA.CREATE[{model}]")
        finally:
            try:
                if files and files.get("input_reference"):
                    files["input_reference"][1].close()
            except Exception:
                pass
    else:
        body = {"model": model, "prompt": prompt_final, "seconds": sec, "size": size}
        r = requests.post(url, headers=_json_headers(), json=body, timeout=60)
        _log_http(r, f"SORA.CREATE[{model}]")

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        raise RuntimeError(f"Sora create error: {err}")

    data = r.json() if r.text else {}
    vid = data.get("id") or data.get("video_id")
    if not vid:
        raise RuntimeError(f"missing video id: {data}")
    return _normalize_video_id(str(vid))

def sora_remix(base_video_id: str, instruction: str) -> str:
    base_video_id = _normalize_video_id(base_video_id)
    url = f"https://api.openai.com/v1/videos/{base_video_id}/remix"

    prompt_final = build_sora_remix_prompt(base_video_id, instruction)
    body = {"prompt": prompt_final}

    r = requests.post(url, headers=_json_headers(), json=body, timeout=60)
    _log_http(r, f"SORA.REMIX[{base_video_id}]")

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        raise RuntimeError(f"Sora remix error: {err}")

    data = r.json() if r.text else {}
    vid = data.get("id") or data.get("video_id")
    if not vid:
        raise RuntimeError(f"missing remix video id: {data}")
    return _normalize_video_id(str(vid))

def sora_status(video_id: str) -> dict:
    video_id = _normalize_video_id(video_id)
    r = requests.get(f"https://api.openai.com/v1/videos/{video_id}", headers=_auth_headers(), timeout=60)
    _log_http(r, "SORA.STATUS")
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json() if r.text else {}

# ================================
# Sora background worker
# ================================




def _download_sora_video_to_file(video_id: str, out_path: Path, *, max_retries: int = 3) -> int:
    """Download a completed OpenAI video to a local file (streaming, with retries)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If already present, keep it.
    try:
        if out_path.exists() and out_path.stat().st_size > 0:
            return int(out_path.stat().st_size)
    except Exception:
        pass

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        resp = None
        try:
            resp = _fetch_sora_content_response(video_id, range_header=None)
            # Follow redirect to the actual file if present.
            if resp.status_code in (302, 303) and resp.headers.get("Location"):
                loc = resp.headers["Location"]
                try:
                    resp.close()
                except Exception:
                    pass
                resp = requests.get(loc, stream=True, timeout=(10, 600))

            if resp.status_code != 200:
                raise RuntimeError(f"download content failed: status={resp.status_code}")

            tmp = out_path.with_suffix(out_path.suffix + ".part")
            size = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    size += len(chunk)

            # Atomic-ish replace.
            tmp.replace(out_path)
            return int(size)

        except Exception as e:
            last_err = e
            try:
                if resp is not None:
                    resp.close()
            except Exception:
                pass

            # Simple backoff.
            time.sleep(min(2.0 * attempt, 8.0))

    raise RuntimeError(f"failed to download video content after {max_retries} attempts: {last_err}")


def _persist_sora_video_to_library(
    *,
    video_id: str,
    user_key: str,
    caption: str,
    kind: str = "create",
    base_video_id: str | None = None,
) -> Path:
    """Persist a completed Sora/OpenAI video into the server-side video library DB + files."""
    # Store under the stable OpenAI video_id so clients can find/remix later.
    dest = (VIDEOS_DIR / f"{video_id}.mp4").resolve()

    size_bytes = _download_sora_video_to_file(video_id, dest)

    now = int(time.time())
    safe_caption = (caption or "").strip()
    if len(safe_caption) > 2000:
        safe_caption = safe_caption[:2000]

    tags = ["sora", kind]
    if base_video_id:
        tags.append(f"base:{base_video_id}")

    conn = _video_conn()
    try:
        conn.execute(
            """
            INSERT INTO videos (
                video_id, owner_key, owner_user_id, owner_display_name,
                caption, status, visibility, tags,
                file_path, mime, size_bytes,
                created_at, updated_at, published_at,
                views, likes, comments, shares
            ) VALUES (
                ?, ?, NULL, NULL,
                ?, 'draft', 'private', ?,
                ?, 'video/mp4', ?,
                ?, ?, NULL,
                0, 0, 0, 0
            )
            ON CONFLICT(video_id) DO UPDATE SET
                owner_key=excluded.owner_key,
                caption=excluded.caption,
                tags=excluded.tags,
                file_path=excluded.file_path,
                mime=excluded.mime,
                size_bytes=excluded.size_bytes,
                updated_at=excluded.updated_at
            """,
            (
                video_id,
                user_key,
                safe_caption,
                json.dumps(tags, ensure_ascii=False),
                str(dest),
                int(size_bytes),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return dest
# ================================
# MiniMax REST (video generation)
# ================================

def _minimax_headers() -> Dict[str, str]:
    if not MINIMAX_API_KEY:
        raise RuntimeError("missing MINIMAX_API_KEY")
    return {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _minimax_parse_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json() if resp is not None else {}
    except Exception:
        return {}

def _minimax_check_base_resp(data: Dict[str, Any], *, ctx: str = "minimax") -> None:
    base = data.get("base_resp") or data.get("baseResp") or {}
    if isinstance(base, dict):
        code = base.get("status_code")
        msg = base.get("status_msg") or base.get("status_message") or base.get("message") or ""
        try:
            code_i = int(code or 0)
        except Exception:
            code_i = 0
        if code_i != 0:
            raise RuntimeError(f"{ctx} base_resp error: {code_i} {msg}".strip())

def minimax_create_t2v(
    prompt: str,
    *,
    duration: Optional[int] = None,
    resolution: Optional[str] = None,
    model: Optional[str] = None,
    prompt_optimizer: bool = True,
    fast_pretreatment: bool = False,
    callback_url: Optional[str] = None,
) -> str:
    url = f"{MINIMAX_BASE_URL}/v1/video_generation"
    dur = _minimax_norm_duration(duration if duration is not None else MINIMAX_DURATION_DEFAULT)
    res = _minimax_norm_resolution(resolution if resolution is not None else MINIMAX_RESOLUTION_DEFAULT)
    body: Dict[str, Any] = {
        "model": (model or MINIMAX_MODEL_T2V_DEFAULT),
        "prompt": (prompt or "").strip()[:2000],
        "duration": dur,
        "resolution": res,
        "prompt_optimizer": bool(prompt_optimizer),
        "fast_pretreatment": bool(fast_pretreatment),
    }
    if callback_url:
        body["callback_url"] = callback_url
    r = requests.post(url, headers=_minimax_headers(), json=body, timeout=MINIMAX_CREATE_TIMEOUT_SEC)
    _log_http(r, f"MINIMAX.T2V[{body.get('model')}]")
    if r.status_code >= 400:
        raise RuntimeError(f"minimax create failed {r.status_code}: {r.text[:400]}")
    data = _minimax_parse_json(r)
    _minimax_check_base_resp(data, ctx="minimax create")
    task_id = data.get("task_id") or data.get("taskId") or data.get("id")
    if not task_id:
        raise RuntimeError(f"minimax create missing task_id: {data}")
    return str(task_id)

def minimax_create_i2v(
    prompt: str,
    *,
    first_frame_image: str,
    duration: Optional[int] = None,
    resolution: Optional[str] = None,
    model: Optional[str] = None,
    prompt_optimizer: bool = True,
    fast_pretreatment: bool = False,
    callback_url: Optional[str] = None,
) -> str:
    url = f"{MINIMAX_BASE_URL}/v1/video_generation"
    dur = _minimax_norm_duration(duration if duration is not None else MINIMAX_DURATION_DEFAULT)
    res = _minimax_norm_resolution(resolution if resolution is not None else MINIMAX_RESOLUTION_DEFAULT)
    body: Dict[str, Any] = {
        "model": (model or MINIMAX_MODEL_I2V_DEFAULT),
        "prompt": (prompt or "").strip()[:2000],
        "first_frame_image": first_frame_image,
        "duration": dur,
        "resolution": res,
        "prompt_optimizer": bool(prompt_optimizer),
        "fast_pretreatment": bool(fast_pretreatment),
    }
    if callback_url:
        body["callback_url"] = callback_url
    r = requests.post(url, headers=_minimax_headers(), json=body, timeout=MINIMAX_CREATE_TIMEOUT_SEC)
    _log_http(r, f"MINIMAX.I2V[{body.get('model')}]")
    if r.status_code >= 400:
        raise RuntimeError(f"minimax i2v failed {r.status_code}: {r.text[:400]}")
    data = _minimax_parse_json(r)
    _minimax_check_base_resp(data, ctx="minimax i2v")
    task_id = data.get("task_id") or data.get("taskId") or data.get("id")
    if not task_id:
        raise RuntimeError(f"minimax i2v missing task_id: {data}")
    return str(task_id)

def minimax_create_s2v(
    prompt: str,
    *,
    subject_image: str,
    model: Optional[str] = None,
    prompt_optimizer: bool = True,
    callback_url: Optional[str] = None,
) -> str:
    """Subject-reference (character consistency). Note: docs show URL list; Data URLs may or may not be accepted."""
    url = f"{MINIMAX_BASE_URL}/v1/video_generation"
    body: Dict[str, Any] = {
        "model": (model or MINIMAX_MODEL_S2V_DEFAULT),
        "prompt": (prompt or "").strip()[:2000],
        "subject_reference": [
            {
                "type": "character",
                "image": [subject_image],
            }
        ],
        "prompt_optimizer": bool(prompt_optimizer),
    }
    if callback_url:
        body["callback_url"] = callback_url
    r = requests.post(url, headers=_minimax_headers(), json=body, timeout=MINIMAX_CREATE_TIMEOUT_SEC)
    _log_http(r, f"MINIMAX.S2V[{body.get('model')}]")
    if r.status_code >= 400:
        raise RuntimeError(f"minimax s2v failed {r.status_code}: {r.text[:400]}")
    data = _minimax_parse_json(r)
    _minimax_check_base_resp(data, ctx="minimax s2v")
    task_id = data.get("task_id") or data.get("taskId") or data.get("id")
    if not task_id:
        raise RuntimeError(f"minimax s2v missing task_id: {data}")
    return str(task_id)

def minimax_query_task(task_id: str) -> Dict[str, Any]:
    tid = str(task_id or "").strip()
    if not tid:
        raise RuntimeError("missing minimax task_id")
    url = f"{MINIMAX_BASE_URL}/v1/query/video_generation"
    r = requests.get(url, headers=_minimax_headers(), params={"task_id": tid}, timeout=MINIMAX_QUERY_TIMEOUT_SEC)
    _log_http(r, "MINIMAX.QUERY")
    if r.status_code >= 400:
        raise RuntimeError(f"minimax query failed {r.status_code}: {r.text[:400]}")
    data = _minimax_parse_json(r)
    _minimax_check_base_resp(data, ctx="minimax query")
    return data

def minimax_retrieve_file(file_id: str) -> Dict[str, Any]:
    fid = str(file_id or "").strip()
    if not fid:
        raise RuntimeError("missing minimax file_id")
    url = f"{MINIMAX_BASE_URL}/v1/files/retrieve"
    r = requests.get(url, headers=_minimax_headers(), params={"file_id": fid}, timeout=MINIMAX_QUERY_TIMEOUT_SEC)
    _log_http(r, "MINIMAX.FILE.RETRIEVE")
    if r.status_code >= 400:
        raise RuntimeError(f"minimax file retrieve failed {r.status_code}: {r.text[:400]}")
    data = _minimax_parse_json(r)
    _minimax_check_base_resp(data, ctx="minimax file retrieve")
    return data

def minimax_get_download_url(file_id: str) -> str:
    data = minimax_retrieve_file(file_id)
    file_obj = (data.get("file") or {}) if isinstance(data, dict) else {}
    dl = (file_obj.get("download_url") or file_obj.get("downloadUrl") or "").strip()
    if not dl:
        raise RuntimeError(f"minimax retrieve file missing download_url: {data}")
    return dl

def bg_sora_worker(job_id: str, timeout_sec: int = 1800) -> None:
    """Background video worker. Historically named *sora*; now supports multiple providers."""
    job = SORA_JOBS.get(job_id)
    if not job:
        return

    try:
        user_key = (job.get("user_key") or job.get("owner_key") or "").strip() or "default"
        client_task_id = (job.get("client_task_id") or "").strip() or None
        content_key = (job.get("content_key") or "").strip() or None

        mode = (job.get("mode") or "").strip().lower()
        if not mode:
            mode = "remix" if job.get("remix_base_video_id") else "create"

        # Provider selection: remix is always Sora.
        provider = _normalize_video_provider(job.get("provider") or "") or ""
        if mode == "remix":
            provider = "sora"
        if not provider:
            provider = _select_video_provider(mode)
        job["provider"] = provider

        ref_path = job.get("ref_path")
        ref_mime = job.get("ref_mime")

        # ----------------
        # Provider: Sora
        # ----------------
        if provider == "sora":
            if mode == "remix":
                base_id = _normalize_video_id(job.get("remix_base_video_id") or "")
                base_id = _resolve_openai_video_id(user_key, base_id) or base_id
                instr = (job.get("remix_instruction") or job.get("prompt") or "").strip()
                if not base_id:
                    raise RuntimeError("missing remix_base_video_id")
                if not instr:
                    instr = "Make subtle improvements. Keep everything else the same."
                video_id = sora_remix(base_id, instr)
            else:
                video_id = sora_create(job.get("prompt") or "", ref_path=ref_path, ref_mime=ref_mime)

            video_id = _normalize_video_id(video_id)
            job.update({"status": "running", "video_id": video_id, "openai_status": "running"})
            if client_task_id:
                _task_upsert(
                    user_id=user_key,
                    client_task_id=client_task_id,
                    status="working",
                    provider="sora",
                    provider_ref_id=video_id,
                )

            if VIDEO_DEDUP_ENABLED and content_key:
                try:
                    _vlock_upsert(
                        user_id=user_key,
                        content_key=content_key,
                        status="working",
                        provider="sora",
                        provider_job_id=job_id,
                        client_task_id=(client_task_id or job_id),
                    )
                except Exception:
                    pass

            deadline = time.time() + int(timeout_sec or 1800)
            finishing_grace_used = False
            transient_fail = 0

            while True:
                try:
                    info = sora_status(video_id)
                    transient_fail = 0
                except Exception as e:
                    transient_fail += 1
                    if transient_fail <= 5:
                        time.sleep(2)
                        continue
                    raise e

                status_raw = str(info.get("status") or info.get("state") or "processing")
                status = status_raw.lower().strip()

                job["openai_status"] = status_raw
                job["status"] = status

                try:
                    prog = int(info.get("progress") or 0)
                    job["progress"] = min(max(int(job.get("progress") or 0), prog), 99)
                except Exception:
                    prog = None

                if prog is not None and prog >= 99 and not finishing_grace_used:
                    deadline = max(deadline, time.time() + 600)
                    finishing_grace_used = True

                if status in ("completed", "succeeded", "done", "success"):
                    # Fallback proxy URL (still works even if we fail to persist).
                    job["url"] = f"/video/stream/{job_id}"
                    job["status"] = "saving"
                    job["progress"] = max(int(job.get("progress") or 0), 95)

                    # Persist completed video as a local asset (portrait-normalized if enabled).
                    openai_vid = job.get("video_id")
                    if not isinstance(openai_vid, str) or not openai_vid:
                        raise RuntimeError("missing openai video_id")

                    base_openai_vid = (
                        _resolve_openai_video_id(
                            user_key,
                            (job.get("remix_base_video_id") or job.get("base_video_id") or ""),
                        )
                        or (job.get("remix_base_video_id") or job.get("base_video_id") or None)
                    )

                    try:
                        asset = _persist_sora_video_as_asset(
                            openai_video_id=openai_vid,
                            user_key=user_key,
                            caption=job.get("prompt") or "",
                            kind=job.get("mode") or "create",
                            base_openai_video_id=base_openai_vid,
                            client_task_id=client_task_id,
                        )
                    except Exception as e:
                        job["save_error"] = str(e)
                        if VIDEO_DEDUP_ENABLED and content_key:
                            try:
                                _vlock_touch_error(user_id=user_key, content_key=content_key)
                            except Exception:
                                pass
                        raise

                    aid = asset.get("asset_id")
                    if not aid:
                        if VIDEO_DEDUP_ENABLED and content_key:
                            try:
                                _vlock_touch_error(user_id=user_key, content_key=content_key)
                            except Exception:
                                pass
                        raise RuntimeError("persist returned empty asset_id")

                    job["asset_id"] = aid
                    job["library_video_id"] = aid
                    job["library_file_path"] = str(asset.get("file_path") or "")
                    job["openai_video_id"] = _normalize_video_id(openai_vid)
                    job["url"] = f"/v1/video/stream/{aid}"
                    job["saved"] = True

                    if VIDEO_DEDUP_ENABLED and content_key:
                        try:
                            _vlock_touch_ready(user_id=user_key, content_key=content_key, asset_id=str(aid))
                        except Exception:
                            pass

                    job["status"] = "done"
                    job["progress"] = 100
                    return

                if status in ("failed", "error", "cancelled", "canceled"):
                    raise RuntimeError(f"sora failed: {status}")

                if time.time() > deadline:
                    raise TimeoutError("sora timeout")

                time.sleep(2)

        # ----------------
        # Provider: MiniMax
        # ----------------
        if provider == "minimax":
            prompt = (job.get("prompt") or "").strip()
            if not prompt:
                prompt = "Generate a video."
            duration = _minimax_norm_duration(job.get("seconds") or job.get("duration") or MINIMAX_DURATION_DEFAULT)
            resolution = _minimax_norm_resolution(job.get("resolution") or MINIMAX_RESOLUTION_DEFAULT)

            task_id: Optional[str] = None
            use_s2v = str(os.getenv("MINIMAX_PREFER_S2V") or os.getenv("MINIMAX_USE_S2V") or "").strip().lower() in ("1", "true", "yes", "on")
            if ref_path:
                data_url = _file_to_data_url(ref_path, ref_mime)
                if use_s2v:
                    try:
                        task_id = minimax_create_s2v(prompt, subject_image=data_url)
                    except Exception as e:
                        log.warning("[MINIMAX] s2v failed; fallback to i2v: %s", e)
                        task_id = minimax_create_i2v(
                            prompt,
                            first_frame_image=data_url,
                            duration=duration,
                            resolution=resolution,
                        )
                else:
                    task_id = minimax_create_i2v(
                        prompt,
                        first_frame_image=data_url,
                        duration=duration,
                        resolution=resolution,
                    )
            else:
                task_id = minimax_create_t2v(prompt, duration=duration, resolution=resolution)

            job.update(
                {
                    "status": "running",
                    "progress": max(int(job.get("progress") or 0), 5),
                    "minimax_task_id": task_id,
                    "openai_status": "Processing",
                }
            )
            if client_task_id:
                _task_upsert(
                    user_id=user_key,
                    client_task_id=client_task_id,
                    status="working",
                    provider="minimax",
                    provider_ref_id=str(task_id or ""),
                )

            if VIDEO_DEDUP_ENABLED and content_key:
                try:
                    _vlock_upsert(
                        user_id=user_key,
                        content_key=content_key,
                        status="working",
                        provider="minimax",
                        provider_job_id=job_id,
                        client_task_id=(client_task_id or job_id),
                    )
                except Exception:
                    pass

            deadline = time.time() + int(timeout_sec or 1800)
            transient_fail = 0

            while True:
                try:
                    info = minimax_query_task(str(task_id or ""))
                    transient_fail = 0
                except Exception as e:
                    transient_fail += 1
                    if transient_fail <= 5:
                        time.sleep(MINIMAX_POLL_INTERVAL_SEC)
                        continue
                    raise e

                status_raw = str(info.get("status") or "Processing")
                st = status_raw.lower().strip()
                job["openai_status"] = status_raw  # keep legacy key for client UI
                job["provider_status"] = status_raw

                # Rough progress mapping (MiniMax query does not expose %)
                if st in ("preparing",):
                    job["status"] = "running"
                    job["progress"] = min(max(int(job.get("progress") or 0), 10), 99)
                elif st in ("queueing", "queued"):
                    job["status"] = "running"
                    job["progress"] = min(max(int(job.get("progress") or 0), 20), 99)
                elif st in ("processing", "running"):
                    job["status"] = "running"
                    job["progress"] = min(max(int(job.get("progress") or 0), 50), 99)
                elif st in ("success", "succeeded", "completed", "done"):
                    file_id = str(info.get("file_id") or "").strip()
                    if not file_id:
                        raise RuntimeError(f"minimax success but missing file_id: {info}")

                    job["minimax_file_id"] = file_id

                    # Fallback proxy URL (still works even if we fail to persist).
                    job["url"] = f"/video/stream/{job_id}"
                    job["status"] = "saving"
                    job["progress"] = max(int(job.get("progress") or 0), 95)

                    try:
                        asset = _persist_minimax_video_as_asset(
                            minimax_file_id=file_id,
                            minimax_task_id=str(task_id or ""),
                            user_key=user_key,
                            caption=prompt,
                            kind=job.get("mode") or "create",
                            client_task_id=client_task_id,
                        )
                    except Exception as e:
                        job["save_error"] = str(e)
                        if VIDEO_DEDUP_ENABLED and content_key:
                            try:
                                _vlock_touch_error(user_id=user_key, content_key=content_key)
                            except Exception:
                                pass
                        raise

                    aid = asset.get("asset_id")
                    if not aid:
                        if VIDEO_DEDUP_ENABLED and content_key:
                            try:
                                _vlock_touch_error(user_id=user_key, content_key=content_key)
                            except Exception:
                                pass
                        raise RuntimeError("persist returned empty asset_id")

                    job["asset_id"] = aid
                    job["library_video_id"] = aid
                    job["library_file_path"] = str(asset.get("file_path") or "")
                    job["url"] = f"/v1/video/stream/{aid}"
                    job["saved"] = True

                    if VIDEO_DEDUP_ENABLED and content_key:
                        try:
                            _vlock_touch_ready(user_id=user_key, content_key=content_key, asset_id=str(aid))
                        except Exception:
                            pass

                    job["status"] = "done"
                    job["progress"] = 100
                    return
                elif st in ("fail", "failed", "error"):
                    raise RuntimeError(f"minimax failed: {status_raw}")

                if time.time() > deadline:
                    raise TimeoutError("minimax timeout")

                time.sleep(MINIMAX_POLL_INTERVAL_SEC)

        raise RuntimeError(f"unsupported provider: {provider}")

    except Exception as e:
        job["error"] = str(e)
        job["status"] = "failed"
        try:
            if (job.get("client_task_id") or "").strip() and (job.get("user_key") or "").strip():
                _task_upsert(
                    user_id=(job.get("user_key") or "default"),
                    client_task_id=(job.get("client_task_id") or "").strip(),
                    status="error",
                    provider=_normalize_video_provider(job.get("provider") or "") or "sora",
                )
        except Exception:
            pass

        if VIDEO_DEDUP_ENABLED and content_key:
            try:
                _vlock_touch_error(user_id=user_key, content_key=str(content_key))
            except Exception:
                pass

        log.exception("[VIDEO] provider=%s job=%s FAILED: %s", job.get("provider"), job_id, e)

def _spawn_sora_job(job_id: str, timeout_sec: int) -> None:
    def runner():
        acquired = False
        try:
            SORA_SEM.acquire()
            acquired = True
            bg_sora_worker(job_id, timeout_sec=timeout_sec)
        finally:
            if acquired:
                try:
                    SORA_SEM.release()
                except Exception:
                    pass

    threading.Thread(target=runner, daemon=True).start()

# ================================
# Streaming proxy helpers
# ================================

def _streaming_proxy_response(r: requests.Response, media_type_default: str = "video/mp4") -> StreamingResponse:
    proxy_headers = {k: r.headers[k] for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"] if k in r.headers}

    def gen():
        try:
            for c in r.iter_content(128 * 1024):
                if c:
                    yield c
        finally:
            try:
                r.close()
            except Exception:
                pass

    return StreamingResponse(
        gen(),
        media_type=r.headers.get("Content-Type", media_type_default),
        headers=proxy_headers,
        status_code=(r.status_code if r.status_code in (200, 206) else 200)
    )

def _fetch_sora_content_response(video_id: str, range_header: Optional[str]) -> requests.Response:
    vid = _normalize_video_id(video_id)
    content_url = f"https://api.openai.com/v1/videos/{vid}/content"

    headers = {"Authorization": f"Bearer {SORA_API_KEY}", "OpenAI-Beta": "video-generation=v1"}
    if OPENAI_ORG_ID:
        headers["OpenAI-Organization"] = OPENAI_ORG_ID
    if range_header:
        headers["Range"] = range_header

    backoffs = [0.5, 1.0, 2.0, 4.0]
    last: Optional[requests.Response] = None

    for i in range(len(backoffs) + 1):
        r = requests.get(content_url, headers=headers, stream=True, allow_redirects=False, timeout=120)

        if r.status_code in (302, 303):
            return r
        if r.status_code in (200, 206):
            return r

        if r.status_code in (404, 409, 425, 500, 502, 503, 504) and i < len(backoffs):
            try:
                r.close()
            except Exception:
                pass
            time.sleep(backoffs[i])
            last = r
            continue

        return r

    return last  # type: ignore

# ================================
# ✅ Chat streaming + server-side incremental TTS
# ================================

def _extract_delta_from_sse_json(obj: Dict[str, Any]) -> str:
    """
    Try best-effort extraction of text delta from:
      - Responses API: {"type":"response.output_text.delta","delta":"..."}
      - Chat Completions: {"choices":[{"delta":{"content":"..."}}]}
      - Other shapes: {"delta":{"text":"..."}}, {"text":"..."}
    """
    if not isinstance(obj, dict):
        return ""
    # responses-like
    d = obj.get("delta")
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        t = d.get("text")
        if isinstance(t, str):
            return t
    t0 = obj.get("text")
    if isinstance(t0, str):
        return t0

    # chat completions-like
    ch = obj.get("choices")
    if isinstance(ch, list) and ch:
        d2 = ch[0].get("delta") if isinstance(ch[0], dict) else None
        if isinstance(d2, dict):
            c = d2.get("content")
            if isinstance(c, str):
                return c
        msg = ch[0].get("message") if isinstance(ch[0], dict) else None
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str):
                return c

    # responses output array
    resp = obj.get("response")
    if isinstance(resp, dict):
        out = resp.get("output")
        if isinstance(out, list):
            for o in out:
                if isinstance(o, dict):
                    cont = o.get("content")
                    if isinstance(cont, list):
                        for c in cont:
                            if isinstance(c, dict):
                                if isinstance(c.get("delta"), str):
                                    return c["delta"]
                                if isinstance(c.get("text"), str):
                                    return c["text"]
    return ""


class SentenceSegmenter:
    """
    Incremental segmenter for TTS:
      - feed(delta) -> list of completed segments
      - flush() -> remaining
    """
    # sentence enders (CN + EN)
    _ENDERS = set(list("。！？!?."))
    # softer punctuation: only cut when buffer is long enough
    _SOFT_ENDERS = set(list("，,;；:：、"))
    # Backward-compatible flags.
    # Some earlier variants referenced `self._HARD_NEWLINE` and also invoked
    # `_find_cut(now)` (passing an extra positional arg). If either happens,
    # we must not crash the whole SSE stream.
    _HARD_NEWLINE = True

    def __init__(self, min_chars: int = 18, max_chars: int = 220) -> None:
        self.buf = ""
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        # Ensure instance has the attribute too (guards against class attr edits).
        self._HARD_NEWLINE = True

    def feed(self, delta: str) -> List[str]:
        out: List[str] = []
        if not delta:
            return out
        self.buf += delta

        # limit runaway
        while True:
            if len(self.buf) < self.min_chars:
                break

            cut = self._find_cut()
            if cut <= 0:
                break

            seg = self.buf[:cut].strip()
            self.buf = self.buf[cut:]
            if seg:
                out.append(seg)

        return out

    def flush(self) -> List[str]:
        seg = self.buf.strip()
        self.buf = ""
        return [seg] if seg else []

    def _find_cut(self, *args, **kwargs) -> int:
        # 1) newline cut (fast response for multi-line)
        if self._HARD_NEWLINE and "\n" in self.buf:
            idx = self.buf.find("\n")
            if idx >= self.min_chars:
                return idx + 1

        # 2) punctuation-based cut (hard enders first)
        best = -1
        for i, ch in enumerate(self.buf):
            if i < self.min_chars:
                continue
            if ch in self._ENDERS:
                # include trailing spaces/newlines
                j = i + 1
                while j < len(self.buf) and self.buf[j] in (" ", "\n", "\t", "\r"):
                    j += 1
                best = j
                break

        if best != -1:
            return best

        # 2b) softer punctuation: only cut when buffer is reasonably long
        soft_min = max(self.min_chars, 24)
        if len(self.buf) >= soft_min:
            for i, ch in enumerate(self.buf):
                if i < soft_min:
                    continue
                if ch in getattr(self, "_SOFT_ENDERS", set()):
                    j = i + 1
                    while j < len(self.buf) and self.buf[j] in (" ", "\n", "\t", "\r"):
                        j += 1
                    return j

        # 3) hard cap
        if len(self.buf) >= self.max_chars:
            return self.max_chars

        return -1


def _tts_headers() -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise RuntimeError("missing OPENAI_API_KEY / SORA_API_KEY")
    h = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/octet-stream",
    }
    if OPENAI_ORG_ID:
        h["OpenAI-Organization"] = OPENAI_ORG_ID
    return h

def _tts_media_type(fmt: str) -> str:
    fmt = (fmt or "mp3").lower().strip()
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "aac": "audio/aac",
        "opus": "audio/opus",
        "flac": "audio/flac",
        "pcm": "audio/pcm",
    }.get(fmt, "audio/mpeg")

def _normalize_tts_voice(voice: str) -> str:
    v = (voice or "").strip() or TTS_VOICE_DEFAULT
    if v.lower() == "realtime":
        v = REALTIME_VOICE_DEFAULT or "alloy"
    return v

# --- TTS text guards (commercial UX) ---
# Many TTS backends have strict input limits. Instead of 413, we:
# 1) strip code blocks, 2) compress/trim, 3) chunk into safe segments.
TTS_SPEECH_MAX_CHARS = int(os.getenv("TTS_SPEECH_MAX_CHARS") or "3500")
TTS_SEGMENT_MAX_CHARS = int(os.getenv("TTS_SEGMENT_MAX_CHARS") or "900")

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

def _tts_sanitize_text(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""

    # Remove fenced code blocks entirely (better than reading symbols).
    txt = _CODE_FENCE_RE.sub("（代码已省略）", raw)

    # Remove inline code `...`
    txt = re.sub(r"`([^`]{1,200})`", r"\1", txt)

    # Drop markdown image/link noise
    txt = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", txt)
    txt = re.sub(r"\[[^\]]+\]\([^\)]*\)", lambda m: m.group(0).split("](")[0].lstrip("["), txt)

    # Collapse excessive whitespace
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()

    # Hard trim to protect TTS service
    if len(txt) > TTS_SPEECH_MAX_CHARS:
        txt = txt[:TTS_SPEECH_MAX_CHARS].rstrip() + "……"

    # If everything was code and got stripped, speak a friendly hint
    if len(txt.strip()) < 6:
        return "代码已生成并显示在屏幕上。我也可以为你解释关键改动点。"

    return txt

def _split_tts_segments(text: str, max_chars: int = TTS_SEGMENT_MAX_CHARS) -> List[str]:
    """Split text into short segments (<= max_chars) by punctuation/newlines."""
    t = (text or "").strip()
    if not t:
        return []
    max_chars = max(120, int(max_chars))

    # Prefer sentence boundaries
    parts = re.split(r"(?<=[。！？!?；;])\s+|\n+", t)
    out: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            out.append(buf.strip())
        buf = ""

    for p in parts:
        p = (p or "").strip()
        if not p:
            continue
        if len(p) > max_chars:
            # Hard split long piece
            i = 0
            while i < len(p):
                out.append(p[i:i+max_chars].strip())
                i += max_chars
            continue

        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf += " " + p
        else:
            flush()
            buf = p

    flush()
    return out

def _tts_bytes(text: str, voice: str, fmt: str = "mp3", instructions: Optional[str] = None) -> bytes:
    text = (text or "").strip()
    if not text:
        return b""
    voice = _normalize_tts_voice(voice)
    fmt = (fmt or "mp3").strip().lower()
    model = TTS_MODEL_DEFAULT

    payload: Dict[str, Any] = {"model": model, "voice": voice, "input": text}
    inst = (instructions or TTS_INSTRUCTIONS_DEFAULT or "").strip()
    if inst:
        payload["instructions"] = inst
    if fmt and fmt != "mp3":
        payload["response_format"] = fmt

    r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=payload, timeout=120)
    rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
    log.info("[TTS.segment] voice=%s fmt=%s -> %s rid=%s bytes=%s",
             voice, fmt, r.status_code, rid, len(r.content or b""))

    if r.status_code >= 400:
        raise RuntimeError(f"openai_tts_error {r.status_code}: {_short(r.text, 300)}")

    return r.content or b""


def _tts_stream_fulltext_to_live(
    tts_id: str,
    text: str,
    voice: str,
    fmt: str = "mp3",
    instructions: Optional[str] = None,
    speed: Any = None,
) -> None:
    """Used by /tts_prepare: make the output robust (no 413) and stream bytes to LiveMP3Stream.

    - Sanitizes long/code-heavy text (don't read code symbols).
    - Splits into safe segments and streams sequentially.
    - Best-effort retries if the upstream TTS model rejects optional params.
    """
    s = _get_live_tts(tts_id)
    if not s:
        return

    voice = _normalize_tts_voice(voice)
    fmt = (fmt or "mp3").strip().lower()

    clean = _tts_sanitize_text(text)
    segments = _split_tts_segments(clean, TTS_SEGMENT_MAX_CHARS)

    # speed: client override -> else env default
    spd: Optional[float] = None
    try:
        if speed is not None:
            spd = float(speed)
    except Exception:
        spd = None
    if spd is None:
        try:
            spd = float(TTS_SPEED_DEFAULT)
        except Exception:
            spd = None
    if spd is not None:
        # common sane range
        spd = max(0.5, min(spd, 1.6))

    inst = (instructions or TTS_INSTRUCTIONS_DEFAULT or "").strip()

    def _post(payload: Dict[str, Any]) -> requests.Response:
        r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=payload, stream=True, timeout=120)
        if r.status_code >= 400 and "speed" in payload:
            try:
                r.close()
            except Exception:
                pass
            p2 = dict(payload)
            p2.pop("speed", None)
            r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p2, stream=True, timeout=120)
        if r.status_code >= 400 and "instructions" in payload:
            try:
                r.close()
            except Exception:
                pass
            p3 = dict(payload)
            p3.pop("instructions", None)
            p3.pop("speed", None)
            r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p3, stream=True, timeout=120)
        return r

    try:
        if not segments:
            s.close()
            return

        for seg in segments:
            seg = (seg or "").strip()
            if not seg:
                continue

            payload: Dict[str, Any] = {"model": TTS_MODEL_DEFAULT, "voice": voice, "input": seg}
            if inst:
                payload["instructions"] = inst
            if spd is not None:
                payload["speed"] = spd
            if fmt and fmt != "mp3":
                payload["response_format"] = fmt

            r = _post(payload)
            rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
            log.info("[TTS.prepare] voice=%s fmt=%s len=%s -> %s rid=%s", voice, fmt, len(seg), r.status_code, rid)

            if r.status_code >= 400:
                # If one segment fails, skip it (do not crash the whole playback)
                try:
                    r.close()
                except Exception:
                    pass
                continue

            for chunk in r.iter_content(chunk_size=TTS_STREAM_CHUNK_SIZE_DEFAULT):
                if chunk:
                    s.push(chunk)

            try:
                r.close()
            except Exception:
                pass

    except Exception as e:
        log.warning("[TTS.prepare] failed: %s", e)
    finally:
        try:
            s.close()
        except Exception:
            pass

def _stream_openai_events(
    model: str,
    messages: List[Dict[str, str]],
    attachments: Optional[List[Dict[str, Any]]] = None,
    *,
    max_output_tokens: Optional[int] = None,
    truncation: str = "auto",
    previous_response_id: Optional[str] = None,
    instructions: Optional[str] = None,
    enable_web_search: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    ✅ Official-aligned streaming for Route-B:

    Primary: OpenAI **Responses API** with `stream=true` (SSE), which emits JSON objects like:
      { "type": "response.output_text.delta", "delta": "..." }
      { "type": "response.output_text.done", "text": "..." }
      { "type": "response.completed", ... }

    Fallback: Chat Completions `stream=true` converted into the above Responses-style events.
    """
    model = (model or CHAT_MODEL_DEFAULT).strip() or CHAT_MODEL_DEFAULT

    # ---------- 1) Responses API (preferred) ----------
    r = None
    try:
        url = "https://api.openai.com/v1/responses"

        # Convert chat messages (+ optional attachments) -> Responses "input" format
        inp = _build_responses_input(messages, attachments or [])

        payload = {
            "model": model,
            "input": inp,
            "stream": True,
            # ✅ Avoid mid-output truncation: reserve enough output+reasoning tokens.
            "max_output_tokens": int(max_output_tokens or CHAT_MAX_OUTPUT_TOKENS_DEFAULT),
            # ✅ Avoid hard errors when conversation grows: let the API truncate older turns if needed.
            "truncation": (truncation or "auto"),
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if instructions:
            payload["instructions"] = instructions

        # ✅ Web search tool (official)
        if enable_web_search and CHAT_ENABLE_WEB_SEARCH_DEFAULT:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"
            # include sources for UI (site icons)
            payload["include"] = ["web_search_call.action.sources"]

        r = requests.post(
            url,
            headers=_chat_headers(),
            json=payload,
            stream=True,
            timeout=(CHAT_STREAM_CONNECT_TIMEOUT_SEC, CHAT_STREAM_READ_TIMEOUT_SEC),
        )
        rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
        log.info("[chat.responses] status=%s rid=%s", r.status_code, rid)

        if r.status_code >= 400:
            raise RuntimeError(f"openai_responses_error {r.status_code}: {_short(r.text, 500)}")

        for raw in r.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith(":"):
                continue  # comment/ping
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("type"):
                yield obj

        try:
            r.close()
        except Exception:
            pass
        return

    except Exception as e:
        log.warning("[chat] Responses streaming failed, fallback to ChatCompletions: %s", e)
        try:
            if r is not None:
                r.close()
        except Exception:
            pass

    # ---------- 2) Chat Completions fallback (converted to Responses-style events) ----------
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": True,
        # Best-effort: avoid truncation when falling back to Chat Completions.
        "max_completion_tokens": min(int(max_output_tokens or CHAT_MAX_OUTPUT_TOKENS_DEFAULT), 8192),
    }
    r = requests.post(
        url,
        headers=_chat_headers(),
        json=payload,
        stream=True,
        timeout=(CHAT_STREAM_CONNECT_TIMEOUT_SEC, CHAT_STREAM_READ_TIMEOUT_SEC),
    )
    rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
    log.info("[chat.completions] status=%s rid=%s", r.status_code, rid)

    # ✅ Compatibility retry: some legacy models only accept max_tokens
    if r.status_code >= 400:
        try:
            body_txt = r.text or ""
        except Exception:
            body_txt = ""
        if ("max_completion_tokens" in body_txt and "not supported" in body_txt) or ("Unsupported parameter" in body_txt and "max_completion_tokens" in body_txt):
            try:
                r.close()
            except Exception:
                pass
            # retry with legacy max_tokens
            payload.pop("max_completion_tokens", None)
            payload["max_tokens"] = min(int(max_output_tokens or CHAT_MAX_OUTPUT_TOKENS_DEFAULT), 8192)
            r = requests.post(
                url,
                headers=_chat_headers(),
                json=payload,
                stream=True,
                timeout=(CHAT_STREAM_CONNECT_TIMEOUT_SEC, CHAT_STREAM_READ_TIMEOUT_SEC),
            )
            rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
            log.info("[chat.completions.retry] status=%s rid=%s", r.status_code, rid)

    if r.status_code >= 400:
        raise RuntimeError(f"openai_chatcompletions_error {r.status_code}: {_short(r.text, 500)}")

    seq = 0
    full_parts: List[str] = []

    for raw in r.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break

        try:
            obj = json.loads(data)
        except Exception:
            continue

        delta = ""
        try:
            delta = obj["choices"][0]["delta"].get("content") or ""
        except Exception:
            delta = _extract_delta_from_sse_json(obj)

        if delta:
            seq += 1
            full_parts.append(delta)
            yield {
                "type": "response.output_text.delta",
                "delta": delta,
                "sequence_number": seq,
                "output_index": 0,
                "content_index": 0,
            }

    full_text = "".join(full_parts).strip()
    if full_text:
        seq += 1
        yield {
            "type": "response.output_text.done",
            "text": full_text,
            "sequence_number": seq,
            "output_index": 0,
            "content_index": 0,
        }

    seq += 1
    yield {"type": "response.completed", "sequence_number": seq}

    try:
        r.close()
    except Exception:
        pass





def _stream_deepseek_events(
    model: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """DeepSeek ChatCompletions SSE -> OpenAI Responses-style delta events.

    We only forward visible `content` deltas. Any reasoning fields are ignored.
    The generator yields a final internal marker:
      {"type":"solara._provider_done","finish_reason":"..."}
    """
    model = (model or DEEPSEEK_MODEL_DEFAULT).strip() or (DEEPSEEK_MODEL_DEFAULT or "deepseek-reasoner")
    url = _deepseek_chat_url()

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
    }
    try:
        mt = int(max_tokens or 0) if max_tokens is not None else 0
    except Exception:
        mt = 0
    if mt > 0:
        # DeepSeek max_tokens semantics are closer to completion tokens.
        payload["max_tokens"] = max(128, min(mt, 16384))

    r = requests.post(
        url,
        headers=_deepseek_headers(stream=True),
        json=payload,
        stream=True,
        timeout=(CHAT_STREAM_CONNECT_TIMEOUT_SEC, float(DEEPSEEK_TIMEOUT_SEC)),
    )
    rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
    log.info("[chat.deepseek] status=%s rid=%s", r.status_code, rid)

    if r.status_code >= 400:
        raise RuntimeError(f"deepseek_error {r.status_code}: {_short(r.text, 600)}")

    seq = 0
    finish_reason: Optional[str] = None
    saw_content = False

    try:
        for raw in r.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                break

            try:
                obj = json.loads(data)
            except Exception:
                continue

            choice = None
            try:
                choice = (obj.get("choices") or [None])[0]
            except Exception:
                choice = None
            if not isinstance(choice, dict):
                continue

            fr = choice.get("finish_reason")
            if fr:
                finish_reason = str(fr)

            delta_obj = choice.get("delta") or {}
            delta = ""
            if isinstance(delta_obj, dict):
                # ✅ visible text
                delta = str(delta_obj.get("content") or "")
                if delta:
                    saw_content = True
                else:
                    # ignore reasoning_content
                    if not saw_content:
                        _ = delta_obj.get("reasoning_content")

            if delta:
                seq += 1
                yield {
                    "type": "response.output_text.delta",
                    "delta": delta,
                    "sequence_number": seq,
                    "output_index": 0,
                    "content_index": 0,
                }
    finally:
        try:
            r.close()
        except Exception:
            pass

    yield {"type": "solara._provider_done", "finish_reason": (finish_reason or "")}
def _build_long_output_instructions(body: Dict[str, Any]) -> Optional[str]:
    """Optional instructions for very long code/text outputs.

    Enable by passing any of:
      - chunk_code=true
      - chunking=true
      - chunk_output=true

    Optional:
      - chunk_max_chars: preferred max characters per part (default 12000)

    The model will be asked to:
      - split output into numbered parts
      - NOT repeat earlier parts
      - if more remains, end with marker: [[SOLARA_CONTINUE]]

    The backend will auto-continue while it sees the marker or upstream emits `response.incomplete`.
    """
    try:
        enabled = bool(body.get("chunk_code") or body.get("chunking") or body.get("chunk_output"))
    except Exception:
        enabled = False
    if not enabled:
        return None

    try:
        max_chars = int(body.get("chunk_max_chars") or "12000")
    except Exception:
        max_chars = 12000
    max_chars = max(2000, min(max_chars, 40000))

    return (
        "When outputting long code or large files, avoid truncation by splitting into numbered parts.\n"
        f"- Keep each part under about {max_chars} characters.\n"
        "- Do NOT repeat earlier parts.\n"
        "- If more content remains after this part, end with the exact marker: [[SOLARA_CONTINUE]]\n"
        "- If you use triple backticks ``` for code blocks, ensure the final output is valid and close any open fences at the end.\n"
    )



# -----------------------------
# ✅ Legacy /chat（单次 HTTP 返回）长输出“不断流”核心实现
#    - 对齐官方 Responses API：max_output_tokens + truncation="auto" + previous_response_id 续写
#    - 处理两种“继续”信号：
#        A) status=incomplete 且 reason=max_output_tokens/max_tokens
#        B) 开启 chunk_code 后，模型按约定输出 [[SOLARA_CONTINUE]]
#    - 商用体验：自动去重（避免续写重复）、可选打包下载（zip）
# -----------------------------

# chat download cache (simple on-disk zip with TTL cleanup)
DOWNLOADS_DIR = BASE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)
CHAT_DOWNLOAD_TTL_SEC = int(os.getenv("CHAT_DOWNLOAD_TTL_SEC") or "7200")  # 2h
CHAT_MAX_TOTAL_CHARS = int(os.getenv("CHAT_MAX_TOTAL_CHARS") or "400000")  # safety cap
DOWNLOADS_LOCK = threading.Lock()

def _cleanup_downloads() -> None:
    now = time.time()
    with DOWNLOADS_LOCK:
        for p in DOWNLOADS_DIR.glob("*.zip"):
            try:
                if (now - p.stat().st_mtime) > CHAT_DOWNLOAD_TTL_SEC:
                    p.unlink(missing_ok=True)
            except Exception:
                pass

def _responses_extract_output_text(resp_obj: Dict[str, Any]) -> str:
    """Extract concatenated assistant output_text from a Responses API JSON object."""
    try:
        out = resp_obj.get("output") or []
        parts: List[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content") or []
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "output_text":
                    parts.append(str(c.get("text") or ""))
        return "".join(parts)
    except Exception:
        return ""

def _responses_incomplete_reason(resp_obj: Dict[str, Any]) -> Optional[str]:
    d = resp_obj.get("incomplete_details")
    if isinstance(d, dict):
        r = d.get("reason")
        return str(r) if r else None
    return None

def _append_without_overlap(accum: str, new: str, *, max_scan: int = 4000) -> str:
    """Append `new` to `accum` while removing common overlap (helps when model repeats prefix)."""
    if not new:
        return accum
    if not accum:
        return new
    a = accum
    b = new
    scan = min(len(a), len(b), max_scan)
    # Try longest suffix/prefix match
    for k in range(scan, 0, -1):
        if a[-k:] == b[:k]:
            return a + b[k:]
    return a + b

def _ensure_code_fences_closed(s: str) -> str:
    """Best-effort: if triple backticks are unbalanced, close them (avoids UI markdown breaking)."""
    if not s:
        return s
    # Count occurrences of ``` (not perfect, but works well in practice)
    cnt = s.count("```")
    if cnt % 2 == 1:
        return s.rstrip() + "\n```\n"
    return s

def _maybe_wrap_code_as_fenced_markdown(s: str) -> str:
    """If the model forgot ``` fences but the whole output looks like code, wrap it.

    This is a last-resort UX guard so iOS can syntax-highlight.
    """
    if not s:
        return s
    if "```" in s:
        return s

    txt = s.strip("\n")
    lines = txt.splitlines()
    nonempty = [ln.strip() for ln in lines if ln.strip()]
    if len(nonempty) < 4:
        return s

    score = 0
    for ln in nonempty[:120]:
        if ln.startswith(("import ", "from ", "def ", "class ", "func ", "let ", "var ", "struct ", "enum ")):
            score += 3
        if any(tok in ln for tok in ("{", "}", ";", "->", "=>")):
            score += 1
        if "(" in ln and ")" in ln:
            score += 1

    avg = score / max(1, len(nonempty))
    if avg < 1.8:
        return s

    lower = txt.lower()
    lang = "plaintext"
    if "import swiftui" in lower or "swiftui" in lower or "avfoundation" in lower or "func " in lower:
        lang = "swift"
    elif "async def " in lower or "\ndef " in ("\n" + lower) or "print(" in lower:
        lang = "python"
    elif "console.log" in lower or "function " in lower or "=>" in lower:
        lang = "javascript"

    return f"```{lang}\n{txt}\n```\n"


def _guess_audio_mime(filename: str, fallback: str = "application/octet-stream") -> str:
    fn = (filename or "").lower()
    if fn.endswith(".m4a") or fn.endswith(".mp4"):
        return "audio/mp4"
    if fn.endswith(".mp3"):
        return "audio/mpeg"
    if fn.endswith(".wav"):
        return "audio/wav"
    if fn.endswith(".aac"):
        return "audio/aac"
    if fn.endswith(".ogg"):
        return "audio/ogg"
    return fallback

def _resolve_local_upload_path(url: str) -> Optional[Path]:
    if not url:
        return None
    try:
        p = urlparse(url)
        path = p.path or ""
    except Exception:
        path = url

    path = (path or "").strip()
    if not path:
        return None

    # Most common: /uploads/<file>
    if "/uploads/" in path:
        rel = path.split("/uploads/", 1)[1].lstrip("/")
        cand = UPLOADS_DIR / rel
        if cand.exists():
            return cand

    # Sometimes: uploads/<file>
    if path.startswith("uploads/"):
        rel = path[len("uploads/"):]
        cand = UPLOADS_DIR / rel
        if cand.exists():
            return cand

    # Fallback: absolute within project (rare)
    if path.startswith("/"):
        cand = BASE_DIR / path.lstrip("/")
        if cand.exists():
            return cand

    return None

def _server_self_url(path_or_url: str) -> str:
    """Convert '/media/xxx' to a URL that THIS backend can fetch.
    IMPORTANT: this is ONLY for server-side fetching bytes, never sent to OpenAI.
    - If url is absolute with a private LAN host, rewrite to 127.0.0.1 to avoid LAN routing issues.
    """
    s = (path_or_url or "").strip()
    if not s:
        raise RuntimeError("empty_url")

    # Absolute URL
    if s.startswith("http://") or s.startswith("https://"):
        try:
            u = urlparse(s)
            host = (u.hostname or "").strip()
            port = u.port
            if host.startswith(("192.168.", "10.", "172.")):
                # keep scheme + path/query, but route to loopback
                port_part = f":{port}" if port else ""
                return f"{u.scheme}://127.0.0.1{port_part}{u.path or ''}" + (f"?{u.query}" if u.query else "")
        except Exception:
            pass
        return s

    # Relative path
    if s.startswith("/"):
        port = (os.getenv("PORT") or "8000").strip()
        return f"http://127.0.0.1:{port}{s}"

    return s


def _read_image_bytes_from_attachment(a: Dict[str, Any]) -> bytes:
    """Read bytes for an image attachment (Scheme-A).
    Priority:
      1) local disk path (if url maps to uploads/)
      2) HTTP fetch from THIS backend (e.g. /media/xxx or http://192.168.x.x/media/xxx)
    """
    url = (a.get("url") or "").strip()

    # 1) local file path (uploads/)
    if url:
        p = _resolve_local_upload_path(url)
        if p and p.exists():
            return p.read_bytes()

    # 2) fetch from this server
    if url:
        fetch_url = _server_self_url(url)
        r = requests.get(fetch_url, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"image_fetch_failed {r.status_code}: {_short(r.text, 200)}")
        data = r.content or b""
        if not data:
            raise RuntimeError("image_fetch_empty")
        return data

    raise RuntimeError("missing_attachment_url")


def _read_audio_bytes_from_url(url: str, mime_hint: str = "") -> Tuple[bytes, str, str]:
    """Return (bytes, filename, mime). Raises on failure."""
    url = (url or "").strip()
    if not url:
        raise RuntimeError("empty_audio_url")

    # 1) Local upload path
    local = _resolve_local_upload_path(url)
    if local and local.exists():
        data = local.read_bytes()
        filename = local.name
        mime = (mime_hint or "").strip() or _guess_audio_mime(filename)
        return data, filename, mime

    # 2) Remote download
    r = requests.get(url, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"download_audio_failed {r.status_code}: {_short(r.text, 200)}")
    data = r.content or b""
    if not data:
        raise RuntimeError("download_audio_empty")

    # best-effort filename
    filename = "audio.m4a"
    try:
        p = urlparse(url)
        tail = (p.path or "").split("/")[-1]
        if tail and "." in tail:
            filename = tail
    except Exception:
        pass

    mime = (mime_hint or "").strip() or (r.headers.get("content-type") or "").split(";")[0].strip() or _guess_audio_mime(filename)
    return data, filename, mime

def _openai_transcribe_audio(audio_bytes: bytes, filename: str, mime: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("missing OPENAI_API_KEY")
    files = {"file": (filename or "audio.m4a", audio_bytes, mime or "application/octet-stream")}
    data: Dict[str, Any] = {"model": TRANSCRIBE_MODEL_DEFAULT}
    if TRANSCRIBE_LANGUAGE_DEFAULT:
        data["language"] = TRANSCRIBE_LANGUAGE_DEFAULT

    r = requests.post(
        TRANSCRIBE_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files=files,
        data=data,
        timeout=TRANSCRIBE_TIMEOUT_SEC,
    )
    rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
    log.info("[transcribe] status=%s rid=%s", r.status_code, rid)
    if r.status_code >= 400:
        raise RuntimeError(f"openai_transcribe_error {r.status_code}: {_short(r.text, 300)}")

    try:
        obj = r.json()
    except Exception:
        obj = {}
    text = ""
    if isinstance(obj, dict):
        text = str(obj.get("text") or "")
    return (text or "").strip()

def _transcribe_audio_attachments_inplace(attachments: List[Dict[str, Any]]) -> str:
    """Fill `transcript` into audio attachments. Return a joined transcript string."""
    texts: List[str] = []
    if not attachments:
        return ""

    now = time.time()

    for a in attachments:
        if not isinstance(a, dict):
            continue
        at = (a.get("type") or "").strip().lower()
        if at != "audio":
            continue

        url = (a.get("url") or "").strip()
        mime = (a.get("mime") or "").strip()

        if not url:
            a["transcript"] = ""
            continue

        # Cache key: url + optional id
        key = f"{url}|{a.get('id') or ''}"
        cached = None
        with _TRANSCRIBE_LOCK:
            v = _TRANSCRIBE_CACHE.get(key)
            if v and (now - float(v[0])) <= float(TRANSCRIBE_CACHE_TTL_SEC):
                cached = v[1]
            elif v:
                _TRANSCRIBE_CACHE.pop(key, None)

        if cached is not None:
            a["transcript"] = cached
            if cached.strip():
                texts.append(cached.strip())
            continue

        transcript = ""
        try:
            audio_bytes, fn, mm = _read_audio_bytes_from_url(url, mime_hint=mime)
            transcript = _openai_transcribe_audio(audio_bytes, filename=fn, mime=mm)
        except Exception as e:
            log.warning("[transcribe] failed for url=%s: %s", _short(url, 120), e)
            transcript = ""

        a["transcript"] = transcript

        with _TRANSCRIBE_LOCK:
            _TRANSCRIBE_CACHE[key] = (now, transcript)

        if transcript.strip():
            texts.append(transcript.strip())

    return "\n".join(texts).strip()

def _build_responses_input(messages: List[Dict[str, str]], attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert legacy chat messages + attachments into Responses API 'input' format.

    IMPORTANT (official schema):
      - user/system/developer messages: content blocks are usually `input_text`
      - assistant messages (history): content blocks MUST be `output_text` (or `refusal`)
    """
    inp: List[Dict[str, Any]] = []

    def _text_block(role: str, text: str) -> Dict[str, Any]:
        r = (role or "user").strip()
        t = (text or "").strip()
        if r == "assistant":
            return {"type": "output_text", "text": t}
        # treat everything else as input-side text
        return {"type": "input_text", "text": t}

    # 1) messages -> blocks
    for m in messages:
        role = (m.get("role") or "user").strip() or "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        inp.append({"role": role, "content": [_text_block(role, content)]})

    # 2) Attachments (best-effort): append to the LAST user message
    if attachments:
        user_idx = None
        for i in range(len(inp) - 1, -1, -1):
            if inp[i].get("role") == "user":
                user_idx = i
                break
        if user_idx is None:
            inp.append({"role": "user", "content": []})
            user_idx = len(inp) - 1

        content_list = inp[user_idx].setdefault("content", [])
        if not isinstance(content_list, list):
            content_list = []
            inp[user_idx]["content"] = content_list

        for a in attachments:
            if not isinstance(a, dict):
                continue
            at = (a.get("type") or "").strip().lower()
            url = (a.get("url") or "").strip()
            mime = (a.get("mime") or "").strip()

            # ✅✅✅ FIX (Scheme-A): image must be sent as DATA URL in image_url (OpenAI must NOT download LAN URLs)
            if at == "image" and url:
                try:
                    raw = _read_image_bytes_from_attachment(a)
                    b64 = base64.b64encode(raw).decode("utf-8")
                    mime2 = (mime or "").strip() or "image/jpeg"
                    if not mime2.startswith("image/"):
                        mime2 = "image/jpeg"
                    data_url = f"data:{mime2};base64,{b64}"
                    content_list.append({"type": "input_image", "image_url": data_url})
                except Exception as e:
                    log.warning("[chat.attach] image->base64 failed url=%s err=%s", _short(url, 160), e)
                    # keep chat alive; degrade gracefully
                    content_list.append({"type": "input_text", "text": f"[image unavailable] {url}"})
                continue
            # ✅ audio: Responses API does NOT accept input_audio. Use transcript -> input_text.
            if at == "audio":
                tr = (a.get("transcript") or "").strip()
                if tr:
                    content_list.append({"type": "input_text", "text": f"[voice transcript]\n{tr}"})
                    continue
                if url:
                    content_list.append({"type": "input_text", "text": f"[voice message] url={url}"})
                    continue
                continue

            # Fallback: embed as text pointer
            if at and (a.get("id") or url):
                content_list.append({
                    "type": "input_text",
                    "text": f"[attachment:{at}] id={a.get('id')} mime={mime} url={url}"
                })

    return inp

def _openai_responses_create_nonstream(
    *,
    model: str,
    inp: List[Dict[str, Any]],
    max_output_tokens: int,
    truncation: str = "auto",
    previous_response_id: Optional[str] = None,
    instructions: Optional[str] = None,
    enable_web_search: bool = False,
) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/responses"
    payload: Dict[str, Any] = {
        "model": model,
        "input": inp,
        "max_output_tokens": int(max_output_tokens),
        "truncation": (truncation or "auto"),
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if instructions:
        payload["instructions"] = instructions

    # ✅ Web search tool (official) - only when enabled by allow_web
    if enable_web_search and CHAT_ENABLE_WEB_SEARCH_DEFAULT:
        payload["tools"] = [{"type": "web_search"}]
        payload["tool_choice"] = "auto"
        payload["include"] = ["web_search_call.action.sources"]

    r = requests.post(
        url,
        headers=_chat_headers(stream=False),
        json=payload,
        timeout=(CHAT_STREAM_CONNECT_TIMEOUT_SEC, CHAT_STREAM_READ_TIMEOUT_SEC),
    )
    rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
    log.info("[chat.responses.nonstream] status=%s rid=%s", r.status_code, rid)

    if r.status_code >= 400:
        raise RuntimeError(f"openai_responses_error {r.status_code}: {_short(r.text, 600)}")

    try:
        return r.json() if r.text else {}
    except Exception:
        return {}

def _create_download_zip(
    *,
    full_text: str,
    parts: List[str],
    meta: Dict[str, Any],
    req_body: Dict[str, Any],
) -> str:
    """Create a zip under downloads/ and return download_id."""
    _cleanup_downloads()
    download_id = uuid.uuid4().hex
    zip_path = DOWNLOADS_DIR / f"{download_id}.zip"

    with DOWNLOADS_LOCK:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("assistant_output.md", full_text)
            z.writestr("assistant_output.txt", full_text)
            z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
            z.writestr("request.json", json.dumps(req_body, ensure_ascii=False, indent=2))
            if parts:
                for i, p in enumerate(parts, start=1):
                    z.writestr(f"parts/part_{i:03d}.md", p)

    return download_id

def _split_into_parts(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    max_chars = max(2000, min(int(max_chars), 80000))
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    i = 0
    n = len(text)
    soft_floor = int(max_chars * 0.62)

    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end]

        cut = -1
        # Prefer paragraph boundary
        j = chunk.rfind("\n\n")
        if j >= soft_floor:
            cut = j + 2
        else:
            # else newline
            j = chunk.rfind("\n")
            if j >= soft_floor:
                cut = j + 1

        if cut <= 0:
            cut = len(chunk)

        part = chunk[:cut].rstrip()
        if part:
            parts.append(part)
        i += cut
        # skip leading newlines
        while i < n and text[i] in "\n\r":
            i += 1

    return parts

def _chat_complete_full_text(
    *,
    model: str,
    messages: List[Dict[str, str]],
    attachments: List[Dict[str, Any]],
    max_output_tokens: int,
    max_continuations: int,
    instructions: Optional[str],
    enable_web_search: bool = False,
) -> Tuple[str, str, str, Optional[Dict[str, Any]], int]:
    """Return: (full_text, status, response_id, incomplete_details, continuations)"""
    model = (model or CHAT_MODEL_DEFAULT).strip() or CHAT_MODEL_DEFAULT
    max_output_tokens = max(256, min(int(max_output_tokens), 60000))
    max_continuations = max(0, min(int(max_continuations), 20))

    CHUNK_MARKER = "[[SOLARA_CONTINUE]]"

    full = ""
    status = "completed"
    response_id = ""
    incomplete_details = None
    continuations = 0

    prev_rid: Optional[str] = None
    current_messages = messages
    current_attachments = attachments

    while True:
        inp = _build_responses_input(current_messages, current_attachments)
        resp = _openai_responses_create_nonstream(
            model=model,
            inp=inp,
            max_output_tokens=max_output_tokens,
            truncation="auto",
            previous_response_id=prev_rid,
            instructions=instructions,
            enable_web_search=enable_web_search,
        )

        response_id = str(resp.get("id") or response_id)
        status = str(resp.get("status") or status)
        incomplete_details = resp.get("incomplete_details") if isinstance(resp.get("incomplete_details"), dict) else None

        part = _responses_extract_output_text(resp) or ""
        # strip marker (avoid leaking it to user)
        tail = part[-512:]
        marker_seen = CHUNK_MARKER in tail
        if marker_seen:
            part = part.replace(CHUNK_MARKER, "")

        # append with de-dup overlap
        full = _append_without_overlap(full, part)

        # safety cap
        if len(full) >= CHAT_MAX_TOTAL_CHARS:
            full = full[:CHAT_MAX_TOTAL_CHARS]
            status = "incomplete"
            incomplete_details = {"reason": "server_max_total_chars"}
            break

        reason = _responses_incomplete_reason(resp) or ""
        need_continue = False

        if status == "incomplete" and reason in ("max_output_tokens", "max_tokens"):
            need_continue = True
        if marker_seen:
            need_continue = True

        if not need_continue:
            break

        if continuations >= max_continuations:
            break

        if not response_id:
            break

        continuations += 1
        prev_rid = response_id

        cont_prompt = (
            "Continue exactly from where you left off. "
            "Output ONLY the remaining code/text. Do NOT repeat earlier parts. "
            "If you were inside a ``` code block, continue inside the same block. "
            "Make sure the final output is valid and close any open code fences at the end."
        )
        if instructions:
            cont_prompt += f" If there is still more after this part, end with {CHUNK_MARKER}."

        current_messages = [{"role": "user", "content": cont_prompt}]
        current_attachments = []

    full = _maybe_wrap_code_as_fenced_markdown(_ensure_code_fences_closed(full)).strip()
    return full, status, response_id, incomplete_details, continuations




def _deepseek_complete_full_text(
    *,
    model: str,
    messages: List[Dict[str, str]],
    attachments: List[Dict[str, Any]],
    max_output_tokens: int,
    max_continuations: int,
    instructions: Optional[str],
) -> Tuple[str, str, str, Optional[Dict[str, Any]], int]:
    """DeepSeek non-stream (implemented via SSE stream accumulation) with auto-continuation.

    Return: (full_text, status, response_id, incomplete_details, continuations)
    NOTE: DeepSeek does not provide OpenAI `previous_response_id`, so continuation is done by
    a tail-anchored prompt to reduce repetition.
    """
    model = (model or DEEPSEEK_MODEL_DEFAULT).strip() or (DEEPSEEK_MODEL_DEFAULT or "deepseek-reasoner")
    max_output_tokens = max(256, min(int(max_output_tokens), 60000))
    max_continuations = max(0, min(int(max_continuations), 20))

    CHUNK_MARKER = "[[SOLARA_CONTINUE]]"

    # Build a stable system prefix (style prompt + memory context + optional long-output instructions)
    prefix: List[Dict[str, str]] = []
    for m in (messages or []):
        if not isinstance(m, dict):
            continue
        r = str(m.get("role") or "").strip()
        if r in ("system", "developer"):
            prefix.append({"role": r, "content": str(m.get("content") or "")})
        else:
            break

    if instructions:
        # ensure instructions exist (avoid duplication)
        if not any((mm.get("content") or "").strip() == instructions.strip() for mm in prefix):
            insert_at = 1 if (prefix and prefix[0].get("role") in ("system", "developer")) else 0
            prefix.insert(insert_at, {"role": "system", "content": instructions})

    def _extract_transcripts(atts: List[Dict[str, Any]]) -> str:
        texts: List[str] = []
        for a in (atts or []):
            if not isinstance(a, dict):
                continue
            if str(a.get("type") or "").strip().lower() != "audio":
                continue
            tr = str(a.get("transcript") or "").strip()
            if tr:
                texts.append(tr)
        return "\n".join(texts).strip()

    transcript = _extract_transcripts(attachments)

    full = ""
    status = "completed"
    response_id = ""  # DeepSeek id not required for our flow
    incomplete_details: Optional[Dict[str, Any]] = None
    continuations = 0

    # Build initial conversation for DeepSeek:
    ds_messages = list(messages or [])
    if transcript:
        ds_messages.append({"role": "user", "content": f"[voice transcript]\n{transcript}"})

    while True:
        part_parts: List[str] = []
        finish_reason = ""
        tail_buf = ""
        marker_seen = False

        # cap completion tokens for DeepSeek
        mt = min(max_output_tokens, 16384)

        for ev in _stream_deepseek_events(model=model, messages=ds_messages, max_tokens=mt):
            typ = ev.get("type") if isinstance(ev, dict) else None
            if typ == "solara._provider_done":
                finish_reason = str(ev.get("finish_reason") or "")
                continue
            if typ != "response.output_text.delta":
                continue

            delta = str(ev.get("delta") or "")
            if not delta:
                continue

            part_parts.append(delta)

            if instructions:
                tail_buf = (tail_buf + delta)[-256:]
                if CHUNK_MARKER in tail_buf:
                    marker_seen = True

        part = "".join(part_parts)
        if CHUNK_MARKER in part:
            part = part.replace(CHUNK_MARKER, "")

        full = _append_without_overlap(full, part)

        if len(full) >= CHAT_MAX_TOTAL_CHARS:
            full = full[:CHAT_MAX_TOTAL_CHARS]
            status = "incomplete"
            incomplete_details = {"reason": "server_max_total_chars"}
            break

        fr = (finish_reason or "").lower()
        need_continue = bool(marker_seen) or fr in ("length", "max_tokens", "max_output_tokens")

        if not need_continue:
            break

        if continuations >= max_continuations:
            status = "incomplete"
            incomplete_details = {"reason": "max_continuations"}
            break

        continuations += 1

        # Tail-anchored continuation to reduce repetition
        tail_ctx = full[-1600:]
        cont_prompt = (
            "Continue exactly from where you left off. Output ONLY the remaining code/text. "
            "Do NOT repeat earlier parts. If you were inside a ``` code block, continue inside the same block. "
            "Make sure the final output is valid and close any open code fences at the end.\n\n"
            "Here is the END of what you already produced (do NOT repeat it):\n-----\n"
            f"{tail_ctx}\n-----\n\nContinue from immediately AFTER the above end."
        )
        if instructions:
            cont_prompt += f" If there is still more after this part, end with {CHUNK_MARKER}."

        ds_messages = prefix + [{"role": "user", "content": cont_prompt}]
        # attachments are folded already

    full = _maybe_wrap_code_as_fenced_markdown(_ensure_code_fences_closed(full)).strip()
    return full, status, response_id, incomplete_details, continuations
def _spawn_chat_worker(
    job: ChatJob,
    model: str,
    messages: List[Dict[str, str]],
    attachments: List[Dict[str, Any]],
    tts_voice: str,
    *,
    provider: str = "openai",
    allow_web: bool = False,
    route_reason: str = "",
    max_output_tokens: int,
    max_continuations: int,
    instructions: Optional[str] = None,
    conversation_id: str = "",
    user_key: str = "default",
    last_user_text: str = "",
    memory_enabled: bool = MEMORY_ENABLED_DEFAULT,
) -> None:
    """
    Background worker (Route-B):

    - Streams provider events into job.events (client renders text by `type`)
    - Extracts `response.output_text.delta` to build incremental text
    - Sentence-segments the incremental text and calls OpenAI TTS per segment
    - Supports auto-continuation:
        * OpenAI: response.incomplete + previous_response_id
        * DeepSeek: finish_reason=length (tail-anchored continuation)
    """

    def run():
        tts_enabled = bool(getattr(job, "tts_stream_enabled", False)) and bool(getattr(job, "tts_id", ""))
        live = _get_live_tts(job.tts_id) if tts_enabled else None
        if tts_enabled and not live:
            job.error = "live_tts_not_found"
            job.push_event({"type": "error", "error": {"message": "live_tts_not_found"}})
            job.close_events()
            return

        provider_norm = (provider or "openai").strip().lower()
        log.info("[chat-worker] chat_id=%s provider=%s model=%s allow_web=%s", job.chat_id, provider_norm, model, bool(allow_web))

        seg_q: Optional["queue.Queue[Optional[str]]"] = queue.Queue() if tts_enabled else None
        segmenter = SentenceSegmenter(min_chars=CHAT_TTS_MIN_CHARS_DEFAULT, max_chars=220) if tts_enabled else None

        # Collect web search sources (for UI site icons)
        web_sources: Dict[str, Dict[str, str]] = {}

        def _push_sources(new_sources: Any) -> None:
            nonlocal web_sources
            if not isinstance(new_sources, list):
                return
            changed = False
            for s in new_sources:
                if not isinstance(s, dict):
                    continue
                url = str(s.get("url") or "").strip()
                if not url:
                    continue
                if url in web_sources:
                    continue
                title = str(s.get("title") or "").strip() or url
                web_sources[url] = {"title": title, "url": url}
                changed = True
            if changed:
                job.push_event({"type": "sources", "sources": list(web_sources.values())})
                job.push_event({"type": "solara.sources", "sources": list(web_sources.values())})

        def _tts_stream_segment(seg: str) -> None:
            seg = (seg or "").strip()
            if not seg:
                return

            voice = _normalize_tts_voice(tts_voice)
            payload: Dict[str, Any] = {"model": TTS_MODEL_DEFAULT, "voice": voice, "input": seg}

            inst = (TTS_INSTRUCTIONS_DEFAULT or "").strip()
            if inst:
                payload["instructions"] = inst

            try:
                spd = float(TTS_SPEED_DEFAULT)
            except Exception:
                spd = None
            if spd is not None:
                payload["speed"] = max(0.5, min(spd, 1.6))

            payload["response_format"] = "mp3"

            def _post(p: Dict[str, Any]) -> requests.Response:
                return requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p, stream=True, timeout=120)

            r = _post(payload)
            if r.status_code >= 400 and "speed" in payload:
                try:
                    r.close()
                except Exception:
                    pass
                p2 = dict(payload)
                p2.pop("speed", None)
                r = _post(p2)
            if r.status_code >= 400 and "instructions" in payload:
                try:
                    r.close()
                except Exception:
                    pass
                p3 = dict(payload)
                p3.pop("instructions", None)
                p3.pop("speed", None)
                r = _post(p3)

            rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
            log.info("[chat-tts.stream] voice=%s fmt=%s -> %s rid=%s", voice, "mp3", r.status_code, rid)

            if r.status_code >= 400:
                raise RuntimeError(f"openai_tts_error {r.status_code}: {_short(r.text, 300)}")

            for chunk in r.iter_content(chunk_size=TTS_STREAM_CHUNK_SIZE_DEFAULT):
                if chunk:
                    live.push(chunk)

            try:
                r.close()
            except Exception:
                pass

        def tts_consumer():
            try:
                while True:
                    seg = seg_q.get()
                    if seg is None:
                        break
                    try:
                        _tts_stream_segment(seg)
                    except Exception as e:
                        log.warning("[chat-tts] segment stream failed: %s", e)
                        continue
            finally:
                try:
                    live.close()
                except Exception:
                    pass
        if tts_enabled and seg_q is not None:
            threading.Thread(target=tts_consumer, daemon=True).start()
        # Meta event
        job.push_event({
            "type": "solara.meta",
            "chat_id": job.chat_id,
            "tts_id": (job.tts_id if tts_enabled else None),
            "tts_url": (f"/tts/live/{job.tts_id}.mp3" if tts_enabled else None),
            "tts_stream": bool(tts_enabled),
            "model": model,
            "provider": provider_norm,
            "allow_web": bool(allow_web),
            "route_reason": (route_reason or ""),
        })

        full_parts: List[str] = []
        continuations = 0
        prev_rid: Optional[str] = None

        CHUNK_MARKER = "[[SOLARA_CONTINUE]]"
        tail_buf = ""

        try:
            current_messages = messages
            current_attachments = attachments

            while True:
                need_continue = False
                marker_seen = False
                last_rid: Optional[str] = None
                incomplete_reason: Optional[str] = None
                finish_reason = ""

                # Choose provider stream
                if provider_norm == "deepseek":
                    ds_msgs = list(current_messages)

                    # DeepSeek doesn't support OpenAI `instructions` param; inject as system message.
                    if instructions:
                        insert_at = 1 if (ds_msgs and (ds_msgs[0].get("role") in ("system", "developer"))) else 0
                        ds_msgs.insert(insert_at, {"role": "system", "content": instructions})

                    # Fold transcripts / file pointers from attachments into text messages
                    if current_attachments:
                        trs: List[str] = []
                        file_urls: List[str] = []
                        for a in (current_attachments or []):
                            if not isinstance(a, dict):
                                continue
                            at = str(a.get("type") or "").strip().lower()
                            if at == "audio":
                                tr = str(a.get("transcript") or "").strip()
                                if tr:
                                    trs.append(tr)
                            if at == "file":
                                u = str(a.get("url") or "").strip()
                                if u:
                                    file_urls.append(u)
                        if trs:
                            ds_msgs.append({"role": "user", "content": "[voice transcript]\n" + "\n".join(trs)})
                        if file_urls:
                            ds_msgs.append({"role": "user", "content": "[file]\n" + "\n".join(file_urls)})

                    stream_iter = _stream_deepseek_events(
                        model=model,
                        messages=ds_msgs,
                        max_tokens=min(int(max_output_tokens or 8192), 16384),
                    )
                else:
                    stream_iter = _stream_openai_events(
                        model=model,
                        messages=current_messages,
                        attachments=current_attachments,
                        max_output_tokens=max_output_tokens,
                        truncation="auto",
                        previous_response_id=prev_rid,
                        instructions=instructions,
                        enable_web_search=bool(allow_web),
                    )

                for ev in stream_iter:
                    typ = ev.get("type") if isinstance(ev, dict) else None

                    # DeepSeek internal done marker
                    if typ == "solara._provider_done":
                        finish_reason = str(ev.get("finish_reason") or "") if isinstance(ev, dict) else ""
                        continue

                    # Capture response id from lifecycle events (OpenAI)
                    if typ in ("response.created", "response.in_progress", "response.completed", "response.incomplete", "response.failed"):
                        resp_obj = ev.get("response") if isinstance(ev, dict) else None
                        if isinstance(resp_obj, dict):
                            rid = resp_obj.get("id")
                            if rid:
                                last_rid = rid

                    # Capture web_search sources from output items (OpenAI)
                    if typ in ("response.output_item.added", "response.output_item.done"):
                        item = ev.get("item") if isinstance(ev, dict) else None
                        if isinstance(item, dict) and item.get("type") == "web_search_call":
                            action = item.get("action") if isinstance(item.get("action"), dict) else {}
                            srcs = action.get("sources")
                            if isinstance(srcs, list):
                                _push_sources(srcs)

                    if typ == "response.completed":
                        resp_obj2 = ev.get("response") if isinstance(ev, dict) else None
                        if isinstance(resp_obj2, dict):
                            srcs2 = resp_obj2.get("sources")
                            if isinstance(srcs2, list):
                                _push_sources(srcs2)

                    if typ == "error":
                        err_obj = ev.get("error") if isinstance(ev, dict) else None
                        msg = ""
                        if isinstance(err_obj, dict):
                            msg = str(err_obj.get("message") or "")
                        if not msg:
                            msg = str(ev.get("message") or "unknown_error")
                        raise RuntimeError(msg)

                    if typ == "response.output_text.delta":
                        delta = str(ev.get("delta") or "")
                        if not delta:
                            continue

                        full_parts.append(delta)

                        if tts_enabled and segmenter is not None and seg_q is not None:
                            for seg in segmenter.feed(delta):
                                seg_q.put(seg)

                        if instructions:
                            tail_buf = (tail_buf + delta)[-256:]
                            if CHUNK_MARKER in tail_buf:
                                marker_seen = True

                        job.push_event(ev)
                        continue

                    if typ == "response.incomplete":
                        resp_obj = ev.get("response") if isinstance(ev, dict) else None
                        inc = None
                        if isinstance(resp_obj, dict):
                            inc = resp_obj.get("incomplete_details")
                        if isinstance(inc, dict):
                            incomplete_reason = str(inc.get("reason") or "")
                        else:
                            incomplete_reason = ""
                        need_continue = True
                        continue

                    if typ in ("response.output_text.done", "response.completed"):
                        continue

                    if isinstance(ev, dict) and typ:
                        job.push_event(ev)

                # Decide whether to continue:
                if marker_seen:
                    need_continue = True

                if provider_norm == "deepseek":
                    fr = (finish_reason or "").lower()
                    if fr in ("length", "max_tokens", "max_output_tokens"):
                        need_continue = True
                        if not incomplete_reason:
                            incomplete_reason = fr

                if not need_continue:
                    break

                if continuations >= max_continuations:
                    break

                if provider_norm != "deepseek":
                    if not last_rid:
                        break

                continuations += 1
                if provider_norm != "deepseek":
                    prev_rid = last_rid

                job.push_event({
                    "type": "solara.continuation",
                    "index": continuations,
                    "reason": (incomplete_reason or ("marker" if marker_seen else "unknown")),
                })

                if provider_norm == "deepseek":
                    tail_ctx = ("".join(full_parts))[-1600:]
                    cont_prompt = (
                        "Continue exactly from where you left off. Output ONLY the remaining code/text. "
                        "Do NOT repeat earlier parts. If you were inside a ``` code block, continue inside the same block. "
                        "Make sure the final output is valid and close any open code fences at the end.\n\n"
                        "Here is the END of what you already produced (do NOT repeat it):\n-----\n"
                        f"{tail_ctx}\n-----\n\nContinue from immediately AFTER the above end."
                    )
                else:
                    cont_prompt = (
                        "Continue exactly from where you left off. "
                        "Output ONLY the remaining code/text. Do NOT repeat earlier parts. "
                        "If you were inside a ``` code block, continue inside the same block. "
                        "Make sure the final output is valid and close any open code fences at the end."
                    )

                if instructions:
                    cont_prompt += f" If there is still more after this part, end with {CHUNK_MARKER}."

                current_messages = [{"role": "user", "content": cont_prompt}]
                current_attachments = []

            # Flush remaining segments
            if tts_enabled and segmenter is not None and seg_q is not None:
                for seg in segmenter.flush():
                    seg_q.put(seg)

                seg_q.put(None)

            job.full_text = ("".join(full_parts)).strip()

            # Persist assistant reply to conversation history (best-effort)
            if conversation_id and job.full_text:
                try:
                    conv_add_message(user_key, conversation_id, "assistant", job.full_text)
                except Exception:
                    pass

            # Ingest into vector memory
            if memory_enabled:
                try:
                    if job.full_text and _should_memory_add(job.full_text):
                        memory_add(user_key, f"助手：{job.full_text.strip()}"[:1200])
                except Exception:
                    pass

            if job.full_text:
                job.push_event({
                    "type": "response.output_text.done",
                    "text": job.full_text,
                    "output_index": 0,
                    "content_index": 0,
                })

            job.push_event({"type": "response.completed"})

        except Exception as e:
            job.error = str(e)
            job.push_event({"type": "error", "error": {"message": str(e)}})
            try:
                seg_q.put(None)
            except Exception:
                pass
            try:
                live.close()
            except Exception:
                pass
        finally:
            job.close_events()

    threading.Thread(target=run, daemon=True).start()


def _sse_data(data_obj: Any) -> str:
    """OpenAI-style SSE frame (no `event:` line)."""
    return f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n"

def _sse_pack(event: str, data_obj: Dict[str, Any]) -> str:
    """Legacy SSE pack (kept for backwards compatibility)."""
    return f"event: {event}\ndata: {json.dumps(data_obj, ensure_ascii=False)}\n\n"


# ================================
# FastAPI app + routes
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[BOOT] start | sora=%s(%ss,%s) | realtime=%s | concurrency=%s | chat=%s | tts=%s",
             SORA_MODEL_DEFAULT, SORA_SECONDS_DEFAULT, SORA_SIZE_DEFAULT,
             REALTIME_MODEL_DEFAULT, SORA_CONCURRENCY,
             CHAT_MODEL_DEFAULT, TTS_MODEL_DEFAULT)
    yield
    log.info("[BOOT] stop")

app = FastAPI(title="ChatAGI-阿杜 Backend", lifespan=lifespan)

# ✅ business routers
# NOTE: order matters — we register our /chat override BEFORE including home_chat_router,
# so even if routers.home_chat also defines /chat, the legacy iOS client will hit the stable handler here.
app.include_router(billing_router)
app.include_router(auth_router)
app.include_router(media_upload_router)
app.include_router(home_automation_router)

# -----------------------------
# ✅ Legacy /chat (single HTTP response, best for current iOS)
#    Goal: NEVER truncate code. If too long -> auto-continue + optional zip download.
# -----------------------------
@app.post("/chat")
async def chat_legacy(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    requested_model = (body.get("model") or "").strip()
    if not CHAT_ALLOW_CLIENT_MODEL:
        requested_model = ""
    allow_web = _extract_allow_web(request, body)
    msgs = body.get("messages") or []
    atts = body.get("attachments") or []

    if not isinstance(msgs, list) or not msgs:
        return JSONResponse({"ok": False, "error": "missing messages"}, status_code=400)

    messages: List[Dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        if content.strip() == "":
            continue
        messages.append({"role": role, "content": content})
    if not messages:
        return JSONResponse({"ok": False, "error": "empty messages"}, status_code=400)

    # ✅ user_key + conversation_id (ChatGPT-style history)
    user_key = _derive_user_key(request, body)
    conversation_id = (body.get("conversation_id") or body.get("conversationId") or "").strip()
    if not conversation_id:
        # use first user text as title
        title_seed = ""
        for mm in reversed(messages):
            if mm.get("role") == "user":
                title_seed = (mm.get("content") or "").strip()
                break
        conversation_id = conv_create(user_key, title_seed[:64] if title_seed else "新对话")

    # ✅ last user text (for memory retrieval + title touch)
    last_user_text = ""
    for mm in reversed(messages):
        if mm.get("role") == "user":
            last_user_text = (mm.get("content") or "").strip()
            break

    # Ensure markdown/code fences for client-side highlight.
    messages = _prepend_style_prompt(messages)

    # ✅ Inject vector memory context (official RAG-style: retrieve -> system)
    if MEMORY_ENABLED_DEFAULT and last_user_text:
        mem_ctx = memory_build_context(user_key, last_user_text, k=MEMORY_TOP_K_DEFAULT, min_score=MEMORY_MIN_SCORE_DEFAULT)
        if mem_ctx:
            # keep style prompt at first
            insert_i = 1 if (messages and messages[0].get("role") == "system") else 0
            messages.insert(insert_i, {"role": "system", "content": mem_ctx})

    # ✅ Persist user message to conversation history (best-effort)
    if last_user_text:
        conv_add_message(user_key, conversation_id, "user", last_user_text)

    attachments: List[Dict[str, Any]] = []
    if isinstance(atts, list) and atts:
        for a in atts:
            if isinstance(a, dict):
                attachments.append(a)

    user_transcript = ""
    if attachments:
        user_transcript = await asyncio.to_thread(_transcribe_audio_attachments_inplace, attachments)

# ✅ Multimodal memory: caption images and store (best-effort)
if MEMORY_ENABLED_DEFAULT and attachments:
    try:
        n_img = 0
        for a in attachments:
            if not isinstance(a, dict):
                continue
            if (a.get("type") or "").strip().lower() != "image":
                continue
            n_img += 1
            if n_img > VISION_CAPTION_MAX_IMAGES_PER_TURN:
                break
            try:
                raw = _read_image_bytes_from_attachment(a)
                b64 = base64.b64encode(raw).decode("utf-8")
                mime2 = ((a.get("mime") or "") or "image/jpeg").strip()
                if not mime2.startswith("image/"):
                    mime2 = "image/jpeg"
                data_url = f"data:{mime2};base64,{b64}"
                cap = _openai_caption_image_data_url(data_url)
                if cap:
                    memory_media_add_image(user_key, source=(a.get("url") or ""), caption=cap, image_bytes=raw)
                    memory_add(user_key, f"用户图片：{cap}")
            except Exception:
                continue
    except Exception:
        pass

    # Long-output safeguards (optional overrides)
    try:
        max_output_tokens = int(body.get("max_output_tokens") or CHAT_MAX_OUTPUT_TOKENS_DEFAULT)
    except Exception:
        max_output_tokens = CHAT_MAX_OUTPUT_TOKENS_DEFAULT
    max_output_tokens = max(256, min(max_output_tokens, 60000))

    try:
        max_continuations = int(body.get("max_continuations") or CHAT_MAX_CONTINUATIONS_DEFAULT)
    except Exception:
        max_continuations = CHAT_MAX_CONTINUATIONS_DEFAULT
    max_continuations = max(0, min(max_continuations, 20))

    # Optional: ask model to chunk long code into parts (then we will continue until finished)
    instructions = _build_long_output_instructions(body)

    # ✅ Smart routing: DeepSeek for default text, OpenAI for web-search / attachments
    provider, route_reason = _route_provider(allow_web=allow_web, attachments=attachments)
    provider, route_reason = _ensure_provider_available(provider, route_reason)
    routed_model = _select_routed_model(provider, requested_model)
    log.info("[chat-route] provider=%s model=%s allow_web=%s reason=%s", provider, routed_model, bool(allow_web), route_reason)

    try:
        if provider == "deepseek":
            full_text, status, rid, inc, conts = _deepseek_complete_full_text(
                model=routed_model,
                messages=messages,
                attachments=attachments,
                max_output_tokens=max_output_tokens,
                max_continuations=max_continuations,
                instructions=instructions,
            )
        else:
            full_text, status, rid, inc, conts = _chat_complete_full_text(
                model=routed_model,
                messages=messages,
                attachments=attachments,
                max_output_tokens=max_output_tokens,
                max_continuations=max_continuations,
                instructions=instructions,
                enable_web_search=bool(allow_web),
            )

        # Optional: server-side parts + zip (commercial UX)
        try:
            chunk_max = int(body.get("chunk_max_chars") or "12000")
        except Exception:
            chunk_max = 12000

        parts: List[str] = []
        # Auto-split if requested OR extremely long
        want_parts = bool(body.get("chunk_code") or body.get("chunking") or body.get("chunk_output") or (len(full_text) > 24000))
        if want_parts:
            parts = _split_into_parts(full_text, chunk_max)

        download_url = None
        if bool(body.get("download") or want_parts or (len(full_text) > 24000)):
            meta = {
                "model": routed_model,
                "response_id": rid,
                "status": status,
                "incomplete_details": inc,
                "continuations": conts,
                "created": int(time.time()),
                "chars": len(full_text),
                "parts": len(parts),
                "max_output_tokens": max_output_tokens,
            }
            dl_id = _create_download_zip(full_text=full_text, parts=parts, meta=meta, req_body=body)
            download_url = f"/chat/download/{dl_id}.zip"

        return {
            "ok": True,
            "provider": provider,
            "model": routed_model,
            "allow_web": bool(allow_web),
            "text": full_text,
            "status": status,
            "response_id": rid,
            "user_transcript": user_transcript,
            "incomplete_details": inc,
            "continuations": conts,
            "parts": parts,               # optional, client can ignore
            "download_url": download_url, # optional
        }

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ✅ Finally include home_chat_router (other endpoints like /session, /rt/intent, etc.)
app.include_router(home_chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def audit(req: Request, call_next):
    t0 = time.time()
    resp = await call_next(req)
    if req.url.path.startswith("/chat/events/") or req.url.path in ("/video", "/video/remix", "/rt/intent", "/session", "/solara/photo", "/chat/prepare", "/chat"):
        ip = req.client.host if req.client else "-"
        log.info("[AUDIT] ip=%s %s %s -> %s in %dms",
                 ip, req.method, req.url.path, resp.status_code, int((time.time()-t0)*1000))
    return resp

# -----------------------------
# ✅ Chat prepare (server-side TTS, streaming text via SSE)
# -----------------------------
# -----------------------------
# ✅ Chat prepare (server-side TTS + streaming text via SSE)
# -----------------------------
@app.post("/chat/prepare")
async def chat_prepare(request: Request):
    """
    POST /chat/prepare
    JSON:
      {
        "model": "gpt-5",
        "messages": [{"role":"user","content":"..."}],
        "conversation_id": "...",        # optional
        "attachments": [...],            # optional (image/audio)
        "tts_voice": "alloy|realtime",   # optional
        "max_output_tokens": 4096,       # optional
        "max_continuations": 2           # optional
      }

    Returns:
      {
        "ok": true,
        "chat_id": "...",
        "conversation_id": "...",
        "user_key": "...",
        "events_url": "/chat/events/<chat_id>",
        "tts_url": "/tts/live/<tts_id>.mp3",
        "user_transcript": "...",
        "expires_in": 600
      }

    Notes:
      - We ALWAYS stream text via SSE (events_url)
      - TTS is generated server-side in parallel; client can choose to play later (manual).
      - We persist conversation + messages here so the drawer history list can show immediately.
      - We also inject long-term memory context (vector DB) when enabled.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    user_key = _derive_user_key(request, body)

    # Conversation id (client may pass it as snake_case or camelCase)
    conversation_id = (body.get("conversation_id") or body.get("conversationId") or "").strip()

    requested_model = (body.get("model") or "").strip()
    allow_web = _extract_allow_web(request, body)
    msgs = body.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        return JSONResponse({"ok": False, "error": "missing messages"}, status_code=400)

    # Normalize messages
    messages: List[Dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user").strip() or "user"
        content = str(m.get("content") or "")
        if content.strip() == "":
            continue
        messages.append({"role": role, "content": content})

    if not messages:
        return JSONResponse({"ok": False, "error": "empty messages"}, status_code=400)

    # Attachments (image/audio)
    atts = body.get("attachments") or []
    attachments: List[Dict[str, Any]] = []
    if isinstance(atts, list) and atts:
        for a in atts:
            if isinstance(a, dict):
                attachments.append(a)

    user_transcript = ""
    if attachments:
        # Do transcription in a background thread (avoid blocking the event loop)
        user_transcript = await asyncio.to_thread(_transcribe_audio_attachments_inplace, attachments)

# ✅ Multimodal memory: caption images and store to vector DB (best-effort, non-blocking)
if MEMORY_ENABLED_DEFAULT and attachments:
    try:
        n_img = 0
        for a in attachments:
            if not isinstance(a, dict):
                continue
            if (a.get("type") or "").strip().lower() != "image":
                continue
            n_img += 1
            if n_img > VISION_CAPTION_MAX_IMAGES_PER_TURN:
                break
            try:
                raw = _read_image_bytes_from_attachment(a)
                b64 = base64.b64encode(raw).decode("utf-8")
                mime2 = ((a.get("mime") or "") or "image/jpeg").strip()
                if not mime2.startswith("image/"):
                    mime2 = "image/jpeg"
                data_url = f"data:{mime2};base64,{b64}"
                cap = _openai_caption_image_data_url(data_url)
                if cap:
                    memory_media_add_image(user_key, source=(a.get("url") or ""), caption=cap, image_bytes=raw)
                    # Also store a text memory hook for better recall with text-only embeddings
                    memory_add(user_key, f"用户图片：{cap}")
            except Exception:
                continue
    except Exception:
        pass

    # Last user text (used for title, memory query, persistence)
    last_user_text = ""
    for mm in reversed(messages):
        if (mm.get("role") or "").strip() == "user":
            last_user_text = (mm.get("content") or "").strip()
            break
    if (not last_user_text) and user_transcript:
        last_user_text = user_transcript.strip()

    # Create/touch conversation record so drawer list can show
    try:
        title_seed = (last_user_text.splitlines()[0].strip() if last_user_text else "").strip()
        if len(title_seed) > 64:
            title_seed = title_seed[:64].rstrip() + "…"

        if conversation_id:
            # Ensure it exists (in case client passes a stale id)
            with _conv_conn() as con:
                cur = con.execute("SELECT 1 FROM conversations WHERE id=? AND user_key=? LIMIT 1", (conversation_id, user_key))
                ok = cur.fetchone() is not None
            if not ok:
                conversation_id = conv_create(user_key, title_seed or "新对话")
            else:
                conv_touch(user_key, conversation_id, last_user_text)
        else:
            conversation_id = conv_create(user_key, title_seed or "新对话")
    except Exception:
        # Never fail the chat because of history db
        conversation_id = conversation_id or uuid.uuid4().hex

    # Persist ONLY the newest user message to DB (avoid duplicating full history)
    if last_user_text:
        try:
            conv_add_message(user_key, conversation_id, "user", last_user_text)
        except Exception:
            pass

        # Add to long-term memory (vector DB)
        if MEMORY_ENABLED_DEFAULT:
            try:
                if _should_memory_add(last_user_text):
                    memory_add(user_key, f"用户：{last_user_text}"[:1200])
            except Exception:
                pass

    # Prepend style prompt (markdown/code fences) for better rendering
    messages = _prepend_style_prompt(messages)

    # Inject memory context for better recall
    if MEMORY_ENABLED_DEFAULT and last_user_text:
        try:
            mem_ctx = memory_build_context(user_key, last_user_text, k=MEMORY_TOP_K_DEFAULT, min_score=MEMORY_MIN_SCORE_DEFAULT)
        except Exception:
            mem_ctx = ""
        if mem_ctx:
            insert_at = 1 if messages and (messages[0].get("role") in ("system", "developer")) else 0
            messages.insert(insert_at, {"role": "system", "content": mem_ctx})

    tts_voice = (body.get("tts_voice") or body.get("voice") or TTS_VOICE_DEFAULT).strip() or TTS_VOICE_DEFAULT

    # Long-output safeguards (optional overrides)
    try:
        max_output_tokens = int(body.get("max_output_tokens") or CHAT_MAX_OUTPUT_TOKENS_DEFAULT)
    except Exception:
        max_output_tokens = CHAT_MAX_OUTPUT_TOKENS_DEFAULT
    max_output_tokens = max(256, min(max_output_tokens, 60000))

    try:
        max_continuations = int(body.get("max_continuations") or CHAT_MAX_CONTINUATIONS_DEFAULT)
    except Exception:
        max_continuations = CHAT_MAX_CONTINUATIONS_DEFAULT
    max_continuations = max(0, min(max_continuations, 10))

    instructions = _build_long_output_instructions(body)

    # ✅ Smart routing: DeepSeek for default text, OpenAI for web-search / attachments
    provider, route_reason = _route_provider(allow_web=allow_web, attachments=attachments)
    provider, route_reason = _ensure_provider_available(provider, route_reason)
    routed_model = _select_routed_model(provider, requested_model)
    log.info("[chat-route] provider=%s model=%s allow_web=%s reason=%s", provider, routed_model, bool(allow_web), route_reason)

    enable_tts_streaming = bool(body.get("tts_stream", False)) and CHAT_ENABLE_TTS_STREAMING
    job = _create_chat_job(enable_tts_streaming=enable_tts_streaming)
    _spawn_chat_worker(
        job=job,
        model=routed_model,
        messages=messages,
        attachments=attachments,
        tts_voice=tts_voice,
        provider=provider,
        allow_web=allow_web,
        route_reason=route_reason,
        max_output_tokens=max_output_tokens,
        max_continuations=max_continuations,
        instructions=instructions,
        conversation_id=conversation_id,
        user_key=user_key,
        memory_enabled=MEMORY_ENABLED_DEFAULT,
    )

    return {
        "ok": True,
        "chat_id": job.chat_id,
        "conversation_id": conversation_id,
        "user_key": user_key,
        "provider": provider,
        "model": routed_model,
        "allow_web": bool(allow_web),
        "events_url": f"/chat/events/{job.chat_id}",
        "tts_url": (f"/tts/live/{job.tts_id}.mp3" if getattr(job, "tts_stream_enabled", False) else None),
        "tts_stream": bool(getattr(job, "tts_stream_enabled", False)),
        "user_transcript": user_transcript,
        "expires_in": 600,
    }

@app.get("/chat/events/{chat_id}")
async def chat_events(chat_id: str, request: Request):
    """
    SSE stream (OpenAI-style):
      data: { "type": "solara.meta", "tts_url": "...", ... }
      data: { "type": "response.output_text.delta", "delta": "..." }
      data: { "type": "response.output_text.done", "text": "..." }
      data: { "type": "response.completed" }
      data: { "type": "error", "error": { "message": "..." } }
    """
    job = _get_chat_job(chat_id)
    if not job:
        return JSONResponse({"ok": False, "error": "chat not found"}, status_code=404)

    async def gen():
        yield "retry: 1500\n\n"
        last_ping = time.time()

        while True:
            try:
                if await request.is_disconnected():
                    break
            except Exception:
                pass

            now = time.time()
            if now - last_ping >= 10:
                last_ping = now
                yield _sse_data({"type": "response.output_text.delta", "delta": ""})

            try:
                item = job.events.get(timeout=0.5)
            except Exception:
                item = None

            if item is None:
                if job.done_evt.is_set():
                    break
                continue

            yield _sse_data(item)

        
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/chat/result/{chat_id}")
def chat_result(chat_id: str):
    job = _get_chat_job(chat_id)
    if not job:
        return JSONResponse({"ok": False, "error": "chat not found"}, status_code=404)
    return {
        "ok": True,
        "chat_id": job.chat_id,
        "text": job.full_text,
        "error": job.error,
        "tts_url": f"/tts/live/{job.tts_id}.mp3",
        "done": bool(job.done_evt.is_set()),
    }

# -----------------------------
# ✅ TTS prepare + live stream
# -----------------------------

# -----------------------------
# ✅ Conversations APIs (ChatGPT-style left list)
# -----------------------------

@app.get("/conversations")
def conversations_list(request: Request, limit: int = 50, offset: int = 0):
    user_key = _derive_user_key(request, {})
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    with _conv_conn() as con:
        cur = con.execute(
            "SELECT id,title,created_at,updated_at,last_preview FROM conversations WHERE user_key=? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (user_key, limit, offset),
        )
        rows = cur.fetchall()
    out = []
    for cid,title,created_at,updated_at,last_preview in rows:
        out.append({
            "id": cid,
            "title": title,
            "created_at": created_at,
            "updated_at": updated_at,
            "last_preview": last_preview or ""
        })
    return {"ok": True, "user_key": user_key, "conversations": out}

@app.post("/conversations")
async def conversations_create(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    user_key = _derive_user_key(request, body)
    title = (body.get("title") or "").strip()
    cid = conv_create(user_key, title)
    return {"ok": True, "user_key": user_key, "id": cid}

@app.get("/conversations/{conversation_id}/messages")
def conversations_messages(conversation_id: str, request: Request, limit: int = 200):
    user_key = _derive_user_key(request, {})
    limit = max(1, min(int(limit), 500))
    with _conv_conn() as con:
        cur = con.execute(
            "SELECT id,role,content,created_at FROM messages WHERE conversation_id=? AND user_key=? ORDER BY created_at ASC LIMIT ?",
            (conversation_id, user_key, limit),
        )
        rows = cur.fetchall()
    msgs = [{"id": mid, "role": role, "content": content, "created_at": created_at} for (mid,role,content,created_at) in rows]
    return {"ok": True, "user_key": user_key, "conversation_id": conversation_id, "messages": msgs}

@app.post("/conversations/{conversation_id}/rename")
async def conversations_rename(conversation_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    user_key = _derive_user_key(request, body)
    title = (body.get("title") or "").strip()[:64] or "新对话"
    with _conv_conn() as con:
        con.execute("UPDATE conversations SET title=? WHERE id=? AND user_key=?", (title, conversation_id, user_key))
    return {"ok": True, "user_key": user_key, "conversation_id": conversation_id, "title": title}

@app.delete("/conversations/{conversation_id}")
def conversations_delete(conversation_id: str, request: Request):
    user_key = _derive_user_key(request, {})
    with _conv_conn() as con:
        con.execute("DELETE FROM messages WHERE conversation_id=? AND user_key=?", (conversation_id, user_key))
        cur = con.execute("DELETE FROM conversations WHERE id=? AND user_key=?", (conversation_id, user_key))
    return {"ok": True, "deleted": int(cur.rowcount or 0)}

# -----------------------------
# ✅ Vector memory APIs
# -----------------------------

@app.get("/memory/list")
def memory_list_api(request: Request, limit: int = 50, offset: int = 0):
    user_key = _derive_user_key(request, {})
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    with _mem_conn() as con:
        cur = con.execute(
            "SELECT id,text,created_at,last_used_at FROM memory_items WHERE user_key=? ORDER BY last_used_at DESC LIMIT ? OFFSET ?",
            (user_key, limit, offset),
        )
        rows = cur.fetchall()
    items = []
    for mid,text,created_at,last_used_at in rows:
        items.append({"id": mid, "text": text, "created_at": created_at, "last_used_at": last_used_at})
    return {"ok": True, "user_key": user_key, "items": items}

@app.post("/memory/add")
async def memory_add_api(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    user_key = _derive_user_key(request, body)
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "missing text"}, status_code=400)
    memory_add(user_key, text)
    return {"ok": True, "user_key": user_key}

@app.get("/memory/search")
def memory_search_api(request: Request, q: str, k: int = MEMORY_TOP_K_DEFAULT):
    user_key = _derive_user_key(request, {})
    hits = memory_search(user_key, q, k=max(1,int(k)), min_score=MEMORY_MIN_SCORE_DEFAULT)
    return {"ok": True, "user_key": user_key, "matches": hits}


@app.get("/memory/media/search")
def memory_media_search_api(request: Request, q: str, k: int = 4):
    user_key = _derive_user_key(request, {})
    hits = memory_media_search(user_key, q, k=max(1, int(k)), min_score=MEMORY_MIN_SCORE_DEFAULT)
    return {"ok": True, "user_key": user_key, "matches": hits}

@app.get("/memory/media/list")
def memory_media_list_api(request: Request, limit: int = 50, offset: int = 0):
    user_key = _derive_user_key(request, {})
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    with _mem_conn() as con:
        cur = con.execute(
            "SELECT id,media_type,source,caption,created_at,last_used_at FROM memory_media WHERE user_key=? ORDER BY last_used_at DESC LIMIT ? OFFSET ?",
            (user_key, limit, offset),
        )
        rows = cur.fetchall()
    items = []
    for mid,mtype,src,cap,created_at,last_used_at in rows:
        items.append({"id": mid, "media_type": mtype, "source": src, "caption": cap, "created_at": created_at, "last_used_at": last_used_at})
    return {"ok": True, "user_key": user_key, "items": items}


@app.post("/memory/clear")
def memory_clear_api(request: Request):
    user_key = _derive_user_key(request, {})
    with _mem_conn() as con:
        cur = con.execute("DELETE FROM memory_items WHERE user_key=?", (user_key,))
    return {"ok": True, "user_key": user_key, "deleted": int(cur.rowcount or 0)}
@app.post("/tts_prepare")
async def tts_prepare(request: Request):
    """
    POST /tts_prepare
    JSON:
      { "text":"...", "voice":"realtime", "format":"mp3" }

    返回:
      { "ok": true, "tts_id": "...", "url": "/tts/live/<tts_id>.mp3", "expires_in": <sec> }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    text = (body.get("text") or body.get("input") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "missing text"}, status_code=400)

    voice = (body.get("voice") or "realtime").strip()
    fmt = (body.get("format") or body.get("response_format") or "mp3")
    instructions = (body.get("instructions") or "").strip()
    speed = body.get("speed")
    fmt = str(fmt).strip().lower()

    tts_id = _create_live_tts_session()
    threading.Thread(
        target=_tts_stream_fulltext_to_live,
        args=(tts_id, text, voice, fmt, instructions, speed),
        daemon=True
    ).start()

    return {"ok": True, "tts_id": tts_id, "url": f"/tts/live/{tts_id}.mp3", "expires_in": TTS_LIVE_TTL_SEC}



@app.websocket("/tts/live/feed")
async def tts_live_feed(ws: WebSocket):
    """Front-end sends realtime delta; backend returns continuous MP3 via /tts/live/{id}.mp3"""
    await ws.accept()

    tts_id = _create_live_tts_session()
    worker = LiveTTSFeedWorker(tts_id)
    worker.start()

    # ready: tell client where to GET the mp3 stream
    try:
        await ws.send_json(
            {
                "ok": True,
                "type": "ready",
                "tts_id": tts_id,
                "tts_url": f"/tts/live/{tts_id}.mp3",
                "speed": float(TTS_SPEED_DEFAULT),
            }
        )
    except Exception:
        worker.close()
        return

    try:
        while True:
            msg = await ws.receive_text()
            try:
                obj = json.loads(msg)
            except Exception:
                continue

            typ = (obj.get("type") or "").strip().lower()
            if typ == "start":
                worker.update_start(
                    voice=obj.get("voice"),
                    fmt=obj.get("format") or obj.get("response_format"),
                    speed=obj.get("speed"),
                    instructions=obj.get("instructions"),
                    min_chars=obj.get("min_chars"),
                    max_chars=obj.get("max_chars"),
                )
                continue

            if typ == "delta":
                worker.push_delta(str(obj.get("text") or ""))
                continue

            if typ == "flush":
                worker.flush()
                continue

            if typ == "end":
                worker.end()
                break

            if typ == "ping":
                try:
                    await ws.send_json({"type": "pong"})
                except Exception:
                    pass
                continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("[TTS.feed] ws error: %s", e)
    finally:
        try:
            worker.close()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

@app.post("/tts_stream.mp3")
async def tts_stream_mp3(request: Request):
    """
    ✅ One-shot streaming TTS endpoint (no /tts_prepare needed)

    POST /tts_stream.mp3
    JSON:
      {
        "text": "....",            # required
        "voice": "alloy|realtime", # optional
        "format": "mp3",           # optional; default mp3
        "instructions": "...",     # optional
        "speed": 1.0               # optional
      }

    Returns:
      Streaming audio bytes (default: audio/mpeg)

    Notes:
      - Designed for true byte-stream playback clients (AudioFileStream/AudioQueue on iOS).
      - Keeps /tts_prepare + /tts/live/{tts_id}.mp3 intact for backward compatibility.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    text = (body.get("text") or body.get("input") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "missing text"}, status_code=400)

    voice = (body.get("voice") or "realtime").strip()
    fmt = (body.get("format") or body.get("response_format") or "mp3")
    instructions = (body.get("instructions") or "").strip()
    speed = body.get("speed")
    fmt = str(fmt).strip().lower() or "mp3"

    media_type_map = {
        "mp3": "audio/mpeg",
        "mpeg": "audio/mpeg",
        "wav": "audio/wav",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "opus": "audio/ogg",
        "ogg": "audio/ogg",
    }
    media_type = media_type_map.get(fmt, "application/octet-stream")

    voice_norm = _normalize_tts_voice(voice)

    def gen() -> Iterator[bytes]:
        t0 = time.time()
        first = True
        yielded_any = False

        clean = _tts_sanitize_text(text)
        segments = _split_tts_segments(clean, TTS_SEGMENT_MAX_CHARS)

        # speed: client override -> else env default
        spd: Optional[float] = None
        try:
            if speed is not None:
                spd = float(speed)
        except Exception:
            spd = None
        if spd is None:
            try:
                spd = float(TTS_SPEED_DEFAULT)
            except Exception:
                spd = None
        if spd is not None:
            spd = max(0.5, min(spd, 1.6))

        inst = (instructions or TTS_INSTRUCTIONS_DEFAULT or "").strip()

        def _post(payload: Dict[str, Any]) -> requests.Response:
            r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=payload, stream=True, timeout=120)
            if r.status_code >= 400 and "speed" in payload:
                try:
                    r.close()
                except Exception:
                    pass
                p2 = dict(payload)
                p2.pop("speed", None)
                r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p2, stream=True, timeout=120)
            if r.status_code >= 400 and "instructions" in payload:
                try:
                    r.close()
                except Exception:
                    pass
                p3 = dict(payload)
                p3.pop("instructions", None)
                p3.pop("speed", None)
                r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p3, stream=True, timeout=120)
            return r

        if not segments:
            return

        for seg in segments:
            seg = (seg or "").strip()
            if not seg:
                continue

            payload: Dict[str, Any] = {"model": TTS_MODEL_DEFAULT, "voice": voice_norm, "input": seg}
            if inst:
                payload["instructions"] = inst
            if spd is not None:
                payload["speed"] = spd
            if fmt and fmt != "mp3":
                payload["response_format"] = fmt

            r = _post(payload)
            rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
            log.info("[TTS.stream] voice=%s fmt=%s len=%s -> %s rid=%s", voice_norm, fmt, len(seg), r.status_code, rid)

            if r.status_code >= 400:
                try:
                    r.close()
                except Exception:
                    pass
                continue

            for chunk in r.iter_content(chunk_size=TTS_STREAM_CHUNK_SIZE_DEFAULT):
                if not chunk:
                    continue

                # MP3 concatenation: strip ID3 on later chunks/segments
                if (not first) and fmt == "mp3":
                    try:
                        chunk = LiveMP3Stream._strip_id3_if_present(chunk)
                    except Exception:
                        pass

                if not chunk:
                    continue

                if not yielded_any:
                    yielded_any = True
                    log.info("[TTS.stream] TTFA=%.3fs voice=%s fmt=%s", time.time() - t0, voice_norm, fmt)

                yield chunk
                first = False

            try:
                r.close()
            except Exception:
                pass

    return StreamingResponse(
        gen(),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Accept-Ranges": "none",
        },
    )


@app.get("/tts/live/{tts_id}.mp3")
async def tts_live(tts_id: str):
    s = _get_live_tts(tts_id)
    if not s:
        return JSONResponse({"ok": False, "error": "tts stream not found"}, status_code=404)

    return StreamingResponse(
        s.generator(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Accept-Ranges": "none",
        },
    )

# ================================
# ✅✅ SORA: Remix endpoint (JSON) — FIXED with final confirm gate
# ================================
@app.post("/video/remix")
async def create_video_remix(request: Request):
    ip = request.client.host if request.client else "unknown"

    try:
        body = await request.json()
    except Exception:
        body = {}

    client_task_id = (body.get("client_task_id") or body.get("clientTaskId") or "").strip()
    viewer = _auth_optional_user(request)
    user_key = (viewer["user_id"] if viewer else _client_id(request))

    base_video_id_raw = (body.get("base_video_id") or body.get("video_id") or "").strip()
    base_video_id = _normalize_video_id(base_video_id_raw)
    # Allow passing our internal asset_id as base id for remix
    base_video_id = _resolve_openai_video_id(user_key, base_video_id) or base_video_id
    if not base_video_id:
        return JSONResponse({"ok": False, "error": "missing base_video_id"}, status_code=400)

    user_confirm_text = (body.get("user_confirm_text") or body.get("confirm_text") or body.get("user_confirm") or "").strip()
    expected_code = (body.get("confirm_code") or body.get("code") or "").strip() or None
    if expected_code is None:
        expected_code = None

    if not _user_has_confirm(user_confirm_text, expected_code=expected_code):
        return JSONResponse(
            {
                "ok": False,
                "error": "need_user_confirm",
                "confirm_phrase": CONFIRM_PHRASE,
                "hint": f"请在最终确认时说/发：{CONFIRM_PHRASE}" + (f" 1234" if expected_code else "")
            },
            status_code=409
        )

    assistant_line = (body.get("assistant_line") or body.get("assistant_rewrite") or body.get("trigger_line") or "").strip()

    instruction = ""
    if assistant_line:
        if not _assistant_line_is_valid(assistant_line, expected_code=expected_code):
            return JSONResponse(
                {
                    "ok": False,
                    "error": "assistant_line_invalid_or_not_confirmed",
                    "required_prefix": TRIGGER_PREFIX,
                    "required_mark": CONFIRM_MARK,
                    "hint": f"助手触发行必须形如：REMIX: <英文> {CONFIRM_MARK}"
                },
                status_code=409
            )
        instruction = _extract_instruction_from_assistant_line(assistant_line)
    else:
        instruction = (body.get("instruction") or body.get("prompt") or "").strip()

    if not instruction:
        return JSONResponse({"ok": False, "error": "missing instruction"}, status_code=400)

    prompt_idem = f"REMIX|{base_video_id}|{instruction}"
    content_key = _video_content_key(user_key=user_key, provider="sora", mode="remix", prompt_idem=prompt_idem)

    # Persistent de-dup: same content => single job/asset
    if VIDEO_DEDUP_ENABLED:
        lock = _vlock_get_active(user_key, content_key)
        if lock:
            old_job_id = (lock.get("provider_job_id") or "").strip()
            old_task_id = (lock.get("client_task_id") or "").strip() or (client_task_id or old_job_id)
            old_asset_id = (lock.get("latest_asset_id") or "").strip()

            # If already saved, return ready directly
            if old_asset_id and (lock.get("status") == "ready"):
                return {
                    "ok": True,
                    "job_id": old_job_id,
                    "client_task_id": old_task_id,
                    "restore_url": "/v1/assets/restore",
                    "status": "ready",
                    "asset_id": old_asset_id,
                    "play_url": f"/v1/video/stream/{old_asset_id}",
                    "remixed_from_video_id": base_video_id,
                }

            old_job = SORA_JOBS.get(old_job_id) if old_job_id else None
            if old_job_id and _job_is_active(old_job):
                return {
                    "ok": True,
                    "job_id": old_job_id,
                    "client_task_id": old_task_id,
                    "restore_url": "/v1/assets/restore",
                    "status_url": f"/video/status/{old_job_id}",
                    "status": old_job.get("status") if old_job else "running",
                    "remixed_from_video_id": base_video_id,
                }

            # Stale lock: let a new request proceed
            try:
                _vlock_delete(user_key, content_key)
            except Exception:
                pass

    _cleanup_recent()
    idem = _idem_key(ip, prompt_idem, "", "")
    rec = RECENT_KEYS.get(idem)
    if rec:
        old_job_id = rec["job_id"]
        old_job = SORA_JOBS.get(old_job_id)
        if _job_is_active(old_job):
            return {
                "ok": True,
                "job_id": old_job_id,
                "client_task_id": ((old_job.get("client_task_id") or "").strip() or old_job_id),
                "restore_url": "/v1/assets/restore",
                "status_url": f"/video/status/{old_job_id}",
                "status": old_job.get("status") if old_job else "running",
                "remixed_from_video_id": base_video_id,
            }
        RECENT_KEYS.pop(idem, None)

    job_id = uuid.uuid4().hex
    SORA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,

        "prompt": instruction or "Make subtle improvements. Keep everything else the same.",
        "prompt_raw": json.dumps(body, ensure_ascii=False),

        "url": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "provider": "sora",
        "user_key": user_key,
        "client_task_id": client_task_id or None,
        "content_key": content_key,

        "ref_path": None,
        "ref_mime": None,

        "mode": "remix",
        "remix_base_video_id": base_video_id,
        "remix_instruction": instruction or "",
        "openai_status": None,

        "confirm_phrase": CONFIRM_PHRASE,
        "user_confirm_text": user_confirm_text,
        "assistant_line": assistant_line or None,
    }
    RECENT_KEYS[idem] = {"job_id": job_id, "ts": time.time()}

    if VIDEO_DEDUP_ENABLED:
        try:
            _vlock_upsert(
                user_id=user_key,
                content_key=content_key,
                status="queued",
                provider="sora",
                provider_job_id=job_id,
                client_task_id=(client_task_id or job_id),
            )
        except Exception:
            pass

    if client_task_id:
        _task_upsert(user_id=user_key, client_task_id=client_task_id, status="queued", provider="sora", provider_job_id=job_id, provider_ref_id=base_video_id)

    _spawn_sora_job(job_id, timeout_sec=1800)

    return {
        "ok": True,
        "job_id": job_id,
        "client_task_id": (client_task_id or job_id),
        "restore_url": "/v1/assets/restore",
        "status_url": f"/video/status/{job_id}",
        "status": "queued",
        "remixed_from_video_id": base_video_id,
    }

# -----------------------------
# SORA: Create endpoint (Form)
# -----------------------------
@app.post("/video")
async def create_video(
    request: Request,
    prompt: str = Form(""),
    client_task_id: str = Form(""),
    image_file: UploadFile = File(None),
    video_file: UploadFile = File(None),
):
    ip = request.client.host if request.client else "unknown"

    viewer = _auth_optional_user(request)
    user_key = (viewer["user_id"] if viewer else _client_id(request))
    client_task_id = (client_task_id or "").strip() or None

    raw_img: Optional[bytes] = None
    img_h = ""
    vid_h = ""
    tmp_video_path: Optional[Path] = None

    if image_file:
        raw_img = await image_file.read()
        img_h = _sha1_bytes(raw_img) if raw_img else ""

    if video_file:
        tmp_video_path = UPLOADS_DIR / f"tmp_upload_{uuid.uuid4().hex}.mp4"
        try:
            vid_h = await _save_upload_to_file_and_sha1(video_file, tmp_video_path)
        except Exception:
            try:
                if tmp_video_path.exists():
                    tmp_video_path.unlink()
            except Exception:
                pass
            raise

    remix = parse_remix_request(prompt or "")
    mode = "create"
    remix_base = ""
    remix_instruction = ""
    prompt_effective = (prompt or "").strip()
    prompt_idem = prompt_effective

    if remix:
        if CONFIRM_PHRASE not in (prompt or ""):
            try:
                if tmp_video_path and tmp_video_path.exists():
                    tmp_video_path.unlink()
            except Exception:
                pass
            return JSONResponse(
                {"ok": False, "error": "need_user_confirm_for_remix", "confirm_phrase": CONFIRM_PHRASE},
                status_code=409
            )

        mode = "remix"
        remix_base = remix.get("base_video_id", "")
        remix_base = _resolve_openai_video_id(user_key, remix_base) or remix_base
        remix_instruction = (remix.get("instruction", "") or "").strip()
        prompt_effective = remix_instruction or "Make subtle improvements. Keep everything else the same."
        prompt_idem = f"REMIX|{remix_base}|{prompt_effective}"

    if not prompt_effective.strip():
        prompt_effective = "Generate a video based on the reference media." if (raw_img or tmp_video_path) else "Generate a video."
        prompt_idem = prompt_effective

    provider = "sora" if mode == "remix" else _select_video_provider("create")

    # Persistent de-dup: same content => single job/asset
    content_key = _video_content_key(user_key=user_key, provider=provider, mode=mode, prompt_idem=prompt_idem, img_h=img_h, vid_h=vid_h)
    if VIDEO_DEDUP_ENABLED:
        lock = _vlock_get_active(user_key, content_key)
        if lock:
            old_job_id = (lock.get("provider_job_id") or "").strip()
            old_task_id = (lock.get("client_task_id") or "").strip() or (client_task_id or old_job_id)
            old_asset_id = (lock.get("latest_asset_id") or "").strip()

            # If already saved, return ready directly
            if old_asset_id and (lock.get("status") == "ready"):
                try:
                    if tmp_video_path and tmp_video_path.exists():
                        tmp_video_path.unlink()
                except Exception:
                    pass
                return {
                    "ok": True,
                    "job_id": old_job_id,
                    "client_task_id": old_task_id,
                    "restore_url": "/v1/assets/restore",
                    "status": "ready",
                    "asset_id": old_asset_id,
                    "play_url": f"/v1/video/stream/{old_asset_id}",
                }

            old_job = SORA_JOBS.get(old_job_id) if old_job_id else None
            if old_job_id and _job_is_active(old_job):
                try:
                    if tmp_video_path and tmp_video_path.exists():
                        tmp_video_path.unlink()
                except Exception:
                    pass
                return {
                    "ok": True,
                    "job_id": old_job_id,
                    "client_task_id": old_task_id,
                    "restore_url": "/v1/assets/restore",
                    "status_url": f"/video/status/{old_job_id}",
                    "status": old_job.get("status") if old_job else "running",
                }

            # Stale lock: let a new request proceed
            try:
                _vlock_delete(user_key, content_key)
            except Exception:
                pass

    _cleanup_recent()
    idem = _idem_key(ip, prompt_idem, img_h, vid_h)
    rec = RECENT_KEYS.get(idem)
    if rec:
        old_job_id = rec["job_id"]
        old_job = SORA_JOBS.get(old_job_id)
        if _job_is_active(old_job):
            try:
                if tmp_video_path and tmp_video_path.exists():
                    tmp_video_path.unlink()
            except Exception:
                pass
            return {
                "ok": True,
                "job_id": old_job_id,
                "client_task_id": ((old_job.get("client_task_id") or "").strip() or old_job_id),
                "restore_url": "/v1/assets/restore",
                "status_url": f"/video/status/{old_job_id}",
                "status": old_job.get("status") if old_job else "running",
            }
        RECENT_KEYS.pop(idem, None)

    job_id = uuid.uuid4().hex
    SORA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": prompt_effective,
        "prompt_raw": (prompt or ""),

        "url": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "provider": provider,
        "user_key": user_key,
        "client_task_id": client_task_id or None,

        "content_key": content_key,

        "ref_path": None,
        "ref_mime": None,

        "mode": mode,
        "remix_base_video_id": remix_base or None,
        "remix_instruction": remix_instruction or None,
        "openai_status": None,
    }
    RECENT_KEYS[idem] = {"job_id": job_id, "ts": time.time()}

    if VIDEO_DEDUP_ENABLED:
        try:
            _vlock_upsert(
                user_id=user_key,
                content_key=content_key,
                status="queued",
                provider=provider,
                provider_job_id=job_id,
                client_task_id=(client_task_id or job_id),
            )
        except Exception:
            pass

    if client_task_id:
        _task_upsert(user_id=user_key, client_task_id=client_task_id, status="queued", provider=provider, provider_job_id=job_id)

    ref_path = None
    ref_mime = None

    try:
        target_wh = _parse_size(SORA_SIZE_DEFAULT)

        if raw_img is not None:
            raw_img_resized = _resize_image_bytes(raw_img, target_wh)
            fn = UPLOADS_DIR / f"{job_id}_img.jpg"
            fn.write_bytes(raw_img_resized)
            ref_path = str(fn)
            ref_mime = "image/jpeg"

        if tmp_video_path is not None and tmp_video_path.exists():
            src_fn = UPLOADS_DIR / f"{job_id}_vid_src.mp4"
            tmp_video_path.replace(src_fn)

            dst_fn = UPLOADS_DIR / f"{job_id}_vid.mp4"
            final_fn = _resize_video_file(str(src_fn), str(dst_fn), target_wh)

            if ref_path is None:
                thumb_fn = UPLOADS_DIR / f"{job_id}_vid_thumb.jpg"
                thumb_path = _video_to_thumb_image(final_fn, str(thumb_fn))
                if thumb_path and Path(thumb_path).exists():
                    ref_path = thumb_path
                    ref_mime = "image/jpeg"

    except Exception as e:
        log.warning("[SORA] ref prepare failed: %s", e)

    SORA_JOBS[job_id]["ref_path"] = ref_path
    SORA_JOBS[job_id]["ref_mime"] = ref_mime

    timeout_sec = 1800 if mode == "remix" else 1200
    _spawn_sora_job(job_id, timeout_sec=timeout_sec)

    return {"ok": True, "job_id": job_id, "client_task_id": (client_task_id or job_id), "restore_url": "/v1/assets/restore", "status_url": f"/video/status/{job_id}", "status": "queued"}

# -----------------------------
# SORA: status / stream / content
# -----------------------------
@app.get("/video/status/{job_id}")
def video_status(job_id: str):
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)

    provider = _normalize_video_provider(job.get("provider") or "") or "sora"
    openai_vid = _normalize_video_id(job.get("video_id") or "") if provider == "sora" else ""
    aid = (job.get("asset_id") or job.get("library_video_id"))

    # Keep legacy keys stable, but add provider-specific fields for debugging.
    seconds = job.get("seconds") or (MINIMAX_DURATION_DEFAULT if provider == "minimax" else SORA_SECONDS_DEFAULT)
    size = job.get("size") or (MINIMAX_RESOLUTION_DEFAULT if provider == "minimax" else SORA_SIZE_DEFAULT)

    return {
        "ok": True,
        "job_id": job_id,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "url": job.get("url"),
        "error": job.get("error"),
        "openai_status": job.get("openai_status"),
        "provider_status": job.get("provider_status"),

        # Legacy fields (Sora)
        "video_id": openai_vid,
        "openai_video_id": openai_vid,

        # Saved asset
        "asset_id": aid,
        "remixed_from_video_id": job.get("remix_base_video_id") if (job.get("mode") == "remix") else None,

        # Provider routing
        "provider": provider,
        "seconds": seconds,
        "size": size,
        "mode": job.get("mode"),
        "remix_base_video_id": job.get("remix_base_video_id"),
        "remix_instruction": job.get("remix_instruction"),

        # MiniMax (optional)
        "minimax_task_id": job.get("minimax_task_id"),
        "minimax_file_id": job.get("minimax_file_id"),
    }

@app.get("/video/stream/{job_id}")
def video_stream(job_id: str, request: Request):
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)

    provider = _normalize_video_provider(job.get("provider") or "") or "sora"

    # If already persisted, stream from local storage (best for playback).
    aid = (job.get("asset_id") or job.get("library_video_id"))
    if aid:
        return RedirectResponse(url=f"/v1/video/stream/{aid}", status_code=307)

    range_header = request.headers.get("range") or request.headers.get("Range")

    if provider == "sora":
        vid = _normalize_video_id(job.get("video_id") or "")
        if not vid:
            return JSONResponse({"ok": False, "error": "video not ready"}, status_code=409)

        r = _fetch_sora_content_response(vid, range_header=range_header)
        if r.status_code in (302, 303):
            loc = r.headers.get("Location")
            if loc:
                return RedirectResponse(loc, status_code=r.status_code)
            return JSONResponse({"ok": False, "error": "redirect without location"}, status_code=502)

        if r.status_code not in (200, 206):
            return JSONResponse({"ok": False, "error": f"upstream {r.status_code}", "detail": r.text[:300]}, status_code=502)

        def it():
            try:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        yield chunk
            finally:
                try:
                    r.close()
                except Exception:
                    pass

        headers = {}
        for k in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
            if k in r.headers:
                headers[k] = r.headers[k]
        return StreamingResponse(it(), status_code=r.status_code, headers=headers)

    if provider == "minimax":
        dl_url = (job.get("minimax_download_url") or job.get("download_url") or "").strip()

        # If we already have file_id, derive download_url; else, try query task.
        if not dl_url:
            fid = str(job.get("minimax_file_id") or "").strip()
            if fid:
                try:
                    dl_url = minimax_get_download_url(fid)
                    job["minimax_download_url"] = dl_url
                except Exception:
                    dl_url = ""

        if not dl_url:
            tid = str(job.get("minimax_task_id") or "").strip()
            if tid:
                try:
                    info = minimax_query_task(tid)
                    fid = str(info.get("file_id") or "").strip()
                    if fid:
                        job["minimax_file_id"] = fid
                        dl_url = minimax_get_download_url(fid)
                        job["minimax_download_url"] = dl_url
                except Exception:
                    dl_url = ""

        if not dl_url:
            return JSONResponse({"ok": False, "error": "video not ready"}, status_code=409)

        headers_up = {}
        if range_header:
            headers_up["Range"] = range_header

        r = requests.get(dl_url, headers=headers_up, stream=True, allow_redirects=False, timeout=MINIMAX_QUERY_TIMEOUT_SEC)

        if r.status_code in (302, 303):
            loc = r.headers.get("Location")
            if loc:
                return RedirectResponse(loc, status_code=r.status_code)
            return JSONResponse({"ok": False, "error": "redirect without location"}, status_code=502)

        if r.status_code not in (200, 206):
            return JSONResponse({"ok": False, "error": f"upstream {r.status_code}", "detail": r.text[:300]}, status_code=502)

        def it():
            try:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        yield chunk
            finally:
                try:
                    r.close()
                except Exception:
                    pass

        headers = {}
        for k in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
            if k in r.headers:
                headers[k] = r.headers[k]
        if "Content-Type" not in headers:
            headers["Content-Type"] = "video/mp4"
        return StreamingResponse(it(), status_code=r.status_code, headers=headers)

    return JSONResponse({"ok": False, "error": f"unsupported provider: {provider}"}, status_code=400)
@app.get("/video/content/{video_id}")
def video_content(video_id: str, range_header: Optional[str] = Header(None, alias="Range")):
    vid = _normalize_video_id(video_id)
    if not vid:
        return JSONResponse({"ok": False, "error": "invalid video_id"}, status_code=400)

    rc = _fetch_sora_content_response(vid, range_header)

    if rc.status_code in (302, 303) and rc.headers.get("Location"):
        loc = rc.headers["Location"]
        try:
            rc.close()
        except Exception:
            pass

        headers2: Dict[str, str] = {}
        if range_header:
            headers2["Range"] = range_header
        r2 = requests.get(loc, headers=headers2, stream=True, timeout=120)
        return _streaming_proxy_response(r2)

    if rc.status_code in (200, 206):
        return _streaming_proxy_response(rc)

    try:
        detail = rc.text[:400]
    except Exception:
        detail = ""
    try:
        rc.close()
    except Exception:
        pass
    return JSONResponse({"ok": False, "error": f"content fetch failed: {rc.status_code}", "detail": detail}, status_code=502)


# =========================
# Assets API (restore / tags) — minimal anti-loss layer
# =========================

@app.post("/v1/assets/restore")
async def v1_assets_restore(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    client_task_id = (body.get("client_task_id") or body.get("clientTaskId") or "").strip()
    asset_hint = (body.get("asset_hint") or body.get("assetHint") or body.get("hint") or "").strip()

    viewer = _auth_optional_user(request)
    user_key = (viewer["user_id"] if viewer else _client_id(request))

    # 1) Restore by stable client_task_id (preferred)
    if client_task_id:
        task = _task_get(user_key, client_task_id)
        if task and task.get("latest_asset_id"):
            asset = _asset_get(user_key, task["latest_asset_id"])
            if asset:
                return _asset_compose_ready_payload(request, asset, task=task)
        if task and (task.get("status") in ("queued", "working")):
            # best effort: include live progress if provider_job_id is still in memory
            job_id = (task.get("provider_job_id") or "").strip()
            prog = None
            st = task.get("status")
            url = None
            if job_id and job_id in SORA_JOBS:
                j = SORA_JOBS.get(job_id) or {}
                prog = j.get("progress")
                url = j.get("url")
                st = j.get("status") or st
            return {"ok": True, "status": "pending", "client_task_id": client_task_id, "provider_job_id": job_id, "progress": prog, "url": url, "hint": "still_processing"}
        if task and task.get("status") == "error":
            return {"ok": False, "status": "error", "client_task_id": client_task_id, "error": "task_failed"}

    # 2) Fallback: restore by hint (asset_id or openai_video_id or caption substring)
    if asset_hint:
        # asset_id exact
        a = _asset_get(user_key, asset_hint)
        if a:
            return _asset_compose_ready_payload(request, a)

        # provider id (OpenAI video_id, MiniMax file_id, etc.)
        provider_id = _normalize_video_id(asset_hint) or asset_hint
        if provider_id:
            conn = _video_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM assets WHERE user_id=? AND source_video_id=? ORDER BY created_at DESC LIMIT 1",
                    (user_key, provider_id),
                ).fetchone()
                if row:
                    return _asset_compose_ready_payload(request, dict(row))
            finally:
                conn.close()

        # caption substring
        conn = _video_conn()
        try:
            row = conn.execute(
                "SELECT * FROM assets WHERE user_id=? AND status='ready' AND storage_key!='' AND (origin_url LIKE ? OR asset_id LIKE ?) ORDER BY created_at DESC LIMIT 1",
                (user_key, f"%{asset_hint}%", f"%{asset_hint}%"),
            ).fetchone()
            if row:
                return _asset_compose_ready_payload(request, dict(row))
        finally:
            conn.close()

    return {"ok": True, "status": "none"}

@app.patch("/v1/assets/{asset_id}/tags")
async def v1_asset_patch_tags(asset_id: str, request: Request):
    viewer = _auth_optional_user(request)
    user_key = (viewer["user_id"] if viewer else _client_id(request))

    asset = _asset_get(user_key, asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="asset not found")

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Allow either {"view_state":"watched"} or {"tags":{"view_state":"watched"}}
    tags = body.get("tags") if isinstance(body, dict) else None
    if isinstance(tags, dict):
        for k, v in tags.items():
            _asset_set_tag(asset_id, str(k), str(v))
    else:
        for k, v in (body or {}).items():
            if k in ("view_state", "lock", "ttl_hint"):
                _asset_set_tag(asset_id, str(k), str(v))

    return {"ok": True, "asset_id": asset_id, "tags": {k: _asset_get_tag(asset_id, k) for k in ("view_state", "lock", "ttl_hint")}}


# -----------------------------
# Photo cache (for voice -> sora intent reference)
# -----------------------------
@app.post("/solara/photo")
async def solara_photo(req: Request, session_id: str = Form(""), image_file: UploadFile = File(...)):
    raw = await image_file.read()
    if not raw:
        return JSONResponse({"ok": False, "error": "empty image"}, status_code=400)

    fid = uuid.uuid4().hex
    p = UPLOADS_DIR / f"photo_{fid}.jpg"

    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        raw = buf.getvalue()
    except Exception:
        pass

    p.write_bytes(raw)

    sk = f"{(req.client.host if req.client else 'unknown')}:{(session_id or 'default').strip()}"
    SOLARA_LAST_MEDIA[sk] = {"type": "image", "path": str(p), "mime": "image/jpeg", "ts": int(time.time())}
    return {"ok": True, "session_id": sk, "path": str(p), "mime": "image/jpeg"}

# ================================
# Realtime session + voice intent -> Sora job
# ================================
REALTIME_SESS_URL = "https://api.openai.com/v1/realtime/sessions"

def _realtime_ephemeral(model: str, voice: str, instructions: Optional[str] = None):
    body: Dict[str, Any] = {"model": model, "voice": voice}
    if instructions:
        body["instructions"] = instructions

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "realtime=v1",
    }
    r = requests.post(REALTIME_SESS_URL, headers=headers, json=body, timeout=20)
    if r.status_code >= 400:
        return None, f"OpenAI error {r.status_code}: {r.text}"

    data = r.json()
    key = ((data.get("client_secret") or {}).get("value") or data.get("ephemeral_key") or data.get("token"))
    if not key:
        return None, "missing ephemeral key"
    return key, None

def _resolve_realtime_instructions(body: Dict[str, Any]) -> Optional[str]:
    provided = (body.get("instructions") or "").strip()
    if provided:
        return provided

    profile = (body.get("profile") or body.get("mode") or "").strip().lower()
    if profile == "companion":
        return None
    if profile == "remix":
        return REMIX_REALTIME_INSTRUCTIONS

    return REALTIME_DEFAULT_INSTRUCTIONS

def _pick_realtime_instructions(req: Request, body: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Home automation: if a binding exists for this client, auto-switch to home-only assistant
    unless caller explicitly requests another profile (e.g. remix).
    Returns: (instructions, resolved_profile)
    """
    requested_profile = (body.get("profile") or body.get("mode") or "default").strip()
    profile_norm = requested_profile.lower().strip() or "default"

    # client_id helps commercial deployments; fallback IP works for local dev
    client_id = (body.get("client_id") or body.get("clientId") or "").strip() or (req.headers.get("x-client-id") or "").strip() or None

    # explicit profile overrides auto-home
    if profile_norm in ("remix", "companion"):
        return _resolve_realtime_instructions(body), profile_norm

    # Force home assistant (debug/lab): export HOME_FORCE_ON=1
    if HOME_FORCE_ON:
        return _home_instructions_ios_protocol(req, client_id=client_id), "home"

    # explicit home
    if profile_norm in ("home", "homeassistant", "home_assistant", "ha"):
        return _home_instructions_ios_protocol(req, client_id=client_id), "home"

    # auto-home if bound
    if home_has_binding(req, client_id=client_id):
        return _home_instructions_ios_protocol(req, client_id=client_id), "home"

    # default behavior
    return _resolve_realtime_instructions(body), profile_norm

@app.post("/session")
async def session_post(req: Request):
    """
    Realtime session bootstrap.
    - If home device is bound (POST /home/gateway_profile), this will auto return home-only assistant instructions.
    - Otherwise falls back to your default instructions / remix instructions.
    """
    try:
        b = await req.json()
    except Exception:
        b = {}

    model = (b.get("model") or REALTIME_MODEL_DEFAULT).strip()
    voice = (b.get("voice") or REALTIME_VOICE_DEFAULT).strip()

    instructions, resolved_profile = _pick_realtime_instructions(req, b)

    key, err = _realtime_ephemeral(model, voice, instructions=instructions)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=502)

    return {
        "ok": True,
        "session_id": uuid.uuid4().hex,
        "rtc_url": f"https://api.openai.com/v1/realtime?model={model}",
        "ephemeral_key": key,
        "modalities": ["text", "audio"],
        "ice_servers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "model": model,
        "voice": voice,
        "profile": resolved_profile,
        "home_bound": bool(resolved_profile == "home"),
    }

def _session_key(req: Request, session_id: str) -> str:
    ip = req.client.host if req.client else "unknown"
    sid = (session_id or "default").strip()
    return f"{ip}:{sid}"

@app.post("/rt/intent")
async def rt_intent(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}

    prompt = (body.get("prompt") or body.get("text") or "").strip()
    if not prompt:
        return JSONResponse({"ok": False, "error": "missing prompt"}, status_code=400)

    session_id = (body.get("session_id") or body.get("conversation_id") or "").strip()
    sk = _session_key(req, session_id)

    job_id = uuid.uuid4().hex

    last_media = SOLARA_LAST_MEDIA.get(sk) or {}
    ref_path = last_media.get("path") if last_media.get("type") == "image" else None
    ref_mime = _guess_mime_from_ext(ref_path) if ref_path else None

    provider = _select_video_provider("create")
    SORA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": prompt,
        "prompt_raw": prompt,

        "url": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "provider": provider,

        "ref_path": ref_path,
        "ref_mime": ref_mime,

        "mode": "create",
        "remix_base_video_id": None,
        "remix_instruction": None,
        "openai_status": None,
    }

    _spawn_sora_job(job_id, timeout_sec=1200)

    return {"ok": True, "job_id": job_id, "status_url": f"/video/status/{job_id}", "session_id": sk}

# ================================
# ✅ FIX: GPT TTS endpoints (NO-STREAM / no body-consuming log)
# - Keep for backward compatibility
# ================================

async def _tts_bytes_impl(request: Request, force_mp3: bool = False):
    try:
        body = await request.json()
    except Exception:
        body = {}

    raw_text = (body.get("text") or body.get("input") or "").strip()
    if not raw_text:
        return JSONResponse({"ok": False, "error": "missing text"}, status_code=400)

    # ✅ Commercial UX: never 413 for long texts. Sanitize + trim + chunk instead.
    text = _tts_sanitize_text(raw_text)

    voice = _normalize_tts_voice((body.get("voice") or "").strip() or TTS_VOICE_DEFAULT)
    model = (body.get("model") or TTS_MODEL_DEFAULT).strip()

    fmt = "mp3" if force_mp3 else (body.get("format") or body.get("response_format") or "mp3")
    fmt = str(fmt).strip().lower() or "mp3"

    instructions = (body.get("instructions") or TTS_INSTRUCTIONS_DEFAULT or "").strip()

    # speed: client override -> else env default
    spd: Optional[float] = None
    try:
        if body.get("speed") is not None:
            spd = float(body.get("speed"))
    except Exception:
        spd = None
    if spd is None:
        try:
            spd = float(TTS_SPEED_DEFAULT)
        except Exception:
            spd = None
    if spd is not None:
        spd = max(0.5, min(spd, 1.6))

    segments = _split_tts_segments(text, TTS_SEGMENT_MAX_CHARS)
    if not segments:
        return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)

    def _post(payload: Dict[str, Any]) -> requests.Response:
        r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=payload, stream=True, timeout=120)
        if r.status_code >= 400 and "speed" in payload:
            try:
                r.close()
            except Exception:
                pass
            p2 = dict(payload)
            p2.pop("speed", None)
            r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p2, stream=True, timeout=120)
        if r.status_code >= 400 and "instructions" in payload:
            try:
                r.close()
            except Exception:
                pass
            p3 = dict(payload)
            p3.pop("instructions", None)
            p3.pop("speed", None)
            r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=p3, stream=True, timeout=120)
        return r

    out = bytearray()
    ok_any = False
    last_err = None

    for seg in segments:
        seg = (seg or "").strip()
        if not seg:
            continue

        payload: Dict[str, Any] = {"model": model, "voice": voice, "input": seg}
        if instructions:
            payload["instructions"] = instructions

        # ✅ Web search tool (official)
        if CHAT_ENABLE_WEB_SEARCH_DEFAULT:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"
            # include sources for UI (site icons)
            payload["include"] = ["web_search_call.action.sources"]
        if spd is not None:
            payload["speed"] = spd
        if fmt and fmt != "mp3":
            payload["response_format"] = fmt

        r = _post(payload)
        rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
        log.info("[TTS.bytes] voice=%s fmt=%s len=%s -> %s rid=%s", voice, fmt, len(seg), r.status_code, rid)

        if r.status_code >= 400:
            last_err = _short(r.text, 300)
            try:
                r.close()
            except Exception:
                pass
            continue

        ok_any = True
        for chunk in r.iter_content(chunk_size=TTS_STREAM_CHUNK_SIZE_DEFAULT):
            if chunk:
                out.extend(chunk)

        try:
            r.close()
        except Exception:
            pass

    if not ok_any:
        return JSONResponse({"ok": False, "error": f"openai_tts_error: {last_err or 'unknown'}"}, status_code=502)

    return Response(content=bytes(out), media_type=_tts_media_type(fmt))

@app.post("/tts")
async def tts(request: Request):
    return await _tts_bytes_impl(request, force_mp3=False)

# -----------------------------
# Health
# -----------------------------


# =========================
# Video Plaza API (MVP)
# =========================
@app.post("/v1/video/upload")
async def v1_video_upload(
    request: Request,
    file: UploadFile = File(...),
    caption: str = Form(""),
):
    """Upload a video as a draft. Client must publish explicitly."""
    viewer = _auth_optional_user(request)
    owner_key = (viewer["user_id"] if viewer else _client_id(request))
    owner_user_id = (viewer["user_id"] if viewer else None)
    owner_display_name = (viewer.get("display_name", "") if viewer else "")

    ct = (file.content_type or "").lower()
    if not ct.startswith("video/"):
        raise HTTPException(status_code=415, detail="Only video/* is supported")

    video_id = uuid.uuid4().hex
    orig_name = file.filename or "video"
    ext = Path(orig_name).suffix.lower()
    if ext not in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}:
        ext = ".mp4"

    dest_path = VIDEOS_DIR / f"{video_id}{ext}"

    size_bytes = 0
    try:
        with open(dest_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > SOLARA_MAX_VIDEO_MB * 1024 * 1024:
                    try:
                        f.close()
                    finally:
                        if dest_path.exists():
                            dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video too large (> {SOLARA_MAX_VIDEO_MB} MB)",
                    )
                f.write(chunk)
    finally:
        try:
            await file.close()
        except Exception:
            pass

    now = int(time.time())
    conn = _video_conn()
    conn.execute(
        """INSERT INTO videos (video_id, owner_key, owner_user_id, owner_display_name, caption, status, file_path, mime, size_bytes, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (video_id, owner_key, owner_user_id, owner_display_name, (caption or "").strip(), "draft", str(dest_path), ct, int(size_bytes), now, now),
    )
    conn.commit()
    conn.close()

    base = str(request.base_url).rstrip("/")
    return {
        "ok": True,
        "video_id": video_id,
        "status": "draft",
        "caption": (caption or "").strip(),
        "owner_key": owner_key,
        "play_url": f"{base}/v1/video/stream/{video_id}",
        "publish_url": f"{base}/v1/video/{video_id}/publish",
    }


@app.post("/v1/video/{video_id}/publish")
async def v1_video_publish(video_id: str, request: Request):
    """Publish a previously uploaded draft video."""
    viewer = _auth_optional_user(request)
    owner_key = (viewer["user_id"] if viewer else _client_id(request))
    conn = _video_conn()
    row = conn.execute(
        "SELECT video_id, owner_key, status FROM videos WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Video not found")
    if row["owner_key"] != owner_key:
        conn.close()
        raise HTTPException(status_code=403, detail="Not allowed")

    if row["status"] != "published":
        now = int(time.time())
        conn.execute(
            "UPDATE videos SET status='published', published_at=?, updated_at=? WHERE video_id=?",
            (now, now, video_id),
        )
        conn.commit()
    conn.close()

    return {"ok": True, "video_id": video_id, "status": "published"}


@app.get("/v1/feed")
async def v1_feed(request: Request, limit: int = 20, cursor: Optional[int] = None):
    """Global plaza feed: all published videos (commercial fields included)."""
    limit = max(1, min(int(limit or 20), 50))

    viewer = _auth_optional_user(request)
    viewer_id = viewer["user_id"] if viewer else None

    conn = _video_conn()
    try:
        base_q = """SELECT v.*"""
        if viewer_id:
            base_q += """,
                CASE WHEN vl.user_id IS NULL THEN 0 ELSE 1 END AS liked_by_me,
                CASE WHEN f.follower_id IS NULL THEN 0 ELSE 1 END AS followed_by_me
            """
        else:
            base_q += """,
                0 AS liked_by_me,
                0 AS followed_by_me
            """

        base_q += """
            FROM videos v
        """

        params: List[Any] = []
        if viewer_id:
            base_q += """
                LEFT JOIN video_likes vl ON vl.video_id = v.video_id AND vl.user_id = ?
                LEFT JOIN follows f ON f.follower_id = ? AND f.following_id = v.owner_user_id
            """
            params.extend([viewer_id, viewer_id])

        base_q += """
            WHERE v.status='published'
              AND v.published_at IS NOT NULL
              AND (v.visibility IS NULL OR v.visibility='public')
        """

        if cursor:
            base_q += " AND v.published_at < ?"
            params.append(int(cursor))

        base_q += " ORDER BY v.published_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(base_q, tuple(params)).fetchall()
    finally:
        conn.close()

    base = str(request.base_url).rstrip("/")
    items: List[Dict[str, Any]] = []
    next_cursor: Optional[int] = None
    for r in rows:
        r = dict(r)
        items.append(
            {
                "video_id": r.get("video_id"),
                "caption": r.get("caption") or "",
                "owner_key": r.get("owner_key"),
                "owner_user_id": r.get("owner_user_id"),
                "owner_display_name": (r.get("owner_display_name") or ""),
                "status": r.get("status"),
                "created_at": r.get("created_at"),
                "published_at": r.get("published_at"),
                "views": int(r.get("views") or 0),
                "likes": int(r.get("likes") or 0),
                "comments": int(r.get("comments") or 0),
                "shares": int(r.get("shares") or 0),
                "liked_by_me": bool(int(r.get("liked_by_me") or 0)),
                "followed_by_me": bool(int(r.get("followed_by_me") or 0)),
                "play_url": f"{base}/v1/video/stream/{r.get('video_id')}",
            }
        )
        next_cursor = r.get("published_at")

    return {"ok": True, "items": items, "next_cursor": next_cursor}
@app.get("/v1/me/videos")
async def v1_me_videos(request: Request, include_drafts: bool = True):
    """Return videos uploaded by current user/client (draft + published)."""
    viewer = _auth_optional_user(request)
    owner_key = (viewer["user_id"] if viewer else _client_id(request))

    conn = _video_conn()
    try:
        if include_drafts:
            rows = conn.execute(
                "SELECT * FROM videos WHERE owner_key=? ORDER BY created_at DESC",
                (owner_key,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM videos
                   WHERE owner_key=? AND status='published' AND published_at IS NOT NULL
                   ORDER BY published_at DESC""",
                (owner_key,),
            ).fetchall()
    finally:
        conn.close()

    base = str(request.base_url).rstrip("/")
    items: List[Dict[str, Any]] = []
    for r in rows:
        r = dict(r)
        items.append(
            {
                "video_id": r.get("video_id"),
                "caption": r.get("caption") or "",
                "owner_key": r.get("owner_key"),
                "owner_user_id": r.get("owner_user_id"),
                "owner_display_name": (r.get("owner_display_name") or ""),
                "status": r.get("status"),
                "visibility": (r.get("visibility") or "public"),
                "tags": (r.get("tags") or ""),
                "created_at": r.get("created_at"),
                "published_at": r.get("published_at"),
                "views": int(r.get("views") or 0),
                "likes": int(r.get("likes") or 0),
                "comments": int(r.get("comments") or 0),
                "shares": int(r.get("shares") or 0),
                "play_url": f"{base}/v1/video/stream/{r.get('video_id')}",
            }
        )

    return {"ok": True, "items": items}
@app.get("/v1/video/stream/{video_id}")
async def v1_video_stream(video_id: str, request: Request):
    conn = _video_conn()
    row = conn.execute(
        "SELECT file_path, mime, status, visibility, owner_key FROM videos WHERE video_id=?",
        (video_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Video not found")

    row = dict(row)

    # Only serve published/public to anonymous users; owners can still access drafts.
    if (row.get("status") != "published") or (row.get("visibility") not in (None, "", "public")):
        viewer = _auth_optional_user(request)
        viewer_key = (viewer.get("user_id") if viewer else _client_id(request))
        if viewer_key != row.get("owner_key"):
            raise HTTPException(status_code=403, detail="forbidden")

    file_path = row.get("file_path")
    mime = (row.get("mime") or "video/mp4")

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file missing")

    # Best-effort: count views
    try:
        now = int(time.time())
        conn2 = _video_conn()
        conn2.execute(
            "UPDATE videos SET views = COALESCE(views,0) + 1, updated_at=? WHERE video_id=?",
            (now, video_id),
        )
        conn2.commit()
        conn2.close()
    except Exception:
        pass

    return FileResponse(
        path=file_path,
        media_type=mime,
        filename=os.path.basename(file_path),
    )
@app.post("/v2/auth/register")
async def v2_auth_register(req: V2RegisterReq):
    username = (req.username or "").strip().lower()
    password = (req.password or "").strip()

    if not username or len(username) < 3 or len(username) > 64:
        raise HTTPException(status_code=400, detail="invalid username")
    if not password or len(password) < 6 or len(password) > 200:
        raise HTTPException(status_code=400, detail="invalid password")

    display_name = (req.display_name or "").strip() or username

    user_id = uuid.uuid4().hex
    now = int(time.time())
    pw_hash = _hash_password(password)

    conn = _video_conn()
    try:
        try:
            conn.execute(
                "INSERT INTO users(user_id, username, display_name, password_hash, avatar_url, created_at) VALUES (?, ?, ?, ?, '', ?)",
                (user_id, username, display_name, pw_hash, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="username already exists")

        token = _issue_access_token(user_id)
        urow = conn.execute(
            "SELECT user_id, username, display_name, avatar_url, created_at FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()

    return {"ok": True, "access_token": token, "token_type": "bearer", "user": _public_user(dict(urow))}

@app.post("/v2/auth/login")
async def v2_auth_login(req: V2LoginReq):
    username = (req.username or "").strip().lower()
    password = (req.password or "").strip()

    conn = _video_conn()
    try:
        urow = conn.execute(
            "SELECT user_id, username, display_name, avatar_url, created_at, password_hash FROM users WHERE username=?",
            (username,),
        ).fetchone()
        if not urow or not _verify_password(password, urow["password_hash"]):
            raise HTTPException(status_code=401, detail="invalid credentials")

        token = _issue_access_token(urow["user_id"])
        user = _public_user(dict(urow))
    finally:
        conn.close()

    return {"ok": True, "access_token": token, "token_type": "bearer", "user": user}

@app.post("/v2/auth/logout")
async def v2_auth_logout(request: Request):
    token = _get_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="unauthorized")
    payload = _decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="unauthorized")
    token_id = payload.get("jti")
    if not token_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    conn = _video_conn()
    try:
        conn.execute("UPDATE sessions SET revoked=1 WHERE token_id=?", (token_id,))
        conn.commit()
    finally:
        conn.close()

    return {"ok": True}

@app.get("/v2/auth/me")
async def v2_auth_me(request: Request):
    user = _auth_required_user(request)

    conn = _video_conn()
    try:
        followers = _count_followers(conn, user["user_id"])
        following = _count_following(conn, user["user_id"])
    finally:
        conn.close()

    return {"ok": True, "user": _public_user(user), "followers": followers, "following": following}

@app.get("/v2/users/suggest")
async def v2_users_suggest(request: Request, limit: int = 20):
    limit = max(1, min(int(limit or 20), 50))
    viewer = _auth_optional_user(request)
    viewer_id = viewer["user_id"] if viewer else None

    conn = _video_conn()
    try:
        if viewer_id:
            rows = conn.execute(
                """SELECT user_id, username, display_name, avatar_url, created_at
                   FROM users WHERE user_id <> ?
                   ORDER BY RANDOM()
                   LIMIT ?""",
                (viewer_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT user_id, username, display_name, avatar_url, created_at
                   FROM users ORDER BY RANDOM() LIMIT ?""",
                (limit,),
            ).fetchall()
    finally:
        conn.close()

    return {"ok": True, "items": [_public_user(dict(r)) for r in rows]}

@app.get("/v2/users/{user_id}")
async def v2_user_profile(user_id: str, request: Request):
    conn = _video_conn()
    try:
        urow = conn.execute(
            "SELECT user_id, username, display_name, avatar_url, created_at FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if not urow:
            raise HTTPException(status_code=404, detail="user not found")
        followers = _count_followers(conn, user_id)
        following = _count_following(conn, user_id)

        viewer = _auth_optional_user(request)
        followed_by_me = False
        if viewer and viewer["user_id"] != user_id:
            f = conn.execute(
                "SELECT 1 FROM follows WHERE follower_id=? AND following_id=?",
                (viewer["user_id"], user_id),
            ).fetchone()
            followed_by_me = bool(f)
    finally:
        conn.close()

    return {
        "ok": True,
        "user": _public_user(dict(urow)),
        "followers": followers,
        "following": following,
        "followed_by_me": followed_by_me,
    }

@app.post("/v2/follow/{target_user_id}")
async def v2_follow(target_user_id: str, request: Request):
    me = _auth_required_user(request)
    if target_user_id == me["user_id"]:
        raise HTTPException(status_code=400, detail="cannot follow self")

    now = int(time.time())
    conn = _video_conn()
    try:
        # ensure user exists
        t = conn.execute("SELECT 1 FROM users WHERE user_id=?", (target_user_id,)).fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="user not found")

        conn.execute(
            "INSERT OR IGNORE INTO follows(follower_id, following_id, created_at) VALUES (?, ?, ?)",
            (me["user_id"], target_user_id, now),
        )
        conn.commit()
        followers = _count_followers(conn, target_user_id)
        following = _count_following(conn, me["user_id"])
    finally:
        conn.close()

    return {"ok": True, "followed": True, "followers": followers, "following": following}

@app.delete("/v2/follow/{target_user_id}")
async def v2_unfollow(target_user_id: str, request: Request):
    me = _auth_required_user(request)
    if target_user_id == me["user_id"]:
        raise HTTPException(status_code=400, detail="cannot unfollow self")

    conn = _video_conn()
    try:
        conn.execute(
            "DELETE FROM follows WHERE follower_id=? AND following_id=?",
            (me["user_id"], target_user_id),
        )
        conn.commit()
        followers = _count_followers(conn, target_user_id)
        following = _count_following(conn, me["user_id"])
    finally:
        conn.close()

    return {"ok": True, "followed": False, "followers": followers, "following": following}

@app.get("/v2/followers/{user_id}")
async def v2_followers(user_id: str, limit: int = 20, cursor: Optional[int] = None):
    limit = max(1, min(int(limit or 20), 50))
    conn = _video_conn()
    try:
        q = """
            SELECT u.user_id, u.username, u.display_name, u.avatar_url, u.created_at, f.created_at AS followed_at
            FROM follows f
            JOIN users u ON u.user_id = f.follower_id
            WHERE f.following_id = ?
        """
        params: List[Any] = [user_id]
        if cursor:
            q += " AND f.created_at < ?"
            params.append(int(cursor))
        q += " ORDER BY f.created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, tuple(params)).fetchall()
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    next_cursor = None
    for r in rows:
        d = dict(r)
        items.append(_public_user(d) | {"followed_at": d.get("followed_at")})
        next_cursor = d.get("followed_at")
    return {"ok": True, "items": items, "next_cursor": next_cursor}

@app.get("/v2/following/{user_id}")
async def v2_following(user_id: str, limit: int = 20, cursor: Optional[int] = None):
    limit = max(1, min(int(limit or 20), 50))
    conn = _video_conn()
    try:
        q = """
            SELECT u.user_id, u.username, u.display_name, u.avatar_url, u.created_at, f.created_at AS followed_at
            FROM follows f
            JOIN users u ON u.user_id = f.following_id
            WHERE f.follower_id = ?
        """
        params: List[Any] = [user_id]
        if cursor:
            q += " AND f.created_at < ?"
            params.append(int(cursor))
        q += " ORDER BY f.created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, tuple(params)).fetchall()
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    next_cursor = None
    for r in rows:
        d = dict(r)
        items.append(_public_user(d) | {"followed_at": d.get("followed_at")})
        next_cursor = d.get("followed_at")
    return {"ok": True, "items": items, "next_cursor": next_cursor}

@app.get("/v2/friends")
async def v2_friends(request: Request, limit: int = 50):
    me = _auth_required_user(request)
    limit = max(1, min(int(limit or 50), 200))

    conn = _video_conn()
    try:
        rows = conn.execute(
            """
            SELECT u.user_id, u.username, u.display_name, u.avatar_url, u.created_at
            FROM users u
            JOIN follows f1 ON f1.following_id = u.user_id AND f1.follower_id = ?
            JOIN follows f2 ON f2.follower_id = u.user_id AND f2.following_id = ?
            ORDER BY u.display_name ASC
            LIMIT ?
            """,
            (me["user_id"], me["user_id"], limit),
        ).fetchall()
    finally:
        conn.close()

    return {"ok": True, "items": [_public_user(dict(r)) for r in rows]}

@app.get("/v2/feed")
async def v2_feed(request: Request, limit: int = 20, cursor: Optional[int] = None, strategy: str = "latest", offset: int = 0):
    """
    Commercial feed endpoint:
      - strategy=latest: cursor based (published_at)
      - strategy=trending: offset based
    """
    limit = max(1, min(int(limit or 20), 50))
    strategy = (strategy or "latest").lower().strip()

    viewer = _auth_optional_user(request)
    viewer_id = viewer["user_id"] if viewer else None

    conn = _video_conn()
    try:
        if strategy == "trending":
            q = """SELECT v.* FROM videos v
                   WHERE v.status='published' AND v.published_at IS NOT NULL
                     AND (v.visibility IS NULL OR v.visibility='public')
                   ORDER BY (COALESCE(v.likes,0)*2 + COALESCE(v.shares,0)*3 + COALESCE(v.comments,0)) DESC,
                            v.published_at DESC
                   LIMIT ? OFFSET ?"""
            rows = conn.execute(q, (limit, max(0, int(offset or 0)))).fetchall()
            next_cursor = None
            next_offset = max(0, int(offset or 0)) + len(rows)
        else:
            q = """SELECT v.* FROM videos v
                   WHERE v.status='published' AND v.published_at IS NOT NULL
                     AND (v.visibility IS NULL OR v.visibility='public')"""
            params: List[Any] = []
            if cursor:
                q += " AND v.published_at < ?"
                params.append(int(cursor))
            q += " ORDER BY v.published_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(q, tuple(params)).fetchall()
            next_offset = None
            next_cursor = rows[-1]["published_at"] if rows else None
    finally:
        conn.close()

    base = str(request.base_url).rstrip("/")
    items: List[Dict[str, Any]] = []
    for r in rows:
        r = dict(r)
        vid = r.get("video_id")
        if not vid:
            continue

        liked_by_me = False
        followed_by_me = False
        if viewer_id:
            conn2 = _video_conn()
            try:
                liked_by_me = bool(
                    conn2.execute(
                        "SELECT 1 FROM video_likes WHERE user_id=? AND video_id=?",
                        (viewer_id, vid),
                    ).fetchone()
                )
                owner_uid = r.get("owner_user_id")
                if owner_uid:
                    followed_by_me = bool(
                        conn2.execute(
                            "SELECT 1 FROM follows WHERE follower_id=? AND following_id=?",
                            (viewer_id, owner_uid),
                        ).fetchone()
                    )
            finally:
                conn2.close()

        items.append(
            {
                "video_id": vid,
                "caption": r.get("caption") or "",
                "owner_key": r.get("owner_key"),
                "owner_user_id": r.get("owner_user_id"),
                "owner_display_name": (r.get("owner_display_name") or ""),
                "published_at": r.get("published_at"),
                "views": int(r.get("views") or 0),
                "likes": int(r.get("likes") or 0),
                "comments": int(r.get("comments") or 0),
                "shares": int(r.get("shares") or 0),
                "liked_by_me": liked_by_me,
                "followed_by_me": followed_by_me,
                "play_url": f"{base}/v1/video/stream/{vid}",
                "rec_reason": ("trending" if strategy == "trending" else "latest"),
            }
        )

    return {"ok": True, "strategy": strategy, "items": items, "next_cursor": next_cursor, "next_offset": next_offset}
@app.post("/v2/video/{video_id}/like")
async def v2_like(video_id: str, request: Request):
    me = _auth_required_user(request)
    now = int(time.time())

    conn = _video_conn()
    try:
        v = conn.execute("SELECT 1 FROM videos WHERE video_id=? AND status='published'", (video_id,)).fetchone()
        if not v:
            raise HTTPException(status_code=404, detail="video not found")

        cur = conn.execute(
            "INSERT OR IGNORE INTO video_likes(user_id, video_id, created_at) VALUES (?, ?, ?)",
            (me["user_id"], video_id, now),
        )
        if cur.rowcount == 1:
            conn.execute("UPDATE videos SET likes = COALESCE(likes,0) + 1, updated_at=? WHERE video_id=?", (now, video_id))
        conn.commit()
        row = conn.execute("SELECT likes FROM videos WHERE video_id=?", (video_id,)).fetchone()
        likes = int(row["likes"] or 0) if row else 0
    finally:
        conn.close()

    return {"ok": True, "liked": True, "likes": likes}

@app.delete("/v2/video/{video_id}/like")
async def v2_unlike(video_id: str, request: Request):
    me = _auth_required_user(request)
    now = int(time.time())

    conn = _video_conn()
    try:
        cur = conn.execute(
            "DELETE FROM video_likes WHERE user_id=? AND video_id=?",
            (me["user_id"], video_id),
        )
        if cur.rowcount == 1:
            conn.execute(
                "UPDATE videos SET likes = MAX(COALESCE(likes,0) - 1, 0), updated_at=? WHERE video_id=?",
                (now, video_id),
            )
        conn.commit()
        row = conn.execute("SELECT likes FROM videos WHERE video_id=?", (video_id,)).fetchone()
        likes = int(row["likes"] or 0) if row else 0
    finally:
        conn.close()

    return {"ok": True, "liked": False, "likes": likes}

@app.get("/v2/video/{video_id}/comments")
async def v2_get_comments(video_id: str, limit: int = 20, cursor: Optional[int] = None):
    limit = max(1, min(int(limit or 20), 50))
    conn = _video_conn()
    try:
        q = """
            SELECT c.comment_id, c.video_id, c.user_id, c.text, c.created_at,
                   u.username, u.display_name, u.avatar_url
            FROM video_comments c
            JOIN users u ON u.user_id = c.user_id
            WHERE c.video_id = ?
        """
        params: List[Any] = [video_id]
        if cursor:
            q += " AND c.created_at < ?"
            params.append(int(cursor))
        q += " ORDER BY c.created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, tuple(params)).fetchall()
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    next_cursor = None
    for r in rows:
        d = dict(r)
        items.append(
            {
                "comment_id": d["comment_id"],
                "video_id": d["video_id"],
                "user": {
                    "user_id": d["user_id"],
                    "username": d["username"],
                    "display_name": d["display_name"],
                    "avatar_url": d.get("avatar_url") or "",
                },
                "text": d["text"],
                "created_at": d["created_at"],
            }
        )
        next_cursor = d["created_at"]
    return {"ok": True, "items": items, "next_cursor": next_cursor}

@app.post("/v2/video/{video_id}/comment")
async def v2_comment(video_id: str, request: Request, body: V2CommentReq):
    me = _auth_required_user(request)
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty comment")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="comment too long")

    now = int(time.time())
    comment_id = uuid.uuid4().hex

    conn = _video_conn()
    try:
        v = conn.execute("SELECT 1 FROM videos WHERE video_id=? AND status='published'", (video_id,)).fetchone()
        if not v:
            raise HTTPException(status_code=404, detail="video not found")

        conn.execute(
            "INSERT INTO video_comments(comment_id, video_id, user_id, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (comment_id, video_id, me["user_id"], text, now),
        )
        conn.execute(
            "UPDATE videos SET comments = COALESCE(comments,0) + 1, updated_at=? WHERE video_id=?",
            (now, video_id),
        )
        conn.commit()
        row = conn.execute("SELECT comments FROM videos WHERE video_id=?", (video_id,)).fetchone()
        comments = int(row["comments"] or 0) if row else 0
    finally:
        conn.close()

    return {"ok": True, "comment_id": comment_id, "comments": comments}

@app.post("/v2/video/{video_id}/share")
async def v2_share(video_id: str, request: Request, body: V2ShareReq):
    me = _auth_required_user(request)
    channel = (body.channel or "").strip()[:64]
    now = int(time.time())
    share_id = uuid.uuid4().hex

    conn = _video_conn()
    try:
        v = conn.execute("SELECT 1 FROM videos WHERE video_id=? AND status='published'", (video_id,)).fetchone()
        if not v:
            raise HTTPException(status_code=404, detail="video not found")

        conn.execute(
            "INSERT INTO video_shares(share_id, video_id, user_id, channel, created_at) VALUES (?, ?, ?, ?, ?)",
            (share_id, video_id, me["user_id"], channel, now),
        )
        conn.execute(
            "UPDATE videos SET shares = COALESCE(shares,0) + 1, updated_at=? WHERE video_id=?",
            (now, video_id),
        )
        conn.commit()
        row = conn.execute("SELECT shares FROM videos WHERE video_id=?", (video_id,)).fetchone()
        shares = int(row["shares"] or 0) if row else 0
    finally:
        conn.close()

    return {"ok": True, "share_id": share_id, "shares": shares}

@app.post("/v2/reco/event")
async def v2_reco_event(request: Request, body: V2RecoEventReq):
    me = _auth_required_user(request)
    et = (body.event_type or "").strip().lower()
    if not et:
        raise HTTPException(status_code=400, detail="missing event_type")

    now = int(time.time())
    event_id = uuid.uuid4().hex
    meta_json = ""
    try:
        if body.meta:
            meta_json = json.dumps(body.meta, ensure_ascii=False)[:4000]
    except Exception:
        meta_json = ""

    conn = _video_conn()
    try:
        conn.execute(
            "INSERT INTO reco_events(event_id, user_id, video_id, event_type, meta_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, me["user_id"], body.video_id, et, meta_json, now),
        )
        conn.commit()
    finally:
        conn.close()

    return {"ok": True, "event_id": event_id}

# ---- DM (REST) ----

@app.post("/v2/dm/thread/{other_user_id}")
async def v2_dm_get_or_create_thread(other_user_id: str, request: Request):
    me = _auth_required_user(request)
    if other_user_id == me["user_id"]:
        raise HTTPException(status_code=400, detail="invalid other_user_id")

    a, b = _normalize_pair(me["user_id"], other_user_id)
    now = int(time.time())

    conn = _video_conn()
    try:
        # ensure other exists
        t = conn.execute("SELECT 1 FROM users WHERE user_id=?", (other_user_id,)).fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="user not found")

        row = conn.execute(
            "SELECT thread_id FROM dm_threads WHERE user_a=? AND user_b=?",
            (a, b),
        ).fetchone()
        if row:
            thread_id = row["thread_id"]
        else:
            thread_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO dm_threads(thread_id, user_a, user_b, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, a, b, now, now),
            )
            conn.commit()
    finally:
        conn.close()

    return {"ok": True, "thread_id": thread_id}

@app.get("/v2/dm/threads")
async def v2_dm_threads(request: Request, limit: int = 50, cursor: Optional[int] = None):
    me = _auth_required_user(request)
    limit = max(1, min(int(limit or 50), 200))

    conn = _video_conn()
    try:
        q = """
            SELECT thread_id, user_a, user_b, created_at, updated_at
            FROM dm_threads
            WHERE user_a=? OR user_b=?
        """
        params: List[Any] = [me["user_id"], me["user_id"]]
        if cursor:
            q += " AND updated_at < ?"
            params.append(int(cursor))
        q += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        threads = conn.execute(q, tuple(params)).fetchall()

        items: List[Dict[str, Any]] = []
        next_cursor = None
        for t in threads:
            t = dict(t)
            other = t["user_b"] if t["user_a"] == me["user_id"] else t["user_a"]
            u = conn.execute(
                "SELECT user_id, username, display_name, avatar_url, created_at FROM users WHERE user_id=?",
                (other,),
            ).fetchone()
            last = conn.execute(
                "SELECT text, created_at FROM dm_messages WHERE thread_id=? ORDER BY created_at DESC LIMIT 1",
                (t["thread_id"],),
            ).fetchone()
            items.append(
                {
                    "thread_id": t["thread_id"],
                    "user": _public_user(dict(u)) if u else {"user_id": other},
                    "updated_at": t["updated_at"],
                    "last_message": (dict(last) if last else None),
                }
            )
            next_cursor = t["updated_at"]
    finally:
        conn.close()

    return {"ok": True, "items": items, "next_cursor": next_cursor}

@app.get("/v2/dm/thread/{thread_id}/messages")
async def v2_dm_messages(thread_id: str, request: Request, limit: int = 50, cursor: Optional[int] = None):
    me = _auth_required_user(request)
    limit = max(1, min(int(limit or 50), 200))

    conn = _video_conn()
    try:
        t = conn.execute(
            "SELECT user_a, user_b FROM dm_threads WHERE thread_id=?",
            (thread_id,),
        ).fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="thread not found")
        if me["user_id"] not in (t["user_a"], t["user_b"]):
            raise HTTPException(status_code=403, detail="forbidden")

        q = "SELECT message_id, thread_id, sender_id, text, created_at FROM dm_messages WHERE thread_id=?"
        params: List[Any] = [thread_id]
        if cursor:
            q += " AND created_at < ?"
            params.append(int(cursor))
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(q, tuple(params)).fetchall()
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    next_cursor = None
    for r in rows:
        d = dict(r)
        items.append(d)
        next_cursor = d["created_at"]
    return {"ok": True, "items": items, "next_cursor": next_cursor}

@app.post("/v2/dm/thread/{thread_id}/messages")
async def v2_dm_send(thread_id: str, request: Request, body: V2DMMessageReq):
    me = _auth_required_user(request)
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty message")
    if len(text) > 4000:
        raise HTTPException(status_code=400, detail="message too long")

    now = int(time.time())
    msg_id = uuid.uuid4().hex

    conn = _video_conn()
    try:
        t = conn.execute(
            "SELECT user_a, user_b FROM dm_threads WHERE thread_id=?",
            (thread_id,),
        ).fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="thread not found")
        if me["user_id"] not in (t["user_a"], t["user_b"]):
            raise HTTPException(status_code=403, detail="forbidden")

        conn.execute(
            "INSERT INTO dm_messages(message_id, thread_id, sender_id, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (msg_id, thread_id, me["user_id"], text, now),
        )
        conn.execute(
            "UPDATE dm_threads SET updated_at=? WHERE thread_id=?",
            (now, thread_id),
        )
        conn.commit()
    finally:
        conn.close()

    return {"ok": True, "message_id": msg_id, "created_at": now}

# ---- WebRTC signaling ----

SIGNALING_CONNECTIONS: Dict[str, WebSocket] = {}

@app.get("/v2/webrtc/ice")
async def v2_webrtc_ice(request: Request):
    _ = _auth_required_user(request)

    ice_servers: List[Dict[str, Any]] = [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]
    if TURN_URLS and TURN_USERNAME and TURN_CREDENTIAL:
        ice_servers.append(
            {"urls": TURN_URLS, "username": TURN_USERNAME, "credential": TURN_CREDENTIAL}
        )

    return {"ok": True, "iceServers": ice_servers}

@app.websocket("/ws/v2/signaling")
async def ws_v2_signaling(websocket: WebSocket):
    # token can be provided via query string (?token=) or Authorization header
    await websocket.accept()

    token = websocket.query_params.get("token") or ""
    if not token:
        auth = websocket.headers.get("authorization") or websocket.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()

    payload = _decode_access_token(token) if token else None
    user_id = payload.get("sub") if payload else None
    token_id = payload.get("jti") if payload else None
    if not user_id or not token_id:
        await websocket.send_text(json.dumps({"type": "error", "error": "unauthorized"}))
        await websocket.close(code=4401)
        return

    # validate session
    conn = _video_conn()
    try:
        srow = conn.execute(
            "SELECT revoked, expires_at FROM sessions WHERE token_id=? AND user_id=?",
            (token_id, user_id),
        ).fetchone()
        if not srow or int(srow["revoked"] or 0) == 1 or int(srow["expires_at"] or 0) < int(time.time()):
            await websocket.send_text(json.dumps({"type": "error", "error": "unauthorized"}))
            await websocket.close(code=4401)
            return
    finally:
        conn.close()

    SIGNALING_CONNECTIONS[user_id] = websocket
    try:
        await websocket.send_text(json.dumps({"type": "ready", "user_id": user_id}))
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                await websocket.send_text(json.dumps({"type": "error", "error": "invalid_json"}))
                continue

            to_id = (data.get("to") or "").strip()
            if not to_id:
                await websocket.send_text(json.dumps({"type": "error", "error": "missing_to"}))
                continue

            data["from"] = user_id
            target = SIGNALING_CONNECTIONS.get(to_id)
            if target:
                await target.send_text(json.dumps(data))
            else:
                await websocket.send_text(json.dumps({"type": "error", "error": "peer_offline", "to": to_id}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if SIGNALING_CONNECTIONS.get(user_id) is websocket:
            SIGNALING_CONNECTIONS.pop(user_id, None)

@app.get("/health")
def health():
    return {
        "ok": True,
        "sora": {
            "model": SORA_MODEL_DEFAULT,
            "seconds": SORA_SECONDS_DEFAULT,
            "size": SORA_SIZE_DEFAULT,
            "concurrency": SORA_CONCURRENCY
        },
        "realtime": {
            "model": REALTIME_MODEL_DEFAULT,
            "voice": REALTIME_VOICE_DEFAULT,
            "has_default_instructions": bool(REALTIME_DEFAULT_INSTRUCTIONS)
        },
        "chat": {
            "model": CHAT_MODEL_DEFAULT,
            "has_streaming_chat": True,
            "has_server_side_tts_stream": True,
            "tts_model": TTS_MODEL_DEFAULT,
            "tts_voice_default": TTS_VOICE_DEFAULT,
        },
        "home_voice": {
            "has_bind": True,
            "bind_paths": ["/home/gateway_profile", "/home/bind", "/ha/bind", "/ha/home/bind", "/homeassistant/bind"],
            "has_dispatch": True,
            "dispatch_paths": ["/home/dispatch", "/ha/dispatch", "/homeassistant/dispatch"],
            "has_instructions": True,
            "instructions_path": "/home/instructions",
        },
        "endpoints": {
            "chat_prepare": "/chat/prepare",
            "chat_events": "/chat/events/{chat_id}",
            "chat_result": "/chat/result/{chat_id}",
            "conversations": "/conversations",
            "conversation_messages": "/conversations/{conversation_id}/messages",
            "memory_list": "/memory/list",
            "memory_search": "/memory/search",
            "memory_add": "/memory/add",
            "memory_clear": "/memory/clear",
            "tts_prepare": "/tts_prepare",
            "tts_live": "/tts/live/{tts_id}.mp3",

            "sora_create": "/video",
            "sora_remix": "/video/remix",
            "sora_status": "/video/status/{job_id}",
            "sora_stream": "/video/stream/{job_id}",
            "sora_content": "/video/content/{video_id}",
            "intent_sora": "/rt/intent",
            "photo": "/solara/photo",
            "realtime_session": "/session",

            "home_bind": "/home/gateway_profile",
            "home_dispatch": "/home/dispatch",
        },
    }
# ---- Fix OpenAPI 500 (Pydantic v2 ForwardRef) ----
try:
    V2RegisterReq.model_rebuild()
    V2LoginReq.model_rebuild()
    V2CommentReq.model_rebuild()
    V2ShareReq.model_rebuild()
    V2RecoEventReq.model_rebuild()
    V2DMMessageReq.model_rebuild()
except Exception:
    pass
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
        access_log=False,
    )













