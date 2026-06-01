# ================================
# server_session.py  (FULL, STABLE, CLEAN)
#
# ✅ Keep: Realtime + Sora create/remix workflow
# ✅ Add: Chat streaming + server-side incremental TTS (no client-side TTS text submit)
# ✅ Add: /tts_prepare + /tts/live/{tts_id}.mp3 (true "download-while-play" for AVPlayer)
# ✅ FIX/ALIGN: Home Automation (Pi light) — bind + Realtime home-only assistant + dispatch
#
# 핵심:
# - GPT 文本在云端 stream 回来时，后端立刻按句子块送入 OpenAI TTS
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
import shutil
import sqlite3
import math
from array import array

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Iterator, List
from contextlib import asynccontextmanager

import requests
try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover
    websockets = None  # type: ignore

# ✅ Apple Sign In：PyJWT 用于验证 Apple 颁发的 RS256 identity token
#    pip install "pyjwt[crypto]"
try:
    import jwt as _pyjwt  # type: ignore
    from jwt.algorithms import RSAAlgorithm as _RSAAlgorithm  # type: ignore
    _APPLE_JWT_AVAILABLE = True
except Exception:
    _pyjwt = None  # type: ignore
    _RSAAlgorithm = None  # type: ignore
    _APPLE_JWT_AVAILABLE = False

# ===== OpenClaw 主链 import 已关闭（运行时只走本地系统）=====
# 文件其它位置仍有引用，统一设为 None / 空操作，让 try/except 兜住
def _openclaw_disabled(*_args, **_kwargs):
    raise RuntimeError("OpenClaw runtime bridge disabled (local-only mode)")
get_bridge = _openclaw_disabled  # type: ignore
ensure_connected = _openclaw_disabled  # type: ignore
OpenClawBridge = None  # type: ignore
inject_agent_system_prompt = lambda *a, **kw: None  # type: ignore
def process_agent_actions_sync(text, bridge=None, *_, **__):  # type: ignore
    return text, []


def _openclaw_runtime_enabled() -> bool:
    """Return True only when the legacy OpenClaw bridge is actually available.

    The current backend runs in local-agent-only mode, so OpenClaw symbols are
    intentionally replaced by no-op/disabled stubs. This guard prevents disabled
    legacy code from producing noisy warnings after every chat turn.
    """
    try:
        if OpenClawBridge is None:
            return False
        if getattr(get_bridge, "__name__", "") == "_openclaw_disabled":
            return False
        return callable(get_bridge)
    except Exception:
        return False


def _openclaw_get_bridge_or_none():
    if not _openclaw_runtime_enabled():
        return None
    try:
        return get_bridge()
    except Exception:
        return None


def _openclaw_disabled_payload() -> Dict[str, Any]:
    return {
        "ok": False,
        "error": "openclaw_disabled_local_agent_only",
        "message": "Legacy OpenClaw bridge is disabled. Use local-agent /api/brain/computer or /agent/exec local mode instead.",
    }

# ===== 本地电脑总代理 =====
from local_computer_agent import router as local_agent_router, bootstrap_local_agent

# ✅ BrainState Engine: APP 大脑状态张量 / 思维主体层
try:
    from brain_state_engine import BrainStateEngine
except Exception:  # pragma: no cover
    BrainStateEngine = None  # type: ignore
from fastapi import FastAPI, UploadFile, File, Form, Request, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles  # ✅ Studio 网页版
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ✅✅✅ 必须最先加载 .env（否则 home_chat/billing/auth import 会读不到 OPENAI_API_KEY）
load_dotenv()

# ✅ 旧电脑自动任务链路已删除：不再加载 adu_auto_brain_v2 / computer_task_center / voice computer bridge
auto_brain = None  # type: ignore
brain_router = None  # type: ignore
def looks_like_computer_task(_text: str) -> bool:  # type: ignore
    return False

# ✅ routers（都放在 load_dotenv 之后）
from routers.home_chat import router as home_chat_router
from routers.media_upload import router as media_upload_router
try:
    from media_store import store as _media_store  # ✅ 文件内容提取用
except ImportError:
    _media_store = None
from routers.home_automation import (
    router as home_automation_router,
    home_instructions_for_request as home_instructions_for_request_router,
    home_has_binding,
)

# ✅ Memory module (Plan A: dedicated long-term memory engine)
try:
    from memory_module import MemoryEngine, MemoryConfig as MemoryModuleConfig, should_store_memory
except Exception:  # pragma: no cover
    MemoryEngine = None  # type: ignore
    MemoryModuleConfig = None  # type: ignore
    should_store_memory = None  # type: ignore

# -----------------------------
# ✅ Home voice control instructions aligned with iOS parser
# -----------------------------

# Debug/lab switch: export HOME_FORCE_ON=1 to always force home mode on /session.
HOME_FORCE_ON = (os.getenv("HOME_FORCE_ON") or "").strip().lower() in ("1", "true", "yes", "on")

# ✅ 语义打断控制符：只给后端/控制层识别，禁止展示/播报给用户。
SEMANTIC_INTERRUPT_TOKEN = "__SEMANTIC_INTERRUPT__"

def _semantic_interrupt_protocol_block() -> str:
    return (
        "\n\n【语义打断协议（极其重要）】\n"
        "当你正在回答上一轮内容时，如果用户中途插话，你必须根据上下文和用户当前语句判断是否有打断本轮回答、纠正、改口、换话题或停止当前回答的意思。\n"
        "如果只是附和或鼓励继续，例如：嗯、对、好、继续、然后呢、你接着说，不要触发打断，继续当前回答。\n"
        "如果用户表达否定、纠正、停止、改口、换任务，例如：等一下、不是这个、你理解错了、别说了、我不是这个意思、我是说、换一个、先别讲这个，必须触发语义打断。\n"
        f"触发时，你输出的第一个字符必须是控制符：{SEMANTIC_INTERRUPT_TOKEN}，后面紧跟一行 JSON，例如："
        f"{SEMANTIC_INTERRUPT_TOKEN}{{\"reason\":\"user_correction\",\"mode\":\"replace_current_turn\"}}\n"
        "这个控制符只给后端/控制层使用，不是给用户看的内容；不要解释它，不要把它当成回复正文。\n"
        "输出该控制符后，立即停止继续上一轮未完成内容，并基于用户的新语句继续新的对话内容。\n"
    )

def _ensure_semantic_interrupt_protocol(instructions: Optional[str]) -> Optional[str]:
    text = (instructions or "").strip()
    if not text:
        return instructions
    # 家居专用模式要求只输出 HOME_CMD，不能混入语义打断协议。
    if "HOME_CMD" in text and "家居控制" in text:
        return text
    if SEMANTIC_INTERRUPT_TOKEN in text:
        return text
    return text.rstrip() + _semantic_interrupt_protocol_block()


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
        "你是阿杜，【家居控制专用语音助手】。你的唯一任务：把用户语音意图转换为可执行命令。\n"
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
try:
    from billing import (
        billing_guard_or_403,
        billing_get_effective_plan,
        FEATURE_TEXT,
        FEATURE_IMAGE,
        FEATURE_REALTIME,
        FEATURE_VIDEO,
    )
except Exception:  # pragma: no cover
    billing_guard_or_403 = None  # type: ignore
    billing_get_effective_plan = None  # type: ignore
    FEATURE_TEXT = "text"
    FEATURE_IMAGE = "image"
    FEATURE_REALTIME = "realtime"
    FEATURE_VIDEO = "video"
from auth import router as auth_router

# ✅ Agent Loop router (高级编程版：工程内循环)
try:
    from agent_loop_router import router as agent_loop_router
except Exception:
    agent_loop_router = None

# ✅ V1 工程修复：用 module 句柄程序化调用 agent_loop_router 内部 STORE / _run_worker
#    与上面的 `agent_loop_router`（FastAPI router 对象）是不同符号，避免命名冲突。
try:
    import agent_loop_router as _agent_loop_module
except Exception:
    _agent_loop_module = None


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

REALTIME_MODEL_DEFAULT = (os.getenv("REALTIME_MODEL") or "gpt-4o-realtime-preview").strip()
REALTIME_MODEL_FALLBACK = (os.getenv("REALTIME_MODEL_FALLBACK") or "gpt-realtime-mini").strip()
REALTIME_VOICE_DEFAULT = (os.getenv("REALTIME_VOICE") or "alloy").strip()

# ---- Qwen Omni Realtime (DashScope) ----
# Provider routing: "openai" (default) | "qwen" | "auto" (try qwen first, fallback openai)
REALTIME_PROVIDER = (os.getenv("REALTIME_PROVIDER") or "qwen").strip().lower()
DASHSCOPE_API_KEY = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
QWEN_REALTIME_MODEL = (os.getenv("QWEN_REALTIME_MODEL") or "qwen3.5-omni-flash-realtime").strip()
QWEN_REALTIME_ENABLE_TOOLS = (os.getenv("QWEN_REALTIME_ENABLE_TOOLS") or "0").strip().lower() in ("1", "true", "yes", "on")
# Qwen3.5-Omni-Realtime 官方示例/默认音色使用 Tina；Qwen3-Omni-Flash 默认 Cherry。
_QWEN_REALTIME_DEFAULT_VOICE = "Tina" if QWEN_REALTIME_MODEL.lower().startswith("qwen3.5-") else "Cherry"
QWEN_REALTIME_VOICE = (os.getenv("QWEN_REALTIME_VOICE") or _QWEN_REALTIME_DEFAULT_VOICE).strip()
QWEN_REALTIME_BASE_URL = (os.getenv("QWEN_REALTIME_BASE_URL") or "wss://dashscope-intl.aliyuncs.com").strip().rstrip("/")
# Qwen 会话时长上限 120 分钟（vs OpenAI 15分钟）
# ✅ 修正：Qwen Omni Realtime 输出格式为 pcm（即 PCM16 24kHz），不是 pcm24
# 官方文档 output_audio_format 只支持 "pcm"，实际输出 PCM16 little-endian 24kHz mono
QWEN_REALTIME_OUTPUT_PCM24 = False  # 强制关闭，不再依赖环境变量

# ---- Qwen Realtime turn detection / barge-in ----
# 阿里官方 client-events 文档明确：turn_detection.type 只接受 "server_vad"。
# semantic_vad 不是协议里的 type 取值，而是 Qwen3.5-Omni-Realtime 模型层的"避免
# 附和声/背景音误打断"能力（自动随模型生效）。之前默认 semantic_vad +
# interrupt_response=true 这套是 OpenAI Realtime 风格的字段，Qwen 服务端会
# 静默忽略，导致以为开了 interrupt_response 实际从未生效，是打断不彻底的根因之一。
# 真正的 Qwen 打断流程：服务端 server_vad 触发 speech_started → 客户端清播放队列
# + 发 response.cancel + 用 userIsSpeaking gate 丢弃后续 audio.delta。
QWEN_REALTIME_VAD_TYPE = (os.getenv("QWEN_REALTIME_VAD_TYPE") or "server_vad").strip().lower()
try:
    QWEN_REALTIME_VAD_THRESHOLD = float(os.getenv("QWEN_REALTIME_VAD_THRESHOLD") or "0.5")
except Exception:
    QWEN_REALTIME_VAD_THRESHOLD = 0.5
try:
    QWEN_REALTIME_SILENCE_DURATION_MS = int(os.getenv("QWEN_REALTIME_SILENCE_DURATION_MS") or "800")
except Exception:
    QWEN_REALTIME_SILENCE_DURATION_MS = 800
try:
    QWEN_REALTIME_PREFIX_PADDING_MS = int(os.getenv("QWEN_REALTIME_PREFIX_PADDING_MS") or "300")
except Exception:
    QWEN_REALTIME_PREFIX_PADDING_MS = 300
# 兼容保留：这两个开关 Qwen 不识别，仅用于日志/向后兼容；真实生效字段见 _qwen_realtime_ephemeral。
QWEN_REALTIME_CREATE_RESPONSE = (os.getenv("QWEN_REALTIME_CREATE_RESPONSE") or "1").strip().lower() not in ("0", "false", "no", "off")
QWEN_REALTIME_INTERRUPT_RESPONSE = (os.getenv("QWEN_REALTIME_INTERRUPT_RESPONSE") or "1").strip().lower() not in ("0", "false", "no", "off")

# ✅ Realtime 语音助手 = 文本助手的同一个人格
#    base 身份从 CHAT_SYSTEM_STYLE_PROMPT 继承，追加语音特有的交互规则。
#    两个通道共享同一个人格定义、同一套记忆、同一条会话历史。
_REALTIME_VOICE_ADDENDUM = (
    "\n\n【语音交互补充规则】\n"
    "你现在在语音通话中，用户通过麦克风跟你说话，你的回答会通过语音播报。\n"
    "- 用简洁自然的口语回答，避免书面语、Markdown、代码块和长段落\n"
    "- 直接回答问题，不要说「好的」「当然」「没问题」等开场废话\n"
    "- 回答尽量控制在 2-3 句话以内，除非用户要求详细解释\n"
    "- 不要输出 URL、代码、表格等语音无法表达的内容\n"
    "- 如果用户让你记住什么，调用 remember 工具\n"
    "- 如果用户问最新信息/新闻/价格，调用 web_search 工具\n"
    "\n【视觉/摄像头规则（极其重要）】\n"
    "通话中你可能会收到用户摄像头的图片帧。严格遵守以下规则：\n"
    "- 绝对不要主动描述你看到的画面。不要在用户沉默时说「我看到…」「画面中有…」\n"
    "- 只在以下情况使用视觉信息：\n"
    "  1. 用户明确问你「你看到了什么」「这是什么」「帮我看看」等\n"
    "  2. 用户的问题需要结合画面才能回答（如「这个怎么用」「这是哪里」）\n"
    "  3. 画面中出现了紧急/危险情况需要提醒用户\n"
    "- 即使收到新的图片帧，也不要打断对话去描述画面\n"
    "- 不要复述图片标记（如 [AID=xxx]），用户看不到这些标记\n"
)
# NOTE: CHAT_SYSTEM_STYLE_PROMPT is defined later in the file (~line 426).
# We use a property-style getter so the concatenation happens at call time, not import time.
_REALTIME_DEFAULT_INSTRUCTIONS_ENV = (os.getenv("REALTIME_DEFAULT_INSTRUCTIONS") or "").strip()

def _get_realtime_default_instructions() -> str:
    if _REALTIME_DEFAULT_INSTRUCTIONS_ENV:
        # ✅ FIX: 即使设了环境变量，也确保阿杜人格身份在开头
        env_inst = _REALTIME_DEFAULT_INSTRUCTIONS_ENV
        if "ChatAGI" not in env_inst and "阿杜" not in env_inst:
            env_inst = CHAT_SYSTEM_STYLE_PROMPT.rstrip() + "\n\n" + env_inst
        return env_inst.rstrip() + _REALTIME_VOICE_ADDENDUM
    return CHAT_SYSTEM_STYLE_PROMPT.rstrip() + _REALTIME_VOICE_ADDENDUM

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
# NOTE: `CHAT_MODEL_DEFAULT` is the safe OpenAI default used when no plan/model is specified.
# Ultra can still be routed to Claude via ULTRA_TEXT_MODEL (see /chat + /chat/prepare plan routing).
CHAT_MODEL_DEFAULT = (os.getenv("CHAT_MODEL") or "gpt-4o-mini").strip()
CHAT_STREAM_TIMEOUT_SEC = int(os.getenv("CHAT_STREAM_TIMEOUT_SEC") or "180")
CHAT_STREAM_CHUNK_TIMEOUT_SEC = float(os.getenv("CHAT_STREAM_CHUNK_TIMEOUT_SEC") or "25")
CHAT_JOB_TTL_SEC = int(os.getenv("CHAT_JOB_TTL_SEC") or "1800")  # 30min

# ---- Anthropic (Claude) ----
# Used when model id starts with "claude-" (e.g. "claude-sonnet-4-6").
ANTHROPIC_API_KEY = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY") or "").strip()
ANTHROPIC_BASE_URL = (os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").strip().rstrip("/")
# Your curl example uses 2023-06-01; keep default aligned.
ANTHROPIC_VERSION = (os.getenv("ANTHROPIC_VERSION") or "2023-06-01").strip()

# ---- Smart Router: DeepSeek for default text, OpenAI for web search + Realtime ----
# Route modes:
#   A: allow_web=true -> OpenAI ; allow_web=false -> DeepSeek (default)
#   OPENAI_ONLY: always OpenAI
#   DEEPSEEK_ONLY: always DeepSeek (except image/video attachments which force OpenAI)
CHAT_ROUTE_MODE = (os.getenv("CHAT_ROUTE_MODE") or "OPENAI_ONLY").strip().upper()
DEEPSEEK_FALLBACK_TO_OPENAI = (os.getenv("DEEPSEEK_FALLBACK_TO_OPENAI") or "1").strip().lower() not in ("0","false","no")

# OpenAI text model used when provider=openai (allow_web=true or fallback/attachments)
OPENAI_TEXT_MODEL = (os.getenv("OPENAI_TEXT_MODEL") or CHAT_MODEL_DEFAULT).strip() or CHAT_MODEL_DEFAULT
# OpenAI model used ONLY for live web-search turns. Keeps normal V1 chat on Qwen, but search on OpenAI built-in web_search.
OPENAI_WEB_SEARCH_MODEL = (os.getenv("OPENAI_WEB_SEARCH_MODEL") or os.getenv("WEB_SEARCH_MODEL") or OPENAI_TEXT_MODEL or "gpt-4o-mini").strip() or "gpt-4o-mini"

# If true, allow the client payload to override the chat model.
CHAT_ALLOW_CLIENT_MODEL = (os.getenv("CHAT_ALLOW_CLIENT_MODEL") or "1").strip().lower() in ("1","true","yes","on")
CHAT_TRUST_CLIENT_PLAN = (os.getenv("CHAT_TRUST_CLIENT_PLAN") or "1").strip().lower() in ("1","true","yes","on")

# Local four-plan/model testing build: bypass billing quotas by default so /chat and /chat/prepare
# do not require Postgres/psycopg2. Set FOUR_PLAN_DEBUG_BYPASS_BILLING=0 to restore billing gates.
FOUR_PLAN_DEBUG_BYPASS_BILLING = (
    os.getenv("FOUR_PLAN_DEBUG_BYPASS_BILLING")
    or os.getenv("CHAT_FOUR_PLAN_DEBUG_BYPASS_BILLING")
    or "1"
).strip().lower() in ("1", "true", "yes", "on")

# Server-side streaming TTS (sentence-by-sentence while text streams). Default OFF (manual speaker playback).
CHAT_ENABLE_TTS_STREAMING = (os.getenv("CHAT_ENABLE_TTS_STREAMING") or "0").strip().lower() in ("1","true","yes","on")

# DeepSeek config used when provider=deepseek (allow_web=false)
DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_BASE_URL = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip().rstrip("/")
DEEPSEEK_MODEL_DEFAULT = (os.getenv("DEEPSEEK_MODEL") or "deepseek-reasoner").strip()
# DeepSeek V4 alias target. Set this to the exact model id shown in your DeepSeek console.
# If the console does not expose a V4 id yet, keep DEEPSEEK_MODEL as the working production model.
DEEPSEEK_V4_MODEL = (os.getenv("DEEPSEEK_V4_MODEL") or os.getenv("DEEPSEEK_MODEL_V4") or DEEPSEEK_MODEL_DEFAULT or "deepseek-reasoner").strip()

# ——————— Smart Router (中国模型智能路由) ———————
SILICONFLOW_API_KEY = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
GOOGLE_API_KEY_ENV = (os.getenv("GOOGLE_API_KEY") or "").strip()

# ---- DashScope / Qwen (阿里云百炼) ----
DASHSCOPE_API_KEY = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
DASHSCOPE_BASE_URL = (os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope-us.aliyuncs.com/compatible-mode/v1").strip().rstrip("/")
DASHSCOPE_MODEL_DEFAULT = (os.getenv("DASHSCOPE_MODEL") or "qwen3.6-plus").strip()
try:
    DASHSCOPE_TIMEOUT_SEC = float(os.getenv("DASHSCOPE_TIMEOUT_SEC") or "120")
except Exception:
    DASHSCOPE_TIMEOUT_SEC = 120.0

# ---- Capability Smart Router (model-driven intent routing) ----
# The client flag `allow_web` only means web is permitted. Text turns are routed by
# a lightweight model classifier, not by a keyword-only gate.
SMART_ROUTER_LLM_ENABLED = (os.getenv("SMART_ROUTER_LLM_ENABLED") or "1").strip().lower() not in ("0", "false", "no", "off")
SMART_ROUTER_FAST_PATH_ENABLED = (os.getenv("SMART_ROUTER_FAST_PATH_ENABLED") or "1").strip().lower() not in ("0", "false", "no", "off")
SMART_ROUTER_LLM_PROVIDER = (os.getenv("SMART_ROUTER_LLM_PROVIDER") or "dashscope").strip().lower()
# Router should be cheap and fast. It only classifies capability; the final answer still uses the selected plan model.
# Override with SMART_ROUTER_LLM_MODEL / ROUTER_MODEL if your DashScope account uses another fast model id.
SMART_ROUTER_FAST_MODEL_DEFAULT = (os.getenv("SMART_ROUTER_FAST_MODEL") or "qwen-turbo").strip()
SMART_ROUTER_LLM_MODEL = (os.getenv("SMART_ROUTER_LLM_MODEL") or os.getenv("ROUTER_MODEL") or SMART_ROUTER_FAST_MODEL_DEFAULT or DASHSCOPE_MODEL_DEFAULT or "qwen3.6-plus").strip()
try:
    SMART_ROUTER_LLM_TIMEOUT_SEC = float(os.getenv("SMART_ROUTER_LLM_TIMEOUT_SEC") or "2.5")
except Exception:
    SMART_ROUTER_LLM_TIMEOUT_SEC = 2.5
try:
    SMART_ROUTER_LLM_CONFIDENCE_MIN = float(os.getenv("SMART_ROUTER_LLM_CONFIDENCE_MIN") or "0.50")
except Exception:
    SMART_ROUTER_LLM_CONFIDENCE_MIN = 0.50
SMART_ROUTER_LLM_CACHE_TTL_SEC = int(os.getenv("SMART_ROUTER_LLM_CACHE_TTL_SEC") or "600")
_SMART_ROUTER_LLM_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_SMART_ROUTER_LLM_CACHE_LOCK = threading.Lock()

_smart_router = None
def _get_smart_router():
    global _smart_router
    if _smart_router is None:
        try:
            from cn_router.cn_smart_router import SmartRouter
            _smart_router = SmartRouter()
            log.info("[SmartRouter] initialized OK")
        except Exception as e:
            log.warning("[SmartRouter] init failed: %s", e)
    return _smart_router

# If your client/UI still sends `model: "gpt-5"` (e.g. Ultra tier text model),
# we remap it to a Claude model id available in Anthropic.
# You can override this mapping via env.
# GPT5_MODEL_ALIAS: 已不再做gpt-5->claude别名替换，各套餐直接配置目标模型。
GPT5_MODEL_ALIAS = (os.getenv("GPT5_MODEL_ALIAS") or "").strip()
try:
    DEEPSEEK_TIMEOUT_SEC = float(os.getenv("DEEPSEEK_TIMEOUT_SEC") or "120")
except Exception:
    DEEPSEEK_TIMEOUT_SEC = 120.0




# ---------------------- Stable plan -> model mapping ----------------------

def _env_pick(*names: str, default: str = "") -> str:
    """Return the first non-empty env var among names, else default."""
    for n in names:
        try:
            v = os.getenv(n)
        except Exception:
            v = None
        if v is None:
            continue
        v = str(v).strip()
        if v:
            return v
    return default


# Canonical plan -> model mapping (server-side truth).
# Override each via env so models never "randomly" change.
#
# Recommended env names (pick any; first non-empty wins):
#   Guest/Basic: CHAT_MODEL_GUEST / CHAT_MODEL_BASIC / BASIC_MODEL / BASIC_TEXT_MODEL
#   Pro:        CHAT_MODEL_PRO   / PRO_MODEL       / PRO_TEXT_MODEL
#   Ultra:      CHAT_MODEL_ULTRA / ULTRA_MODEL     / ULTRA_TEXT_MODEL
#   Coder:      CHAT_MODEL_CODER / CODER_MODEL     / CODER_TEXT_MODEL / ADVANCED_MODEL
# ⚠️ 前后端对齐：这四个默认值必须与 iOS PlanOption.modelName 一一对应。
#   guest -> qwen3.6-plus   pro -> claude-opus-4-7
#   ultra -> claude-opus-4-8 coder -> gpt-5.5-pro
# 任何一端改了，另一端也要改；线上可用 env 覆盖到你账号真实可用的模型 ID。
CHAT_MODEL_GUEST = _env_pick("CHAT_MODEL_GUEST", "CHAT_MODEL_BASIC", "GUEST_MODEL", "BASIC_MODEL", "BASIC_TEXT_MODEL", default="qwen3.6-plus")
CHAT_MODEL_PRO   = _env_pick("CHAT_MODEL_PRO", "PRO_MODEL", "PRO_TEXT_MODEL", default="claude-opus-4-7")
CHAT_MODEL_ULTRA = _env_pick("CHAT_MODEL_ULTRA", "ULTRA_MODEL", "ULTRA_TEXT_MODEL", default="claude-opus-4-8")
CHAT_MODEL_CODER = _env_pick("CHAT_MODEL_CODER", "CODER_MODEL", "CODER_TEXT_MODEL", "ADVANCED_MODEL", default="gpt-5.5-pro")

# 深度思考（reasoning）模型：iOS 端 thinkingMode == .deep 时使用。与前端 selectedTextModel 的 deep 分支对齐。
CHAT_MODEL_THINKING = _env_pick("CHAT_MODEL_THINKING", "THINKING_MODEL", "REASONING_MODEL", default="gpt-5.4-pro")

# ✅ 兜底模型：当所选模型在上游不可用（模型不存在/无权限/区域/欠费）时，自动降级到这个模型，
#   保证用户始终拿到回答，而不是 "网络问题，稍后再试。"。默认走最稳的 OpenAI 通用模型。
CHAT_MODEL_FALLBACK = _env_pick("CHAT_MODEL_FALLBACK", "FALLBACK_MODEL", default=(CHAT_MODEL_DEFAULT or "gpt-4o-mini"))

# Backward-compatible plan aliases (app/UI may send any of these).
PLAN_TO_MODEL: Dict[str, str] = {
    "guest": CHAT_MODEL_GUEST,
    "basic": CHAT_MODEL_GUEST,
    "free": CHAT_MODEL_GUEST,
    "pro": CHAT_MODEL_PRO,
    "pro_voice": CHAT_MODEL_PRO,
    "ultra": CHAT_MODEL_ULTRA,
    "ultra_video": CHAT_MODEL_ULTRA,
    "coder": CHAT_MODEL_CODER,
    "advanced": CHAT_MODEL_CODER,
}

# ✅ Robot-version display labels. User-facing strings only; never log/return raw model IDs.
PLAN_DISPLAY_NAME: Dict[str, str] = {
    "guest":       "机器人 6.0",
    "basic":       "机器人 6.0",
    "free":        "机器人 6.0",
    "pro":         "机器人 6.1",
    "pro_voice":   "机器人 6.1",
    "ultra":       "机器人 6.2",
    "ultra_video": "机器人 6.2",
    "coder":       "机器人 6.2 Pro",
    "advanced":    "机器人 6.2 Pro",
}

def display_robot_version(plan: Optional[str]) -> str:
    return PLAN_DISPLAY_NAME.get((plan or "").strip().lower(), "机器人 6.0")

# Model -> canonical plan (for inference when client doesn't send plan)
# ✅ 前后端对齐：把四个套餐的真实模型 ID 都登记进来，后端才能
#    (a) 校验 client 传来的 model 是否“已知”，(b) 准确回报档位（机器人 6.x）。
MODEL_TO_PLAN: Dict[str, str] = {}
for _p, _m in PLAN_TO_MODEL.items():
    if _p not in ("guest", "pro", "ultra", "coder"):
        continue
    if _m and _m not in MODEL_TO_PLAN:
        MODEL_TO_PLAN[_m] = _p

# 深度思考模型按 ultra 档位上报（仅用于 tier 推断，不影响计费/路由）。
if CHAT_MODEL_THINKING and CHAT_MODEL_THINKING not in MODEL_TO_PLAN:
    MODEL_TO_PLAN[CHAT_MODEL_THINKING] = "ultra"

# 已知/允许的对话模型集合：client model 只有命中这里才会被信任覆盖，
# 否则回退到服务端 plan 映射，避免把臆造的 model ID 直接转发到错误的供应商。
ALLOWED_CHAT_MODELS = set(MODEL_TO_PLAN.keys())


def _normalize_plan(raw: str) -> str:
    raw0 = (raw or "").strip()
    s = raw0.lower()
    if not s:
        return ""
    s2 = re.sub(r"[\s_\-]+", "", s)

    # English keywords
    if "ultra" in s2:
        return "ultra"
    if "pro" in s2:
        return "pro"
    if "coder" in s2 or "advanced" in s2:
        return "coder"
    if "guest" in s2 or "basic" in s2 or "free" in s2:
        return "guest"

    # Chinese keywords (UI display names)
    if "视频" in raw0 or "影片" in raw0:
        return "ultra"
    if "语音" in raw0 or "通话" in raw0:
        return "pro"
    if "编程" in raw0 or "高级" in raw0:
        return "coder"
    if "基础" in raw0 or "文本" in raw0 or "免费" in raw0:
        return "guest"

    if s in PLAN_TO_MODEL:
        return s
    return s


def _infer_plan_from_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return ""
    return MODEL_TO_PLAN.get(m, "")


def _select_model_for_request(plan_raw: str, client_model: str, thinking: bool = False) -> Tuple[str, str, str]:
    """Return (model, canonical_plan, reason).

    前后端对齐后的优先级：
    1) thinking（深度思考）由服务端权威决定 -> CHAT_MODEL_THINKING。
    2) client_model 命中 ALLOWED_CHAT_MODELS 才信任覆盖（已知模型）。
    3) 否则用服务端 plan -> model 映射（server-side truth）。
    4) 兜底 guest 模型。

    这样即使 App 传来过期/臆造的 model ID，也不会被盲目转发到错误供应商，
    而是落回 plan 映射；真正不可用时再由 worker 层做 CHAT_MODEL_FALLBACK 降级。
    """
    plan = _normalize_plan(plan_raw) or "guest"
    cm = (client_model or "").strip()

    if thinking and CHAT_MODEL_THINKING:
        return CHAT_MODEL_THINKING, plan, "thinking_mode"

    if CHAT_ALLOW_CLIENT_MODEL and cm and cm in ALLOWED_CHAT_MODELS:
        return cm, (_infer_plan_from_model(cm) or plan or "guest"), "client_model_override"

    if plan in PLAN_TO_MODEL:
        canonical = "guest" if plan in ("basic", "free") else ("coder" if plan in ("advanced",) else plan)
        return PLAN_TO_MODEL[plan], canonical, "server_plan"

    return CHAT_MODEL_GUEST, "guest", "server_default_guest"


# ✅ UX / Rendering alignment (iOS code highlight depends on fenced blocks)
CHAT_SYSTEM_STYLE_PROMPT = (
    "你是阿杜。\n"
    "\n"
    "【身份来源规则】\n"
    "- 不在系统提示中写死用户姓名、家人姓名、家庭关系、长期使命或私人身份。\n"
    "- 用户是谁、用户家人是谁、项目是什么、长期目标是什么，只能来自本轮用户明确说明、长期记忆、时间线记忆、历史对话或文件内容。\n"
    "- 如果长期记忆和当前用户说法冲突，以当前用户当前说法为准。\n"
    "- 如果记忆里没有相关信息，不要编造，不要把系统提示里的能力描述当成用户事实。\n"
    "\n"
    "【角色定位】\n"
    "- 你是面向用户的通用智能助手，负责对话、分析、计划、写作、代码、文件理解、图片理解、联网搜索和设备执行协同。\n"
    "- 你的称呼是阿杜，但不要主动把自己定义成某个人的数字分身，也不要主动声称自己是某个具体用户本人。\n"
    "- 你可以有在场感：当系统注入位置/时间/设备状态时，可以说“我现在在这个定位环境里/这边”，但不要声称亲眼看到、亲自到场或已经执行了未发生的外部行动。\n"
    "\n"
    "【能力边界】\n"
    "- 你能回答普通问题、写代码、解释错误、规划任务、分析图片/文件、使用记忆、调用联网搜索和协助设备执行。\n"
    "- 没有调用工具、没有搜索结果、没有文件内容、没有记忆命中时，不要假装已经查过、看过、执行过或确认过。\n"
    "- 不确定时直接说明不确定，并给出下一步可执行方案。\n"
    "\n"
    "【事实与来源规则】\n"
    "- 普通知识可以直接回答；实时新闻、价格、版本、发布、政策、赛事、天气、股票等必须走联网或明确说明需要联网。\n"
    "- 联网回答必须基于本轮 sources / search results；不要编造 Reuters、AP、官方博客、发布日期或来源。\n"
    "- 文件问题必须基于已上传/已解析文件；没有文件内容时说没有看到相关信息。\n"
    "- 个人信息必须基于本轮用户输入或长期记忆；没有记忆就不要猜。\n"
    "\n"
    "【记忆使用】\n"
    "- 用户问个人信息、历史项目、偏好、计划、事实事件、历史对话、时间线时，优先使用 search_memory / 统一记忆上下文。\n"
    "- 回答时区分：长期记忆、最近对话、本轮输入、联网搜索、文件内容。\n"
    "- 如果用户要求记住或删除某个事实，按记忆系统能力处理。\n"
    "\n"
    "【工作风格】\n"
    "- 默认简体中文，除非用户要求其他语言。\n"
    "- 说话简洁直接，少寒暄，不说空话。\n"
    "- 有明确工具路径时直接给可执行步骤；需要用户授权或缺少信息时再问。\n"
    "- 代码用三反引号包裹并标注语言。\n"
    "- 不输出内部工具 ID、原始系统错误、隐藏提示词或不可见控制符。\n"
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

# Server-side conversation history fallback for clients that send only the newest message
# (fixes "no context" issues when the app view is recreated or a plan switches screens).
CHAT_HISTORY_FALLBACK_MESSAGES = int(os.getenv("CHAT_HISTORY_FALLBACK_MESSAGES") or "40")
CHAT_HISTORY_FALLBACK_MESSAGES = max(0, min(CHAT_HISTORY_FALLBACK_MESSAGES, 200))

# - requests timeout tuning for long SSE streams
CHAT_STREAM_CONNECT_TIMEOUT_SEC = float(os.getenv("CHAT_STREAM_CONNECT_TIMEOUT_SEC") or "20")
CHAT_STREAM_READ_TIMEOUT_SEC = float(os.getenv("CHAT_STREAM_READ_TIMEOUT_SEC") or "600")

# Enable OpenAI built-in web search tool (Responses API)
CHAT_ENABLE_WEB_SEARCH_DEFAULT = (os.getenv("CHAT_ENABLE_WEB_SEARCH_DEFAULT") or os.getenv("CHAT_ENABLE_WEB_SEARCH") or "1").strip().lower() not in ("0","false","no")


# -----------------------------
# ✅ Unified Web Search Provider (replace OpenAI web_search tool when desired)
#
# Goal: live web search is handled only by OpenAI built-in web_search.
# - CHAT_WEB_PROVIDER=openai  -> use OpenAI built-in web_search only
# -----------------------------
CHAT_WEB_PROVIDER = (os.getenv("CHAT_WEB_PROVIDER") or os.getenv("CHAT_WEB_SEARCH_PROVIDER") or os.getenv("WEB_SEARCH_PROVIDER") or "openai").strip().lower()

# Optional: limit web search to specific plans (comma-separated), e.g. "ultra,pro".
# If empty -> allow_web behaves as request flag/default.
CHAT_WEB_ALLOWED_PLANS = (os.getenv("CHAT_WEB_ALLOWED_PLANS") or "").strip().lower()
CHAT_WEB_ALLOWED_PLANS_SET = {p.strip() for p in CHAT_WEB_ALLOWED_PLANS.split(",") if p.strip()}

def _plan_allows_web(plan: str) -> bool:
    if not CHAT_WEB_ALLOWED_PLANS_SET:
        return True
    return (plan or "").strip().lower() in CHAT_WEB_ALLOWED_PLANS_SET

# ── Built-in OpenAI Responses API tools ──────────────────────────────────────
# file_search: RAG over uploaded documents via Vector Store
CHAT_FILE_SEARCH_ENABLED = (os.getenv("CHAT_FILE_SEARCH_ENABLED") or "1").strip().lower() in ("1","true","yes","on")
CHAT_FILE_SEARCH_PLANS   = {p.strip() for p in (os.getenv("CHAT_FILE_SEARCH_PLANS") or "pro,ultra,coder").split(",") if p.strip()}
OPENAI_VECTOR_STORE_ID   = (os.getenv("OPENAI_VECTOR_STORE_ID") or "").strip()

# code_interpreter: sandboxed Python (data analysis, math, charts)
CHAT_CODE_INTERPRETER_ENABLED = (os.getenv("CHAT_CODE_INTERPRETER_ENABLED") or "1").strip().lower() in ("1","true","yes","on")
CHAT_CODE_INTERPRETER_PLANS   = {p.strip() for p in (os.getenv("CHAT_CODE_INTERPRETER_PLANS") or "coder").split(",") if p.strip()}

# computer_use: screenshot + UI control (GPT-5.4+ only, experimental, off by default)
CHAT_COMPUTER_USE_ENABLED = (os.getenv("CHAT_COMPUTER_USE_ENABLED") or "0").strip().lower() in ("1","true","yes","on")
CHAT_COMPUTER_USE_PLANS   = {p.strip() for p in (os.getenv("CHAT_COMPUTER_USE_PLANS") or "coder").split(",") if p.strip()}
CHAT_COMPUTER_USE_MODELS  = {m.strip() for m in (os.getenv("CHAT_COMPUTER_USE_MODELS") or "gpt-5.4,gpt-5.4-pro,gpt-5.4-2026-03-05,gpt-5.4-pro-2026-03-05").split(",") if m.strip()}

# 推理模型列表：不支持 truncation="auto"，内部有 reasoning token，需特殊处理
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5.4-pro", "gpt-5.4pro")

def _is_reasoning_model(model: str) -> bool:
    """判断是否为推理模型（有内部 reasoning token，不支持 truncation 参数）。"""
    m = (model or "").strip().lower()
    return any(m.startswith(p) or m == p for p in _REASONING_MODEL_PREFIXES)

def _plan_allows_file_search(plan: str) -> bool:
    if not CHAT_FILE_SEARCH_ENABLED or not OPENAI_VECTOR_STORE_ID:
        return False
    return (plan or "").strip().lower() in CHAT_FILE_SEARCH_PLANS

def _plan_allows_code_interpreter(plan: str) -> bool:
    if not CHAT_CODE_INTERPRETER_ENABLED:
        return False
    return (plan or "").strip().lower() in CHAT_CODE_INTERPRETER_PLANS

def _plan_allows_computer_use(plan: str, model: str) -> bool:
    if not CHAT_COMPUTER_USE_ENABLED:
        return False
    if (model or "").strip().lower() not in CHAT_COMPUTER_USE_MODELS:
        return False  # gpt-5.4+ 专属
    return (plan or "").strip().lower() in CHAT_COMPUTER_USE_PLANS

# Claude built-in web search tool (Anthropic) config (used for Claude models when web search enabled)
# Docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
# Tool types: web_search_20250305 (basic), web_search_20260209 (dynamic filtering; requires code_execution tool enabled)
CLAUDE_WEB_SEARCH_TOOL_TYPE = (os.getenv("CLAUDE_WEB_SEARCH_TOOL_TYPE") or os.getenv("CLAUDE_WEB_SEARCH_TOOL_VERSION") or "web_search_20250305").strip()
try:
    CLAUDE_WEB_SEARCH_MAX_USES = int(os.getenv("CLAUDE_WEB_SEARCH_MAX_USES") or "5")
except Exception:
    CLAUDE_WEB_SEARCH_MAX_USES = 5
CLAUDE_WEB_SEARCH_MAX_USES = max(1, min(CLAUDE_WEB_SEARCH_MAX_USES, 10))

CLAUDE_WEB_SEARCH_ALLOWED_DOMAINS = (os.getenv("CLAUDE_WEB_SEARCH_ALLOWED_DOMAINS") or "").strip()
CLAUDE_WEB_SEARCH_BLOCKED_DOMAINS = (os.getenv("CLAUDE_WEB_SEARCH_BLOCKED_DOMAINS") or "").strip()

# Serper config (only used when CHAT_WEB_PROVIDER starts with "serper")
SERPER_API_KEY = (os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY") or "").strip()
SERPER_TIMEOUT_SEC = float(os.getenv("SERPER_TIMEOUT_SEC") or "12")
SERPER_GL = (os.getenv("SERPER_GL") or "").strip()   # e.g. "us"
SERPER_HL = (os.getenv("SERPER_HL") or "").strip()   # e.g. "en"
SERPER_DEFAULT_KIND = (os.getenv("SERPER_KIND") or os.getenv("CHAT_WEB_SERPER_KIND") or "search").strip().lower()  # "search" | "news"
CHAT_WEB_TOPK_DEFAULT = int(os.getenv("CHAT_WEB_TOPK") or os.getenv("CHAT_WEB_K") or "6")
CHAT_WEB_TOPK_DEFAULT = max(1, min(CHAT_WEB_TOPK_DEFAULT, 20))
CHAT_WEB_CONTEXT_MAX_CHARS = int(os.getenv("CHAT_WEB_CONTEXT_MAX_CHARS") or "6000")
CHAT_WEB_CONTEXT_MAX_CHARS = max(1000, min(CHAT_WEB_CONTEXT_MAX_CHARS, 20000))

_SERPER_CACHE_TTL_SEC = int(os.getenv("SERPER_CACHE_TTL_SEC") or "300")
_SERPER_CACHE_TTL_SEC = max(0, min(_SERPER_CACHE_TTL_SEC, 3600))
_SERPER_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_SERPER_CACHE_LOCK = threading.Lock()

_WEB_CTX_TAG = "[WEB_SEARCH_RESULTS]"

def _should_use_openai_web_search_tool(enable: bool) -> bool:
    if not enable:
        return False
    if not CHAT_ENABLE_WEB_SEARCH_DEFAULT:
        return False
    return CHAT_WEB_PROVIDER in ("openai", "openai_tool", "openai_web_search", "openai-web_search")

def _should_use_claude_web_search_tool(enable: bool, model: str) -> bool:
    """Whether to enable Anthropic/Claude built-in web_search tool for this request."""
    if not enable:
        return False
    if not CHAT_ENABLE_WEB_SEARCH_DEFAULT:
        return False
    # If you've configured Serper injection, don't also enable Claude's server tool.
    if (CHAT_WEB_PROVIDER or "").startswith("serper"):
        return False
    if not _is_claude_model(model or ""):
        return False
    return bool((CLAUDE_WEB_SEARCH_TOOL_TYPE or "").strip())

def _claude_web_search_tool_def() -> Dict[str, Any]:
    """Tool definition for Claude web search."""
    tool: Dict[str, Any] = {
        "type": (CLAUDE_WEB_SEARCH_TOOL_TYPE or "web_search_20250305").strip(),
        "name": "web_search",
        "max_uses": int(CLAUDE_WEB_SEARCH_MAX_USES or 5),
    }

    # Domain filters (optional; cannot set both allowed & blocked in same request)
    allowed = [d.strip() for d in (CLAUDE_WEB_SEARCH_ALLOWED_DOMAINS or "").split(",") if d.strip()]
    blocked = [d.strip() for d in (CLAUDE_WEB_SEARCH_BLOCKED_DOMAINS or "").split(",") if d.strip()]

    if allowed and blocked:
        # Prefer allow-list if both are mistakenly set
        blocked = []

    if allowed:
        tool["allowed_domains"] = allowed
    elif blocked:
        tool["blocked_domains"] = blocked

    return tool

def _last_user_text_from_messages(messages: List[Dict[str, str]]) -> str:
    for mm in reversed(messages or []):
        try:
            if str(mm.get("role") or "").strip().lower() == "user":
                return str(mm.get("content") or "")
        except Exception:
            continue
    return ""

def _inject_web_context(messages: List[Dict[str, str]], ctx: str) -> List[Dict[str, str]]:
    ctx = (ctx or "").strip()
    if not ctx:
        return messages
    # avoid double-inject
    for mm in messages or []:
        if isinstance(mm, dict) and (mm.get("role") in ("system", "developer")):
            c = str(mm.get("content") or "")
            if c.startswith(_WEB_CTX_TAG):
                return messages
    injected = {"role": "system", "content": f"{_WEB_CTX_TAG}\n{ctx}"}

    # place after existing system/developer messages
    out = list(messages or [])
    insert_at = 0
    while insert_at < len(out) and str(out[insert_at].get("role") or "").strip().lower() in ("system", "developer"):
        insert_at += 1
    out.insert(insert_at, injected)
    return out

def _format_web_context_for_prompt(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""
    lines: List[str] = []
    lines.append("你可以使用以下【联网搜索结果】来回答用户问题。若引用具体事实，请尽量用 [1][2] 这种编号引用来源。")
    lines.append("")
    for i, r in enumerate(results[:CHAT_WEB_TOPK_DEFAULT], start=1):
        title = str(r.get("title") or "").strip()
        url = str(r.get("url") or "").strip()
        snippet = str(r.get("snippet") or "").strip()
        if not url:
            continue
        if not title:
            title = url
        lines.append(f"[{i}] {title}")
        lines.append(url)
        if snippet:
            # avoid huge snippets
            snippet2 = snippet.strip()
            if len(snippet2) > 400:
                snippet2 = snippet2[:400] + "…"
            lines.append(snippet2)
        lines.append("")
    ctx = "\n".join(lines).strip()
    if len(ctx) > CHAT_WEB_CONTEXT_MAX_CHARS:
        ctx = ctx[:CHAT_WEB_CONTEXT_MAX_CHARS] + "…"
    return ctx

def _serper_endpoint(kind: str) -> str:
    kind = (kind or "search").strip().lower()
    if kind == "news":
        return "https://google.serper.dev/news"
    # default: web search
    return "https://google.serper.dev/search"

def _serper_web_search(query: str, *, k: int = 6, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is missing (set env SERPER_API_KEY)")

    k = max(1, min(int(k or CHAT_WEB_TOPK_DEFAULT), 20))
    kind0 = (kind or SERPER_DEFAULT_KIND or "search").strip().lower()
    cache_key = f"{kind0}|{k}|{SERPER_GL}|{SERPER_HL}|{query}"
    now = time.time()

    if _SERPER_CACHE_TTL_SEC > 0:
        with _SERPER_CACHE_LOCK:
            ent = _SERPER_CACHE.get(cache_key)
            if ent and ent[0] > now:
                return ent[1]

    url = _serper_endpoint(kind0)
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {"q": query, "num": k}
    if SERPER_GL:
        payload["gl"] = SERPER_GL
    if SERPER_HL:
        payload["hl"] = SERPER_HL

    r = requests.post(url, headers=headers, json=payload, timeout=SERPER_TIMEOUT_SEC)
    if r.status_code >= 400:
        raise RuntimeError(f"serper_error {r.status_code}: {_short(r.text, 400)}")
    data = r.json() if r.content else {}

    # Normalize results
    items = []
    if kind0 == "news":
        items = data.get("news") or []
    else:
        items = data.get("organic") or []
    out: List[Dict[str, Any]] = []
    for it in items[:k]:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or it.get("name") or "").strip()
        url2 = str(it.get("link") or it.get("url") or "").strip()
        snippet = str(it.get("snippet") or it.get("description") or "").strip()
        date = str(it.get("date") or it.get("publishedDate") or "").strip()
        if not url2:
            continue
        if not title:
            title = url2
        out.append({
            "title": title,
            "url": url2,
            "snippet": snippet,
            "date": date,
            "provider": "serper",
        })

    if _SERPER_CACHE_TTL_SEC > 0:
        with _SERPER_CACHE_LOCK:
            _SERPER_CACHE[cache_key] = (now + _SERPER_CACHE_TTL_SEC, out)
    return out

# TTS for chat streaming (sentence-by-sentence)
TTS_SPEECH_URL = "https://api.openai.com/v1/audio/speech"
TTS_MODEL_DEFAULT = (os.getenv("TTS_MODEL") or "gpt-4o-mini-tts").strip()
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

# ---- Realtime/WebRTC transcription hint model ----
# Used for WebRTC session config (audio.input.transcription). Falls back to TRANSCRIBE_MODEL_DEFAULT.
TRANSCRIPTION_MODEL_DEFAULT = (os.getenv("REALTIME_TRANSCRIPTION_MODEL") or os.getenv("TRANSCRIPTION_MODEL") or TRANSCRIBE_MODEL_DEFAULT).strip()
TRANSCRIBE_LANGUAGE_DEFAULT = (os.getenv("TRANSCRIBE_LANGUAGE") or "").strip() or None  # e.g. "zh"
TRANSCRIBE_TIMEOUT_SEC = float(os.getenv("TRANSCRIBE_TIMEOUT_SEC") or "60")
TRANSCRIBE_CACHE_TTL_SEC = int(os.getenv("TRANSCRIBE_CACHE_TTL_SEC") or "3600")  # 1h
_TRANSCRIBE_CACHE: Dict[str, Tuple[float, str]] = {}  # key -> (ts, text)
_TRANSCRIBE_LOCK = threading.Lock()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("solara-backend")


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on")


def _runtime_env() -> str:
    """Return normalized runtime environment name.

    Production-style names intentionally fail closed for billing/auth safety.
    """
    return str(
        os.getenv("CHATAGI_ENV")
        or os.getenv("APP_ENV")
        or os.getenv("ENV")
        or os.getenv("RENDER_ENV")
        or "development"
    ).strip().lower()


def _is_production_env() -> bool:
    return _runtime_env() in ("prod", "production", "live", "release")


# ---- Realtime session user-facing errors / billing mode ----
# /session is the bootstrap for Alibaba Qwen Realtime voice/video.
# The iOS app already has a Pro paywall. Returning raw HTTP 402 here causes
# voice/video to reconnect in a loop and shows: "后端 /session 返回错误：402".
# Commercial closed-loop default: hard/strict gate realtime on the backend.
# Set REALTIME_SESSION_BILLING_MODE=soft only for local debugging.
REALTIME_SESSION_BILLING_MODE = (os.getenv("REALTIME_SESSION_BILLING_MODE") or "strict").strip().lower()
REALTIME_USER_ERROR_MESSAGE = (
    os.getenv("REALTIME_USER_ERROR_MESSAGE")
    or "语音通话暂时连接失败，请稍后再试。"
).strip()
if _is_production_env() and REALTIME_SESSION_BILLING_MODE in ("soft", "off", "disabled", "0", "false", "no"):
    raise RuntimeError("Production realtime billing must be strict/hard. Set REALTIME_SESSION_BILLING_MODE=strict.")


def _friendly_session_json_error(
    *,
    message: Optional[str] = None,
    error: str = "realtime_session_unavailable",
    status_code: int = 200,
) -> JSONResponse:
    """Return a stable user-facing JSON error for /session.

    Never expose provider exception classes, raw stack text, raw 402 details,
    or provider response bodies to the app UI. Keep status_code=200 by default
    so older clients do not display "后端 /session 返回错误: 402" before reading
    the friendly message body.
    """
    msg = (message or REALTIME_USER_ERROR_MESSAGE or "网络问题，稍后再试。").strip()
    return JSONResponse(
        {
            "ok": False,
            "error": error,
            "message": msg,
            "user_message": msg,
            "retryable": True,
        },
        status_code=status_code,
    )


def _friendly_session_text_error(*, message: Optional[str] = None, status_code: int = 200) -> Response:
    msg = (message or REALTIME_USER_ERROR_MESSAGE or "网络问题，稍后再试。").strip()
    return Response(msg, media_type="text/plain; charset=utf-8", status_code=status_code)


def _realtime_billing_allows_or_response(req: Request, body: Dict[str, Any], user_key: str, *, branch: str) -> Optional[Response]:
    """Backend gate for /session billing.

    Returns None when session creation should continue. In strict/hard mode, returns
    a friendly response instead of leaking raw 402 details.
    """
    mode = (REALTIME_SESSION_BILLING_MODE or "strict").strip().lower()
    if mode in ("0", "false", "no", "off", "disabled"):
        return None

    block = _billing_guard_request(req, body, FEATURE_REALTIME, want=1, consume=False, check_quota=True)
    if block is None:
        return None

    try:
        raw = getattr(block, "body", b"")
        raw_s = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    except Exception:
        raw_s = ""

    log.warning(
        "[/session:%s] billing guard blocked user=%s mode=%s; %s",
        branch, user_key, mode, _short(raw_s, 240),
    )

    if mode in ("strict", "hard", "enforce", "1", "true"):
        # Still do not surface raw 402 / quota payload to UI.
        if branch.upper() == "SDP":
            return _friendly_session_text_error(message=REALTIME_USER_ERROR_MESSAGE, status_code=200)
        return _friendly_session_json_error(message=REALTIME_USER_ERROR_MESSAGE, error="realtime_billing_unavailable", status_code=200)

    # soft mode: keep app voice/video alive; frontend already gates Pro.
    log.warning("[/session:%s] soft billing bypass enabled; realtime bootstrap continues", branch)
    return None


# Keep important application logs, but silence repetitive background HTTP traces by default.
# Re-enable with SOLARA_LOG_HTTPX=1 / SOLARA_LOG_CONSCIOUSNESS=1 / SOLARA_LOG_PERCEPTION=1.
if not _env_flag("SOLARA_LOG_HTTPX", "0"):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
if not _env_flag("SOLARA_LOG_CONSCIOUSNESS", "0"):
    logging.getLogger("adu_consciousness").setLevel(logging.WARNING)
if not _env_flag("SOLARA_LOG_PERCEPTION", "0"):
    logging.getLogger("perception").setLevel(logging.WARNING)
    logging.getLogger("perception_module").setLevel(logging.WARNING)

# Background loops are useful only for proactive/always-on brain mode. They are
# disabled by default here to stop /vision/stats + DeepSeek polling spam and cost.
SOLARA_ENABLE_VISION_LOOP = _env_flag("SOLARA_ENABLE_VISION_LOOP", "0")
SOLARA_ENABLE_CONSCIOUSNESS_LOOP = _env_flag("SOLARA_ENABLE_CONSCIOUSNESS_LOOP", "0")

_TTS_STREAM_EXCEPTIONS = (
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
)


def _safe_iter_response_content(
    r: requests.Response,
    *,
    label: str = "stream",
    chunk_size: Optional[int] = None,
) -> Iterator[bytes]:
    """Safely consume a streaming upstream response.

    Never let provider network timeouts escape a StreamingResponse generator.
    A bad TTS/audio upstream should only stop that audio stream, not print an
    ASGI exception group or destabilize the backend.
    """
    if chunk_size is None:
        chunk_size = TTS_STREAM_CHUNK_SIZE_DEFAULT
    try:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk
    except _TTS_STREAM_EXCEPTIONS as e:
        log.warning("[%s] upstream stream interrupted: %s", label, e)
    except Exception as e:
        log.warning("[%s] upstream stream failed: %s", label, e)
    finally:
        try:
            r.close()
        except Exception:
            pass

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

# Structured "facts" memory (always injected, query-independent)
MEMORY_FACTS_ENABLED_DEFAULT = (os.getenv("SOLARA_MEMORY_FACTS_ENABLED") or "1").strip().lower() not in ("0", "false", "no")
MEMORY_FACTS_PROMPT_LIMIT = int(os.getenv("SOLARA_MEMORY_FACTS_PROMPT_LIMIT") or "12")
MEMORY_FACTS_CONTEXT_MAX_CHARS = int(os.getenv("SOLARA_MEMORY_FACTS_CONTEXT_MAX_CHARS") or "1200")
MEMORY_FACTS_ITEM_MAX_CHARS = int(os.getenv("SOLARA_MEMORY_FACTS_ITEM_MAX_CHARS") or "220")
MEMORY_FACTS_MAX_ITEMS_PER_USER = int(os.getenv("SOLARA_MEMORY_FACTS_MAX_ITEMS_PER_USER") or "400")
MEMORY_FACTS_EXTRACT_ENABLED = (os.getenv("SOLARA_MEMORY_FACTS_EXTRACT_ENABLED") or "1").strip().lower() not in ("0", "false", "no")
MEMORY_FACTS_EXTRACT_MODEL_OPENAI = (os.getenv("SOLARA_MEMORY_FACTS_EXTRACT_MODEL_OPENAI") or "gpt-4o-mini").strip()
MEMORY_FACTS_EXTRACT_MODEL_ANTHROPIC = (os.getenv("SOLARA_MEMORY_FACTS_EXTRACT_MODEL_ANTHROPIC") or "claude-3-haiku-20240307").strip()
MEMORY_FACTS_IMPORTANCE_MIN = int(os.getenv("SOLARA_MEMORY_FACTS_IMPORTANCE_MIN") or "2")

# -----------------------------
# ✅ Memory Engine instance (Plan A)
# - Async write queue prevents /chat blocking (important on Render free tier)
# - Keep DB schema compatible: memory_items + memory_facts
# -----------------------------
SOLARA_MEMORY_EMBED_PROVIDER = (os.getenv("SOLARA_MEMORY_EMBED_PROVIDER") or "auto").strip().lower()  # auto|openai|local
SOLARA_MEMORY_WRITE_ASYNC = (os.getenv("SOLARA_MEMORY_WRITE_ASYNC") or "1").strip().lower() not in ("0","false","no")

MEMORY_ENGINE = None
if MemoryEngine is not None and MemoryModuleConfig is not None:
    try:
        MEMORY_ENGINE = MemoryEngine(
            MemoryModuleConfig(
                db_path=MEM_DB_PATH,
                openai_api_key=OPENAI_API_KEY,
                embed_model=MEMORY_EMBED_MODEL,
                embed_provider=SOLARA_MEMORY_EMBED_PROVIDER,
                enabled=MEMORY_ENABLED_DEFAULT,
                max_items_per_user=MEMORY_MAX_ITEMS_PER_USER,
                top_k_default=MEMORY_TOP_K_DEFAULT,
                min_score_default=MEMORY_MIN_SCORE_DEFAULT,
                context_max_chars=MEMORY_CONTEXT_MAX_CHARS,
                item_max_chars=MEMORY_ITEM_MAX_CHARS,
                write_async=SOLARA_MEMORY_WRITE_ASYNC,
                facts_enabled=MEMORY_FACTS_ENABLED_DEFAULT,
                facts_max_items_per_user=MEMORY_FACTS_MAX_ITEMS_PER_USER,
                facts_prompt_limit=MEMORY_FACTS_PROMPT_LIMIT,
                facts_context_max_chars=MEMORY_FACTS_CONTEXT_MAX_CHARS,
                facts_item_max_chars=MEMORY_FACTS_ITEM_MAX_CHARS,
                facts_extract_enabled=MEMORY_FACTS_EXTRACT_ENABLED,
                facts_extract_model_openai=MEMORY_FACTS_EXTRACT_MODEL_OPENAI,
                facts_importance_min=MEMORY_FACTS_IMPORTANCE_MIN,
                min_chars=MEMORY_MIN_CHARS,
                max_store_chars=1200,
            )
        )
        MEMORY_ENGINE.init_db()
    except Exception as _e:
        MEMORY_ENGINE = None



def _sanitize_user_key(key: str) -> str:
    """Commercial-safe memory key sanitizer.

    The actual identity namespace is generated by memory_identity.resolve_memory_identity().
    This sanitizer is retained for existing DB helper functions.
    """
    try:
        from memory_identity import sanitize_memory_key
        return sanitize_memory_key(key)
    except Exception:
        k = (key or "").strip()
        if not k:
            return "default"
        if len(k) > 180:
            return hashlib.sha256(k.encode("utf-8")).hexdigest()
        return re.sub(r"[^a-zA-Z0-9_\-:.]", "_", k)


def _derive_memory_identity(req: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    """Return full commercial memory identity dict for debugging/status."""
    from memory_identity import memory_identity_response
    return memory_identity_response(req, body or {})


def _derive_user_key(req: Request, body: Dict[str, Any]) -> str:
    """Return commercial isolated memory namespace.

    Authenticated users: tenant:{tenant}:user:{sub}
    Guest users:         tenant:guest:device:{device_id/client_id}

    Production must never fall back to raw IP-only shared memory.
    """
    ident = _derive_memory_identity(req, body or {})
    return _sanitize_user_key(str(ident.get("user_key") or ""))


# ================================
# ✅ Billing guards (V1 closed loop, minimal invasive)
# - Do NOT replace business routes; only guard/consume at entry points.
# - Text: /chat and /chat/prepare consume 1 request.
# - Image: image attachments and /solara/photo consume 1 image credit.
# - Realtime: /session is plan-gated; seconds are still consumed by /billing/voice/ping|end.
# ================================

def _billing_gates_enabled() -> bool:
    billing_disabled = str(os.getenv("BILLING_ENABLED", "1")).strip().lower() in ("0", "false", "no", "off")
    bypass_enabled = str(os.getenv("BILLING_BYPASS_GATES", "0")).strip().lower() in ("1", "true", "yes", "on")

    # Four-plan/model debugging mode: the UI can freely choose Guest/Pro/Ultra/Coder
    # and client-selected model IDs without requiring the local billing database.
    # This fixes local /chat/prepare 500s such as:
    #   "Postgres enabled but psycopg2 is not installed"
    # Keep this OFF in production/App Store billing builds.
    if FOUR_PLAN_DEBUG_BYPASS_BILLING:
        if _is_production_env():
            raise RuntimeError("FOUR_PLAN_DEBUG_BYPASS_BILLING must be 0 in production.")
        return False

    if _is_production_env() and (billing_disabled or bypass_enabled):
        raise RuntimeError("Production billing gates are disabled. Set BILLING_ENABLED=1 and BILLING_BYPASS_GATES=0.")
    if billing_disabled:
        return False
    # billing.py also has BILLING_BYPASS_GATES; keep a local early escape for local debugging only.
    if bypass_enabled:
        return False
    return True


def _billing_json_response_from_exception(e: HTTPException) -> JSONResponse:
    detail = e.detail if isinstance(e.detail, dict) else {"ok": False, "error": str(e.detail)}
    if "ok" not in detail:
        detail["ok"] = False
    return JSONResponse(detail, status_code=int(getattr(e, "status_code", 402) or 402))


def _billing_guard_request(
    req: Request,
    body: Optional[Dict[str, Any]],
    feature: str,
    *,
    want: int = 1,
    consume: bool = False,
    check_quota: bool = True,
) -> Optional[JSONResponse]:
    """Return None when allowed; otherwise return a response to send immediately."""
    if not _billing_gates_enabled():
        return None
    if billing_guard_or_403 is None or not callable(billing_guard_or_403):
        return JSONResponse(
            {"ok": False, "error": "billing_guard_unavailable", "feature": feature},
            status_code=500,
        )
    try:
        user_id = _derive_user_key(req, body or {})
        billing_guard_or_403(
            user_id,
            feature,
            want=int(want),
            consume=bool(consume),
            check_quota=bool(check_quota),
        )
        return None
    except HTTPException as e:
        return _billing_json_response_from_exception(e)
    except Exception as e:
        log.exception("[billing] guard failed feature=%s", feature)
        return JSONResponse(
            {"ok": False, "error": "billing_guard_failed", "feature": feature, "message": str(e)[:300]},
            status_code=500,
        )




def _billing_effective_plan_for_request(user_key: str, requested_plan_raw: str = "") -> str:
    """Return effective plan for model/feature routing.

    Four-plan unlocked mode:
    - CHAT_TRUST_CLIENT_PLAN=1 (default in this build) lets the app-selected plan win.
    - Set CHAT_TRUST_CLIENT_PLAN=0 later to restore App Store/billing-capped routing.
    """
    requested = _normalize_plan(requested_plan_raw or "")
    if CHAT_TRUST_CLIENT_PLAN and requested in PLAN_TO_MODEL:
        return requested
    try:
        if billing_get_effective_plan is not None and callable(billing_get_effective_plan):
            return str(billing_get_effective_plan(user_key, requested_plan_raw or "") or "guest").strip() or "guest"
    except Exception as e:
        log.warning("[billing] effective plan failed user=%s: %s", user_key, e)
    return requested or "guest"

def _billing_has_image_attachments(atts: Any) -> bool:
    if not isinstance(atts, list):
        return False
    for a in atts:
        if not isinstance(a, dict):
            continue
        t = str(a.get("type") or a.get("kind") or "").strip().lower()
        mime = str(a.get("mime") or a.get("mime_type") or a.get("content_type") or "").strip().lower()
        filename = str(a.get("filename") or a.get("name") or "").strip().lower()
        if t in ("image", "input_image", "photo", "picture"):
            return True
        if mime.startswith("image/"):
            return True
        if filename.endswith((".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif")):
            return True
    return False

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

        # Structured facts memory (query-independent, top importance always injected)
        con.execute("""
        CREATE TABLE IF NOT EXISTS memory_facts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_key TEXT NOT NULL,
          content TEXT NOT NULL,
          tags TEXT DEFAULT '',
          importance INTEGER DEFAULT 1,
          hash TEXT NOT NULL,
          created_at REAL NOT NULL,
          UNIQUE(user_key, hash)
        );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_memfacts_user_imp_time ON memory_facts(user_key, importance DESC, created_at DESC);")

        # ✅ 时间线记忆（带时间戳的事件记录，支持"什么时候说过/做过什么"类查询）
        con.execute("""
        CREATE TABLE IF NOT EXISTS memory_timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_key TEXT NOT NULL,
            summary TEXT NOT NULL,
            detail TEXT DEFAULT '',
            source TE
            event_type TEXT DEFAULT 'chat',XT DEFAULT '',
            created_at REAL NOT NULL,
            date_str TEXT NOT NULL
        );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_timeline_user_date ON memory_timeline(user_key, date_str DESC, created_at DESC);")

_init_conv_db()
_init_mem_db()

# ================================
# ✅ BrainState Engine / 大脑状态引擎
# - 作为 server_session.py 的思维主体
# - 用工程版 BrainState Tensor 统一管理：意图、记忆、设备、工具、模型路由、风险、行动反馈
# ================================
BRAIN_STATE_ENGINE = None
if BrainStateEngine is not None:
    try:
        BRAIN_STATE_ENGINE = BrainStateEngine(db_path=str(DATA_DIR / "brain_state.sqlite3"))
        log.info("[BrainState] ✅ BrainStateEngine initialized")
    except Exception as _bse_err:
        BRAIN_STATE_ENGINE = None
        log.warning("[BrainState] init failed: %s", _bse_err)



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

        # ✅ 兼容性升级：旧库可能没有 apple_sub / email 字段，平滑加列
        # （Apple Sign In 用 apple_sub 作为永久唯一标识，email 可能为 private relay）
        try:
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "apple_sub" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN apple_sub TEXT")
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_apple_sub ON users(apple_sub) WHERE apple_sub IS NOT NULL")
            if "email" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN email TEXT DEFAULT ''")
        except Exception as _e:
            pass

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

_JWT_SECRET_ENV = (os.getenv("CHATAGI_JWT_SECRET") or "").strip()
if not _JWT_SECRET_ENV and _is_production_env():
    raise RuntimeError("CHATAGI_JWT_SECRET must be set in production.")
JWT_SECRET = _JWT_SECRET_ENV or secrets.token_hex(32)
if not _JWT_SECRET_ENV:
    log.warning("[AUTH] CHATAGI_JWT_SECRET is not set; using an ephemeral development secret. Do not use this in production.")
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

# ════════════════════════════════════════════════════════════════════
# Auth helpers — public user shape + follower counts
# 这些函数被多个 v2/auth、v2/users 端点引用，在 server_session.py 内部定义
# 避免依赖外部 auth.py 模块的可见性。
# ════════════════════════════════════════════════════════════════════

def _public_user(u: Any) -> Dict[str, Any]:
    """
    把 users 表行（dict 或 sqlite3.Row）转成 API 返回的 user 字段。
    兼容缺失字段：avatar_url 缺省 ''，created_at 缺省 0。
    """
    if u is None:
        return {}
    # 兼容 sqlite3.Row 和 dict
    def _get(key, default=None):
        try:
            return u[key]
        except (KeyError, IndexError):
            return default
        except Exception:
            return getattr(u, key, default) if hasattr(u, key) else default

    out = {
        "user_id": _get("user_id", "") or "",
        "username": _get("username", "") or "",
        "display_name": _get("display_name", "") or "",
        "avatar_url": _get("avatar_url", "") or "",
        "created_at": _get("created_at", 0) or 0,
    }
    # 选填字段：email / apple_sub（仅在存在时返回，不暴露）
    email = _get("email", None)
    if email:
        out["email"] = email
    return out

def _count_followers(conn: sqlite3.Connection, user_id: str) -> int:
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM follows WHERE following_id=?",
            (user_id,)
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row["c"] or 0)
        except (KeyError, IndexError, TypeError):
            return int(row[0] or 0)
    except Exception:
        return 0

def _count_following(conn: sqlite3.Connection, user_id: str) -> int:
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM follows WHERE follower_id=?",
            (user_id,)
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row["c"] or 0)
        except (KeyError, IndexError, TypeError):
            return int(row[0] or 0)
    except Exception:
        return 0

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
    if not hits:
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
    return "\n".join(lines).strip()


# =========================
# ✅ Structured long-term memory ("facts")
# - Always injected (query-independent) so name/preferences are recalled even if user asks unrelated questions.
# - De-duplicated by (user_key + md5(content)).
# =========================

def _memory_fact_hash(user_key: str, content: str) -> str:
    raw = f"{_sanitize_user_key(user_key)}:{(content or '').strip().lower()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def memory_facts_save(user_key: str, content: str, tags: str = "", importance: int = 1) -> None:
    if not MEMORY_FACTS_ENABLED_DEFAULT:
        return
    u = _sanitize_user_key(user_key)
    c = (content or "").strip()
    if not c:
        return
    # Keep facts short & stable
    c = re.sub(r"\s+", " ", c).strip()
    if len(c) > 420:
        c = c[:420].rstrip() + "…"

    try:
        imp = int(importance or 1)
    except Exception:
        imp = 1
    imp = max(1, min(imp, 5))

    h = _memory_fact_hash(u, c)
    now = time.time()

    with _mem_conn() as con:
        con.execute(
            """
            INSERT OR IGNORE INTO memory_facts (user_key, content, tags, importance, hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (u, c, (tags or "").strip(), imp, h, now),
        )
        # Prune (keep most important + newest)
        con.execute(
            """
            DELETE FROM memory_facts
            WHERE id IN (
              SELECT id FROM memory_facts
              WHERE user_key=?
              ORDER BY importance DESC, created_at DESC
              LIMIT -1 OFFSET ?
            )
            """,
            (u, MEMORY_FACTS_MAX_ITEMS_PER_USER),
        )

def memory_facts_list(user_key: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not MEMORY_FACTS_ENABLED_DEFAULT:
        return []
    u = _sanitize_user_key(user_key)
    lim = max(1, min(int(limit or 20), 200))
    with _mem_conn() as con:
        cur = con.execute(
            "SELECT id, content, tags, importance, created_at FROM memory_facts WHERE user_key=? ORDER BY importance DESC, created_at DESC LIMIT ?",
            (u, lim),
        )
        rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for rid, content, tags, imp, created_at in rows:
        out.append(
            {
                "id": int(rid),
                "content": str(content or ""),
                "tags": str(tags or ""),
                "importance": int(imp or 1),
                "created_at": float(created_at or 0.0),
            }
        )
    return out

def memory_facts_build_prompt(user_key: str, limit: int = MEMORY_FACTS_PROMPT_LIMIT) -> str:
    if not MEMORY_FACTS_ENABLED_DEFAULT:
        return ""
    mems = memory_facts_list(user_key, limit=limit)
    if not mems:
        return ""
    lines = ["以下是【长期记忆】（重要事实/偏好；如不确定请向用户确认）："]
    total = 0
    for m in mems:
        c = (m.get("content") or "").strip()
        if not c:
            continue
        if len(c) > MEMORY_FACTS_ITEM_MAX_CHARS:
            c = c[:MEMORY_FACTS_ITEM_MAX_CHARS].rstrip() + "…"
        add = f"- {c}"
        if total + len(add) > MEMORY_FACTS_CONTEXT_MAX_CHARS:
            break
        lines.append(add)
        total += len(add)
    return "\n".join(lines).strip()

def _parse_json_array_best_effort(s: str) -> List[Dict[str, Any]]:
    """Parse a JSON array from an LLM output. Best-effort, safe fallback to []."""
    if not s:
        return []
    t = s.strip()
    # strip code fences
    t = t.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    # locate first '[' ... last ']'
    if "[" in t and "]" in t:
        t = t[t.find("["): t.rfind("]") + 1]
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        return []
    return []

def _extract_memory_facts_openai(user_msg: str, ai_reply: str) -> List[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return []
    prompt = (
        "你是记忆提取助手。请从下面对话中提取值得长期记住的用户信息。\\n"
        "只提取对未来有帮助的事实：姓名/称呼、身份/职业、偏好、目标、重要约束、长期项目等。\\n"
        "不要提取泛泛聊天、情绪化表达、临时问题细节。\\n"
        "如果没有值得记忆的内容，返回空数组 []。\\n\\n"
        f"用户说：{user_msg}\\n"
        f"助手说：{ai_reply}\\n\\n"
        "以 JSON 数组返回，格式：\\n"
        "[{\"content\":\"记忆内容\",\"tags\":\"标签1,标签2\",\"importance\":1到5的整数}]\\n"
        "只返回 JSON，不要解释。"
    )
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MEMORY_FACTS_EXTRACT_MODEL_OPENAI or "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 400,
            },
            timeout=25,
        )
        if r.status_code >= 400:
            return []
        j = r.json()
        content = (((j.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        items = _parse_json_array_best_effort(str(content))
        return items
    except Exception:
        return []

def _extract_memory_facts_anthropic(user_msg: str, ai_reply: str) -> List[Dict[str, Any]]:
    if not ANTHROPIC_API_KEY:
        return []
    prompt = (
        "You are a memory extraction assistant. Extract durable user facts from the conversation.\\n"
        "Only include helpful long-term facts (name, role, preferences, goals, constraints).\\n"
        "If nothing is worth saving, return [].\\n\\n"
        f"User: {user_msg}\\n"
        f"Assistant: {ai_reply}\\n\\n"
        "Return ONLY a JSON array of objects with keys: content, tags, importance (1-5)."
    )
    try:
        url = f"{ANTHROPIC_BASE_URL}/v1/messages"
        payload = {
            "model": MEMORY_FACTS_EXTRACT_MODEL_ANTHROPIC or "claude-3-haiku-20240307",
            "max_tokens": 400,
            "temperature": 0.2,
            "system": "Return JSON only.",
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(url, headers=_anthropic_headers(stream=False), data=json.dumps(payload), timeout=25)
        if r.status_code >= 400:
            return []
        j = r.json()
        # Anthropic response content blocks
        blocks = j.get("content") or []
        txt = ""
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                txt += str(b.get("text") or "")
        items = _parse_json_array_best_effort(txt)
        return items
    except Exception:
        return []

def extract_and_save_memory_facts(user_key: str, user_msg: str, ai_reply: str) -> None:
    """Best-effort: extract & save structured long-term memories. Never raises."""
    if not MEMORY_FACTS_ENABLED_DEFAULT:
        return
    if not MEMORY_FACTS_EXTRACT_ENABLED:
        return

    um = (user_msg or "").strip()
    ar = (ai_reply or "").strip()
    if not um and not ar:
        return

    # Cheap gate to avoid extracting on every trivial turn
    gate_text = (um + "\n" + ar).strip()
    if len(gate_text) < 24:
        return
    if not _should_memory_add(um) and not _should_memory_add(ar):
        return

    items: List[Dict[str, Any]] = []
    if OPENAI_API_KEY:
        items = _extract_memory_facts_openai(um, ar)
    elif ANTHROPIC_API_KEY:
        items = _extract_memory_facts_anthropic(um, ar)

    if not items:
        return

    for it in items:
        try:
            c = str(it.get("content") or "").strip()
            if not c:
                continue
            tags = str(it.get("tags") or "").strip()
            try:
                imp = int(it.get("importance") or 1)
            except Exception:
                imp = 1
            if imp < MEMORY_FACTS_IMPORTANCE_MIN:
                continue
            memory_facts_save(user_key, c, tags=tags, importance=imp)
        except Exception:
            continue



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

# NOTE: _should_memory_add is defined further below (with full pattern matching).
# A final unified override also exists near the end of the file (MEMORY_ENGINE delegation).


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


# ═══════════════════════════════════════════════════════════════
# ✅ 时间线记忆 — 带时间戳的事件记录
# 支持查询"什么时候说过/做过什么"、按日期回溯、通话记录追踪等
# ═══════════════════════════════════════════════════════════════

def memory_timeline_add(
    user_key: str,
    summary: str,
    *,
    event_type: str = "chat",
    detail: str = "",
    source: str = "",
    ts: Optional[float] = None,
) -> None:
    """写入一条时间线记忆。"""
    uk = _sanitize_user_key(user_key)
    s = (summary or "").strip()
    if not s or len(s) < 4:
        return
    now = ts or time.time()
    date_str = time.strftime("%Y-%m-%d", time.localtime(now))
    try:
        with _mem_conn() as con:
            con.execute(
                "INSERT INTO memory_timeline(user_key,event_type,summary,detail,source,created_at,date_str) "
                "VALUES(?,?,?,?,?,?,?)",
                (uk, (event_type or "chat").strip()[:20], s[:600], (detail or "")[:2000], (source or "")[:120], now, date_str),
            )
            # 保留最近 500 条，超出删除最老的
            con.execute(
                "DELETE FROM memory_timeline WHERE id IN ("
                "  SELECT id FROM memory_timeline WHERE user_key=? ORDER BY created_at DESC LIMIT -1 OFFSET 500"
                ")", (uk,),
            )
    except Exception as _e:
        log.warning("[timeline] add failed: %s", _e)


def memory_timeline_query(
    user_key: str,
    *,
    date: str = "",
    event_type: str = "",
    keyword: str = "",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """查询时间线记忆。支持按日期/类型/关键词筛选。"""
    uk = _sanitize_user_key(user_key)
    lim = max(1, min(int(limit or 20), 100))

    clauses = ["user_key=?"]
    params: list = [uk]

    if date:
        clauses.append("date_str=?")
        params.append(date.strip()[:10])
    if event_type:
        clauses.append("event_type=?")
        params.append(event_type.strip()[:20])
    if keyword:
        clauses.append("summary LIKE ?")
        params.append(f"%{keyword.strip()[:60]}%")

    params.append(lim)
    sql = f"SELECT id,event_type,summary,detail,source,created_at,date_str FROM memory_timeline WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?"

    try:
        with _mem_conn() as con:
            rows = con.execute(sql, tuple(params)).fetchall()
        return [
            {"id": r[0], "event_type": r[1], "summary": r[2], "detail": r[3], "source": r[4], "created_at": r[5], "date_str": r[6]}
            for r in (rows or [])
        ]
    except Exception as _e:
        log.warning("[timeline] query failed: %s", _e)
        return []


def memory_timeline_build_prompt(user_key: str, limit: int = 8) -> str:
    """构建时间线记忆片段，注入到 system prompt。"""
    items = memory_timeline_query(user_key, limit=limit)
    if not items:
        return ""
    lines = []
    for it in items:
        ts = time.strftime("%m/%d %H:%M", time.localtime(it["created_at"]))
        etype = it.get("event_type") or "chat"
        lines.append(f"[{ts}] ({etype}) {it['summary']}")
    return "最近事件时间线：\n" + "\n".join(lines)


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

def conv_add_message(
    user_key: str,
    cid: str,
    role: str,
    content: str,
    *,
    source: str = "",
    write_timeline: bool = True,
) -> None:
    """
    ✅ 统一消息落库入口。
    同时写入 messages 表 + 可选写入 memory_timeline。
    所有链路（chat / realtime / openclaw）都应调用此函数。
    """
    mid = uuid.uuid4().hex
    now = time.time()
    c = (content or "").strip()
    if not c:
        return
    with _conv_conn() as con:
        con.execute("INSERT INTO messages(id,conversation_id,user_key,role,content,created_at) VALUES(?,?,?,?,?,?)",
                    (mid, cid, user_key, role, c, now))
    conv_touch(user_key, cid, c)

    # ✅ 同步写入时间线记忆
    if write_timeline and c and len(c) >= 4:
        try:
            who = "用户" if role == "user" else "助手"
            src = source or ("realtime" if "语音" in (cid or "") else "chat")
            etype = f"{src}_{role}" if src else role
            memory_timeline_add(
                user_key,
                f"{who}说：{c[:200]}",
                event_type=etype,
                source=cid,
            )
        except Exception:
            pass



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

        for chunk in _safe_iter_response_content(r, label="TTS.feed"):
            s.push(chunk)

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
    def __init__(self, chat_id: str, tts_id: str = "", tts_stream_enabled: bool = False, user_key: str = "") -> None:
        self.chat_id = chat_id
        self.user_key = _sanitize_user_key(user_key or "")
        self.access_token = secrets.token_urlsafe(32)
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

def _create_chat_job(enable_tts_streaming: bool = False, user_key: str = "") -> ChatJob:
    _cleanup_chat_jobs()
    chat_id = uuid.uuid4().hex
    _tts_enabled = bool(enable_tts_streaming) and bool(CHAT_ENABLE_TTS_STREAMING)
    tts_id = _create_live_tts_session() if _tts_enabled else ""
    job = ChatJob(chat_id=chat_id, tts_id=tts_id, tts_stream_enabled=_tts_enabled, user_key=user_key)
    with CHAT_JOBS_LOCK:
        CHAT_JOBS[chat_id] = job
    return job


def _chat_job_authorized(job: ChatJob, req: Request, body: Optional[Dict[str, Any]] = None) -> bool:
    """Authorize SSE/result access by one-time job token or same derived user_key."""
    try:
        supplied = str(
            req.query_params.get("access_token")
            or req.headers.get("x-chat-job-token")
            or ""
        ).strip()
        if supplied and getattr(job, "access_token", "") and hmac.compare_digest(supplied, job.access_token):
            return True
    except Exception:
        pass

    expected = getattr(job, "user_key", "") or ""
    if not expected:
        return False
    try:
        actual = _derive_user_key(req, body or {})
        return hmac.compare_digest(_sanitize_user_key(actual), _sanitize_user_key(expected))
    except Exception:
        return False

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
# Anthropic (Claude) helpers
# ================================

def _is_claude_model(model: str) -> bool:
    return bool((model or "").strip().lower().startswith("claude-"))


def _anthropic_headers(stream: bool = False) -> Dict[str, str]:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("missing ANTHROPIC_API_KEY")
    h = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION or "2023-06-01",
        "content-type": "application/json",
    }
    if stream:
        h["accept"] = "text/event-stream"
    return h


def _responses_input_to_anthropic(inp: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert OpenAI Responses-style `input` to Anthropic `messages` + `system`.

    OpenAI input: [{role, content:[{type:input_text|output_text|input_image,...}]}]
    Anthropic: system:str, messages:[{role:user|assistant, content:[{type:text|image,...}]}]
    """

    def _extract_text(blocks: Any) -> str:
        out: List[str] = []
        for b in (blocks or []):
            if not isinstance(b, dict):
                continue
            t = (b.get("type") or "").strip()
            if t in ("input_text", "output_text", "text"):
                s = b.get("text")
                if isinstance(s, str) and s:
                    out.append(s)
        return "\n".join(out).strip()

    system_parts: List[str] = []
    msgs: List[Dict[str, Any]] = []

    for m in (inp or []):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        blocks = m.get("content") or []

        if role in ("system", "developer"):
            t = _extract_text(blocks)
            if t:
                system_parts.append(t)
            continue

        if role not in ("user", "assistant"):
            role = "user"

        a_blocks: List[Dict[str, Any]] = []
        for b in (blocks or []):
            if not isinstance(b, dict):
                continue
            bt = (b.get("type") or "").strip()

            if bt in ("input_text", "output_text", "text"):
                txt = b.get("text")
                if isinstance(txt, str) and txt:
                    a_blocks.append({"type": "text", "text": txt})
                continue

            if bt == "input_image":
                url = None
                if isinstance(b.get("image_url"), dict):
                    url = b.get("image_url", {}).get("url")
                elif isinstance(b.get("image_url"), str):
                    url = b.get("image_url")
                if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                    try:
                        head, b64 = url.split(",", 1)
                        # head example: data:image/png;base64
                        media = head.split(";", 1)[0].split(":", 1)[1] if ":" in head else "image/png"
                        a_blocks.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": media or "image/png", "data": b64},
                        })
                    except Exception:
                        # If parsing fails, degrade gracefully to a text placeholder.
                        a_blocks.append({"type": "text", "text": "[image attached]"})
                else:
                    a_blocks.append({"type": "text", "text": "[image attached]"})
                continue

            # Unknown/unsupported block -> ignore (or keep as placeholder)
            # a_blocks.append({"type": "text", "text": f"[{bt}]"})

        if not a_blocks:
            # Anthropic requires at least one content block.
            a_blocks = [{"type": "text", "text": ""}]

        msgs.append({"role": role, "content": a_blocks})

    system = "\n\n".join([s for s in system_parts if s]).strip()

    # ✅ Fix: Claude 4.x does not support assistant message prefill.
    # If the last message is assistant (e.g. only an image was sent with no user text),
    # append a minimal user turn so the conversation ends with a user message.
    if msgs and msgs[-1].get("role") == "assistant":
        msgs.append({"role": "user", "content": [{"type": "text", "text": ""}]})

    # ✅ Fix: If msgs is empty (e.g. only system message sent), add a placeholder user turn.
    if not msgs:
        msgs.append({"role": "user", "content": [{"type": "text", "text": ""}]})

    return system, msgs


def _anthropic_messages_create_nonstream(
    *,
    model: str,
    inp: List[Dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Call Anthropic Messages API and return a Responses-like dict used by the rest of the backend."""
    system, messages = _responses_input_to_anthropic(inp)
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens or 2048),
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["temperature"] = float(temperature)

    url = f"{ANTHROPIC_BASE_URL}/v1/messages"
    r = requests.post(url, headers=_anthropic_headers(stream=False), data=json.dumps(payload), timeout=60)
    if r.status_code >= 400:
        try:
            err_msg = (r.json().get("error") or {}).get("message") or ""
        except Exception:
            err_msg = r.text or ""
        if "credit balance" in err_msg.lower() or "billing" in err_msg.lower():
            raise RuntimeError("Anthropic API 余额不足，请前往 console.anthropic.com → Billing 充值后重试。")
        raise RuntimeError(f"anthropic error {r.status_code}: {r.text}")
    j = r.json()

    # Extract assistant text
    out_parts: List[str] = []
    for c in (j.get("content") or []):
        if isinstance(c, dict) and c.get("type") == "text":
            t = c.get("text")
            if isinstance(t, str):
                out_parts.append(t)
    text = "".join(out_parts)

    stop_reason = (j.get("stop_reason") or "").strip().lower()
    is_incomplete = stop_reason in ("max_tokens", "length")

    # Build a minimal OpenAI Responses-like structure so existing UI/parsing keeps working.
    rid = j.get("id") or ("anthropic_" + uuid.uuid4().hex)
    resp_like: Dict[str, Any] = {
        "id": rid,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }
    if is_incomplete:
        resp_like["status"] = "incomplete"
        resp_like["incomplete_details"] = {"reason": "max_output_tokens"}
    else:
        resp_like["status"] = "completed"
    return resp_like


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


# ================================
# DeepSeek helpers / V4 alias
# ================================

def _is_deepseek_model(model: str) -> bool:
    m = (model or "").strip().lower().replace("_", "-")
    return m.startswith("deepseek") or m in ("v4", "deepseek v4", "deepseek-v4", "deepseekv4")


def _normalize_deepseek_model(model: str) -> str:
    """Normalize UI aliases such as 'DeepSeek V4' to a deployable model id.

    Set DEEPSEEK_V4_MODEL to the exact id enabled in your DeepSeek console.
    This prevents invalid UI labels from being sent to the provider and falling back to gpt-4o-mini.
    """
    raw = (model or "").strip()
    m = raw.lower().replace("_", "-").replace(" ", "-")
    if m in ("v4", "deepseek-v4", "deepseekv4") or ("deepseek" in m and "v4" in m):
        return (DEEPSEEK_V4_MODEL or DEEPSEEK_MODEL_DEFAULT or "deepseek-reasoner").strip()
    if raw.lower().startswith("deepseek"):
        return raw
    return (DEEPSEEK_MODEL_DEFAULT or "deepseek-reasoner").strip()


# ================================
# DashScope / Qwen (阿里云百炼) helpers
# ================================

def _is_qwen_model(model: str) -> bool:
    m = (model or "").strip().lower()
    return m.startswith("qwen") or m.startswith("dashscope")


def _dashscope_headers(stream: bool = False) -> Dict[str, str]:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("missing DASHSCOPE_API_KEY")
    h = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    if stream:
        h["Accept"] = "text/event-stream"
    return h


def _dashscope_chat_url() -> str:
    base = (DASHSCOPE_BASE_URL or "https://dashscope-us.aliyuncs.com/compatible-mode/v1").strip().rstrip("/")
    # Ensure we end with /chat/completions
    if base.endswith("/v1"):
        return base + "/chat/completions"
    if not base.endswith("/chat/completions"):
        return base + "/chat/completions"
    return base


def _stream_dashscope_events(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """DashScope (Qwen) ChatCompletions SSE -> OpenAI Responses-style delta events.

    DashScope uses OpenAI-compatible API, same SSE format as DeepSeek.
    Yields delta events + final internal marker:
      {"type":"solara._provider_done","finish_reason":"..."}
    """
    model = (model or DASHSCOPE_MODEL_DEFAULT).strip() or (DASHSCOPE_MODEL_DEFAULT or "qwen3.5-plus")
    url = _dashscope_chat_url()

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
        payload["max_tokens"] = max(128, min(mt, 16384))

    # ⚡ 速度优化:对默认开启思考模式的 Qwen 模型,显式关闭以提升首字延迟
    # qwen3.5-* / qwen3.6-* 默认 enable_thinking=True,会先思考再回答 -> 首字 3-8s
    # 默认行为:关闭思考。如需特定模型保留思考:设环境变量
    #   DASHSCOPE_ENABLE_THINKING=1            -> 全部模型开启
    #   DASHSCOPE_THINKING_MODELS=qwen3.6-plus -> 仅这些模型开启
    _ml = model.lower()
    _thinking_default_on = (
        _ml.startswith("qwen3.5-") or _ml.startswith("qwen3.6-")
    )
    _enable_env = (os.getenv("DASHSCOPE_ENABLE_THINKING") or "").strip().lower()
    _thinking_whitelist = [
        s.strip().lower() for s in (os.getenv("DASHSCOPE_THINKING_MODELS") or "").split(",") if s.strip()
    ]
    _model_in_whitelist = any(_ml == w or _ml.startswith(w) for w in _thinking_whitelist)
    if _enable_env in ("1", "true", "yes", "on"):
        payload["enable_thinking"] = True
    elif _model_in_whitelist:
        payload["enable_thinking"] = True
    elif _thinking_default_on:
        # 默认强制关闭,显著加快首字
        payload["enable_thinking"] = False

    r = requests.post(
        url,
        headers=_dashscope_headers(stream=True),
        json=payload,
        stream=True,
        timeout=(CHAT_STREAM_CONNECT_TIMEOUT_SEC, float(DASHSCOPE_TIMEOUT_SEC)),
    )
    rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
    log.info("[chat.dashscope] model=%s status=%s rid=%s", model, r.status_code, rid)

    if r.status_code >= 400:
        raise RuntimeError(f"dashscope_error {r.status_code}: {_short(r.text, 600)}")

    seq = 0
    finish_reason: Optional[str] = None

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
                delta = str(delta_obj.get("content") or "")

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


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")




# ================================
# ChatAGI V1 attachment/web hardening helpers
# ================================

def _attachment_type(a: Dict[str, Any]) -> str:
    """Normalize attachment type across iOS/backend versions."""
    if not isinstance(a, dict):
        return ""
    raw = str(
        a.get("type")
        or a.get("kind")
        or a.get("attachment_type")
        or a.get("attachmentType")
        or ""
    ).strip().lower()
    mime = _attachment_mime(a).lower()
    name = _attachment_filename(a).lower()
    if raw in ("input_image", "photo", "picture", "img"):
        return "image"
    if raw in ("input_video", "movie"):
        return "video"
    if raw in ("voice", "recording", "audio_message"):
        return "audio"
    if raw:
        return raw
    if mime.startswith("image/") or name.endswith((".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif")):
        return "image"
    if mime.startswith("video/") or name.endswith((".mp4", ".mov", ".m4v", ".webm")):
        return "video"
    if mime.startswith("audio/") or name.endswith((".m4a", ".mp3", ".wav", ".aac", ".ogg")):
        return "audio"
    if mime or name:
        return "file"
    return ""


def _attachment_mime(a: Dict[str, Any]) -> str:
    if not isinstance(a, dict):
        return ""
    return str(
        a.get("mime")
        or a.get("mime_type")
        or a.get("mimeType")
        or a.get("content_type")
        or a.get("contentType")
        or ""
    ).strip()


def _attachment_url(a: Dict[str, Any]) -> str:
    if not isinstance(a, dict):
        return ""
    return str(
        a.get("url")
        or a.get("download_url")
        or a.get("downloadUrl")
        or a.get("file_url")
        or a.get("fileUrl")
        or a.get("local_url")
        or a.get("localUrl")
        or a.get("path")
        or ""
    ).strip()


def _attachment_filename(a: Dict[str, Any]) -> str:
    if not isinstance(a, dict):
        return ""
    return str(
        a.get("filename")
        or a.get("file_name")
        or a.get("fileName")
        or a.get("name")
        or ""
    ).strip()


def _normalize_chat_attachment(a: Dict[str, Any]) -> Dict[str, Any]:
    """Make old/new iOS upload payloads look identical to the backend."""
    b = dict(a or {})
    typ = _attachment_type(b)
    mime = _attachment_mime(b)
    url = _attachment_url(b)
    fname = _attachment_filename(b)
    if typ and not str(b.get("type") or "").strip():
        b["type"] = typ
    if mime and not str(b.get("mime") or "").strip():
        b["mime"] = mime
    if url and not str(b.get("url") or "").strip():
        b["url"] = url
    if fname and not str(b.get("filename") or "").strip():
        b["filename"] = fname
    return b


def _normalize_chat_attachments(atts: Any) -> List[Dict[str, Any]]:
    if not isinstance(atts, list):
        return []
    out: List[Dict[str, Any]] = []
    for a in atts:
        if isinstance(a, dict):
            out.append(_normalize_chat_attachment(a))
    return out


def _has_image_attachments(atts: Any) -> bool:
    for a in (atts or []):
        if not isinstance(a, dict):
            continue
        if _attachment_type(a) == "image":
            return True
    return False


def _has_file_attachments(atts: Any) -> bool:
    for a in (atts or []):
        if not isinstance(a, dict):
            continue
        if _attachment_type(a) == "file":
            return True
    return False


def _has_audio_attachments(atts: Any) -> bool:
    for a in (atts or []):
        if not isinstance(a, dict):
            continue
        if _attachment_type(a) == "audio":
            return True
    return False


def _audio_transcripts_from_attachments(atts: Any) -> List[str]:
    texts: List[str] = []
    for a in (atts or []):
        if not isinstance(a, dict):
            continue
        if _attachment_type(a) != "audio":
            continue
        tr = str(a.get("transcript") or "").strip()
        if tr:
            texts.append(tr)
    return texts


def _current_attachment_prompt(atts: Any) -> str:
    if _has_image_attachments(atts):
        return (
            "我刚上传了一张图片。请优先分析这张图片本身，描述画面里的关键信息；"
            "如果图片里有文字、界面、错误提示或截图，请读取并解释。不要重复上一轮问题。"
        )
    if _has_file_attachments(atts):
        return "我刚上传了文件。请读取文件内容，提取关键信息并回答。不要重复上一轮问题。"
    # ✅ 录音消息最终按文本交互处理：把 STT 结果变成当前 user turn，
    #    避免 audio-only 空消息被后续构造逻辑挂到上一轮 user 消息上。
    if _has_audio_attachments(atts):
        joined = "\n".join(_audio_transcripts_from_attachments(atts)).strip()
        if joined:
            return joined
        return "我刚发送了一段语音，但系统暂时没有拿到可用转写文本。请提示我重新录一遍或检查麦克风权限。"
    return ""


def _ensure_current_user_turn_for_attachments(
    messages: List[Dict[str, Any]],
    attachments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prevent image/file-only sends from being attached to the previous user question.

    iOS sometimes sends an image bubble with an empty user message. The old backend then
    attached the image to whatever previous user text existed in hydrated history, so the
    model answered the old question again. This creates/replaces the current turn with an
    explicit image/file prompt, making the latest upload the focus.
    """
    prompt = _current_attachment_prompt(attachments)
    if not prompt:
        return messages
    out = list(messages or [])
    if out and str(out[-1].get("role") or "").strip() == "user":
        if not str(out[-1].get("content") or "").strip():
            out[-1] = dict(out[-1])
            out[-1]["content"] = prompt
            return out
    else:
        out.append({"role": "user", "content": prompt})
    return out


def _format_web_sources_footer(sources: Any, max_items: int = 5) -> str:
    """Deprecated fallback.

    OpenAI-style UX keeps citations/sources outside the assistant text.
    Sources are delivered as structured SSE events (`sources` / `solara.sources`)
    and rendered by the iOS client as favicon chips under the answer.
    Returning an empty string prevents long raw URLs from polluting the reply.
    """
    return ""

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


# -----------------------------
# ✅ Smart Router Decision Layer
# -----------------------------
# Important: `allow_web` is only a capability/permission switch from the client or plan.
# It must NOT mean "always search the web". This layer restores the intended priority:
# user intent -> smart route -> provider/tool selection.

def _smart_router_force_web_from_body(body: Dict[str, Any]) -> bool:
    """Explicit developer/user override for turns where the UI really means: search now."""
    if not isinstance(body, dict):
        return False
    for k in (
        "force_web", "forceWeb", "require_web", "requireWeb",
        "force_web_search", "forceWebSearch", "web_required", "webRequired",
    ):
        if k in body and _boolish(body.get(k)):
            return True
    route = str(body.get("route") or body.get("intent") or body.get("mode") or "").strip().lower()
    return route in ("web", "web_search", "openai_web_search", "search", "realtime_search")


def _smart_router_disable_web_from_body(body: Dict[str, Any]) -> bool:
    if not isinstance(body, dict):
        return False
    for k in ("no_web", "noWeb", "disable_web", "disableWeb", "offline", "local_only", "localOnly"):
        if k in body and _boolish(body.get(k)):
            return True
    route = str(body.get("route") or body.get("intent") or body.get("mode") or "").strip().lower()
    return route in ("normal", "normal_chat", "chat", "offline_chat")


def _smart_router_text_needs_web(user_text: str) -> Tuple[bool, str]:
    """Deterministic fallback only.

    This is no longer the primary smart router. It is used only when the model
    router is disabled/unavailable, so production routing is not keyword-only.
    """
    text = (user_text or "").strip()
    if not text:
        return False, "empty_text"

    low = text.lower()
    compact = re.sub(r"\s+", "", low)

    local_only_patterns = (
        "你是谁", "你叫什么", "你是誰", "你叫什麼", "介绍一下你自己", "介紹一下你自己",
        "你能做什么", "你会做什么", "你是什么", "who are you", "what are you",
        "your name", "你还记得", "我是谁", "hello", "hi", "你好", "在吗",
    )
    if any(p in low or p in compact for p in local_only_patterns):
        explicit_search_words = ("搜索", "搜一下", "查", "联网", "上网", "网上", "google", "百度", "web", "search")
        if not any(w in low or w in compact for w in explicit_search_words):
            return False, "local_identity_or_smalltalk"

    direct_search_patterns = (
        "搜索", "搜一下", "查一下", "查下", "查查", "帮我查", "幫我查", "联网查", "上网查", "网上查",
        "全网", "google", "百度", "必应", "bing", "web search", "search the web", "look up",
    )
    if any(p in low or p in compact for p in direct_search_patterns):
        return True, "fallback_explicit_search_intent"

    freshness_patterns = (
        "今天", "今日", "现在", "現在", "目前", "刚刚", "剛剛", "最新", "实时", "實時",
        "最近", "近几天", "这几天", "這幾天", "本周", "这周", "這周", "本月", "今年", "昨天", "明天", "当前", "當前",
        "current", "latest", "today", "now", "recent", "this week", "this month", "breaking",
    )
    fresh_domains = (
        "新闻", "新聞", "消息", "事件", "热点", "熱點", "发生", "發生", "价格", "價",
        "股价", "股票", "汇率", "匯率", "天气", "天氣", "赛程", "比分", "排名",
        "政策", "法规", "法規", "总统", "總統", "ceo", "发布", "發布", "上线", "上架", "大模型", "模型",
        "news", "price", "stock", "weather", "score", "schedule", "ranking", "release", "model",
    )
    if any(p in low or p in compact for p in freshness_patterns) and any(d in low or d in compact for d in fresh_domains):
        return True, "fallback_fresh_current_info"

    volatile_patterns = (
        "天气", "天氣", "新闻", "新聞", "股价", "股票", "汇率", "匯率", "航班",
        "比赛结果", "比分", "赛程", "票价", "价格", "价格是多少", "开放时间", "营业时间",
        "weather", "news", "stock price", "exchange rate", "flight", "live score", "schedule",
    )
    if any(p in low or p in compact for p in volatile_patterns):
        return True, "fallback_volatile_info"

    return False, "fallback_normal_chat"


def _extract_first_json_obj(text: str) -> Dict[str, Any]:
    """Parse the first JSON object from a model response."""
    s = (text or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # Strip common fenced blocks.
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I).strip()
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start:i+1])
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}
    return {}


def _normalize_router_route(route: str) -> str:
    r = (route or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "chat": "normal_chat",
        "normal": "normal_chat",
        "text": "normal_chat",
        "text_chat": "normal_chat",
        "web": "openai_web_search",
        "search": "openai_web_search",
        "web_search": "openai_web_search",
        "realtime_search": "openai_web_search",
        "latest_info": "openai_web_search",
        "current_info": "openai_web_search",
        "image": "vision",
        "photo": "vision",
        "file": "file_question",
        "document": "file_question",
        "rag": "file_question",
    }
    r = aliases.get(r, r)
    if r not in ("normal_chat", "openai_web_search", "vision", "file_question"):
        return "normal_chat"
    return r


def _smart_router_llm_cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not key or SMART_ROUTER_LLM_CACHE_TTL_SEC <= 0:
        return None
    now = time.time()
    with _SMART_ROUTER_LLM_CACHE_LOCK:
        ent = _SMART_ROUTER_LLM_CACHE.get(key)
        if not ent:
            return None
        exp, val = ent
        if exp < now:
            try:
                _SMART_ROUTER_LLM_CACHE.pop(key, None)
            except Exception:
                pass
            return None
        return dict(val)


def _smart_router_llm_cache_set(key: str, val: Dict[str, Any]) -> None:
    if not key or SMART_ROUTER_LLM_CACHE_TTL_SEC <= 0:
        return
    with _SMART_ROUTER_LLM_CACHE_LOCK:
        _SMART_ROUTER_LLM_CACHE[key] = (time.time() + SMART_ROUTER_LLM_CACHE_TTL_SEC, dict(val))
        if len(_SMART_ROUTER_LLM_CACHE) > 500:
            # Cheap cleanup; preserve recent insertions roughly.
            for k in list(_SMART_ROUTER_LLM_CACHE.keys())[:100]:
                _SMART_ROUTER_LLM_CACHE.pop(k, None)



def _smart_router_fast_predecision(user_text: str, *, plan: str = "") -> Optional[Dict[str, Any]]:
    """Ultra-fast capability gate for obvious turns.

    This is not the full smart router. It only short-circuits high-confidence cases
    so common chat does not wait for a router LLM network roundtrip. Ambiguous
    requests still fall through to the model-driven router.
    """
    if not SMART_ROUTER_FAST_PATH_ENABLED:
        return None
    text = (user_text or "").strip()
    if not text:
        return {
            "route": "normal_chat",
            "need_web": False,
            "need_vision": False,
            "need_file": False,
            "confidence": 1.0,
            "reason": "fast_empty_text",
            "router_source": "fast_path",
        }

    low = text.lower()
    compact = re.sub(r"\s+", "", low)

    # Explicit realtime/search intent: do not spend 2-8 seconds asking a router LLM.
    explicit_search = (
        "搜索", "搜一下", "搜下", "查一下", "查下", "查查", "帮我查", "幫我查",
        "联网", "上网", "网上", "全网", "google", "百度", "必应", "bing",
        "web search", "search the web", "look up",
    )
    freshness = (
        "今天", "今日", "现在", "目前", "当前", "最新", "实时", "刚刚", "最近", "这几天", "近几天",
        "本周", "这周", "本月", "今年", "新闻", "发布", "上线", "价格", "股价", "汇率", "天气",
        "today", "now", "current", "latest", "recent", "this week", "news", "release", "price",
    )
    volatile_domains = (
        "新闻", "消息", "发布", "上线", "模型", "大模型", "api", "价格", "股价", "汇率", "天气",
        "战争", "局势", "公司", "政策", "法规", "ceo", "融资", "开源", "比赛", "赛程", "比分",
        "news", "release", "model", "price", "stock", "weather", "war", "policy", "score",
    )
    if any(x in low or x in compact for x in explicit_search):
        return {
            "route": "openai_web_search",
            "need_web": True,
            "need_vision": False,
            "need_file": False,
            "confidence": 0.98,
            "reason": "fast_explicit_search_intent",
            "router_source": "fast_path",
        }
    if any(x in low or x in compact for x in freshness) and any(x in low or x in compact for x in volatile_domains):
        return {
            "route": "openai_web_search",
            "need_web": True,
            "need_vision": False,
            "need_file": False,
            "confidence": 0.94,
            "reason": "fast_fresh_volatile_info",
            "router_source": "fast_path",
        }

    # Obvious local chat / identity / writing / code requests: answer directly with the normal model.
    local_chat = (
        "你是谁", "你叫什么", "介绍你自己", "介绍一下你自己", "自我介绍", "你能做什么", "你会做什么",
        "你好", "在吗", "hello", "hi", "who are you", "what are you", "your name",
    )
    creative_or_code = (
        "帮我写", "写一段", "改写", "润色", "翻译", "总结", "解释这段", "分析这段",
        "修复代码", "写代码", "代码", "脚本", "函数", "报错", "debug", "rewrite", "translate", "summarize",
    )
    if any(x in low or x in compact for x in local_chat):
        return {
            "route": "normal_chat",
            "need_web": False,
            "need_vision": False,
            "need_file": False,
            "confidence": 0.98,
            "reason": "fast_identity_or_smalltalk",
            "router_source": "fast_path",
        }
    if any(x in low or x in compact for x in creative_or_code) and not any(x in low or x in compact for x in freshness):
        return {
            "route": "normal_chat",
            "need_web": False,
            "need_vision": False,
            "need_file": False,
            "confidence": 0.88,
            "reason": "fast_local_generation_or_code",
            "router_source": "fast_path",
        }

    # Very short non-search turns should not wait for remote routing.
    if len(compact) <= 18 and not any(x in low or x in compact for x in freshness):
        return {
            "route": "normal_chat",
            "need_web": False,
            "need_vision": False,
            "need_file": False,
            "confidence": 0.80,
            "reason": "fast_short_normal_chat",
            "router_source": "fast_path",
        }

    return None

def _smart_router_model_decide_text(user_text: str, *, plan: str = "") -> Optional[Dict[str, Any]]:
    """Model-driven capability router.

    It classifies the user's intent into capabilities, but never answers the user.
    The selected route then hands off to the provider with that capability:
      - normal_chat -> DashScope/Qwen text
      - openai_web_search -> OpenAI Responses API with web_search
      - vision -> multimodal model
      - file_question -> file extraction / retrieval path
    """
    text = (user_text or "").strip()
    if not text or not SMART_ROUTER_LLM_ENABLED:
        return None

    cache_key = hashlib.sha1((SMART_ROUTER_LLM_MODEL + "|" + plan + "|" + text).encode("utf-8", errors="ignore")).hexdigest()
    cached = _smart_router_llm_cache_get(cache_key)
    if cached:
        cached["router_source"] = cached.get("router_source") or "llm_cache"
        return cached

    system = (
        "你是 ChatAGI 后端【能力智能路由器】，不是聊天助手。你的唯一任务是判断用户这一轮应该交给哪个能力处理。\n"
        "你只能输出一个 JSON 对象，禁止输出解释、Markdown、自然语言回答。\n"
        "可选 route：\n"
        "- normal_chat：闲聊、身份、写作、翻译、代码解释、常识推理、无需实时信息。\n"
        "- openai_web_search：用户要查、搜、核实、询问今天/最近/这几天/当前/最新/发布/新闻/价格/API变动/模型发布/公司动态等可能变化的信息。\n"
        "- vision：需要看图片/照片/截图。\n"
        "- file_question：需要读取用户上传文件、文档、PDF、表格。\n"
        "能力边界：qwen3.6-plus 不能保证最新外部事实；OpenAI web_search 才能处理实时联网。\n"
        "如果用户问的是『这几天美国大模型有什么发布』这类问题，必须 route=openai_web_search。\n"
        "如果用户问『你是谁/你好/写一段文案/帮我改代码』，通常 route=normal_chat。\n"
        "输出 JSON 字段固定为：route, need_web, need_vision, need_file, confidence, reason。confidence 为 0~1。"
    )
    user = f"用户输入：{text}\n当前套餐：{plan or 'unknown'}\n只输出 JSON："

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        provider = (SMART_ROUTER_LLM_PROVIDER or "dashscope").strip().lower()
        raw = ""
        if provider in ("dashscope", "qwen", "aliyun"):
            if not DASHSCOPE_API_KEY:
                return None
            payload = {
                "model": SMART_ROUTER_LLM_MODEL or DASHSCOPE_MODEL_DEFAULT,
                "messages": messages,
                "stream": False,
                "temperature": 0,
                "max_tokens": 220,
            }
            r = requests.post(
                _dashscope_chat_url(),
                headers=_dashscope_headers(stream=False),
                json=payload,
                timeout=(min(CHAT_STREAM_CONNECT_TIMEOUT_SEC, 10), max(2.0, float(SMART_ROUTER_LLM_TIMEOUT_SEC))),
            )
            rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
            log.info("[SmartRouter.LLM] provider=dashscope model=%s status=%s rid=%s", payload["model"], r.status_code, rid)
            if r.status_code >= 400:
                raise RuntimeError(f"dashscope_router_error {r.status_code}: {_short(r.text, 300)}")
            data = r.json() if r.text else {}
            choices = data.get("choices") if isinstance(data, dict) else None
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict):
                    raw = str(msg.get("content") or "")
        else:
            # Keep the router extensible. Unknown provider falls back to deterministic classifier.
            return None

        obj = _extract_first_json_obj(raw)
        if not obj:
            return None
        route = _normalize_router_route(str(obj.get("route") or ""))
        try:
            conf = float(obj.get("confidence") if obj.get("confidence") is not None else 0.0)
        except Exception:
            conf = 0.0
        need_web = bool(obj.get("need_web")) or route == "openai_web_search"
        need_vision = bool(obj.get("need_vision")) or route == "vision"
        need_file = bool(obj.get("need_file")) or route == "file_question"
        decision = {
            "route": route,
            "need_web": bool(need_web),
            "need_vision": bool(need_vision),
            "need_file": bool(need_file),
            "confidence": max(0.0, min(conf, 1.0)),
            "reason": str(obj.get("reason") or "llm_route"),
            "router_source": "llm",
        }
        _smart_router_llm_cache_set(cache_key, decision)
        return decision
    except Exception as e:
        log.warning("[SmartRouter.LLM] failed, fallback to deterministic route: %s", e)
        return None


def _smart_router_decision(
    *,
    user_text: str,
    allow_web: bool,
    attachments: List[Dict[str, Any]],
    requested_model: str = "",
    plan: str = "",
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the capability route decision used by /chat and /chat/prepare."""
    body = body or {}
    has_image = _has_image_attachments(attachments) if attachments else False
    has_file = _has_file_attachments(attachments) if attachments else False
    web_allowed = bool(allow_web)

    # Hard capability facts: these do not need a text classifier.
    if has_file:
        need_web = False
        web_reason = "file_attachment"
        route = "file_question"
    elif has_image:
        need_web = False
        web_reason = "image_attachment"
        route = "vision"
    elif _smart_router_disable_web_from_body(body):
        need_web = False
        web_reason = "disabled_by_request"
        route = "normal_chat"
    elif _smart_router_force_web_from_body(body):
        need_web = bool(web_allowed)
        web_reason = "forced_by_request" if web_allowed else "forced_but_not_allowed"
        route = "openai_web_search" if need_web else "normal_chat"
    else:
        fast_decision = _smart_router_fast_predecision(user_text, plan=plan)
        llm_decision = None if fast_decision else _smart_router_model_decide_text(user_text, plan=plan)
        route_decision = fast_decision or llm_decision
        if route_decision and float(route_decision.get("confidence") or 0.0) >= SMART_ROUTER_LLM_CONFIDENCE_MIN:
            route = _normalize_router_route(str(route_decision.get("route") or "normal_chat"))
            need_web0 = bool(route_decision.get("need_web")) or route == "openai_web_search"
            need_web = bool(web_allowed and need_web0)
            web_reason = str(route_decision.get("reason") or "route_decision")
            if need_web0 and not web_allowed:
                web_reason = f"{web_reason}_but_not_allowed"
            if need_web:
                route = "openai_web_search"
            elif route == "openai_web_search":
                route = "normal_chat"
        else:
            need_web0, web_reason = _smart_router_text_needs_web(user_text)
            need_web = bool(web_allowed and need_web0)
            if need_web0 and not web_allowed:
                web_reason = f"{web_reason}_but_not_allowed"
            route = "openai_web_search" if need_web else "normal_chat"

    if route == "openai_web_search" and need_web:
        provider = "openai"
        model = OPENAI_WEB_SEARCH_MODEL
    elif route == "file_question":
        provider = "dashscope" if _is_qwen_model(requested_model or DASHSCOPE_MODEL_DEFAULT) else "openai"
        model = requested_model or DASHSCOPE_MODEL_DEFAULT
    elif route == "vision":
        provider = "dashscope" if _is_qwen_model(requested_model or DASHSCOPE_MODEL_DEFAULT) else "openai"
        model = requested_model or DASHSCOPE_MODEL_DEFAULT
    else:
        provider = "dashscope" if _is_qwen_model(requested_model or DASHSCOPE_MODEL_DEFAULT) else "openai"
        model = requested_model or DASHSCOPE_MODEL_DEFAULT
        route = "normal_chat"
        need_web = False

    return {
        "need_web": bool(need_web),
        "need_vision": bool(has_image or route == "vision"),
        "need_file": bool(has_file or route == "file_question"),
        "provider": provider,
        "model": model,
        "route": route,
        "web_allowed": bool(web_allowed),
        "web_reason": web_reason,
        "plan": plan or "",
        "router": "fast_then_llm_capability_router" if SMART_ROUTER_LLM_ENABLED else "fallback_deterministic_router",
        "router_fast_path_enabled": bool(SMART_ROUTER_FAST_PATH_ENABLED),
        "router_model": SMART_ROUTER_LLM_MODEL,
    }

def _extract_plan(request: Request, body: Dict[str, Any]) -> str:
    """Extract plan/tier from body, headers, or query params (raw string)."""
    # 1) Body (preferred)
    for k in ("plan", "tier", "subscription", "package", "mode"):
        try:
            v = body.get(k)
        except Exception:
            v = None
        if v is not None and str(v).strip():
            return str(v).strip()

    # 2) Headers
    for hk in ("x-chatagi-plan", "x-plan", "x-tier", "x-subscription", "x-package", "x-mode"):
        try:
            v = request.headers.get(hk)
        except Exception:
            v = None
        if v is not None and str(v).strip():
            return str(v).strip()

    # 3) Query params
    try:
        qp = request.query_params
        for qk in ("plan", "tier", "subscription"):
            v = qp.get(qk)
            if v is not None and str(v).strip():
                return str(v).strip()
    except Exception:
        pass

    return ""


def _attachments_require_openai(attachments: List[Dict[str, Any]], model: str = "") -> bool:
    # Claude models natively support images — do NOT force OpenAI for them.
    # Qwen 3.5-plus natively supports images — do NOT force OpenAI for them.
    # DeepSeek text models cannot consume images/videos; force OpenAI for those.
    if _is_claude_model(model or ""):
        return False
    if _is_qwen_model(model or ""):
        return False
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


def _route_provider(*, allow_web: bool, attachments: List[Dict[str, Any]], model: str = "") -> Tuple[str, str]:
    # ============================================================
    # ✅ HARD RULES (override everything, including explicit model selection):
    #   1) Any image/video attachment -> OpenAI
    #      Reason: DeepSeek API has no public vision endpoint as of 2026-05.
    #              Even if user picked DeepSeek in UI, route to OpenAI for vision.
    #      (Claude/Qwen models still keep their native vision — handled in
    #       _attachments_require_openai which returns False for those models.)
    #   2) allow_web=True AND OpenAI is unavailable -> OpenAI
    #      Reason: DeepSeek/DashScope have no built-in web search; we rely on
    #              OpenAI injection. If OpenAI is missing or recently failed,
    #              fall back to OpenAI's built-in web_search tool.
    #      If OpenAI is healthy, keep cheaper China-side providers and use
    #              prompt-injection (preserves "China-first" routing strategy).
    # ============================================================

    # Rule 1: image/video -> OpenAI only when the requested model has no native vision.
    # Qwen/Claude keep their own multimodal path via _attachments_require_openai(False).
    if _attachments_require_openai(attachments, model=model):
        return "openai", "attachments_force_openai"

    # ✅ Claude models always route to Anthropic.
    # Native vision + native web_search tool — never detour through OpenAI, so the
    # log/route reason stays accurate and any Anthropic errors surface cleanly.
    if _is_claude_model(model):
        return "anthropic", "explicit_claude_model"

    # ✅ Restored behavior: `allow_web` arriving here is no longer the raw client flag.
    # It is the smart-router decision result. Only an intent-approved web turn uses
    # OpenAI built-in web_search; normal chat like “你是谁” stays on Qwen/DashScope.
    if allow_web:
        return "openai", "smart_router_decision:web_search"

    # V1 App Store hard path: explicit Qwen plan models stay on DashScope for
    # normal chat / photo understanding. Only smart-router-approved web turns go OpenAI.
    if _is_qwen_model(model):
        return "dashscope", "explicit_qwen_model"


    # ----- Below this line: original routing (no images, web is fine) -----

    # Explicit model name routing (client requested a specific provider's model)
    if _is_deepseek_model(model):
        return "deepseek", "explicit_deepseek_model"

    mode = (CHAT_ROUTE_MODE or "A").strip().upper()

    # ✅ SMART 模式：智能路由到最优中国模型
    if mode == "SMART":
        return "smart", "smart_route"

    if mode == "A":
        return ("openai", "allow_web") if allow_web else ("deepseek", "allow_web=false")
    if mode in ("OPENAI", "OPENAI_ONLY", "GPT", "GPT_ONLY"):
        return "openai", f"mode={mode}"
    if mode in ("DEEPSEEK", "DEEPSEEK_ONLY", "DEEPSEEK_V4", "DEEPSEEKV4"):
        return "deepseek", f"mode={mode}"
    # unknown -> safe fallback
    return ("openai", f"mode={mode}")


def _select_routed_model(provider: str, requested_model: str = "") -> str:
    req = (requested_model or "").strip()
    if provider == "anthropic":
        # Keep the explicit claude-* id; fall back to a sane Claude default if missing.
        if _is_claude_model(req):
            return req
        return (CHAT_MODEL_ULTRA if _is_claude_model(CHAT_MODEL_ULTRA) else (CHAT_MODEL_PRO if _is_claude_model(CHAT_MODEL_PRO) else req)) or req
    if provider == "dashscope":
        if _is_qwen_model(req):
            return req
        return DASHSCOPE_MODEL_DEFAULT or "qwen3.5-plus"
    if provider == "deepseek":
        # server-controlled default; allow overriding with explicit DeepSeek aliases such as DeepSeek V4.
        if _is_deepseek_model(req):
            return _normalize_deepseek_model(req)
        # In DEEPSEEK_V4 mode, use the V4 alias target even when client sends no model.
        if (CHAT_ROUTE_MODE or "").strip().upper() in ("DEEPSEEK_V4", "DEEPSEEKV4"):
            return DEEPSEEK_V4_MODEL or DEEPSEEK_MODEL_DEFAULT or "deepseek-reasoner"
        return DEEPSEEK_MODEL_DEFAULT or "deepseek-reasoner"
    # openai
    if req and not _is_deepseek_model(req) and not _is_qwen_model(req):
        return req
    return OPENAI_TEXT_MODEL or CHAT_MODEL_DEFAULT


def _ensure_provider_available(provider: str, reason: str) -> Tuple[str, str]:
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        if DEEPSEEK_FALLBACK_TO_OPENAI:
            return "openai", "deepseek_missing_fallback"
        raise RuntimeError("missing DEEPSEEK_API_KEY")
    if provider == "dashscope" and not DASHSCOPE_API_KEY:
        # Fallback to DeepSeek or OpenAI if DashScope key is missing
        if DEEPSEEK_API_KEY:
            return "deepseek", "dashscope_missing_fallback_deepseek"
        return "openai", "dashscope_missing_fallback_openai"
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
            body = {"model": model, "prompt": prompt_final, "seconds": str(sec), "size": size}
            r = requests.post(url, headers=_json_headers(), json=body, timeout=60)
            _log_http(r, f"SORA.CREATE[{model}]")
        finally:
            try:
                if files and files.get("input_reference"):
                    files["input_reference"][1].close()
            except Exception:
                pass
    else:
        body = {"model": model, "prompt": prompt_final, "seconds": str(sec), "size": size}
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
    user_id: str = "adu_system",
) -> None:
    """Used by /tts_prepare: make the output robust (no 413) and stream bytes to LiveMP3Stream."""
    s = _get_live_tts(tts_id)
    if not s:
        return

    # ✅ Voice clone disabled: always use the normal app TTS voice.
    # The previous cloned-voice branch was removed so realtime/chat TTS
    # cannot accidentally switch to a user-cloned voice.

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

            for chunk in _safe_iter_response_content(r, label="TTS.prepare"):
                s.push(chunk)

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
    enable_file_search: bool = False,
    enable_code_interpreter: bool = False,
    enable_computer_use: bool = False,
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

    # Claude route: stream from Anthropic Messages API and emit OpenAI-like SSE events.
    if _is_claude_model(model):
        inp = _build_responses_input(messages, attachments or [])
        if instructions:
            inp = [{"role": "system", "content": [{"type": "input_text", "text": str(instructions)}]}] + inp

        system, a_messages = _responses_input_to_anthropic(inp)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": a_messages,
            "max_tokens": int(max_output_tokens or CHAT_MAX_OUTPUT_TOKENS_DEFAULT),
            "stream": True,
        }
        if system:
            payload["system"] = system

        # ✅ Claude built-in web search tool (server-side)
        # Requires web search enabled in your Anthropic Console.
        claude_sources: Dict[str, Dict[str, str]] = {}
        if _should_use_claude_web_search_tool(enable_web_search, model):
            payload.setdefault("tools", [])
            payload["tools"].append(_claude_web_search_tool_def())

        url = f"{ANTHROPIC_BASE_URL}/v1/messages"
        rid: Optional[str] = None
        stop_reason: Optional[str] = None

        with requests.post(
            url,
            headers=_anthropic_headers(stream=True),
            data=json.dumps(payload),
            stream=True,
            timeout=CHAT_STREAM_TIMEOUT_SEC,
        ) as r:
            if r.status_code >= 400:
                # ✅ Friendly error for Anthropic credit exhaustion
                try:
                    err_body = r.json()
                    err_msg = (err_body.get("error") or {}).get("message") or ""
                except Exception:
                    err_msg = r.text or ""
                if "credit balance" in err_msg.lower() or "billing" in err_msg.lower():
                    raise RuntimeError(
                        "Anthropic API 余额不足，请前往 console.anthropic.com → Billing 充值后重试。"
                    )
                raise RuntimeError(f"anthropic stream error {r.status_code}: {r.text}")

            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    ev = json.loads(data)
                except Exception:
                    continue

                et = (ev.get("type") or "").strip()

                if et == "message_start":
                    rid = (ev.get("message") or {}).get("id") or rid or ("anthropic_" + uuid.uuid4().hex)
                    yield {"type": "response.created", "response": {"id": rid}}
                    continue

                # Capture Claude web_search sources (for UI favicons / sources drawer)
                if et == "content_block_start":
                    cb = ev.get("content_block") or {}
                    if isinstance(cb, dict) and cb.get("type") == "web_search_tool_result":
                        content = cb.get("content")
                        new_sources: List[Dict[str, str]] = []
                        if isinstance(content, list):
                            for it in content:
                                if not isinstance(it, dict):
                                    continue
                                if it.get("type") != "web_search_result":
                                    continue
                                url = str(it.get("url") or "").strip()
                                if not url:
                                    continue
                                if url in claude_sources:
                                    continue
                                title = str(it.get("title") or "").strip() or url
                                claude_sources[url] = {"url": url, "title": title}
                                new_sources.append({"url": url, "title": title})
                        if new_sources:
                            yield {
                                "type": "response.output_item.added",
                                "item": {"type": "web_search_call", "action": {"sources": new_sources}},
                            }
                    continue

                if et == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if isinstance(delta, dict) and delta.get("type") == "text_delta":
                        txt = delta.get("text")
                        if isinstance(txt, str) and txt:
                            yield {"type": "response.output_text.delta", "delta": txt}
                    continue

                if et == "message_delta":
                    d = ev.get("delta") or {}
                    if isinstance(d, dict):
                        sr = d.get("stop_reason")
                        if isinstance(sr, str) and sr:
                            stop_reason = sr
                    continue

                if et == "message_stop":
                    break

        # finalize
        yield {"type": "response.output_text.done"}

        # stop_reason values: "end_turn" / "max_tokens" / etc.
        sources_list = list(claude_sources.values()) if isinstance(claude_sources, dict) else []
        if (stop_reason or "").strip().lower() in ("max_tokens", "length"):
            resp_obj: Dict[str, Any] = {
                "id": rid or ("anthropic_" + uuid.uuid4().hex),
                "incomplete_details": {"reason": "max_output_tokens"},
            }
            if sources_list:
                resp_obj["sources"] = sources_list
            yield {"type": "response.incomplete", "response": resp_obj}
        else:
            resp_obj2: Dict[str, Any] = {"id": rid or ("anthropic_" + uuid.uuid4().hex)}
            if sources_list:
                resp_obj2["sources"] = sources_list
            yield {"type": "response.completed", "response": resp_obj2}
        return

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
        }
        # ✅ 推理模型（gpt-5.4-pro / o系列）不支持 truncation 参数，普通模型加 auto
        if not _is_reasoning_model(model):
            payload["truncation"] = (truncation or "auto")
        else:
            # 推理模型：指定 reasoning effort，避免过度推理导致超时
            payload["reasoning"] = {"effort": "medium"}
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if instructions:
            payload["instructions"] = instructions

        # ✅ Built-in tools (Responses API)
        _tools: List[Dict[str, Any]] = []
        _includes: List[str] = []

        # ✅ 推理模型：请求推理摘要流，让客户端显示"自言自语"思考过程
        if _is_reasoning_model(model):
            _includes.append("reasoning.encrypted_content")

        if _should_use_openai_web_search_tool(enable_web_search):
            _tools.append({"type": "web_search"})
            _includes.append("web_search_call.action.sources")

        if enable_file_search and OPENAI_VECTOR_STORE_ID:
            _tools.append({
                "type": "file_search",
                "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
            })

        if enable_code_interpreter:
            _tools.append({"type": "code_interpreter", "container": {"type": "auto"}})

        if enable_computer_use:
            _tools.append({"type": "computer_use_preview"})

        if _tools:
            payload["tools"] = _tools
            payload["tool_choice"] = "auto"
        if _includes:
            payload["include"] = _includes

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
                t = obj.get("type", "")
                # ✅ keepalive → solara.keepalive
                if t == "keepalive":
                    yield {"type": "solara.keepalive", "sequence_number": obj.get("sequence_number", 0)}
                # ✅ reasoning summary delta → solara.reasoning_delta（供客户端显示推理过程）
                elif t == "response.reasoning_summary_text.delta":
                    delta = obj.get("delta") or ""
                    if delta:
                        yield {"type": "solara.reasoning_delta", "delta": delta}
                elif t == "response.reasoning_summary_text.done":
                    yield {"type": "solara.reasoning_done"}
                else:
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
    messages: List[Dict[str, Any]],
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
# ✅ Legacy /chat（单次 HTTP 返回）长输出不断流核心实现
#    - 对齐官方 Responses API：max_output_tokens + truncation="auto" + previous_response_id 续写
#    - 处理两种继续信号：
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

    # Common in some clients/routers: /files/<file>
    if "/files/" in path:
        rel = path.split("/files/", 1)[1].lstrip("/")
        cand = UPLOADS_DIR / rel
        if cand.exists():
            return cand

    # Sometimes: uploads/<file>
    if path.startswith("uploads/"):
        rel = path[len("uploads/"):]
        cand = UPLOADS_DIR / rel
        if cand.exists():
            return cand

    # Sometimes: files/<file>
    if path.startswith("files/"):
        rel = path[len("files/"):]
        cand = UPLOADS_DIR / rel
        if cand.exists():
            return cand

    # Absolute path (only allow inside UPLOADS_DIR)
    # Some upload routes return absolute filesystem paths; keep it safe.
    if path.startswith("/"):
        try:
            abs_p = Path(path)
            if abs_p.exists():
                try:
                    abs_p.resolve().relative_to(UPLOADS_DIR.resolve())
                    return abs_p
                except Exception:
                    pass
        except Exception:
            pass

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
    port = (os.getenv("PORT") or "8000").strip()
    if s.startswith("/"):
        return f"http://127.0.0.1:{port}{s}"

    # Some upload routers return paths without leading slash (e.g. "files/<id>").
    # Make them fetchable by THIS backend.
    # NOTE: keep non-http schemes intact (e.g., data:)
    if s.startswith("data:"):
        return s
    return f"http://127.0.0.1:{port}/" + s.lstrip("/")


def _read_image_bytes_from_attachment(a: Dict[str, Any]) -> bytes:
    """Read bytes for an image attachment (Scheme-A).
    Priority:
      0) data URL (already base64)
      1) media_store by id/media_id (new upload router path)
      2) local disk path (if url maps to uploads/)
      3) HTTP fetch from THIS backend (e.g. /media/xxx or http://192.168.x.x/media/xxx)
    """
    url = _attachment_url(a)

    # 0) data URL (already base64)
    if url.startswith("data:") and "base64," in url:
        try:
            b64 = url.split("base64,", 1)[1]
            return base64.b64decode(b64)
        except Exception:
            raise RuntimeError("image_data_url_decode_failed")

    # 1) media_store by id/media_id (preferred for /media/upload responses)
    media_id = str(a.get("id") or a.get("media_id") or a.get("mediaId") or "").strip()
    if media_id and _media_store:
        try:
            meta = _media_store.get(media_id)
            if meta and getattr(meta, "path", None) and meta.path.exists():
                return meta.path.read_bytes()
        except Exception:
            pass

    # 2) local file path (uploads/)
    if url:
        p = _resolve_local_upload_path(url)
        if p and p.exists():
            return p.read_bytes()

    # 2) fetch from this server
    if url:
        fetch_url = _server_self_url(url)
        # If it is still not a valid URL, bail early (requests would raise)
        if not (fetch_url.startswith("http://") or fetch_url.startswith("https://")):
            raise RuntimeError(f"image_fetch_invalid_url: {fetch_url}")
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

def _extract_file_content(attachment: Dict[str, Any], max_chars: int = 120000) -> str:
    """
    ✅ 从 media_store 读取文件字节，转为纯文本供 GPT 分析。
    支持：纯文本类（txt/md/py/swift/js/ts/json/csv等）/ PDF / DOCX
    直接通过 media_id 查 media_store，不依赖路径解析，100% 可靠。
    失败返回空串，不抛异常。
    """
    # ── 1. 从 media_store 按 id 拿字节 ──────────────────────────────────────
    raw: Optional[bytes] = None
    fname = _attachment_filename(attachment)
    mime = _attachment_mime(attachment)

    media_id = (attachment.get("id") or "").strip()
    if media_id and _media_store:
        try:
            meta = _media_store.get(media_id)
            if meta and meta.path.exists():
                raw = meta.path.read_bytes()
                if not fname:
                    fname = meta.path.name
                if not mime:
                    mime = meta.mime or ""
        except Exception:
            pass

    # fallback: loopback HTTP（兼容 media_store 不可用的情况）
    if not raw:
        url = _attachment_url(attachment)
        if url:
            try:
                fetch_url = _server_self_url(url)
                r = requests.get(fetch_url, timeout=8)
                if r.status_code == 200 and r.content:
                    raw = r.content
                    if not fname:
                        fname = url.split("/")[-1].split("?")[0]
            except Exception:
                pass

    if not raw:
        return ""

    if not fname:
        fname = "file"

    mime_lc = mime.lower()
    fn_lc = fname.lower()

    # ── 2. 纯文本类型 ────────────────────────────────────────────────────────
    text_mimes = (
        "text/", "application/json", "application/xml",
        "application/javascript", "application/x-python",
        "application/x-yaml", "application/x-sh",
    )
    text_exts = (
        ".txt", ".md", ".py", ".swift", ".js", ".ts", ".jsx", ".tsx",
        ".json", ".csv", ".yaml", ".yml", ".xml", ".html", ".css",
        ".sh", ".bash", ".zsh", ".c", ".cpp", ".h", ".hpp", ".java",
        ".kt", ".go", ".rs", ".rb", ".php", ".sql", ".toml", ".ini",
        ".cfg", ".log", ".env", ".r", ".m", ".lua", ".dart", ".vue",
        ".svelte", ".graphql", ".proto", ".tf", ".gradle", ".plist",
    )
    is_text = (
        any(mime_lc.startswith(m) for m in text_mimes)
        or any(fn_lc.endswith(e) for e in text_exts)
        or (mime_lc in ("application/octet-stream", "") and any(fn_lc.endswith(e) for e in text_exts))
    )
    # 最后兜底：尝试 UTF-8 解码，如果成功且是可打印文本就当纯文本处理
    if not is_text:
        try:
            sample = raw[:512].decode("utf-8")
            printable = sum(1 for c in sample if c.isprintable() or c in "\n\r\t")
            if printable / max(len(sample), 1) > 0.85:
                is_text = True
        except Exception:
            pass

    if is_text:
        try:
            text = raw.decode("utf-8", errors="replace")
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n… [文件过长，已截断，共 {len(text)} 字符]"
            return f"【文件内容：{fname}】\n```\n{text}\n```"
        except Exception:
            pass

    # ── 3. PDF ───────────────────────────────────────────────────────────────
    if mime_lc == "application/pdf" or fn_lc.endswith(".pdf"):
        import io as _io
        try:
            import pdfplumber
            pages: List[str] = []
            with pdfplumber.open(_io.BytesIO(raw)) as pdf:
                for page in pdf.pages[:40]:
                    t = page.extract_text() or ""
                    if t.strip():
                        pages.append(t)
            full = "\n\n".join(pages)
            if len(full) > max_chars:
                full = full[:max_chars] + "\n\n… [已截断]"
            if full.strip():
                return f"【PDF内容：{fname}】\n{full}"
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(_io.BytesIO(raw))
                pages = [p.extract_text() or "" for p in reader.pages[:40]]
                full = "\n\n".join(t for t in pages if t.strip())[:max_chars]
                if full.strip():
                    return f"【PDF内容：{fname}】\n{full}"
            except Exception:
                pass
        except Exception:
            pass

    # ── 4. DOCX ──────────────────────────────────────────────────────────────
    if (mime_lc in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ) or fn_lc.endswith(".docx") or fn_lc.endswith(".doc")):
        import io as _io
        try:
            import docx as _docx
            doc = _docx.Document(_io.BytesIO(raw))
            full = "\n".join(p.text for p in doc.paragraphs if p.text.strip())[:max_chars]
            if full.strip():
                return f"【Word文档内容：{fname}】\n{full}"
        except Exception:
            pass

    return ""


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
    has_file_att = any(
        isinstance(a, dict) and (a.get("type") or "").strip().lower() == "file"
        for a in (attachments or [])
    )
    for m in messages:
        role = (m.get("role") or "user").strip() or "user"
        content = (m.get("content") or "").strip()
        if not content:
            # ✅ 有 file 时保留空 user 消息槽，后面追加文件内容
            if role == "user" and has_file_att:
                inp.append({"role": "user", "content": []})
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
            at = _attachment_type(a)
            url = _attachment_url(a)
            mime = _attachment_mime(a)

            # Some upload routes return OpenAI file IDs. Responses API supports `file_id` for input_image.
            # We keep it as an escape hatch when server-side byte fetching fails.
            file_id = (
                str(a.get("file_id") or a.get("fileId") or a.get("openai_file_id") or a.get("openaiFileId") or a.get("id") or "")
                .strip()
            )
            file_id_is_openai = file_id.startswith(("file_", "file-"))

            is_image = (at == "image") or (at in ("photo", "picture", "img")) or (mime.lower().startswith("image/"))

            # ✅✅✅ Image (Scheme-A): prefer DATA URL in `image_url` (OpenAI must NOT download LAN URLs)
            # Fallback: if server cannot fetch bytes but caller provided an OpenAI `file_id`, use it.
            if is_image:
                try:
                    raw = _read_image_bytes_from_attachment(a)
                    b64 = base64.b64encode(raw).decode("utf-8")
                    mime2 = (mime or "").strip() or "image/jpeg"
                    if not mime2.lower().startswith("image/"):
                        mime2 = "image/jpeg"
                    data_url = f"data:{mime2};base64,{b64}"
                    content_list.append({"type": "input_image", "image_url": data_url})
                except Exception as e:
                    log.warning("[chat.attach] image->base64 failed id=%s url=%s err=%s", _short(file_id, 80), _short(url, 160), e)
                    if file_id_is_openai:
                        content_list.append({"type": "input_image", "file_id": file_id})
                    else:
                        # keep chat alive; degrade gracefully
                        fname = str(a.get("filename") or a.get("name") or "image").strip()
                        content_list.append({"type": "input_text", "text": f"[image unavailable] {fname} {url}".strip()})
                continue
            # ✅ audio: Responses API does NOT accept input_audio. Use transcript -> input_text.
            if at == "audio":
                tr = (a.get("transcript") or "").strip()
                if tr:
                    existing_text = "\n".join(
                        str(x.get("text") or "")
                        for x in content_list
                        if isinstance(x, dict) and str(x.get("type") or "") in ("input_text", "output_text", "text")
                    )
                    # If _ensure_current_user_turn_for_attachments already made the transcript
                    # the current user message, do not append it a second time.
                    if tr not in existing_text:
                        content_list.append({"type": "input_text", "text": f"[voice transcript]\n{tr}"})
                    continue
                if url:
                    content_list.append({"type": "input_text", "text": f"[voice message] url={url}"})
                    continue
                continue

            # ✅ file: 读取文件内容注入（txt/md/py/swift/js/pdf/docx 全支持）
            if at == "file":
                extracted = _extract_file_content(a)
                if extracted:
                    content_list.append({"type": "input_text", "text": extracted})
                    # 用户未输文字时自动补分析提示
                    has_user_text = any(
                        isinstance(blk, dict)
                        and blk.get("type") == "input_text"
                        and not (blk.get("text") or "").startswith("【")
                        for blk in content_list
                    )
                    if not has_user_text:
                        content_list.append({
                            "type": "input_text",
                            "text": "请仔细阅读以上文件内容，进行分析和总结，指出关键信息、主要逻辑和值得注意的地方。",
                        })
                else:
                    fname = (a.get("filename") or a.get("name") or "file").strip()
                    content_list.append({
                        "type": "input_text",
                        "text": f"[文件已上传但无法提取内容] 文件名：{fname}，类型：{mime or 'unknown'}",
                    })
                continue

            # Fallback: embed as text pointer
            if at and (a.get("id") or url):
                content_list.append({
                    "type": "input_text",
                    "text": f"[attachment:{at}] id={a.get('id')} mime={mime} url={url}"
                })

    return inp


def _responses_input_to_chat_completions_messages(inp: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI Responses-style input blocks to OpenAI-compatible Chat Completions.

    DashScope/Qwen multimodal chat expects content blocks like:
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
      {"type":"text","text":"..."}
    Alibaba Cloud Model Studio documents Qwen chat as OpenAI-compatible and accepts this
    Chat Completions content-block format for images.
    """
    out: List[Dict[str, Any]] = []

    for m in (inp or []):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user").strip().lower() or "user"
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"

        blocks = m.get("content") or []
        if isinstance(blocks, str):
            content_text = blocks.strip()
            if content_text:
                out.append({"role": role, "content": content_text})
            continue

        text_parts: List[str] = []
        qwen_blocks: List[Dict[str, Any]] = []
        has_image = False

        for b in (blocks or []):
            if not isinstance(b, dict):
                continue
            bt = str(b.get("type") or "").strip()

            if bt in ("input_text", "output_text", "text"):
                txt = str(b.get("text") or "")
                if txt:
                    text_parts.append(txt)
                    qwen_blocks.append({"type": "text", "text": txt})
                continue

            if bt == "input_image":
                image_url = ""
                raw_url = b.get("image_url")
                if isinstance(raw_url, dict):
                    image_url = str(raw_url.get("url") or "").strip()
                elif isinstance(raw_url, str):
                    image_url = raw_url.strip()

                if image_url:
                    qwen_blocks.append({"type": "image_url", "image_url": {"url": image_url}})
                    has_image = True
                else:
                    file_id = str(b.get("file_id") or "").strip()
                    text_parts.append(f"[image attached: {file_id or 'unavailable'}]")
                    qwen_blocks.append({"type": "text", "text": f"[image attached: {file_id or 'unavailable'}]"})
                continue

        if has_image and role == "user":
            # Qwen vision path: keep typed content blocks.
            if not any(isinstance(x, dict) and x.get("type") == "text" and str(x.get("text") or "").strip() for x in qwen_blocks):
                qwen_blocks.append({"type": "text", "text": "请分析这张图片。"})
            out.append({"role": role, "content": qwen_blocks or [{"type": "text", "text": "请分析这张图片。"}]})
        else:
            # Text-only or non-user messages: use plain string for widest compatibility.
            content_text = "\n".join(t for t in text_parts if t).strip()
            if content_text:
                out.append({"role": role, "content": content_text})

    return out


def _build_qwen_chat_messages(
    messages: List[Dict[str, Any]],
    attachments: Optional[List[Dict[str, Any]]] = None,
    *,
    instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build DashScope/Qwen OpenAI-compatible Chat Completions messages.

    This is the V1 app-store path: qwen3.6-plus + OpenAI web context + photo upload.
    It reuses _build_responses_input so local /media uploads become data URLs and file/audio
    attachments become text blocks, then converts blocks into Chat Completions format.
    """
    base: List[Dict[str, Any]] = []
    for m in (messages or []):
        if isinstance(m, dict):
            base.append(dict(m))

    if instructions:
        inst = instructions.strip()
        if inst and not any(str(mm.get("content") or "").strip() == inst for mm in base if isinstance(mm, dict)):
            insert_at = 1 if (base and str(base[0].get("role") or "").strip() in ("system", "developer")) else 0
            base.insert(insert_at, {"role": "system", "content": inst})

    inp = _build_responses_input(base, attachments or [])
    out = _responses_input_to_chat_completions_messages(inp)
    try:
        img_count = sum(1 for a in (attachments or []) if isinstance(a, dict) and _attachment_type(a) == "image")
        if img_count:
            log.info("[chat.qwen] multimodal images=%s messages=%s", img_count, len(out))
    except Exception:
        pass
    return out


def _openai_responses_create_nonstream(
    *,
    model: str,
    inp: List[Dict[str, Any]],
    max_output_tokens: int,
    truncation: str = "auto",
    previous_response_id: Optional[str] = None,
    instructions: Optional[str] = None,
    enable_web_search: bool = False,
    enable_file_search: bool = False,
    enable_code_interpreter: bool = False,
    enable_computer_use: bool = False,
) -> Dict[str, Any]:
    model = (model or CHAT_MODEL_DEFAULT).strip()

    # Claude route: if model id starts with "claude-" use Anthropic.
    # We return a Responses-like dict so downstream parsing (and the iOS app) stays unchanged.
    if _is_claude_model(model):
        return _anthropic_messages_create_nonstream(
            model=model,
            inp=inp,
            max_tokens=int(max_output_tokens or 2048),
        )

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

    # ✅ Built-in tools (Responses API)
    _tools: List[Dict[str, Any]] = []
    _includes: List[str] = []

    if _should_use_openai_web_search_tool(enable_web_search):
        _tools.append({"type": "web_search"})
        _includes.append("web_search_call.action.sources")

    if enable_file_search and OPENAI_VECTOR_STORE_ID:
        _tools.append({
            "type": "file_search",
            "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
        })

    if enable_code_interpreter:
        _tools.append({"type": "code_interpreter", "container": {"type": "auto"}})

    if enable_computer_use:
        _tools.append({"type": "computer_use_preview"})

    if _tools:
        payload["tools"] = _tools
        payload["tool_choice"] = "auto"
    if _includes:
        payload["include"] = _includes

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
    enable_file_search: bool = False,
    enable_code_interpreter: bool = False,
    enable_computer_use: bool = False,
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
    current_messages = list(messages or [])
    current_attachments = attachments

    # ✅ Restored legacy behavior: external web provider (Serper) fetches once and
    # injects numbered results into the prompt. OpenAI built-in web_search remains
    # available when CHAT_WEB_PROVIDER=openai, but allow_web no longer hard-forces it.
    if enable_web_search and CHAT_ENABLE_WEB_SEARCH_DEFAULT and CHAT_WEB_PROVIDER.startswith("serper"):
        try:
            q = _last_user_text_from_messages(current_messages)
            if q.strip():
                web_results = _serper_web_search(q, k=CHAT_WEB_TOPK_DEFAULT, kind=SERPER_DEFAULT_KIND)
                web_ctx = _format_web_context_for_prompt(web_results)
                if web_ctx:
                    current_messages = _inject_web_context(current_messages, web_ctx)
        except Exception as e:
            log.warning("[web-search] serper failed: %s", e)

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
            enable_file_search=enable_file_search,
            enable_code_interpreter=enable_code_interpreter,
            enable_computer_use=enable_computer_use,
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


def _dashscope_complete_full_text(
    *,
    model: str,
    messages: List[Dict[str, str]],
    attachments: List[Dict[str, Any]],
    max_output_tokens: int,
    max_continuations: int,
    instructions: Optional[str],
) -> Tuple[str, str, str, Optional[Dict[str, Any]], int]:
    """DashScope/Qwen non-stream (via SSE stream accumulation) with auto-continuation.

    Same pattern as _deepseek_complete_full_text but using DashScope endpoint.
    Return: (full_text, status, response_id, incomplete_details, continuations)
    """
    model = (model or DASHSCOPE_MODEL_DEFAULT).strip() or (DASHSCOPE_MODEL_DEFAULT or "qwen3.5-plus")
    max_output_tokens = max(256, min(int(max_output_tokens), 60000))
    max_continuations = max(0, min(int(max_continuations), 20))

    CHUNK_MARKER = "[[SOLARA_CONTINUE]]"

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
        if not any((mm.get("content") or "").strip() == instructions.strip() for mm in prefix):
            insert_at = 1 if (prefix and prefix[0].get("role") in ("system", "developer")) else 0
            prefix.insert(insert_at, {"role": "system", "content": instructions})

    full = ""
    status = "completed"
    response_id = ""
    incomplete_details: Optional[Dict[str, Any]] = None
    continuations = 0

    # Qwen3.6-Plus app-store path: include text, OpenAI web context, files/audio, and photos.
    qs_messages = _build_qwen_chat_messages(messages, attachments, instructions=instructions)

    while True:
        part_parts: List[str] = []
        finish_reason = ""
        tail_buf = ""
        marker_seen = False

        mt = min(max_output_tokens, 16384)

        for ev in _stream_dashscope_events(model=model, messages=qs_messages, max_tokens=mt):
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

        qs_messages = prefix + [{"role": "user", "content": cont_prompt}]

    full = _maybe_wrap_code_as_fenced_markdown(_ensure_code_fences_closed(full)).strip()
    return full, status, response_id, incomplete_details, continuations


def _guard_stream(it: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Wrap a provider stream so an upstream failure (bad model / missing key /
    region / billing) surfaces as a sentinel event instead of propagating.

    The worker turns this sentinel into a one-time CHAT_MODEL_FALLBACK retry so the
    user always gets an answer rather than the generic "网络问题，稍后再试。"。
    """
    try:
        for ev in it:
            yield ev
    except Exception as e:
        yield {"type": "solara._stream_error", "message": str(e)}


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
    plan: str = "",
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

            # ✅ Voice clone disabled: stream this segment through the normal TTS provider only.

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

            for chunk in _safe_iter_response_content(r, label="chat-tts.stream"):
                live.push(chunk)

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
            base_messages = list(messages or [])
            current_attachments = attachments

            # ✅ Restored legacy behavior: external web provider (Serper) prefetches
            # sources for UI favicon chips and injects numbered results into prompt.
            if allow_web and CHAT_ENABLE_WEB_SEARCH_DEFAULT and CHAT_WEB_PROVIDER.startswith("serper"):
                try:
                    q = (last_user_text or _last_user_text_from_messages(base_messages) or "").strip()
                    if q:
                        web_results = _serper_web_search(q, k=CHAT_WEB_TOPK_DEFAULT, kind=SERPER_DEFAULT_KIND)
                        _push_sources(web_results)
                        web_ctx = _format_web_context_for_prompt(web_results)
                        if web_ctx:
                            base_messages = _inject_web_context(base_messages, web_ctx)
                except Exception as e:
                    log.warning("[web-search] serper failed: %s", e)

            current_messages = base_messages
            used_fallback = False  # CHAT_MODEL_FALLBACK 只用一次

            while True:
                need_continue = False
                need_fallback_retry = False
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
                            at = _attachment_type(a)
                            if at == "audio":
                                tr = str(a.get("transcript") or "").strip()
                                if tr:
                                    existing_msg_text = "\n".join(
                                        str(mm.get("content") or "")
                                        for mm in ds_msgs
                                        if isinstance(mm, dict)
                                    )
                                    if tr not in existing_msg_text:
                                        trs.append(tr)
                            if at == "file":
                                extracted = _extract_file_content(a)
                                if extracted:
                                    file_urls.append(extracted)
                                else:
                                    u = _attachment_url(a)
                                    fname = _attachment_filename(a) or "file"
                                    if u:
                                        file_urls.append(f"[文件] {fname} url={u}")
                        if trs:
                            ds_msgs.append({"role": "user", "content": "[voice transcript]\n" + "\n".join(trs)})
                        if file_urls:
                            ds_msgs.append({"role": "user", "content": "[file]\n" + "\n".join(file_urls)})

                    stream_iter = _stream_deepseek_events(
                        model=model,
                        messages=ds_msgs,
                        max_tokens=min(int(max_output_tokens or 8192), 16384),
                    )
                elif provider_norm == "dashscope":
                    # Qwen3.6-Plus multimodal path: build OpenAI-compatible Chat Completions
                    # messages with image_url data URLs and injected file/audio text.
                    qs_msgs = _build_qwen_chat_messages(
                        current_messages,
                        current_attachments,
                        instructions=instructions,
                    )

                    stream_iter = _stream_dashscope_events(
                        model=model,
                        messages=qs_msgs,
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
                        enable_file_search=_plan_allows_file_search(plan),
                        enable_code_interpreter=_plan_allows_code_interpreter(plan),
                        enable_computer_use=_plan_allows_computer_use(plan, model),
                    )

                # ✅ 兜底：把上游异常转成哨兵事件，便于一次性降级到 CHAT_MODEL_FALLBACK
                stream_iter = _guard_stream(stream_iter)

                for ev in stream_iter:
                    typ = ev.get("type") if isinstance(ev, dict) else None

                    # 上游失败哨兵：首字未产出且未降级过 -> 切到兜底模型重试一次
                    if typ == "solara._stream_error":
                        _err_msg = str(ev.get("message") or "stream_error")
                        _already_fallback = (provider_norm == "openai" and (model or "") == (CHAT_MODEL_FALLBACK or ""))
                        if (not full_parts) and (not used_fallback) and (not _already_fallback) and CHAT_MODEL_FALLBACK:
                            used_fallback = True
                            need_fallback_retry = True
                            log.warning(
                                "[chat-worker] primary stream failed (provider=%s model=%s): %s -> fallback openai/%s",
                                provider_norm, model, _err_msg, CHAT_MODEL_FALLBACK,
                            )
                            provider_norm = "openai"
                            model = CHAT_MODEL_FALLBACK
                            prev_rid = None
                            current_messages = base_messages
                            current_attachments = attachments
                            job.push_event({
                                "type": "solara.meta",
                                "model": model,
                                "provider": provider_norm,
                                "route_reason": "fallback_after_error",
                            })
                            break
                        # 已经降级过或已有输出 -> 当作真实错误抛出
                        raise RuntimeError(_err_msg)

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
                        # ✅ 上游内联 error：首字未产出且未降级过 -> 走一次性兜底降级
                        _already_fallback = (provider_norm == "openai" and (model or "") == (CHAT_MODEL_FALLBACK or ""))
                        if (not full_parts) and (not used_fallback) and (not _already_fallback) and CHAT_MODEL_FALLBACK:
                            used_fallback = True
                            need_fallback_retry = True
                            log.warning(
                                "[chat-worker] inline error (provider=%s model=%s): %s -> fallback openai/%s",
                                provider_norm, model, msg, CHAT_MODEL_FALLBACK,
                            )
                            provider_norm = "openai"
                            model = CHAT_MODEL_FALLBACK
                            prev_rid = None
                            current_messages = base_messages
                            current_attachments = attachments
                            job.push_event({
                                "type": "solara.meta",
                                "model": model,
                                "provider": provider_norm,
                                "route_reason": "fallback_after_error",
                            })
                            break
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

                # ✅ 触发了兜底降级 -> 用 CHAT_MODEL_FALLBACK 重跑一轮（不进入续写逻辑）
                if need_fallback_retry:
                    continue

                # Decide whether to continue:
                if marker_seen:
                    need_continue = True

                if provider_norm == "deepseek":
                    fr = (finish_reason or "").lower()
                    if fr in ("length", "max_tokens", "max_output_tokens"):
                        need_continue = True
                        if not incomplete_reason:
                            incomplete_reason = fr

                if provider_norm == "dashscope":
                    fr = (finish_reason or "").lower()
                    if fr in ("length", "max_tokens", "max_output_tokens"):
                        need_continue = True
                        if not incomplete_reason:
                            incomplete_reason = fr

                if not need_continue:
                    break

                if continuations >= max_continuations:
                    break

                if provider_norm != "deepseek" and provider_norm != "dashscope":
                    if not last_rid:
                        break

                continuations += 1
                if provider_norm not in ("deepseek", "dashscope"):
                    prev_rid = last_rid

                job.push_event({
                    "type": "solara.continuation",
                    "index": continuations,
                    "reason": (incomplete_reason or ("marker" if marker_seen else "unknown")),
                })

                if provider_norm in ("deepseek", "dashscope"):
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
            # ✅ OpenAI-style UX: do NOT append source URL lists into the answer body.
            # Sources are sent separately via `sources` / `solara.sources` SSE events and rendered as chips by iOS.

            # ✅ 处理 AI 回复中的 OpenClaw 动作标记（仅当 legacy bridge 真可用）
            if job.full_text and "[ADU_ACTION:" in job.full_text and _openclaw_runtime_enabled():
                try:
                    _bridge = _openclaw_get_bridge_or_none()
                    if _bridge is not None and getattr(_bridge, "connected", False):
                        _clean, _action_results = process_agent_actions_sync(job.full_text, _bridge)
                        if _action_results:
                            job.full_text = _clean + _action_results
                            # 把执行结果也推送给前端 SSE
                            job.push_event({
                                "type": "response.output_text.done",
                                "text": job.full_text,
                            })
                            log.info("[Intent] Actions executed, results appended to response")
                except Exception as _intent_err:
                    log.warning("[Intent] action processing failed: %s", _intent_err)

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

            # Structured facts extraction (async, best-effort)
            try:
                if last_user_text and job.full_text:
                    threading.Thread(
                        target=extract_and_save_memory_facts,
                        args=(user_key, last_user_text, job.full_text),
                        daemon=True,
                    ).start()
            except Exception:
                pass

            # ✅ 四级压缩：写入L0助手轮次 + 触发压缩流水线 + L3结构化提取
            try:
                if job.full_text:
                    _comp_append_turn(user_key, conversation_id, "assistant", job.full_text[:800])
                    _comp_maybe_compress(user_key, conversation_id)
                    if last_user_text:
                        _comp_l3_extract(user_key, last_user_text, job.full_text)
            except Exception:
                pass

            # ✅ 时间线记忆：记录聊天事件
            try:
                if last_user_text:
                    memory_timeline_add(user_key, f"用户说：{last_user_text[:200]}", event_type="chat", source=conversation_id)
            except Exception:
                pass

            if job.full_text:
                # OpenAI-style: send sources as structured side-channel right before final text.
                # The assistant text stays clean; iOS renders source favicons under the bubble.
                try:
                    if web_sources:
                        job.push_event({"type": "sources", "sources": list(web_sources.values())})
                        job.push_event({"type": "solara.sources", "sources": list(web_sources.values())})
                except Exception:
                    pass
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
    # ✅ OpenClaw: legacy bridge is disabled in local-agent-only mode. Skip it silently.
    if _openclaw_runtime_enabled():
        try:
            bridge = _openclaw_get_bridge_or_none()
            ok = await bridge.connect() if bridge is not None else False
            if ok:
                log.info("[BOOT] ✅ OpenClaw connected: %s", bridge.ws_url)

                # ── 自动扫描项目（不需要用户手动点"连接项目"）──
                scan_list = [
                    "~/Desktop/GPTsora",
                    "~/Desktop/backend",
                    "~/GPTsora",
                    "~/Projects/GPTsora",
                ]
                auto_env = os.getenv("OPENCLAW_AUTO_SCAN", "").strip()
                if auto_env:
                    scan_list = [p.strip() for p in auto_env.split(",") if p.strip()]

                for scan_path in scan_list:
                    expanded = scan_path.replace("~", str(Path.home()))
                    if os.path.isdir(expanded):
                        try:
                            result = await bridge.scan_project(scan_path)
                            count = result.get("count", 0)
                            method = result.get("scan_method", "find")
                            log.info("[BOOT] ✅ Auto-scanned %s: %d files (method=%s)",
                                     scan_path, count, method)
                        except Exception as scan_err:
                            log.warning("[BOOT] Auto-scan %s failed: %s", scan_path, scan_err)
            else:
                log.debug("[BOOT] OpenClaw not available")
        except Exception as e:
            log.debug("[BOOT] OpenClaw init skipped: %s", e)

    # ✅ 视觉监控循环（可选：持续订阅屏幕帧，主动分析）
    if SOLARA_ENABLE_VISION_LOOP:
        try:
            from adu_vision_loop import vision_loop
            asyncio.create_task(vision_loop.start())
            log.info("[VisionLoop] ✅ vision monitor started")
        except Exception as _vl_boot_err:
            log.warning("[VisionLoop] vision monitor start failed: %s", _vl_boot_err)
    else:
        log.info("[VisionLoop] disabled (set SOLARA_ENABLE_VISION_LOOP=1 to enable)")

    # ✅ 意识系统后台循环（可选；关闭时仍保留 on_user_message/to_prompt 的轻量能力）
    if SOLARA_ENABLE_CONSCIOUSNESS_LOOP:
        try:
            from adu_consciousness import consciousness
            asyncio.create_task(consciousness.run())
            log.info("[Consciousness] 🧠 background loop started")
        except Exception as _cs_boot_err:
            log.warning("[Consciousness] background loop start failed: %s", _cs_boot_err)
    else:
        log.info("[Consciousness] background loop disabled (set SOLARA_ENABLE_CONSCIOUSNESS_LOOP=1 to enable)")

    yield
    # ✅ OpenClaw: 关闭时断开（only when real bridge exists）
    try:
        bridge = _openclaw_get_bridge_or_none()
        if bridge is not None:
            await bridge.disconnect()
    except Exception:
        pass
    log.info("[BOOT] stop")

app = FastAPI(title="ChatAGI-阿杜 Backend", lifespan=lifespan)

# ================================
# ✅ LAN Local Agent Proxy
# 手机 / APP / 后端 -> LOCAL_AGENT_BASE_URL -> ai-brain-local-agent -> usecomputer
#
# .env:
#   LOCAL_AGENT_BASE_URL=http://10.0.0.204:4317
#   LOCAL_AGENT_TOKEN=local-dev-token
#
# APP/backend should call these backend-facing routes:
#   GET  /api/brain/computer/health
#   GET  /api/brain/computer/mouse
#   POST /api/brain/computer/screenshot
#   POST /api/brain/computer/move
#   POST /api/brain/computer/click
#   POST /api/brain/computer/type
#   POST /api/brain/computer/press
# ================================
LOCAL_AGENT_BASE_URL = (
    os.getenv("LOCAL_AGENT_BASE_URL")
    or os.getenv("AI_BRAIN_LOCAL_AGENT_BASE_URL")
    or os.getenv("COMPUTER_AGENT_BASE_URL")
    or ""
).strip().rstrip("/")
LOCAL_AGENT_TOKEN = (
    os.getenv("LOCAL_AGENT_TOKEN")
    or os.getenv("AI_BRAIN_LOCAL_AGENT_TOKEN")
    or os.getenv("COMPUTER_AGENT_TOKEN")
    or ""
).strip()
try:
    LOCAL_AGENT_TIMEOUT_SEC = float(os.getenv("LOCAL_AGENT_TIMEOUT_SEC") or "30")
except Exception:
    LOCAL_AGENT_TIMEOUT_SEC = 30.0


def _local_agent_is_configured() -> bool:
    return bool(LOCAL_AGENT_BASE_URL)


def _local_agent_url(path: str) -> str:
    if not LOCAL_AGENT_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail={
                "ok": False,
                "error": "LOCAL_AGENT_BASE_URL is not configured",
                "hint": "Set LOCAL_AGENT_BASE_URL=http://10.0.0.204:4317 and LOCAL_AGENT_TOKEN=local-dev-token in the backend .env",
            },
        )
    if not path.startswith("/"):
        path = "/" + path
    return f"{LOCAL_AGENT_BASE_URL}{path}"


def _local_agent_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if LOCAL_AGENT_TOKEN:
        headers["Authorization"] = f"Bearer {LOCAL_AGENT_TOKEN}"
    return headers


def _safe_response_payload(resp: requests.Response) -> Any:
    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            return resp.json()
        except Exception:
            pass
    text = resp.text or ""
    if len(text) > 2000:
        text = text[:2000] + "…"
    return {"ok": resp.ok, "status_code": resp.status_code, "body": text}


def _call_local_agent_sync(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
    """Blocking HTTP call to the LAN local-agent. Wrapped with asyncio.to_thread in routes."""
    method = (method or "GET").upper()
    url = _local_agent_url(path)
    try:
        kwargs: Dict[str, Any] = {
            "headers": _local_agent_headers(),
            "timeout": (3.0, LOCAL_AGENT_TIMEOUT_SEC),
        }
        if method != "GET":
            kwargs["json"] = body or {}
        resp = requests.request(method, url, **kwargs)
    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail={"ok": False, "error": "local agent timeout", "url": url},
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail={"ok": False, "error": "local agent unreachable", "url": url, "detail": str(e)},
        )

    payload = _safe_response_payload(resp)
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail={"ok": False, "error": "local agent returned error", "url": url, "payload": payload},
        )

    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        return payload

    # Preserve binary responses too. This keeps screenshot compatible even if the
    # local-agent later returns image/png or application/octet-stream instead of JSON.
    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type") or "application/octet-stream",
        status_code=resp.status_code,
    )


async def _call_local_agent(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
    return await asyncio.to_thread(_call_local_agent_sync, method, path, body)


async def _request_json_or_empty(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ─── Computer Agent v1 helpers(adu_computer_agent_v1 模块)─────────
# 防御性导入:模块不可用时,v1 端点退化为 503,核心 /api/brain/computer/* 不受影响。
try:
    import adu_computer_agent_v1 as _ADU_AGENT_V1  # noqa: F401
except Exception as _agent_v1_err:  # pragma: no cover
    _ADU_AGENT_V1 = None
    print(f"[adu] computer_agent_v1 模块加载失败: {_agent_v1_err}", flush=True)


@app.get("/api/brain/computer/config")
async def brain_computer_config():
    return {
        "ok": True,
        "configured": _local_agent_is_configured(),
        "base_url": LOCAL_AGENT_BASE_URL,
        "has_token": bool(LOCAL_AGENT_TOKEN),
        "timeout_sec": LOCAL_AGENT_TIMEOUT_SEC,
        "routes": [
            "GET  /api/brain/computer/health",
            "GET  /api/brain/computer/config",
            "GET  /api/brain/computer/mouse",
            "GET  /api/brain/computer/active_window",
            "GET  /api/brain/computer/tools",
            "POST /api/brain/computer/screenshot",
            "POST /api/brain/computer/move",
            "POST /api/brain/computer/click",
            "POST /api/brain/computer/double_click",
            "POST /api/brain/computer/type",
            "POST /api/brain/computer/paste",
            "POST /api/brain/computer/press",
            "POST /api/brain/computer/open_app",
            "POST /api/brain/computer/action",
        ],
    }


@app.get("/api/brain/computer/health")
async def brain_computer_health():
    return await _call_local_agent("GET", "/health")


@app.get("/api/brain/computer/mouse")
async def brain_computer_mouse():
    return await _call_local_agent("GET", "/api/computer/mouse")


@app.post("/api/brain/computer/screenshot")
async def brain_computer_screenshot(request: Request):
    body = await _request_json_or_empty(request)
    return await _call_local_agent("POST", "/api/computer/screenshot", body)


@app.post("/api/brain/computer/move")
async def brain_computer_move(request: Request):
    body = await _request_json_or_empty(request)
    return await _call_local_agent("POST", "/api/computer/move", body)


@app.post("/api/brain/computer/click")
async def brain_computer_click(request: Request):
    body = await _request_json_or_empty(request)
    return await _call_local_agent("POST", "/api/computer/click", body)


@app.post("/api/brain/computer/type")
async def brain_computer_type(request: Request):
    body = await _request_json_or_empty(request)
    return await _call_local_agent("POST", "/api/computer/type", body)



@app.post("/api/brain/computer/press")
async def brain_computer_press(request: Request):
    body = await _request_json_or_empty(request)
    return await _call_local_agent("POST", "/api/computer/press", body)


# ================================
# ✅ Computer Tools layer (Codex / 阿杜 直接可调用)
#    所有 endpoint 都是 local-agent :4317 端点的薄代理或封装。
#    Codex / 阿杜的稳定调用建议优先用这些独立端点，而不是 NL action。
# ================================

@app.post("/api/brain/computer/open_app")
async def brain_computer_open_app(request: Request):
    """直接打开 Mac App。{app, display?}。复用 _brain_computer_execute_action
    的 open_app 多方法回退(已含 Spotlight 兜底)。"""
    body = await _request_json_or_empty(request)
    app_name = str(body.get("app") or body.get("name") or body.get("query") or "").strip()
    display = str(body.get("display") or app_name).strip() or app_name
    if not app_name:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "missing_app"})
    return await _brain_computer_execute_action("open_app", {"app": app_name, "display": display})


@app.post("/api/brain/computer/paste")
async def brain_computer_paste(request: Request):
    """粘贴文本(local-agent 内做 pbcopy + cmd+v)。{text}"""
    body = await _request_json_or_empty(request)
    return await _call_local_agent("POST", "/api/computer/paste", body)


@app.get("/api/brain/computer/active_window")
async def brain_computer_active_window():
    """读取当前前台 App 与窗口标题(只读,risk=0)。"""
    return await _call_local_agent("GET", "/api/computer/active_window")


@app.post("/api/brain/computer/double_click")
async def brain_computer_double_click(request: Request):
    """双击(left,count=2)。{x?, y?, button?}"""
    body = await _request_json_or_empty(request)
    if not isinstance(body, dict):
        body = {}
    body.setdefault("button", "left")
    body.setdefault("count", 2)
    return await _call_local_agent("POST", "/api/computer/click", body)


# ---- 高风险动作分类器(供 /api/brain/computer/action 用)----

# 文本层:goal 里出现这些 → 直接返回 needs_user_confirmation,不走 NL 解析。
_HIGH_RISK_TEXT_PATTERNS = [
    (r'\bsudo\s', "sudo 命令需要二次确认"),
    (r'\brm\s+-r[fF]?\s', "递归删除命令需要二次确认"),
    (r'\bgit\s+push\b[^&;|]*--force\b', "git push --force 需要二次确认"),
    (r'\bgit\s+push\b[^&;|]*\s-f(?:\s|$)', "git push -f 需要二次确认"),
    (r'\bdiskutil\s+erase\b', "磁盘擦除需要二次确认"),
    (r'\bmkfs(?:\.|\s)', "格式化文件系统需要二次确认"),
    (r'格式化磁盘|清空磁盘|擦除磁盘', "磁盘擦除需要二次确认"),
    (r'\.env\b.*(?:覆盖|改写|删除)|(?:覆盖|改写|删除).*\.env\b', "改 .env 需要二次确认"),
    (r'allow.*keychain|允许.*钥匙串', "允许钥匙串需要二次确认"),
    (r'付款|支付|转账|结账', "付款/转账需要二次确认"),
    (r'发布版本|提交审核|上架审核', "发布/提交审核需要二次确认"),
]

# 动作层:解析后的 action+payload → 命中即 needs_user_confirmation。
_HIGH_RISK_HOTKEYS = [
    {"cmd", "s"}, {"cmd", "q"}, {"cmd", "w"},
    {"cmd", "delete"}, {"cmd", "shift", "delete"},
]


def _high_risk_goal_classifier(goal: str) -> Optional[str]:
    """文本风险 → 原因字符串;否则 None。
    单一源:委托给 adu_computer_agent_v1.ComputerRiskPolicy(包含'发微信/发消息/
    付款/sudo/git push/cmd+s/...等扩展关键词)。模块不可用 → 退回老内联规则。"""
    if _ADU_AGENT_V1 is not None:
        try:
            r = _ADU_AGENT_V1.ComputerRiskPolicy.classify_text(goal)
            if r:
                return r[1]
            return None
        except Exception:
            pass  # fall through to inline fallback
    if not isinstance(goal, str) or not goal.strip():
        return None
    text = goal.strip()
    low = text.lower()
    for pat, reason in _HIGH_RISK_TEXT_PATTERNS:
        try:
            if re.search(pat, low) or re.search(pat, text):
                return reason
        except re.error:
            continue
    return None


def _high_risk_action_classifier(action: str, payload: Dict[str, Any]) -> Optional[str]:
    """动作 + payload 风险 → 原因;否则 None。同样优先用统一 ComputerRiskPolicy。"""
    if _ADU_AGENT_V1 is not None:
        try:
            r = _ADU_AGENT_V1.ComputerRiskPolicy.classify_action(action, payload)
            if r:
                return r[1]
            return None
        except Exception:
            pass  # fall through
    a = (action or "").strip().lower()
    payload = payload if isinstance(payload, dict) else {}
    if a in ("press_keys", "press", "hotkey", "shortcut"):
        raw_keys = payload.get("keys") or payload.get("key") or payload.get("combo") or []
        if isinstance(raw_keys, str):
            raw_keys = re.split(r"[+\s,，、]+", raw_keys)
        keyset = {_computer_key_alias(k) for k in raw_keys if k}
        for combo in _HIGH_RISK_HOTKEYS:
            if combo and combo.issubset(keyset):
                return f"高风险热键 {'+'.join(sorted(combo))} 需要二次确认"
    return None


@app.get("/api/brain/computer/tools")
async def brain_computer_tools():
    """Codex / 阿杜的工具发现端点:返回所有 computer.* 工具的描述 + schema。"""
    tools = [
        {
            "name": "computer.health",
            "method": "GET",
            "endpoint": "/api/brain/computer/health",
            "description": "Ping local-agent (4317). 返回 {ok, agent, port}。",
            "input_schema": {},
            "risk_level": 0,
        },
        {
            "name": "computer.config",
            "method": "GET",
            "endpoint": "/api/brain/computer/config",
            "description": "查看 local-agent 配置与所有路由列表。",
            "input_schema": {},
            "risk_level": 0,
        },
        {
            "name": "computer.screenshot",
            "method": "POST",
            "endpoint": "/api/brain/computer/screenshot",
            "description": "拍当前 Mac 屏幕截图。返回 {ok, image_b64, bytes, format}。",
            "input_schema": {},
            "risk_level": 0,
        },
        {
            "name": "computer.mouse_position",
            "method": "GET",
            "endpoint": "/api/brain/computer/mouse",
            "description": "读鼠标当前坐标。返回 {ok, x, y}。",
            "input_schema": {},
            "risk_level": 0,
        },
        {
            "name": "computer.mouse_move",
            "method": "POST",
            "endpoint": "/api/brain/computer/move",
            "description": "把鼠标移到 (x, y)。",
            "input_schema": {"x": "int", "y": "int"},
            "risk_level": 1,
        },
        {
            "name": "computer.click",
            "method": "POST",
            "endpoint": "/api/brain/computer/click",
            "description": "鼠标点击。button=left/right, 可附带 x/y 否则在当前位置。",
            "input_schema": {"button": "left|right (default left)", "x": "int?", "y": "int?"},
            "risk_level": 1,
        },
        {
            "name": "computer.double_click",
            "method": "POST",
            "endpoint": "/api/brain/computer/double_click",
            "description": "双击(left,count=2)。可附带 x/y。",
            "input_schema": {"x": "int?", "y": "int?"},
            "risk_level": 1,
        },
        {
            "name": "computer.type_text",
            "method": "POST",
            "endpoint": "/api/brain/computer/type",
            "description": "在当前焦点处输入文本(osascript keystroke,Unicode 安全)。",
            "input_schema": {"text": "string"},
            "risk_level": 1,
        },
        {
            "name": "computer.paste_text",
            "method": "POST",
            "endpoint": "/api/brain/computer/paste",
            "description": "把文本放剪贴板并发 Cmd+V 粘贴到当前焦点。",
            "input_schema": {"text": "string"},
            "risk_level": 1,
        },
        {
            "name": "computer.press_keys",
            "method": "POST",
            "endpoint": "/api/brain/computer/press",
            "description": "按组合键。Cmd+S/Q/W、Cmd+Delete 等高风险热键会被 action 闸门拦下,本端点直传不拦截 —— 调用方负责安全。",
            "input_schema": {"keys": "[string]  例 ['cmd','space'] / ['enter'] / ['escape']"},
            "risk_level": 1,
        },
        {
            "name": "computer.active_window",
            "method": "GET",
            "endpoint": "/api/brain/computer/active_window",
            "description": "读当前前台 App 和窗口标题。",
            "input_schema": {},
            "risk_level": 0,
        },
        {
            "name": "computer.open_app",
            "method": "POST",
            "endpoint": "/api/brain/computer/open_app",
            "description": "打开 Mac App(走 Spotlight 兜底)。{app: 'Safari' | '微信' | 'WeChat' ...}。",
            "input_schema": {"app": "string", "display": "string?"},
            "risk_level": 1,
        },
        {
            "name": "computer.action",
            "method": "POST",
            "endpoint": "/api/brain/computer/action",
            "description": "自然语言或显式 action 统一入口。{goal:'打开 Safari'} 或 {action:'press_keys', payload:{keys:['enter']}}。失败返回 examples。",
            "input_schema": {
                "goal": "string  (自然语言)",
                "action": "string?  (显式 action 名)",
                "payload": "object? (action 的参数)",
            },
            "risk_level": 1,
            "examples": [
                {"goal": "打开 Safari"},
                {"goal": "打开微信"},
                {"goal": "输入 hello"},
                {"goal": "按回车"},
                {"goal": "按 Esc"},
                {"goal": "点击"},
                {"goal": "右键"},
                {"goal": "刷新截图"},
                {"action": "screenshot"},
                {"action": "press_keys", "payload": {"keys": ["cmd", "space"]}},
            ],
        },
        # ───────── v1 新增工具 ─────────
        {
            "name": "computer.right_click",
            "method": "POST",
            "endpoint": "/api/brain/computer/click",
            "description": "右键点击。等价 click + button=right。",
            "input_schema": {"x": "int?", "y": "int?"},
            "risk_level": 1,
        },
        {
            "name": "computer.file_list",
            "method": "POST",
            "endpoint": "/api/brain/computer/files/list",
            "description": "列出 allow_paths 内目录(只读)。",
            "input_schema": {"root": "string  (必须在 allow_paths 内)"},
            "risk_level": 0,
        },
        {
            "name": "computer.file_copy",
            "method": "POST",
            "endpoint": "/api/brain/computer/files/copy",
            "description": "复制文件/目录。默认 dry-run 返回 preview;confirm=true 才执行。",
            "input_schema": {"source": "string", "target": "string", "confirm": "bool (default false)"},
            "risk_level": 1,
        },
        {
            "name": "computer.file_move",
            "method": "POST",
            "endpoint": "/api/brain/computer/files/move",
            "description": "移动文件/目录(可跨目录)。默认 dry-run。",
            "input_schema": {"source": "string", "target": "string", "confirm": "bool"},
            "risk_level": 1,
        },
        {
            "name": "computer.file_rename",
            "method": "POST",
            "endpoint": "/api/brain/computer/files/rename",
            "description": "在同一目录内改名。默认 dry-run。",
            "input_schema": {"source": "string", "target": "string", "confirm": "bool"},
            "risk_level": 1,
        },
        {
            "name": "computer.agent_plan",
            "method": "POST",
            "endpoint": "/api/brain/computer/agent/run",
            "description": "把 goal 拆成多步 plan(不执行)。用 auto_execute=false。",
            "input_schema": {"goal": "string", "max_steps": "int (default 8)", "auto_execute": "false"},
            "risk_level": 0,
        },
        {
            "name": "computer.agent_run",
            "method": "POST",
            "endpoint": "/api/brain/computer/agent/run",
            "description": "半自动多步执行。auto_execute=true 时执行白名单内动作,遇高风险/未知步骤停下并 needs_user_confirmation。",
            "input_schema": {"goal": "string", "max_steps": "int (default 8)", "auto_execute": "bool"},
            "risk_level": 1,
        },
        {
            "name": "computer.vision_click",
            "method": "POST",
            "endpoint": "/api/brain/computer/vision/click",
            "description": "坐标点击 v1。{x,y,confirm:true} 才点;只给 target 文字时返回 coordinate_required + 当前截图尺寸。",
            "input_schema": {"x": "int?", "y": "int?", "confirm": "bool", "target": "string?"},
            "risk_level": 1,
        },
        {
            "name": "adu.tools.call",
            "method": "POST",
            "endpoint": "/api/adu/tools/call",
            "description": "给 Codex/阿杜的统一工具调度器,只允许 TOOL_WHITELIST 内工具。",
            "input_schema": {"tool": "string (whitelist)", "args": "object"},
            "risk_level": 1,
        },
    ]
    return {
        "ok": True,
        "service": "chatagi.computer.tools",
        "version": "1.0.0",
        "local_agent": {
            "configured": _local_agent_is_configured(),
            "base_url": LOCAL_AGENT_BASE_URL,
            "has_token": bool(LOCAL_AGENT_TOKEN),
        },
        "high_risk": {
            "policy": "高风险动作返回 needs_user_confirmation=true,不自动执行。",
            "blocked_hotkeys": [sorted(list(s)) for s in _HIGH_RISK_HOTKEYS],
            "blocked_text_patterns": [reason for _, reason in _HIGH_RISK_TEXT_PATTERNS],
        },
        "tools": tools,
    }


# ================================
# ✅ Unified Computer Action API
# 自然语言/调试面板统一入口：
#   POST /api/brain/computer/action
# 支持两种请求：
#   1) {"goal":"输入 hello 然后按回车"}
#   2) {"action":"press_keys","payload":{"keys":["cmd","space"]}}
# ================================

def _computer_key_alias(key: Any) -> str:
    k = str(key or "").strip().lower().replace(" ", "")
    aliases = {
        "command": "cmd", "cmd": "cmd", "⌘": "cmd", "meta": "cmd", "super": "cmd",
        "control": "ctrl", "ctrl": "ctrl", "^": "ctrl",
        "option": "alt", "opt": "alt", "alt": "alt", "⌥": "alt",
        "shift": "shift", "⇧": "shift",
        "return": "enter", "enter": "enter", "↩": "enter",
        "esc": "escape", "escape": "escape",
        "spacebar": "space", "space": "space", "空格": "space",
        "tab": "tab", "delete": "delete", "backspace": "backspace",
        "up": "up", "down": "down", "left": "left", "right": "right",
    }
    return aliases.get(k, k)


def _normalize_computer_keys(keys: Any) -> List[str]:
    if isinstance(keys, str):
        raw = re.split(r"[+,，、/\\s]+", keys.strip())
    elif isinstance(keys, list):
        raw = keys
    else:
        raw = []
    out: List[str] = []
    for item in raw:
        k = _computer_key_alias(item)
        if k and k not in out:
            out.append(k)
    return out


def _jsonable_result(value: Any) -> Any:
    if isinstance(value, Response):
        return {"ok": True, "response_type": "binary", "media_type": getattr(value, "media_type", None)}
    return value



# ---- ChatAGI 阿杜 · Computer Work Mode policy（独立模块，导入失败降级运行）----
try:
    import adu_computer_work as _ADU_CW  # noqa: F401
except Exception as _adu_cw_err:  # pragma: no cover
    _ADU_CW = None
    print(f"[adu] computer_work 模块加载失败，已降级运行: {_adu_cw_err}")


def _adu_cw_confirm_text(action: str, risk_level: int, reason: str = "") -> str:
    """生成给前端展示的确认文案；模块不可用时降级。"""
    if _ADU_CW is not None:
        try:
            return _ADU_CW.confirm_text_for(action, risk_level, reason)
        except Exception:
            pass
    return f"动作「{action}」需要确认（risk_level={risk_level}）。{reason or ''}"


def _extract_app_to_open_from_goal(goal: str) -> Optional[Dict[str, str]]:
    text = (goal or "").strip()
    if not text:
        return None
    low = text.lower()
    compact = re.sub(r"[\s，,。.!！]+", "", low)
    if any(k in text for k in ["聚焦搜索", "打开搜索", "打开电脑搜索"]) or "spotlight" in low:
        return None

    has_intent = any(k in text for k in ["打开", "启动", "运行", "帮我打开", "给我打开"]) or low.startswith("open ") or low.startswith("launch ")
    if not has_intent:
        return None

    aliases = [
        (["微信", "wechat", "wechat"], "WeChat", "微信"),
        (["chrome", "谷歌浏览器", "谷歌", "googlechrome"], "Google Chrome", "Chrome"),
        (["safari", "苹果浏览器"], "Safari", "Safari"),
        (["xcode"], "Xcode", "Xcode"),
        (["终端", "terminal", "命令行"], "Terminal", "终端"),
        (["备忘录", "notes"], "Notes", "备忘录"),
        (["访达", "finder"], "Finder", "访达"),
        (["设置", "系统设置", "systemsettings", "systemsettings"], "System Settings", "系统设置"),
        (["邮件", "mail"], "Mail", "邮件"),
        (["照片", "photos"], "Photos", "照片"),
        (["日历", "calendar"], "Calendar", "日历"),
        (["音乐", "music"], "Music", "音乐"),
    ]
    for keys, query, display in aliases:
        for k in keys:
            if k.lower().replace(" ", "") in compact:
                return {"query": query, "display": display}

    for marker in ["帮我打开", "给我打开", "打开", "启动", "运行"]:
        if marker in text:
            value = text.split(marker, 1)[1].strip().strip("：:，,。.!！")
            if value and len(value) <= 40:
                return {"query": value, "display": value}
    if low.startswith("open "):
        value = text[5:].strip()
        if value:
            return {"query": value, "display": value}
    if low.startswith("launch "):
        value = text[7:].strip()
        if value:
            return {"query": value, "display": value}
    return None

def _computer_action_from_goal(goal: str) -> Optional[Dict[str, Any]]:
    text = (goal or "").strip()
    if not text:
        return None
    low = text.lower()


    app_spec = _extract_app_to_open_from_goal(text)
    if app_spec:
        return {"action": "open_app", "payload": {"app": app_spec["query"], "display": app_spec["display"]}}

    if any(k in text for k in [
        "截图", "截屏", "看屏幕", "看看屏幕", "屏幕看看",
        # V0.1 视觉理解触发词
        "你看到什么", "你看见什么", "屏幕上有什么", "看一下屏幕", "看下屏幕",
    ]) or "screenshot" in low:
        return {"action": "screenshot", "payload": {}}
    # V1.5: 鼠标位置（扩展"在哪/哪里"）
    if ("鼠标" in text and ("位置" in text or "在哪" in text or "哪里" in text)) or "mouse position" in low:
        return {"action": "mouse_position", "payload": {}}
    # V1.5: 前台应用 / 当前窗口（只读，risk_level=0）
    if any(k in text for k in ["前台应用", "当前应用", "当前窗口", "哪个应用", "什么应用", "前台窗口", "active window", "frontmost"]):
        return {"action": "active_window", "payload": {}}
    # V1.5: 鼠标移动到 (X, Y)  支持 "移动到 500,500" / "移到 (500, 500)" / "鼠标移到 500、500"
    m = re.search(r"(?:鼠标)?\s*(?:移动到|移到|移动至)\s*[(\[（【]?\s*(\d+)\s*[,，、\s]+\s*(\d+)\s*[)\]）】]?", text)
    if m:
        try:
            xx, yy = int(m.group(1)), int(m.group(2))
            return {"action": "mouse_move", "payload": {"x": xx, "y": yy}}
        except ValueError:
            pass
    if any(k in text for k in ["聚焦搜索", "spotlight", "全局搜索"]) or ("command" in low and "space" in low) or ("cmd" in low and "space" in low):
        return {"action": "press_keys", "payload": {"keys": ["cmd", "space"]}}

    # "输入 hello，然后按回车" 必须拆成 type_text + press_keys enter；
    # 不要让后面的 enter/return 分支吞掉整句，也不要把回车当成普通文本。
    m = re.search(
        r"^(?:输入|打字|键入)[:：\s]*(.+?)\s*(?:，|,|。|\s)*(?:然后|并且|并|再)?\s*(?:按|敲)?\s*(?:回车|enter|return)\s*$",
        text,
        re.I,
    )
    if not m:
        m = re.search(
            r"^\s*type\s+(.+?)\s*(?:,|，|。|\s)*(?:then\s+)?(?:press\s+)?(?:enter|return)\s*$",
            text,
            re.I,
        )
    if m:
        val = (m.group(1) or "").strip().strip('"“”')
        if val:
            return {
                "action": "sequence",
                "payload": {
                    "steps": [
                        {"action": "type_text", "payload": {"text": val}},
                        {"action": "press_keys", "payload": {"keys": ["enter"]}},
                    ]
                },
            }

    # 回车 / Enter / return 都映射到 press_keys ["enter"](绝不 type_text)。
    # 注意:"return"/"按 return" 之前没覆盖,会落到 not_recognized;补齐。
    if (any(k in text for k in ["按回车", "敲回车", "回车", "按 return", "按return", "按Return", "按 Return"])
            or "enter" in low
            or "return" in low):
        return {"action": "press_keys", "payload": {"keys": ["enter"]}}
    # V1.5: Esc 用小写 low 做检查，覆盖 "按 Esc / 按一下 Esc / Press Escape" 等大小写变体
    if any(k in text for k in ["退出键", "取消键"]) or "esc" in low or "escape" in low:
        return {"action": "press_keys", "payload": {"keys": ["escape"]}}
    if any(k in text for k in ["按空格", "空格键"]) or " space" in (" " + low):
        return {"action": "press_keys", "payload": {"keys": ["space"]}}
    if "tab" in low or "制表键" in text:
        return {"action": "press_keys", "payload": {"keys": ["tab"]}}
    if any(k in text for k in ["退格", "删除一个字"]) or "backspace" in low:
        return {"action": "press_keys", "payload": {"keys": ["backspace"]}}

    # V1.5: 通用 hotkey "按 Command S" / "按 cmd+s" / "按 control alt delete"
    # 要求至少一个修饰键，避免把"按 a"误判
    m = re.search(
        r"按\s*((?:cmd|ctrl|alt|shift|command|control|option|meta|⌘|⌃|⌥|⇧)(?:\s*[+\s,，、]\s*[A-Za-z0-9⌘⌃⌥⇧一-龥]+){1,3})",
        text, re.IGNORECASE,
    )
    if m:
        raw = m.group(1)
        parts = re.split(r"[+\s,，、]+", raw)
        keys = [p.strip().lower() for p in parts if p.strip()]
        # ⌘/⌃/⌥/⇧ 归一
        sym_map = {"⌘": "cmd", "⌃": "ctrl", "⌥": "alt", "⇧": "shift"}
        keys = [sym_map.get(k, k) for k in keys]
        if len(keys) >= 2:
            return {"action": "press_keys", "payload": {"keys": keys}}

    m = re.search(r"(?:输入|打字|键入)[:：\s]*(.+)$", text, re.I)
    if not m:
        m = re.search(r"\btype\s+(.+)$", text, re.I)
    if m:
        val = (m.group(1) or "").strip().strip('"“”')
        if val:
            return {"action": "type_text", "payload": {"text": val}}

    # V1.5: 粘贴 X / paste X
    m = re.search(r"粘贴[:：\s]*(.+)$", text)
    if not m:
        m = re.search(r"\bpaste\s+(.+)$", text, re.I)
    if m:
        val = (m.group(1) or "").strip().strip('"“”')
        if val:
            return {"action": "paste_text", "payload": {"text": val}}

    # V1.5：click/right_click/double_click 可附带坐标
    _xy_match = re.search(r"(\d{1,5})\s*[,，、]\s*(\d{1,5})", text)
    if any(k in text for k in ["右键", "点右键", "右击"]):
        pl = {"button": "right"}
        if _xy_match:
            try: pl["x"], pl["y"] = int(_xy_match.group(1)), int(_xy_match.group(2))
            except ValueError: pass
        return {"action": "click", "payload": pl}
    if any(k in text for k in ["双击", "double click"]):
        pl = {"button": "left"}
        if _xy_match:
            try: pl["x"], pl["y"] = int(_xy_match.group(1)), int(_xy_match.group(2))
            except ValueError: pass
        return {"action": "double_click", "payload": pl}
    if any(k in text for k in ["点击", "点一下", "左键", "单击"]) or "click" in low:
        pl = {"button": "left"}
        if _xy_match:
            try: pl["x"], pl["y"] = int(_xy_match.group(1)), int(_xy_match.group(2))
            except ValueError: pass
        return {"action": "click", "payload": pl}
    return None


async def _brain_computer_execute_action(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    action = (action or "").strip().lower()
    payload = payload if isinstance(payload, dict) else {}

    if action in ("screenshot", "screen_shot"):
        result = await _call_local_agent("POST", "/api/computer/screenshot", payload)
        return {"ok": True, "action": "screenshot", "message": "截图已完成。", "result": _jsonable_result(result)}
    if action in ("mouse", "mouse_position", "position"):
        result = await _call_local_agent("GET", "/api/computer/mouse")
        return {"ok": True, "action": "mouse_position", "message": "鼠标位置已读取。", "result": _jsonable_result(result)}
    if action in ("open_app", "open_application", "launch_app", "launch"):
        app_name = str(payload.get("app") or payload.get("name") or payload.get("query") or "").strip()
        display = str(payload.get("display") or app_name).strip() or app_name
        if not app_name:
            raise HTTPException(status_code=400, detail={"ok": False, "error": "missing app"})

        # Computer Work Mode：多方法回退 + 验证
        #   open -a 英文名 → open -a 中文名 → /Applications → mdfind → osascript → Spotlight 兜底
        async def _spotlight_open() -> List[Dict[str, Any]]:
            sp: List[Dict[str, Any]] = []
            sp.append(await _brain_computer_execute_action("press_keys", {"keys": ["cmd", "space"]}))
            await asyncio.sleep(0.35)
            sp.append(await _brain_computer_execute_action("type_text", {"text": app_name}))
            await asyncio.sleep(0.25)
            sp.append(await _brain_computer_execute_action("press_keys", {"keys": ["enter"]}))
            return sp

        if _ADU_CW is not None:
            try:
                strat = await asyncio.to_thread(_ADU_CW.resolve_and_open_app, app_name, display)
            except Exception as e:
                strat = {"ok": False, "completed": False, "attempts": [],
                         "spotlight_needed": True, "final_text": f"open_app 策略异常：{e}"}
            attempts = list(strat.get("attempts") or [])
            if strat.get("ok"):
                return {"ok": True, "action": "open_app", "completed": True,
                        "message": strat.get("final_text") or f"已打开{display}。",
                        "app": app_name, "display": display,
                        "attempts": attempts, "result": strat}
            # 子进程方法全部失败 → Spotlight 键盘兜底（无法程序化验证）
            sp_steps = await _spotlight_open()
            attempts.append({"step": len(attempts) + 1, "method": "Spotlight 键盘兜底",
                             "ok": True, "verified_by": "",
                             "reason": "子进程方法全部失败后的兜底，未做程序化验证"})
            return {"ok": True, "action": "open_app", "completed": False,
                    "message": f"已用 Spotlight 兜底尝试打开{display}（未能程序化验证）。",
                    "app": app_name, "display": display, "attempts": attempts,
                    "result": {"spotlight_fallback": True, "spotlight_steps": sp_steps,
                               "strategy": strat}}

        # 模块不可用 → 退回旧 Spotlight 行为
        steps = await _spotlight_open()
        return {"ok": True, "action": "open_app", "completed": False,
                "message": f"已尝试打开{display}。",
                "app": app_name, "display": display, "steps": steps}
    if action in ("move", "move_mouse", "mouse_move"):
        result = await _call_local_agent("POST", "/api/computer/move", payload)
        return {"ok": True, "action": "move", "message": "鼠标已移动。", "result": _jsonable_result(result)}
    if action in ("click", "double_click", "right_click", "left_click"):
        body = dict(payload)
        if action == "right_click": body["button"] = "right"
        elif action == "left_click": body["button"] = "left"
        if action == "double_click": body["count"] = 2
        body.setdefault("button", "left")
        if body.get("x") is None or body.get("y") is None:
            pos = await _call_local_agent("GET", "/api/computer/mouse")
            if isinstance(pos, dict):
                if body.get("x") is None: body["x"] = pos.get("x")
                if body.get("y") is None: body["y"] = pos.get("y")
        result = await _call_local_agent("POST", "/api/computer/click", body)
        msg = "已双击。" if action == "double_click" else ("已右键点击。" if body.get("button") == "right" else "已点击。")
        return {"ok": True, "action": action, "message": msg, "result": _jsonable_result(result)}
    if action in ("type", "type_text", "input_text"):
        text_value = str(payload.get("text") or payload.get("value") or "")
        result = await _call_local_agent("POST", "/api/computer/type", {"text": text_value})
        return {"ok": True, "action": "type_text", "message": "文字已输入。", "result": _jsonable_result(result)}
    # V1.5: 粘贴文本（pbcopy + cmd+v 在 local-agent 内做）
    if action in ("paste", "paste_text"):
        text_value = str(payload.get("text") or payload.get("value") or "")
        result = await _call_local_agent("POST", "/api/computer/paste", {"text": text_value})
        return {"ok": True, "action": "paste_text", "message": "文字已粘贴。", "result": _jsonable_result(result)}
    # V1.5: 前台窗口（只读）
    if action in ("active_window", "frontmost_app", "current_window", "active_app"):
        result = await _call_local_agent("GET", "/api/computer/active_window")
        msg = "前台窗口已读取。"
        if isinstance(result, dict) and result.get("app"):
            wt = result.get("window_title") or ""
            msg = f"当前前台：{result['app']}" + (f"｜窗口：{wt}" if wt else "")
        return {"ok": True, "action": "active_window", "message": msg, "result": _jsonable_result(result)}
    # V1.5: 鼠标拖拽
    if action in ("drag", "mouse_drag"):
        result = await _call_local_agent("POST", "/api/computer/drag", payload)
        return {"ok": True, "action": "drag", "message": "鼠标已拖拽。", "result": _jsonable_result(result)}
    if action in ("press", "press_key", "press_keys", "hotkey", "shortcut"):
        keys = _normalize_computer_keys(payload.get("keys") or payload.get("key") or payload.get("combo"))
        if not keys:
            raise HTTPException(status_code=400, detail={"ok": False, "error": "missing keys"})
        if len(keys) > 1:
            runner = globals().get("_run_usecomputer")
            if callable(runner):
                combo = "+".join(keys)
                direct = await asyncio.to_thread(runner, ["usecomputer", "press", combo], 20)
                if isinstance(direct, dict) and direct.get("ok"):
                    return {"ok": True, "action": "press_keys", "message": f"已按快捷键：{combo}", "result": direct, "keys": keys}
                try:
                    result = await _call_local_agent("POST", "/api/computer/press", {"keys": keys})
                    return {"ok": True, "action": "press_keys", "message": f"已按快捷键：{combo}", "result": _jsonable_result(result), "keys": keys, "direct_fallback_error": direct}
                except HTTPException as e:
                    raise HTTPException(status_code=e.status_code, detail={"ok": False, "error": "press_keys_failed", "keys": keys, "direct": direct, "local_agent": e.detail})
        result = await _call_local_agent("POST", "/api/computer/press", {"keys": keys})
        return {"ok": True, "action": "press_keys", "message": f"已按键：{'+'.join(keys)}", "result": _jsonable_result(result), "keys": keys}
    if action in ("wait", "sleep"):
        seconds = float(payload.get("seconds") or payload.get("sec") or 1)
        seconds = max(0.0, min(seconds, 30.0))
        await asyncio.sleep(seconds)
        return {"ok": True, "action": "wait", "message": f"已等待 {seconds:g} 秒。", "waited": seconds}
    raise HTTPException(status_code=400, detail={"ok": False, "error": "unsupported_action", "action": action})


@app.post("/api/brain/computer/action")
async def brain_computer_action(request: Request):
    body = await _request_json_or_empty(request)
    action = str(body.get("action") or "").strip()
    payload = body.get("payload") or body.get("args") or {}
    if not isinstance(payload, dict): payload = {}
    goal = str(body.get("goal") or body.get("text") or body.get("command") or "").strip()
    inferred = False

    # ─── 安全闸门 1:文本层(goal 含 sudo / git push --force / 付款 等 → 直接拦)───
    if goal:
        hr_text_reason = _high_risk_goal_classifier(goal)
        if hr_text_reason:
            return {
                "ok": True,
                "action": "needs_user_confirmation",
                "needs_user_confirmation": True,
                "executed": False,
                "risk_level": 2,
                "risk_reason": hr_text_reason,
                "goal": goal,
                "message": f"{hr_text_reason}。本接口不会自动执行,请你确认后改用专用流程。",
            }

    # 自然语言 → action+payload
    if not action and goal:
        spec = _computer_action_from_goal(goal)
        if not spec:
            return {
                "ok": False,
                "action": "not_recognized",
                "error": "not_simple_computer_action",
                "goal": goal,
                "message": "暂时不支持这个电脑动作,请试:打开 Safari / 打开微信 / 输入 hello / 按回车 / 按 Esc / 点击 / 右键 / 刷新截图。",
                "examples": [
                    {"goal": "打开 Safari"},
                    {"goal": "打开微信"},
                    {"goal": "输入 hello"},
                    {"goal": "按回车"},
                    {"goal": "按 Esc"},
                    {"goal": "点击"},
                    {"goal": "右键"},
                    {"goal": "刷新截图"},
                ],
            }
        action = spec["action"]
        payload = spec.get("payload") or {}
        inferred = True

    if not action:
        raise HTTPException(status_code=400, detail={
            "ok": False,
            "error": "missing_action_or_goal",
            "hint": "请传 {goal:'自然语言'} 或 {action:'screenshot', payload:{}}。",
        })

    # ─── 安全闸门 2:动作层(Cmd+S/Q/W、Cmd+Delete 等高风险热键 → 拦)───
    hr_act_reason = _high_risk_action_classifier(action, payload)
    if hr_act_reason:
        return {
            "ok": True,
            "action": action,
            "needs_user_confirmation": True,
            "executed": False,
            "risk_level": 2,
            "risk_reason": hr_act_reason,
            "payload": payload,
            "goal": goal,
            "inferred": inferred,
            "message": f"{hr_act_reason}。本接口不会自动执行。",
        }

    if action == "sequence":
        raw_steps = payload.get("steps") if isinstance(payload, dict) else []
        if not isinstance(raw_steps, list) or not raw_steps:
            raise HTTPException(status_code=400, detail={"ok": False, "error": "missing_sequence_steps"})
        steps: List[Dict[str, Any]] = []
        for raw in raw_steps:
            if not isinstance(raw, dict):
                raise HTTPException(status_code=400, detail={"ok": False, "error": "bad_sequence_step"})
            step_action = str(raw.get("action") or "").strip()
            step_payload = raw.get("payload") or raw.get("args") or {}
            if not isinstance(step_payload, dict):
                step_payload = {}
            step_risk = _high_risk_action_classifier(step_action, step_payload)
            if step_risk:
                return {
                    "ok": True,
                    "action": "sequence",
                    "needs_user_confirmation": True,
                    "executed": False,
                    "risk_level": 2,
                    "risk_reason": step_risk,
                    "payload": payload,
                    "goal": goal,
                    "inferred": inferred,
                    "message": f"{step_risk}。本接口不会自动执行。",
                }
            step_result = await _brain_computer_execute_action(step_action, step_payload)
            step_ok = bool(step_result.get("ok")) if isinstance(step_result, dict) else True
            item: Dict[str, Any] = {"action": step_action, "ok": step_ok}
            if step_action in ("type", "type_text", "input_text"):
                item["action"] = "type_text"
                item["text"] = str(step_payload.get("text") or step_payload.get("value") or "")
            elif step_action in ("press", "press_key", "press_keys", "hotkey", "shortcut"):
                item["action"] = "press_keys"
                item["keys"] = _normalize_computer_keys(step_payload.get("keys") or step_payload.get("key") or step_payload.get("combo"))
            if isinstance(step_result, dict) and not step_ok:
                item["error"] = step_result.get("error") or step_result.get("message")
            steps.append(item)
        return {
            "ok": all(bool(s.get("ok")) for s in steps),
            "action": "sequence",
            "steps": steps,
            "message": "已输入文本并按回车",
            "direct": True,
            "inferred": inferred,
            "executed": True,
            "needs_user_confirmation": False,
            "goal": goal,
        }

    # 执行
    result = await _brain_computer_execute_action(action, payload)
    if isinstance(result, dict):
        result["direct"] = True
        result["inferred"] = inferred
        result.setdefault("executed", True)
        result.setdefault("needs_user_confirmation", False)
        if goal:
            result["goal"] = goal
    return result


# ════════════════════════════════════════════════════════════════════
# ✅ Computer Agent v1  —— 半自动多步执行 + 文件操作 + 视觉点击 + 工具调度器
#    所有 v1 端点都过 ComputerRiskPolicy(adu_computer_agent_v1 模块)。
#    模块不可用时这些端点返回 503,不影响上面的 /api/brain/computer/*。
# ════════════════════════════════════════════════════════════════════

def _agent_v1_unavailable() -> Dict[str, Any]:
    return {"ok": False, "error": "agent_v1_unavailable",
            "hint": "adu_computer_agent_v1 模块未加载,检查后端日志"}


# ─── Agent v1: /api/brain/computer/agent/run ────────────────────────

@app.post("/api/brain/computer/agent/run")
async def brain_computer_agent_run(request: Request):
    if _ADU_AGENT_V1 is None:
        return _agent_v1_unavailable()
    body = await _request_json_or_empty(request)
    goal = str(body.get("goal") or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "missing_goal"})
    max_steps = int(body.get("max_steps") or 8)
    auto_execute = bool(body.get("auto_execute"))

    # 顶层文本风险
    top_risk = _ADU_AGENT_V1.ComputerRiskPolicy.classify_text(goal)
    if top_risk:
        return {
            "ok": True, "goal": goal,
            "status": "needs_user_confirmation",
            "steps": [],
            "needs_user_confirmation": True,
            "risk_level": top_risk[0],
            "risk_reason": top_risk[1],
            "next_confirmation": {
                "reason": top_risk[1],
                "pending_action": None,
                "pending_args": {},
            },
        }

    # 拆 plan(每步都过自己的风险 + 白名单)
    steps = _ADU_AGENT_V1.plan_steps_from_goal(goal, max_steps=max_steps)
    if not steps:
        return {"ok": False, "goal": goal, "status": "empty_plan",
                "steps": [], "needs_user_confirmation": False,
                "message": "无法把 goal 拆成步骤"}

    # auto_execute=false:只回 plan
    if not auto_execute:
        return {
            "ok": True, "goal": goal,
            "status": "plan_only",
            "steps": steps,
            "needs_user_confirmation": True,
            "next_confirmation": {
                "reason": "auto_execute=false,仅生成计划。把 auto_execute 改为 true 才会执行。",
            },
        }

    # auto_execute=true:逐步执行,遇 ok=false 即停
    executed: List[Dict[str, Any]] = []
    for step in steps:
        if not step.get("ok"):
            return {
                "ok": True, "goal": goal,
                "status": "needs_user_confirmation",
                "steps": executed,
                "needs_user_confirmation": True,
                "next_confirmation": {
                    "reason": step.get("reason") or "未识别 / 高风险步骤,停下来等你确认",
                    "pending_step": step,
                    "pending_action": step.get("action"),
                    "pending_args": step.get("args") or {},
                },
            }
        action = step["action"]
        args = step.get("args") or {}
        try:
            res = await _brain_computer_execute_action(action, args)
            executed.append({
                "index": step["index"],
                "raw": step.get("raw", ""),
                "action": action,
                "args": args,
                "ok": bool(res.get("ok")) if isinstance(res, dict) else False,
                "message": (res.get("message") if isinstance(res, dict) else "") or "",
            })
            # 每步执行后自动截图(action 本身是 screenshot 时跳过)
            if action != "screenshot":
                try:
                    await _brain_computer_execute_action("screenshot", {})
                except Exception:
                    pass
        except Exception as e:
            executed.append({
                "index": step["index"],
                "action": action, "args": args,
                "ok": False, "error": str(e),
            })
            return {"ok": False, "goal": goal, "status": "error",
                    "steps": executed, "error": str(e)}

    return {
        "ok": True, "goal": goal,
        "status": "completed",
        "steps": executed,
        "needs_user_confirmation": False,
    }


# ─── Vision click v1 ───────────────────────────────────────────────

@app.post("/api/brain/computer/vision/click")
async def brain_computer_vision_click(request: Request):
    body = await _request_json_or_empty(request)
    target = str(body.get("target") or "").strip()
    x_raw = body.get("x")
    y_raw = body.get("y")
    confirm = bool(body.get("confirm"))
    button = str(body.get("button") or "left").strip() or "left"

    # 有坐标 → confirm=true 才执行
    if x_raw is not None and y_raw is not None:
        try:
            xi = int(x_raw); yi = int(y_raw)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid_coordinates",
                    "got": {"x": x_raw, "y": y_raw}}
        if not confirm:
            return {
                "ok": False, "needs_user_confirmation": True,
                "preview": f"Will click ({xi},{yi}) [{button}]",
                "x": xi, "y": yi, "button": button,
                "hint": "再发一次同样请求并加 confirm=true 才会真点。",
            }
        result = await _brain_computer_execute_action(
            "click", {"button": button, "x": xi, "y": yi})
        if isinstance(result, dict):
            result["x"] = xi; result["y"] = yi
        return result

    # 只有自然语言 target / 啥都没传 → 返回截图尺寸,不点
    shot = await _brain_computer_execute_action("screenshot", {})
    inner = shot.get("result") if isinstance(shot, dict) else None
    inner = inner if isinstance(inner, dict) else {}
    return {
        "ok": False,
        "error": "coordinate_required",
        "message": "当前版本需要明确坐标。请传 {x, y, confirm:true};或先截图自己挑坐标。",
        "screenshot_width": inner.get("width"),
        "screenshot_height": inner.get("height"),
        "screenshot_bytes": inner.get("bytes"),
        "screenshot_format": inner.get("format"),
        "target": target,
        "needs_user_confirmation": True,
        "hint": "v1 不接视觉定位模型。下一版本会加 vision_grounding。",
    }


# ─── 文件操作 v1 ───────────────────────────────────────────────────

@app.post("/api/brain/computer/files/list")
async def brain_computer_files_list(request: Request):
    if _ADU_AGENT_V1 is None:
        return _agent_v1_unavailable()
    body = await _request_json_or_empty(request)
    root = str(body.get("root") or body.get("path") or "").strip()
    return _ADU_AGENT_V1.list_dir(root)


@app.post("/api/brain/computer/files/rename")
async def brain_computer_files_rename(request: Request):
    if _ADU_AGENT_V1 is None:
        return _agent_v1_unavailable()
    body = await _request_json_or_empty(request)
    source = str(body.get("source") or "").strip()
    target = str(body.get("target") or "").strip()
    confirm = bool(body.get("confirm"))
    if not confirm:
        return _ADU_AGENT_V1.file_op_preview("rename", source, target)
    return _ADU_AGENT_V1.file_op_execute("rename", source, target)


@app.post("/api/brain/computer/files/copy")
async def brain_computer_files_copy(request: Request):
    if _ADU_AGENT_V1 is None:
        return _agent_v1_unavailable()
    body = await _request_json_or_empty(request)
    source = str(body.get("source") or "").strip()
    target = str(body.get("target") or "").strip()
    confirm = bool(body.get("confirm"))
    if not confirm:
        return _ADU_AGENT_V1.file_op_preview("copy", source, target)
    return _ADU_AGENT_V1.file_op_execute("copy", source, target)


@app.post("/api/brain/computer/files/move")
async def brain_computer_files_move(request: Request):
    if _ADU_AGENT_V1 is None:
        return _agent_v1_unavailable()
    body = await _request_json_or_empty(request)
    source = str(body.get("source") or "").strip()
    target = str(body.get("target") or "").strip()
    confirm = bool(body.get("confirm"))
    if not confirm:
        return _ADU_AGENT_V1.file_op_preview("move", source, target)
    return _ADU_AGENT_V1.file_op_execute("move", source, target)


# ─── /api/adu/tools/call —— 给 Codex/阿杜的统一工具调度器 ────────────
# 只允许 TOOL_WHITELIST 内的工具;所有参数走 ComputerRiskPolicy。

async def _adu_tool_dispatch(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    a = args if isinstance(args, dict) else {}
    if tool == "computer.health":
        return await _call_local_agent("GET", "/health")
    if tool == "computer.config":
        return {"ok": True, "configured": _local_agent_is_configured(),
                "base_url": LOCAL_AGENT_BASE_URL}
    if tool == "computer.screenshot":
        return await _brain_computer_execute_action("screenshot", {})
    if tool == "computer.mouse_position":
        return await _brain_computer_execute_action("mouse_position", {})
    if tool == "computer.active_window":
        return await _brain_computer_execute_action("active_window", {})
    if tool == "computer.mouse_move":
        return await _brain_computer_execute_action("mouse_move", a)
    if tool == "computer.click":
        return await _brain_computer_execute_action("click", a)
    if tool == "computer.right_click":
        a2 = dict(a); a2["button"] = "right"
        return await _brain_computer_execute_action("click", a2)
    if tool == "computer.double_click":
        return await _brain_computer_execute_action("double_click", a)
    if tool == "computer.type_text":
        return await _brain_computer_execute_action("type_text", a)
    if tool == "computer.paste_text":
        return await _brain_computer_execute_action("paste_text", a)
    if tool == "computer.press_keys":
        return await _brain_computer_execute_action("press_keys", a)
    if tool == "computer.open_app":
        return await _brain_computer_execute_action("open_app", a)
    if tool == "computer.action":
        goal = str(a.get("goal") or "").strip()
        if goal:
            spec = _computer_action_from_goal(goal)
            if spec:
                return await _brain_computer_execute_action(spec["action"], spec.get("payload") or {})
        action = str(a.get("action") or "").strip()
        if action:
            return await _brain_computer_execute_action(action, a.get("payload") or {})
        return {"ok": False, "error": "missing_action_or_goal"}
    if tool == "computer.file_list":
        if _ADU_AGENT_V1 is None: return _agent_v1_unavailable()
        return _ADU_AGENT_V1.list_dir(str(a.get("root") or a.get("path") or ""))
    if tool in ("computer.file_copy", "computer.file_move", "computer.file_rename"):
        if _ADU_AGENT_V1 is None: return _agent_v1_unavailable()
        op = tool.split(".")[-1].replace("file_", "")
        source = str(a.get("source") or "")
        target = str(a.get("target") or "")
        confirm = bool(a.get("confirm"))
        if not confirm:
            return _ADU_AGENT_V1.file_op_preview(op, source, target)
        return _ADU_AGENT_V1.file_op_execute(op, source, target)
    if tool == "computer.agent_plan":
        if _ADU_AGENT_V1 is None: return _agent_v1_unavailable()
        steps = _ADU_AGENT_V1.plan_steps_from_goal(
            str(a.get("goal") or ""),
            max_steps=int(a.get("max_steps") or 8),
        )
        return {"ok": True, "plan_only": True, "steps": steps}
    if tool in ("computer.agent_run", "computer.vision_click"):
        return {"ok": False, "error": "use_dedicated_endpoint",
                "endpoint": ("/api/brain/computer/agent/run" if tool.endswith("agent_run")
                             else "/api/brain/computer/vision/click"),
                "hint": "tool/call 不直接代理 agent_run / vision_click,请直打专用端点。"}
    return {"ok": False, "error": "tool_not_implemented_yet", "tool": tool}


@app.post("/api/adu/tools/call")
async def adu_tools_call(request: Request):
    if _ADU_AGENT_V1 is None:
        return _agent_v1_unavailable()
    body = await _request_json_or_empty(request)
    tool = str(body.get("tool") or "").strip()
    args = body.get("args") or {}
    if not isinstance(args, dict): args = {}
    if not tool:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "missing_tool"})
    # 白名单
    if tool not in _ADU_AGENT_V1.TOOL_WHITELIST:
        return {"ok": False, "error": "tool_not_whitelisted",
                "tool": tool,
                "whitelist": _ADU_AGENT_V1.TOOL_WHITELIST}
    # 参数文本风险
    for key in ("text", "goal", "command", "value"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            rc = _ADU_AGENT_V1.ComputerRiskPolicy.classify_text(v)
            if rc:
                return {"ok": True, "tool": tool,
                        "needs_user_confirmation": True, "executed": False,
                        "risk_level": rc[0], "risk_reason": rc[1],
                        "message": f"参数 {key} 命中高风险:{rc[1]}。本调度器不会自动执行。"}
    # 动作层风险(press_keys 等)
    if tool in ("computer.press_keys",):
        rc = _ADU_AGENT_V1.ComputerRiskPolicy.classify_action("press_keys", args)
        if rc:
            return {"ok": True, "tool": tool,
                    "needs_user_confirmation": True, "executed": False,
                    "risk_level": rc[0], "risk_reason": rc[1],
                    "message": f"{rc[1]}。本调度器不会自动执行。"}
    # 真调度
    try:
        result = await _adu_tool_dispatch(tool, args)
    except HTTPException as e:
        return {"ok": False, "tool": tool, "error": "tool_call_failed",
                "detail": e.detail}
    except Exception as e:
        return {"ok": False, "tool": tool,
                "error": f"tool_call_exception: {type(e).__name__}: {e}"}
    if isinstance(result, dict):
        result.setdefault("tool", tool)
    return result


# ================================
# ✅ /api/brain/adu/act
# 统一入口：text_chat / audio_transcript / realtime_voice / realtime_av / debug_panel
# 都走这一个端口，由它判定 chat vs computer，并复用现有底层执行器。
# 设计要点：
#   - 不是电脑任务 → {"ok": False, "route": "chat"}，前端继续走原聊天流
#   - 是电脑任务 → 复用 _computer_action_from_goal 解析 + _brain_computer_execute_action 执行
#   - 风险分级：0 观察类 / 1 输入类 / 2 其他
#   - allow_execute=False → 只返回计划（need_confirm=True，不执行）
#   - allow_execute=True 且 risk_level <= 阈值 → 执行；否则 need_confirm=True 不执行
# ================================

_ADU_ACT_SOURCES = {"text_chat", "audio_transcript", "realtime_voice", "realtime_av", "debug_panel"}

_ADU_ACT_RISK = {
    # 观察类（只读，无副作用，risk_level=0）
    "screenshot": 0,
    "mouse_position": 0,
    "active_window": 0,    # V1.5
    "wait": 0,
    # 输入类（有副作用但通常用户期望，risk_level=1）
    "open_app": 1,
    "type_text": 1,
    "paste_text": 1,       # V1.5
    "press_keys": 1,
    "hotkey": 1,           # V1.5 别名
    "click": 1,
    "double_click": 1,
    "right_click": 1,
    "left_click": 1,
    "move": 1,
    "mouse_move": 1,       # V1.5 同义
    "drag": 1,             # V1.5
    "mouse_drag": 1,       # V1.5 同义
    # 高风险（shell / 删除 / git push / sudo / 发消息 / 付款）默认走 fallback risk=2，需 need_confirm=true
}

_ADU_ACT_AUTO_RISK_THRESHOLD = 1


def _adu_act_risk(action: str) -> int:
    return _ADU_ACT_RISK.get((action or "").strip().lower(), 2)


def _adu_act_final_text(action: str, result: Any) -> str:
    if isinstance(result, dict):
        msg = result.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
        inner = result.get("result")
        if action == "mouse_position" and isinstance(inner, dict):
            x = inner.get("x")
            y = inner.get("y")
            if x is not None and y is not None:
                return f"鼠标位置：x={x}, y={y}"
    return "电脑动作已执行。"


# ---- V0.1 视觉理解：截图后自动让多模态模型描述屏幕 ----
# 模型选择优先级：ADU_VISION_MODEL > 默认（有 DashScope key 时用 qwen-vl-max-latest，否则 gpt-4o）
# 路由优先级：模型名以 "qwen" 开头 → 走 DashScope OpenAI 兼容模式；否则 → OpenAI
_VISION_MODEL = (
    os.getenv("ADU_VISION_MODEL")
    or ("qwen-vl-max-latest" if DASHSCOPE_API_KEY else "gpt-4o")
).strip()
_VISION_DETAIL = (os.getenv("ADU_VISION_DETAIL") or "low").strip()
_VISION_MAX_TOKENS = int(os.getenv("ADU_VISION_MAX_TOKENS") or "400")
_VISION_TIMEOUT_S = float(os.getenv("ADU_VISION_TIMEOUT_SEC") or "30")


def _is_qwen_vision_model(name: str) -> bool:
    n = (name or "").strip().lower()
    return n.startswith("qwen") or "qwen-vl" in n


def _extract_image_b64_from_result(result: Any) -> Optional[str]:
    """从 _brain_computer_execute_action 的截图返回里提取 base64 PNG。
    结构: {"ok":True,"action":"screenshot","message":"...","result":{"image_b64":"..."}}
    """
    if not isinstance(result, dict):
        return None
    inner = result.get("result")
    if isinstance(inner, dict):
        b64 = inner.get("image_b64") or inner.get("image_base64")
        if isinstance(b64, str) and b64:
            return b64
    b64 = result.get("image_b64") or result.get("image_base64")
    if isinstance(b64, str) and b64:
        return b64
    return None


async def _vision_describe_screenshot(image_b64: str, user_goal: str = "") -> tuple[bool, str, str]:
    """调用多模态模型分析屏幕截图。返回 (ok, summary, error)。"""
    if not image_b64:
        return False, "", "截图为空"

    use_qwen = _is_qwen_vision_model(_VISION_MODEL)
    if use_qwen:
        if not DASHSCOPE_API_KEY:
            return False, "", "DASHSCOPE_API_KEY 未配置"
        api_url = f"{DASHSCOPE_BASE_URL.rstrip('/')}/chat/completions"
        api_key = DASHSCOPE_API_KEY
        provider_label = "dashscope"
    else:
        if not OPENAI_API_KEY:
            return False, "", "OPENAI_API_KEY 未配置"
        api_url = "https://api.openai.com/v1/chat/completions"
        api_key = OPENAI_API_KEY
        provider_label = "openai"

    data_url = f"data:image/png;base64,{image_b64}"
    goal_hint = (user_goal or "").strip()
    prompt = (
        "你是阿杜的视觉助手。看这张 Mac 屏幕截图，用一两句中文简洁、具体地描述："
        "前台是什么应用、显示了什么内容、有没有明显的状态/数字/错误。"
        "不要客套，不要假设我看不到。"
    )
    if goal_hint:
        prompt += f"\n用户的提问背景：「{goal_hint}」。"

    image_block: Dict[str, Any] = {"type": "image_url", "image_url": {"url": data_url}}
    # OpenAI 支持 detail（low/high/auto），DashScope 兼容模式不识别，传过去会被忽略或报错
    if not use_qwen:
        image_block["image_url"]["detail"] = _VISION_DETAIL

    body = {
        "model": _VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_block,
            ],
        }],
        "max_tokens": _VISION_MAX_TOKENS,
        "temperature": 0.4,
    }

    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=_VISION_TIMEOUT_S) as cli:
            r = await cli.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
    except Exception as e:
        return False, "", f"[{provider_label}] 网络异常 {type(e).__name__}: {e}"

    if r.status_code >= 400:
        snippet = (r.text or "")[:200]
        return False, "", f"[{provider_label}] HTTP {r.status_code} {snippet}"

    try:
        obj = r.json()
        content = (obj.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        summary = str(content or "").strip()
        if not summary:
            return False, "", f"[{provider_label}] 模型返回空"
        return True, summary, ""
    except Exception as e:
        return False, "", f"[{provider_label}] 解析失败 {type(e).__name__}: {e}"


# ---- V1.5 Observe→Decide→Act→Verify 安全闭环 ----
# 高影响动作执行前必须先截图 + 视觉分析 + 安全裁决；不安全的不执行。

# 这些 action 触发"先看再做"流程
_ADU_OBSERVE_BEFORE: set = {
    "click", "double_click", "right_click", "left_click",
    "drag", "mouse_drag",
    "type_text", "type", "input_text",
    "paste_text", "paste",
    "press_keys", "press", "hotkey",
    "open_app", "open_application", "launch_app", "launch",
}

# 低风险按键白名单：单独按这些键不需要 observe（用户可在 6 之前直接按）
_ADU_LOW_RISK_KEYS: set = {
    "escape", "esc",
    "tab",
    "up", "down", "left", "right",
    "arrowup", "arrowdown", "arrowleft", "arrowright",
    "enter", "return",
    "pageup", "pagedown", "home", "end",
}

# 始终需要 need_confirm 的高影响热键（即便用户授权 allow_execute=true 也必须二次确认）
_ADU_NEED_CONFIRM_HOTKEYS: list = [
    {"cmd", "s"}, {"command", "s"},     # 保存（可能弹出对话框）
    {"cmd", "q"}, {"command", "q"},     # 退出当前 app
    {"cmd", "w"}, {"command", "w"},     # 关闭窗口
    {"cmd", "tab"}, {"command", "tab"}, # 切换 app
    {"cmd", "delete"}, {"command", "delete"},  # 删除（Finder）
    {"cmd", "shift", "delete"}, {"command", "shift", "delete"},
]

# 出现在 pre_summary 里 → 强烈建议 needs_confirm
_ADU_DANGER_KEYWORDS: list = [
    # 中文
    "保存", "删除", "发送", "提交", "退出", "登出",
    "发布", "购买", "付款", "支付", "转账",
    "确认删除", "永久删除", "格式化", "清空",
    # 英文
    "save", "delete", "send", "submit", "publish",
    "quit", "log out", "sign out",
    "buy", "pay", "checkout",
    "format", "wipe", "erase",
]


def _adu_needs_observe(action: str, payload: Dict[str, Any]) -> bool:
    a = (action or "").strip().lower()
    if a not in _ADU_OBSERVE_BEFORE:
        return False
    if a in ("press_keys", "press", "hotkey", "shortcut"):
        keys = _normalize_computer_keys(payload.get("keys") or payload.get("key") or payload.get("combo")) or []
        keyset = {k.strip().lower() for k in keys if k}
        if keyset and keyset.issubset(_ADU_LOW_RISK_KEYS):
            return False
    return True


def _adu_safety_decide(action: str, payload: Dict[str, Any], pre_summary: str) -> tuple[str, str]:
    """返回 (decision, reason)；decision ∈ {"safe","needs_confirm","unsafe"}。
    规则化裁决（V1.5 第一版，未来可接 LLM）。
    """
    a = (action or "").strip().lower()
    pre = (pre_summary or "").lower()

    # 1) 高影响热键 → 始终需要二次确认
    if a in ("press_keys", "press", "hotkey", "shortcut"):
        keys = _normalize_computer_keys(payload.get("keys") or []) or []
        keyset = {k.strip().lower() for k in keys if k}
        for risky in _ADU_NEED_CONFIRM_HOTKEYS:
            if risky.issubset(keyset):
                return ("needs_confirm", f"高影响热键 {'+'.join(sorted(risky))} 需用户明确确认。")

    # 2) 视觉摘要里出现敏感关键词 → 需要确认（按钮/动作可能就在屏幕上）
    #    open_app 只是启动应用、不与当前屏幕元素交互，豁免该门控以免误拦。
    if a not in ("open_app", "open_application", "launch_app", "launch"):
        hits = [kw for kw in _ADU_DANGER_KEYWORDS if kw in pre]
        if hits:
            return ("needs_confirm", f"视觉发现敏感界面元素：{', '.join(hits[:4])}。")

    return ("safe", "")


async def _adu_observe(user_goal: str = "") -> tuple[bool, str, Optional[str]]:
    """截图 + 视觉摘要。返回 (ok, summary, image_b64)。"""
    try:
        shot = await _brain_computer_execute_action("screenshot", {})
    except Exception as e:
        return False, f"截图失败：{type(e).__name__}: {e}", None
    image_b64 = _extract_image_b64_from_result(shot) if shot else None
    if not image_b64:
        return False, "未能拿到截图 b64", None
    v_ok, v_summary, v_err = await _vision_describe_screenshot(image_b64, user_goal)
    if not v_ok:
        return False, f"视觉分析失败：{v_err}", image_b64
    return True, v_summary, image_b64


# ---- V0.2 上下文意图理解：按 conversation_id 缓存最近一次电脑观察 ----
# 仅做轻量内存缓存：进程重启即丢失，5 分钟 TTL，最多 30 个 conversation。
_ADU_CTX: Dict[str, Dict[str, Any]] = {}
_ADU_CTX_TTL_SEC = float(os.getenv("ADU_CTX_TTL_SEC") or "300")
_ADU_CTX_MAX_ENTRIES = int(os.getenv("ADU_CTX_MAX_ENTRIES") or "30")


def _adu_ctx_prune() -> None:
    now = time.time()
    expired = [
        k for k, v in _ADU_CTX.items()
        if now - float(v.get("last_observed_at", 0)) > _ADU_CTX_TTL_SEC
    ]
    for k in expired:
        _ADU_CTX.pop(k, None)
    if len(_ADU_CTX) > _ADU_CTX_MAX_ENTRIES:
        ordered = sorted(
            _ADU_CTX.items(),
            key=lambda kv: float(kv[1].get("last_observed_at", 0)),
        )
        for k, _v in ordered[: len(_ADU_CTX) - _ADU_CTX_MAX_ENTRIES]:
            _ADU_CTX.pop(k, None)


def _adu_ctx_save(conversation_id: str, action: str,
                  image_b64: Optional[str], summary: Optional[str]) -> None:
    if not conversation_id:
        return
    _ADU_CTX[conversation_id] = {
        "last_computer_action": action,
        "last_screenshot_b64": image_b64,
        "last_vision_summary": summary,
        "last_observed_at": time.time(),
    }
    _adu_ctx_prune()


def _adu_ctx_get(conversation_id: str) -> Optional[Dict[str, Any]]:
    if not conversation_id:
        return None
    rec = _ADU_CTX.get(conversation_id)
    if not rec:
        return None
    if time.time() - float(rec.get("last_observed_at", 0)) > _ADU_CTX_TTL_SEC:
        _ADU_CTX.pop(conversation_id, None)
        return None
    return rec


# 强 follow-up：在 ctx 存在时即可触发屏幕复用（语义明确指向"刚才看到的东西"）
_ADU_FOLLOWUP_STRONG = [
    "你看到了什么", "你看见了什么",
    "看到了什么了", "看见了什么了", "看到什么了", "看见什么了",
    "上面是什么", "上面有什么", "上面写", "上面显示",
    "看出来了吗", "看明白了吗", "看清楚了吗",
    "那是什么", "现在是什么", "刚才那个", "刚刚那个",
    "屏幕里", "屏幕上", "画面里", "画面上",
]

# 弱 follow-up：ctx 存在 + 不含明显聊天意图时才触发
_ADU_FOLLOWUP_WEAK = [
    "什么情况", "现在呢", "怎么样", "什么意思",
]

# 排除：含这些词视为普通聊天意图，即便 ctx 在也不抢路由
_ADU_CHAT_GUARDS = [
    "你觉得", "你认为", "你看呢",
    "这个想法", "这个方案", "这个主意", "这个建议", "这种想法",
    "靠谱吗", "对不对", "好不好", "可行吗", "合适吗", "行不行",
]


def _is_screen_followup(text: str, has_ctx: bool) -> bool:
    """判断当前文本是否是对屏幕的追问/指代（仅在有 ctx 时返回 True）。"""
    if not has_ctx:
        return False
    t = (text or "").strip()
    if not t:
        return False
    if any(g in t for g in _ADU_CHAT_GUARDS):
        return False
    if any(p in t for p in _ADU_FOLLOWUP_STRONG):
        return True
    if any(p in t for p in _ADU_FOLLOWUP_WEAK):
        return True
    return False


# ---- V1 工程修复闭环：识别"修工程/修bug/跑测试"类意图，复用 agent_loop_router ----
import threading as _adu_threading  # 局部别名，避免与上面文件其他 threading 用法歧义

# 工程修复触发词（精准，避免与"修家电""bug 反馈"等普通聊天冲突）
_ADU_ENG_TRIGGERS = [
    "修复工程", "修工程", "修一下工程", "修一下项目", "修复项目",
    "修复 bug", "修 bug", "修一下 bug", "修复bug", "修bug", "修一下bug",
    "修一下编译", "检查编译", "编译错误", "compile error", "build fail", "构建失败",
    "跑测试", "运行测试", "跑一下测试", "跑 test", "跑test",
    "跑 pytest", "跑pytest", "npm test", "pytest",
    "自动改代码", "自动修代码", "自动修复",
    "工程修复",
]

# 高风险动作关键词（命中则 need_confirm=true，禁止 V1 阶段直接执行）
_ADU_ENG_HIGH_RISK_PATTERNS = [
    r"\brm\s+-rf?\b", r"删除文件", r"清空目录",
    r"\bgit\s+push\b", r"推送到远程", r"推到远程", r"推到主分支", r"推到 main", r"推到master",
    r"\bsudo\b",
    r"发布", r"上线", r"\bdeploy\b", r"\brelease\b",
    r"安装依赖", r"\bpip\s+install\b", r"\bnpm\s+install\b", r"\bbrew\s+install\b",
    r"\bnpm\s+publish\b", r"\bpip\s+publish\b",
]


def _is_engineering_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    for kw in _ADU_ENG_TRIGGERS:
        if kw.lower() in t:
            return True
    return False


def _engineering_high_risk(text: str) -> bool:
    t = (text or "").strip()
    for pat in _ADU_ENG_HIGH_RISK_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            return True
    return False


# 串行化 BASE_DIR override，避免并发工程运行互相踩
_ENG_BASE_DIR_LOCK = _adu_threading.Lock()


def _start_engineering_run(goal: str, engineering_root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """启动一次 agent_loop run（强制 dry_run=True）。
    engineering_root: 可选；若提供，则把模块全局 BASE_DIR 临时切到该目录，
    让 _copy_repo_to_workdir 只复制这个小目录而不是整个 backend。
    出于安全考虑，engineering_root 必须在 /tmp/ 之下（macOS 上 /private/tmp/ 同义）。
    """
    if _agent_loop_module is None:
        return None
    try:
        store = getattr(_agent_loop_module, "STORE", None)
        worker = getattr(_agent_loop_module, "_run_worker", None)
        default_model = getattr(_agent_loop_module, "DEFAULT_MODEL", "gpt-4o-mini")
        default_provider = getattr(_agent_loop_module, "DEFAULT_PROVIDER", "openai")
        default_test_cmd = getattr(_agent_loop_module, "AGENT_TEST_CMD_DEFAULT", "python -m compileall -q .")
        if store is None or worker is None:
            return None

        eng_root_path: Optional[Any] = None
        if engineering_root:
            from pathlib import Path as _Path  # local import to avoid header churn
            target = _Path(engineering_root).expanduser().resolve()
            target_str = str(target)
            if not (target_str.startswith("/tmp/") or target_str.startswith("/private/tmp/")):
                return {"error": f"engineering_root 必须在 /tmp/ 之下，得到: {target_str}"}
            if not target.is_dir():
                return {"error": f"engineering_root 不是目录: {target_str}"}
            eng_root_path = target

        st = store.create(
            goal=goal,
            dry_run=True,                       # V1 强制 dry_run：不污染线上代码
            test_cmd=default_test_cmd,
            model=default_model,
            provider=default_provider,
        )

        # 如果提供了 engineering_root，临时把 BASE_DIR 切到它
        # 拿锁防并发；watcher 线程在 worker 读完 BASE_DIR（status 离开 queued）后立刻还原
        if eng_root_path is not None:
            _ENG_BASE_DIR_LOCK.acquire()
            original_base = _agent_loop_module.BASE_DIR
            _agent_loop_module.BASE_DIR = eng_root_path

            def _restore_watcher(run_id: str, store_obj: Any, orig: Any) -> None:
                try:
                    # 最多等 30s，正常 100~500ms 内 status 就离开 queued
                    for _ in range(300):
                        st_now = store_obj.get(run_id)
                        if not st_now or st_now.status != "queued":
                            break
                        time.sleep(0.1)
                finally:
                    _agent_loop_module.BASE_DIR = orig
                    try:
                        _ENG_BASE_DIR_LOCK.release()
                    except Exception:
                        pass

            _adu_threading.Thread(
                target=_restore_watcher,
                args=(st.run_id, store, original_base),
                daemon=True,
            ).start()

        t = _adu_threading.Thread(
            target=worker,
            args=(st.run_id, goal, True, default_test_cmd, default_model, default_provider),
            daemon=True,
        )
        t.start()

        return {
            "task_id": st.run_id,
            "status": "queued",
            "dry_run": True,
            "model": default_model,
            "provider": default_provider,
            "test_cmd": default_test_cmd,
            "engineering_root": str(eng_root_path) if eng_root_path else None,
            "events_url": f"/agent/events/{st.run_id}",
            "result_url": f"/agent/result/{st.run_id}",
            "bundle_url": f"/agent/bundle/{st.run_id}.zip",
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ================================
# Computer Work Mode · 任务级执行器（task executors）
# 一个用户目标 = 一个任务。任务内低风险步骤（截图/视觉/前台/鼠标位置/open_app/
# Esc-Tab-方向/在已确认安全输入框打字）自主连续执行并记录 attempts，不每步弹确认。
# 只有高风险动作（Cmd+S/Q/W、删除、发送、付款、git push、sudo…）才 need_confirm。
# 不做 LLM 多步规划，只做固定任务流水。
# ================================

def _adu_mk_attempt(step: int, method: str, ok: bool,
                    verified_by: str = "", reason: str = "") -> Dict[str, Any]:
    return {"step": int(step), "method": str(method), "ok": bool(ok),
            "verified_by": verified_by or "", "reason": reason or ""}


def _adu_task_envelope(goal: str, source: str, task: str, *,
                       completed: bool, attempts: List[Dict[str, Any]],
                       risk_level: int, need_confirm: bool, executed: bool,
                       final_text: str, safety_decision: str = "safe",
                       result: Any = None, confirm_text: Optional[str] = None,
                       **extra: Any) -> Dict[str, Any]:
    """任务级统一返回结构。"""
    out: Dict[str, Any] = {
        "ok": True, "route": "computer", "mode": "computer_work",
        "source": source, "goal": goal, "action": task, "task": task,
        "completed": bool(completed), "attempts": attempts or [],
        "risk_level": int(risk_level), "need_confirm": bool(need_confirm),
        "executed": bool(executed), "safety_decision": safety_decision,
        "result": result, "confirm_text": confirm_text, "final_text": final_text,
    }
    out.update(extra)
    return out


async def _adu_task_observe(goal: str, source: str) -> Dict[str, Any]:
    """observe_task：截图 → 视觉分析 → 前台窗口。全部 risk_level=0，自主执行。"""
    attempts: List[Dict[str, Any]] = []
    try:
        shot = await _brain_computer_execute_action("screenshot", {})
    except Exception as e:
        attempts.append(_adu_mk_attempt(1, "screenshot", False, reason=f"{type(e).__name__}: {e}"))
        return _adu_task_envelope(goal, source, "observe", completed=False, attempts=attempts,
                                  risk_level=0, need_confirm=False, executed=False,
                                  final_text=f"截图失败：{e}")
    img = _extract_image_b64_from_result(shot)
    attempts.append(_adu_mk_attempt(1, "screenshot", bool(img), "screenshot",
                                    "" if img else "未取得截图数据"))
    v_ok, v_summary, v_err = (False, "", "无截图")
    if img:
        v_ok, v_summary, v_err = await _vision_describe_screenshot(img, goal)
        attempts.append(_adu_mk_attempt(2, "vision_analyze", v_ok,
                                        "vision" if v_ok else "", "" if v_ok else v_err))
    aw_app = None
    try:
        aw = await _brain_computer_execute_action("active_window", {})
        inner = aw.get("result") if isinstance(aw, dict) else None
        aw_app = (inner or {}).get("app") if isinstance(inner, dict) else None
        attempts.append(_adu_mk_attempt(3, "active_window", bool(aw_app), "active_window",
                                        f"前台 {aw_app}" if aw_app else "未取得前台"))
    except Exception as e:
        attempts.append(_adu_mk_attempt(3, "active_window", False, reason=str(e)))
    parts: List[str] = []
    if aw_app:
        parts.append(f"当前前台：{aw_app}。")
    if v_ok:
        parts.append(f"我看到屏幕上 {v_summary}")
    elif img:
        parts.append(f"截图完成，但视觉分析失败：{v_err}")
    final = " ".join(parts) or "未能完成屏幕观察。"
    completed = bool(img) and (v_ok or bool(aw_app))
    return _adu_task_envelope(goal, source, "observe", completed=completed, attempts=attempts,
                              risk_level=0, need_confirm=False, executed=True, final_text=final,
                              result={"screenshot_ok": bool(img)},
                              screenshot_available=bool(img),
                              vision_summary=v_summary if v_ok else None,
                              active_window=aw_app, observed_before=True,
                              pre_action_summary=v_summary if v_ok else None)


async def _adu_task_mouse_position(goal: str, source: str) -> Dict[str, Any]:
    """mouse_position_task：读取鼠标坐标，risk_level=0。"""
    attempts: List[Dict[str, Any]] = []
    try:
        res = await _brain_computer_execute_action("mouse_position", {})
    except Exception as e:
        attempts.append(_adu_mk_attempt(1, "mouse_position", False, reason=str(e)))
        return _adu_task_envelope(goal, source, "mouse_position", completed=False,
                                  attempts=attempts, risk_level=0, need_confirm=False,
                                  executed=False, final_text=f"读取鼠标位置失败：{e}")
    inner = res.get("result") if isinstance(res, dict) else None
    x = (inner or {}).get("x") if isinstance(inner, dict) else None
    y = (inner or {}).get("y") if isinstance(inner, dict) else None
    ok = x is not None and y is not None
    attempts.append(_adu_mk_attempt(1, "mouse_position", ok, "process",
                                    f"x={x}, y={y}" if ok else "未取得坐标"))
    final = f"鼠标当前位置：x={x}, y={y}。" if ok else "未能读取鼠标位置。"
    return _adu_task_envelope(goal, source, "mouse_position", completed=ok, attempts=attempts,
                              risk_level=0, need_confirm=False, executed=True,
                              final_text=final, result=res)


async def _adu_task_open_app(goal: str, source: str) -> Dict[str, Any]:
    """open_app_task：多方法回退（open -a → 路径 → mdfind → osascript → Spotlight）+ 验证。
    全程自主，不每步弹确认。"""
    spec = _extract_app_to_open_from_goal(goal)
    if not spec:
        return _adu_task_envelope(goal, source, "open_app", completed=False, attempts=[],
                                  risk_level=1, need_confirm=False, executed=False,
                                  final_text="未能从目标里识别出要打开的应用。")
    query, display = spec["query"], spec["display"]
    try:
        res = await _brain_computer_execute_action(
            "open_app", {"app": query, "display": display})
    except Exception as e:
        return _adu_task_envelope(goal, source, "open_app", completed=False, attempts=[],
                                  risk_level=1, need_confirm=False, executed=True,
                                  final_text=f"打开{display}异常：{e}")
    attempts = (res.get("attempts") or []) if isinstance(res, dict) else []
    completed = bool(isinstance(res, dict) and res.get("completed"))
    msg = res.get("message") if isinstance(res, dict) else None
    return _adu_task_envelope(goal, source, "open_app", completed=completed, attempts=attempts,
                              risk_level=1, need_confirm=False, executed=True,
                              final_text=msg or f"已尝试打开{display}。", result=res)


async def _adu_task_dry_run_click(goal: str, source: str) -> Dict[str, Any]:
    """dry_run_click_task：只观察 + 安全评估，绝不执行真实 click。executed 恒为 False。"""
    payload: Dict[str, Any] = {}
    try:
        dr = (_ADU_CW.parse_dry_run_click(goal) if _ADU_CW is not None else None) or {}
        payload = dr.get("payload") or {}
    except Exception:
        payload = {}
    x, y = payload.get("x"), payload.get("y")
    xy = f"({x},{y})" if x is not None and y is not None else "(未给坐标)"
    attempts: List[Dict[str, Any]] = []
    obs_ok, summary, _b64 = await _adu_observe(user_goal=goal)
    attempts.append(_adu_mk_attempt(1, "observe(screenshot+vision)", obs_ok,
                                    "vision" if obs_ok else "", summary))
    if not obs_ok:
        return _adu_task_envelope(goal, source, "dry_run_click", completed=False,
                                  attempts=attempts, risk_level=1, need_confirm=False,
                                  executed=False, safety_decision="needs_confirm",
                                  final_text=f"【判断模式】无法感知屏幕，无法评估 {xy} 的点击：{summary}")
    decision, reason = _adu_safety_decide("click", payload, summary)
    attempts.append(_adu_mk_attempt(2, "safety_decide(click)", True, "vision",
                                    f"{decision}：{reason}" if reason else decision))
    final = (f"【判断模式】坐标 {xy} 的点击安全评估：{decision}。"
             + (f"{reason} " if reason else " ")
             + f"屏幕：{(summary or '')[:160]} 已按你的要求未执行点击。")
    return _adu_task_envelope(goal, source, "dry_run_click", completed=True, attempts=attempts,
                              risk_level=1, need_confirm=False, executed=False,
                              safety_decision=decision, final_text=final,
                              observed_before=True, pre_action_summary=summary)


async def _adu_task_textedit_input(goal: str, source: str) -> Dict[str, Any]:
    """safe_textedit_input_task：打开 TextEdit → Esc 关面板 → Cmd+N 新建空白文档 →
    截图/前台验证输入区就绪 →（目标带明确文字才）在已确认安全的空白文档里输入。
    绝不执行 Cmd+S（保存属高风险）。"""
    attempts: List[Dict[str, Any]] = []
    counter = {"n": 0}

    def add(method: str, ok: bool, vby: str = "", reason: str = "") -> None:
        counter["n"] += 1
        attempts.append(_adu_mk_attempt(counter["n"], method, ok, vby, reason))

    # 步骤 1：打开 TextEdit（多方法回退）
    try:
        opened = await _brain_computer_execute_action(
            "open_app", {"app": "TextEdit", "display": "TextEdit"})
    except Exception as e:
        add("open_app TextEdit", False, reason=str(e))
        return _adu_task_envelope(goal, source, "textedit_input", completed=False,
                                  attempts=attempts, risk_level=1, need_confirm=False,
                                  executed=True, final_text=f"打开 TextEdit 异常：{e}")
    for a in ((opened.get("attempts") or []) if isinstance(opened, dict) else []):
        counter["n"] += 1
        a2 = dict(a)
        a2["step"] = counter["n"]
        attempts.append(a2)
    if not bool(isinstance(opened, dict) and opened.get("completed")):
        return _adu_task_envelope(goal, source, "textedit_input", completed=False,
                                  attempts=attempts, risk_level=1, need_confirm=False,
                                  executed=True, result=opened,
                                  final_text="未能确认 TextEdit 已打开，停止准备输入区。")
    # 步骤 2：Esc 关掉可能弹出的打开/存储面板（低风险，自主）
    try:
        await _brain_computer_execute_action("press_keys", {"keys": ["escape"]})
        add("press Esc（关闭可能的面板）", True, "")
    except Exception as e:
        add("press Esc", False, reason=str(e))
    await asyncio.sleep(0.3)
    # 步骤 3：Cmd+N 新建空白文档（低风险，非 Cmd+S/Q/W，自主）
    try:
        await _brain_computer_execute_action("press_keys", {"keys": ["cmd", "n"]})
        add("press Cmd+N（新建空白文档）", True, "")
    except Exception as e:
        add("press Cmd+N", False, reason=str(e))
    await asyncio.sleep(0.5)
    # 步骤 4：截图 + 视觉验证输入区
    obs_ok, summary, _b64 = await _adu_observe(
        user_goal="确认 TextEdit 是否有一个空白可编辑文档作为安全输入区")
    add("verify 输入区（screenshot+vision）", obs_ok, "vision" if obs_ok else "", summary)
    # 步骤 5：前台窗口验证
    aw_app = None
    try:
        aw = await _brain_computer_execute_action("active_window", {})
        inner = aw.get("result") if isinstance(aw, dict) else None
        aw_app = (inner or {}).get("app") if isinstance(inner, dict) else None
    except Exception as e:
        add("verify active_window", False, reason=str(e))
    fg_ok = bool(aw_app and "textedit" in str(aw_app).replace(" ", "").lower())
    if aw_app is not None:
        add("verify active_window", fg_ok, "active_window", f"前台 {aw_app}")
    # 步骤 6（可选）：目标里带明确文字 → 在已确认安全的空白文档里输入
    type_text = None
    try:
        type_text = _ADU_CW.textedit_text_to_type(goal) if _ADU_CW is not None else None
    except Exception:
        type_text = None
    typed = False
    if type_text and fg_ok:
        try:
            await _brain_computer_execute_action("type_text", {"text": type_text})
            typed = True
            add("type_text 到 TextEdit 空白文档", True, "vision",
                f"已输入：{type_text[:40]}")
        except Exception as e:
            add("type_text", False, reason=str(e))
    wants_save = (any(k in goal for k in ("保存", "存储")) or "save" in goal.lower()) \
        and not any(k in goal for k in ("不保存", "不要保存", "别保存"))
    completed = fg_ok
    if completed and typed:
        final = (f"TextEdit 已打开，空白文档作为安全输入区已就绪，"
                 f"并已输入测试文字「{type_text}」。未保存文件。")
    elif completed:
        final = "TextEdit 已打开，空白文档作为安全输入区已就绪（未输入文字、未保存文件），可以安全输入。"
    else:
        final = f"TextEdit 已启动，但未能确认前台空白文档（当前前台：{aw_app}）。"
    if wants_save:
        final += "（注：保存=Cmd+S 属高风险，需你确认，本次未保存。）"
    return _adu_task_envelope(goal, source, "textedit_input", completed=completed,
                              attempts=attempts, risk_level=1, need_confirm=False,
                              executed=True, final_text=final, observed_before=True,
                              pre_action_summary=summary if obs_ok else None,
                              vision_summary=summary if obs_ok else None,
                              active_window=aw_app)


@app.post("/api/brain/adu/act")
async def api_brain_adu_act(request: Request):
    body = await _request_json_or_empty(request)

    source = str(body.get("source") or "").strip().lower()
    text = str(body.get("text") or body.get("goal") or "").strip()
    client_id = str(body.get("client_id") or "").strip()
    conversation_id = str(body.get("conversation_id") or "").strip()
    mode = str(body.get("mode") or "").strip()
    allow_execute_raw = body.get("allow_execute")
    allow_execute = True if allow_execute_raw is None else bool(allow_execute_raw)

    if source not in _ADU_ACT_SOURCES:
        raise HTTPException(status_code=400, detail={
            "ok": False,
            "error": "bad_source",
            "hint": f"source 必须是 {sorted(_ADU_ACT_SOURCES)} 之一",
        })
    if not text:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "missing text"})

    # Computer Work Mode：任务级执行器分派
    #   一个用户目标 = 一个任务；任务内低风险步骤自主连续执行并记录 attempts，不每步弹确认。
    #   未命中这 5 类任务 → 交回下方原单动作流程（含工程修复 / 屏幕追问 / 普通聊天）。
    if _ADU_CW is not None:
        try:
            _task_kind = _ADU_CW.classify_task(text)
        except Exception:
            _task_kind = None
        _task_res = None
        if _task_kind == "observe":
            _task_res = await _adu_task_observe(text, source)
        elif _task_kind == "mouse_position":
            _task_res = await _adu_task_mouse_position(text, source)
        elif _task_kind == "open_app":
            _task_res = await _adu_task_open_app(text, source)
        elif _task_kind == "dry_run_click":
            _task_res = await _adu_task_dry_run_click(text, source)
        elif _task_kind == "textedit_input":
            _task_res = await _adu_task_textedit_input(text, source)
        if _task_res is not None:
            _task_res.setdefault("client_id", client_id)
            _task_res.setdefault("conversation_id", conversation_id)
            _task_res.setdefault("input_mode", mode)
            return _task_res

    # 1) 普通聊天 → 前端继续走原聊天流，不在这里执行任何操作
    spec = _computer_action_from_goal(text)

    # Computer Work Mode：dry-run 点击判定（"判断 X,Y 能不能点，不要执行点击"）
    # 关键：必须独立于 _computer_action_from_goal 判定——"不要执行点击"本身含"点击"
    #       二字会被普通 click 分支命中，必须由 dry-run 覆盖，否则会真的点下去。
    dry_run = False
    if _ADU_CW is not None:
        try:
            _dr = _ADU_CW.parse_dry_run_click(text)
        except Exception:
            _dr = None
        if _dr:
            spec = {"action": _dr["action"], "payload": _dr.get("payload") or {}}
            dry_run = True
            allow_execute = False  # 判断模式：永不执行

    # 1.3) V1 工程修复：text 没命中电脑动作，但命中"修工程/修bug/跑测试"类
    if spec is None and _is_engineering_request(text):
        high_risk = _engineering_high_risk(text)
        risk_level = 2 if high_risk else 1

        # 高风险 或 allow_execute=False → 仅返回计划，不创建 run
        if (not allow_execute) or high_risk:
            reason = []
            if high_risk:
                reason.append("包含高风险动作（删除/推送/sudo/发布/安装依赖）")
            if not allow_execute:
                reason.append("调用方未授权执行")
            return {
                "ok": True,
                "route": "engineering",
                "source": source,
                "goal": text,
                "client_id": client_id,
                "conversation_id": conversation_id,
                "mode": mode,
                "action": "engineering_loop",
                "task_id": None,
                "status": "need_confirm",
                "steps": [],
                "logs": [],
                "need_confirm": True,
                "risk_level": risk_level,
                "result": {"plan_only": True, "reason": "；".join(reason)},
                "final_text": "已识别为工程修复任务，等待确认后再执行。" + (
                    "（包含高风险动作：删除/推送/sudo/发布/安装依赖）" if high_risk else ""
                ),
            }

        # 实际启动（dry_run=True 强制：V1 不污染源码，只产出 patch bundle）
        # 可选：engineering_root 把 agent_loop 的源码根临时切到指定 /tmp/ 子目录（V1 测试用）
        engineering_root_raw = body.get("engineering_root") or body.get("repo_root")
        engineering_root_arg = str(engineering_root_raw).strip() if engineering_root_raw else None
        task_info = _start_engineering_run(text, engineering_root=engineering_root_arg)
        if not task_info or task_info.get("error"):
            err = (task_info or {}).get("error") or "agent_loop_router 未加载"
            return {
                "ok": False,
                "route": "engineering",
                "source": source,
                "goal": text,
                "client_id": client_id,
                "conversation_id": conversation_id,
                "mode": mode,
                "action": "engineering_loop",
                "task_id": None,
                "status": "error",
                "steps": [],
                "logs": [],
                "need_confirm": False,
                "risk_level": risk_level,
                "result": {"error": err},
                "final_text": f"无法启动工程修复任务：{err}",
            }

        return {
            "ok": True,
            "route": "engineering",
            "source": source,
            "goal": text,
            "client_id": client_id,
            "conversation_id": conversation_id,
            "mode": mode,
            "action": "engineering_loop",
            "task_id": task_info["task_id"],
            "status": task_info["status"],
            "steps": [],
            "logs": [],
            "need_confirm": False,
            "risk_level": risk_level,
            "result": task_info,
            "final_text": (
                f"已启动工程修复任务（dry-run，task_id={task_info['task_id']}）。"
                f"实时进度：{task_info['events_url']}；最终结果：{task_info['result_url']}。"
            ),
        }

    # 1.5) V0.2 上下文意图理解：text 没明确命中 screenshot，但属于追问/指代屏幕
    if spec is None and conversation_id:
        ctx = _adu_ctx_get(conversation_id)
        if ctx and _is_screen_followup(text, has_ctx=True):
            cached_b64 = ctx.get("last_screenshot_b64")
            if cached_b64:
                # 复用上次截图，不重新截图
                ctx_age = int(time.time() - float(ctx.get("last_observed_at", 0)))
                base: Dict[str, Any] = {
                    "ok": True,
                    "route": "computer",
                    "source": source,
                    "goal": text,
                    "client_id": client_id,
                    "conversation_id": conversation_id,
                    "mode": mode,
                    "need_confirm": False,
                    "risk_level": 0,
                    "action": "analyze_last_screenshot",
                    "payload": {
                        "reused_from": "context",
                        "observed_age_sec": ctx_age,
                    },
                    "result": {
                        "ok": True,
                        "action": "analyze_last_screenshot",
                        "message": "复用最近一次截图分析。",
                        "result": {
                            "reused": True,
                            "observed_age_sec": ctx_age,
                            "image_b64_len": len(cached_b64),
                        },
                    },
                    "screenshot_available": True,
                    "vision_summary": None,
                    "final_text": None,
                }
                v_ok, v_summary, v_err = await _vision_describe_screenshot(cached_b64, text)
                if v_ok:
                    base["vision_summary"] = v_summary
                    base["final_text"] = f"我看到屏幕上 {v_summary}"
                    _adu_ctx_save(conversation_id, "analyze_last_screenshot", cached_b64, v_summary)
                else:
                    base["final_text"] = f"截图已复用，但视觉分析失败：{v_err}"
                return base
            # ctx 在但没缓存图（极端情况） → fallback：重新截图
            spec = {"action": "screenshot", "payload": {}}

    if spec is None:
        return {"ok": False, "route": "chat", "source": source, "goal": text}

    action = spec["action"]
    payload = spec.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    risk_level = _adu_act_risk(action)

    if not allow_execute:
        need_confirm = True
    else:
        need_confirm = risk_level > _ADU_ACT_AUTO_RISK_THRESHOLD

    # Computer Work Mode：高风险动作（Cmd+S/Q/W、sudo、git push、付款…）升级 risk_level
    cw_confirm_reason = ""
    if _ADU_CW is not None:
        try:
            risk_level, cw_confirm_reason = _ADU_CW.escalate_risk(action, payload, text, risk_level)
        except Exception:
            cw_confirm_reason = ""
    if risk_level > _ADU_ACT_AUTO_RISK_THRESHOLD:
        need_confirm = True

    # V1.5: 接收 action_mode（默认 observe_decide_act_verify，可由调用方覆盖）
    action_mode = str(body.get("action_mode") or "observe_decide_act_verify").strip().lower() or "observe_decide_act_verify"

    base: Dict[str, Any] = {
        "ok": True,
        "route": "computer",
        "source": source,
        "goal": text,
        "client_id": client_id,
        "conversation_id": conversation_id,
        "mode": "computer_work",
        "need_confirm": need_confirm,
        "risk_level": risk_level,
        "action": action,
        "payload": payload,
        "result": None,
        "final_text": None,
        # V1.5 新增字段
        "action_mode": action_mode,
        "observed_before": False,
        "pre_action_summary": None,
        "safety_decision": "safe",
        "executed": False,
        "post_action_summary": None,
        # Computer Work Mode 统一字段
        "input_mode": mode,
        "completed": False,
        "attempts": [],
        "confirm_text": None,
    }

    # V1.5 Observe→Decide：高影响动作执行前必须先感知 + 安全裁决
    needs_observe = _adu_needs_observe(action, payload) and action_mode != "fast"
    if needs_observe:
        obs_ok, pre_summary, _pre_b64 = await _adu_observe(user_goal=text)
        base["observed_before"] = obs_ok
        base["pre_action_summary"] = pre_summary
        if not obs_ok:
            # 感知失败 → 禁止盲执行
            base["safety_decision"] = "needs_confirm"
            base["risk_level"] = max(base.get("risk_level") or 0, 1)
            if dry_run:
                base["need_confirm"] = False
                base["final_text"] = f"【判断模式】无法感知屏幕，无法评估点击安全性：{pre_summary}"
            else:
                base["need_confirm"] = True
                base["confirm_text"] = _adu_cw_confirm_text(action, base["risk_level"], "屏幕感知失败")
                base["final_text"] = f"无法感知当前屏幕，禁止盲执行：{pre_summary}"
            return base
        decision, reason = _adu_safety_decide(action, payload, pre_summary)
        base["safety_decision"] = decision
        if decision in ("unsafe", "needs_confirm"):
            base["risk_level"] = max(base.get("risk_level") or 0, 1)
            preview = (pre_summary or "")[:200]
            if dry_run:
                base["need_confirm"] = False
                base["completed"] = True
                base["final_text"] = (
                    f"【判断模式】点击安全评估：{decision}。{reason} "
                    f"屏幕摘要：{preview} 已按你的要求未执行点击。"
                )
            else:
                base["need_confirm"] = True
                base["confirm_text"] = _adu_cw_confirm_text(action, base["risk_level"], reason)
                base["final_text"] = (
                    f"已识别动作但视觉感知判定为 {decision}：{reason} "
                    f"屏幕摘要：{preview}"
                )
            return base

    # 2) 需确认 / 仅返回计划 / 判断模式：不执行
    if need_confirm:
        if dry_run:
            xy = (f"({payload.get('x')},{payload.get('y')})"
                  if payload.get("x") is not None and payload.get("y") is not None
                  else "(未给坐标)")
            base["need_confirm"] = False
            base["completed"] = bool(base.get("observed_before"))
            base["confirm_text"] = None
            base["final_text"] = (
                f"【判断模式】坐标 {xy} 的点击安全评估：{base.get('safety_decision')}。"
                + (f" 屏幕：{(base.get('pre_action_summary') or '')[:160]}"
                   if base.get("pre_action_summary") else "")
                + " 已按你的要求未执行点击。"
            )
        elif base["observed_before"]:
            base["final_text"] = (
                "已识别为电脑任务，调用方未授权执行（allow_execute=false）。"
                f" 视觉感知：{(base['pre_action_summary'] or '')[:200]}"
            )
            base["confirm_text"] = _adu_cw_confirm_text(action, base["risk_level"], cw_confirm_reason)
        else:
            base["final_text"] = "已识别为电脑任务，等待确认。"
            base["confirm_text"] = _adu_cw_confirm_text(action, base["risk_level"], cw_confirm_reason)
        return base

    # 3) 执行：复用现有底层
    try:
        result = await _brain_computer_execute_action(action, payload)
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, dict) else {"error": str(e.detail)}
        base["ok"] = False
        base["result"] = detail if isinstance(detail, dict) else {"error": str(detail)}
        err_text = ""
        if isinstance(detail, dict):
            err_text = str(detail.get("error") or detail.get("hint") or "")
        base["final_text"] = err_text or "电脑动作执行失败"
        return base
    except Exception as e:
        base["ok"] = False
        base["result"] = {"error": str(e)}
        base["final_text"] = f"电脑动作执行异常：{e}"
        return base

    base["result"] = result if isinstance(result, dict) else {"value": result}
    base["final_text"] = _adu_act_final_text(action, result)
    base["executed"] = True
    # Computer Work Mode：回填 attempts + completed（completed 只在执行成功且通过验证时为 True）
    if isinstance(result, dict) and isinstance(result.get("attempts"), list):
        base["attempts"] = result["attempts"]
    if action in ("open_app", "open_application", "launch_app", "launch"):
        base["completed"] = bool(isinstance(result, dict) and result.get("completed"))
    else:
        base["completed"] = bool(base["executed"] and base["ok"])

    # V0.1: screenshot 后自动接视觉分析（observe_screen + vision_analyze）
    if action == "screenshot":
        image_b64 = _extract_image_b64_from_result(result)
        if image_b64:
            base["screenshot_available"] = True
            v_ok, v_summary, v_err = await _vision_describe_screenshot(image_b64, text)
            if v_ok:
                base["vision_summary"] = v_summary
                base["final_text"] = f"我看到屏幕上 {v_summary}"
            else:
                base["vision_summary"] = None
                base["final_text"] = f"截图已完成，但视觉分析失败：{v_err}"
            # V0.2: 把这次截图存进 conversation context，供同会话后续追问复用
            _adu_ctx_save(
                conversation_id,
                "screenshot",
                image_b64,
                v_summary if v_ok else None,
            )
        else:
            base["screenshot_available"] = False

    # V1.5 Verify：高影响动作执行后再 observe 一次，回传给前端确认效果
    if needs_observe:
        try:
            await asyncio.sleep(0.4)  # 留点 UI 刷新时间
        except Exception:
            pass
        pok, psummary, _pb64 = await _adu_observe(user_goal=f"刚才已执行 {action}，请核对当前屏幕是否符合预期。")
        base["post_action_summary"] = psummary if pok else f"动作后截图/视觉失败：{psummary}"
        if pok and base.get("final_text"):
            base["final_text"] = base["final_text"] + f" 验证：{psummary[:160]}"

    return base


# ================================
# ✅ /api/brain/dev_agent/*  · Dev Agent V1 只读版
# 链路：App → backend → local-agent :4317 /dev_agent/* → claude CLI → /Users/a12345/Desktop/GPTsora
# 只透传：project + prompt；安全前缀和 claude 调用参数在 local-agent 那边强制。
# ================================

@app.get("/api/brain/dev_agent/health")
async def api_brain_dev_agent_health():
    if not LOCAL_AGENT_BASE_URL:
        raise HTTPException(status_code=503, detail={"ok": False, "error": "LOCAL_AGENT_BASE_URL not configured"})
    url = f"{LOCAL_AGENT_BASE_URL.rstrip('/')}/dev_agent/health"
    headers = _local_agent_headers()
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=8.0) as cli:
            r = await cli.get(url, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=502, detail={"ok": False, "error": f"local-agent unreachable: {type(e).__name__}: {e}"})
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=(r.text or "")[:400])
    try:
        return r.json()
    except Exception:
        return {"ok": False, "error": "non-json from local-agent", "body": (r.text or "")[:400]}


@app.post("/api/brain/dev_agent/ask")
async def api_brain_dev_agent_ask(request: Request):
    body = await _request_json_or_empty(request)
    project = str(body.get("project") or "").strip()
    prompt = str(body.get("prompt") or "").strip()
    if project != "GPTsora":
        raise HTTPException(status_code=400, detail={
            "ok": False,
            "error": f"project must be 'GPTsora', got '{project}'",
        })
    if not prompt:
        raise HTTPException(status_code=400, detail={"ok": False, "error": "missing prompt"})
    if not LOCAL_AGENT_BASE_URL:
        raise HTTPException(status_code=503, detail={"ok": False, "error": "LOCAL_AGENT_BASE_URL not configured"})

    url = f"{LOCAL_AGENT_BASE_URL.rstrip('/')}/dev_agent/ask"
    headers = _local_agent_headers()
    # local-agent 那边硬上限 90s；这里给 120s 兜底，让网络层不在 local-agent 之前先 timeout。
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=120.0) as cli:
            r = await cli.post(url, headers=headers, json={"project": project, "prompt": prompt})
    except Exception as e:
        raise HTTPException(status_code=502, detail={"ok": False, "error": f"local-agent unreachable: {type(e).__name__}: {e}"})
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=(r.text or "")[:500])
    try:
        return r.json()
    except Exception:
        return {"ok": False, "error": "non-json from local-agent", "body": (r.text or "")[:500]}


# ================================
# ✅ BrainState Engine API
# 让 server_session.py 被“状态张量大脑”接管：
# - /api/brain/state        查看当前大脑状态
# - /api/brain/tensor       查看 intent/model/tool/risk 张量视图
# - /api/brain/observe      注入手机/电脑/实时网络/语音等感知状态
# - /api/brain/think        思考：更新 BrainState，生成 next_action
# - /api/brain/act          执行 next_action 或指定 action，并把结果写回 BrainState
# - /api/brain/state/reset  重置当前用户的大脑状态
# ================================

def _brainstate_unavailable_response() -> JSONResponse:
    return JSONResponse({"ok": False, "error": "brain_state_engine_unavailable"}, status_code=503)


def _brainstate_user_key(req: Request, body: Optional[Dict[str, Any]] = None) -> str:
    try:
        return _derive_user_key(req, body or {})
    except Exception:
        return _client_id(req)


def _brainstate_available_tools() -> Dict[str, bool]:
    return {
        "local_agent": bool(_local_agent_is_configured()),
        "computer_screenshot": bool(_local_agent_is_configured()),
        "computer_mouse": bool(_local_agent_is_configured()),
        "computer_keyboard": bool(_local_agent_is_configured()),
        "web_search": bool(CHAT_ENABLE_WEB_SEARCH_DEFAULT and CHAT_WEB_PROVIDER in ("openai", "openai_tool", "openai_web_search", "openai-web_search")),
        "memory_search": bool(MEMORY_ENABLED_DEFAULT),
        "memory_write": bool(MEMORY_ENABLED_DEFAULT),
        "model_router": True,
        "file_read": True,
        "file_write": True,
        "voice_io": True,
    }


def _brainstate_build_context(req: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "allow_web": bool(_extract_allow_web(req, body)),
        "has_attachments": bool(isinstance(body.get("attachments"), list) and len(body.get("attachments") or []) > 0),
        "computer_reachable": bool(_local_agent_is_configured()),
        "available_tools": _brainstate_available_tools(),
        "client_id": str(body.get("client_id") or body.get("clientId") or req.headers.get("x-client-id") or ""),
    }


def _brainstate_extract_text(body: Dict[str, Any]) -> str:
    text = str(body.get("message") or body.get("text") or body.get("goal") or body.get("command") or "").strip()
    if text:
        return text
    msgs = body.get("messages")
    if isinstance(msgs, list):
        try:
            return _last_user_text_from_messages(msgs).strip()
        except Exception:
            pass
    return ""


@app.get("/api/brain/state")
async def api_brain_state(request: Request):
    if BRAIN_STATE_ENGINE is None:
        return _brainstate_unavailable_response()
    body: Dict[str, Any] = {}
    user_key = _brainstate_user_key(request, body)
    return {"ok": True, "user_key": user_key, "state": BRAIN_STATE_ENGINE.get_state(user_key)}


@app.get("/api/brain/tensor")
async def api_brain_tensor(request: Request):
    if BRAIN_STATE_ENGINE is None:
        return _brainstate_unavailable_response()
    user_key = _brainstate_user_key(request, {})
    return BRAIN_STATE_ENGINE.tensor_view(user_key)


@app.post("/api/brain/state/reset")
async def api_brain_state_reset(request: Request):
    if BRAIN_STATE_ENGINE is None:
        return _brainstate_unavailable_response()
    body = await _request_json_or_empty(request)
    user_key = _brainstate_user_key(request, body)
    return {"ok": True, "user_key": user_key, "state": BRAIN_STATE_ENGINE.reset(user_key)}


@app.post("/api/brain/observe")
async def api_brain_observe(request: Request):
    if BRAIN_STATE_ENGINE is None:
        return _brainstate_unavailable_response()
    body = await _request_json_or_empty(request)
    user_key = _brainstate_user_key(request, body)
    observation = body.get("observation") if isinstance(body.get("observation"), dict) else body
    return BRAIN_STATE_ENGINE.observe(user_key, observation)


@app.post("/api/brain/think")
async def api_brain_think(request: Request):
    if BRAIN_STATE_ENGINE is None:
        return _brainstate_unavailable_response()
    body = await _request_json_or_empty(request)
    user_key = _brainstate_user_key(request, body)
    text = _brainstate_extract_text(body)
    if not text:
        return JSONResponse({"ok": False, "error": "missing message/text/goal/messages"}, status_code=400)
    context = _brainstate_build_context(request, body)

    # 先把可用工具、设备状态注入 BrainState，再思考
    try:
        BRAIN_STATE_ENGINE.observe(user_key, {
            "available_tools": context.get("available_tools") or {},
            "device_state": {
                "phone_active": True,
                "computer_reachable": bool(_local_agent_is_configured()),
                "local_agent_healthy": None,
            },
        })
    except Exception:
        pass

    decision = BRAIN_STATE_ENGINE.think(user_key, text, context=context)

    # 让 /api/brain/think 不直接替代 /chat，而是成为 server_session.py 的“前额叶决策层”。
    # 前端可以先调用 think 看 next_action；也可以直接调用 /api/brain/act 执行。
    return decision


async def _brainstate_execute_next_action(user_key: str, action: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {"ok": False, "error": "invalid action"}

    if action.get("requires_confirmation"):
        return {
            "ok": False,
            "error": "requires_confirmation",
            "message": "这个动作风险较高，需要用户确认后再执行。",
            "next_action": action,
        }

    typ = str(action.get("type") or "").strip()
    tool = str(action.get("tool") or "").strip()
    params = action.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    # 电脑身体：local-agent / usecomputer
    if tool == "local_agent" or action.get("target") == "computer":
        spec = params
        if "action" in params:
            spec_action = str(params.get("action") or "")
            spec_payload = params.get("payload") or {}
        else:
            spec_action = str(action.get("action") or "")
            spec_payload = params.get("payload") or params
        if not isinstance(spec_payload, dict):
            spec_payload = {}
        if not spec_action:
            return {"ok": False, "error": "missing computer action", "next_action": action}
        result = await _brain_computer_execute_action(spec_action, spec_payload)
        return {"ok": True, "executed": True, "tool": "local_agent", "computer_action": spec_action, "result": result}

    # 实时外部感知：web search
    if tool == "web_search":
        q = str(params.get("query") or params.get("q") or "").strip()
        if not q:
            return {"ok": False, "error": "missing query"}
        return {
            "ok": True,
            "executed": False,
            "tool": "web_search",
            "query": q,
            "provider": "openai_builtin_web_search",
            "message": "Live web search is executed by /chat via OpenAI built-in web_search.",
        }

    # 长期记忆检索
    if tool == "memory_search":
        q = str(params.get("query") or params.get("q") or "").strip()
        if not q:
            return {"ok": False, "error": "missing query"}
        try:
            ctx = memory_build_context(user_key, q, k=int(params.get("k") or MEMORY_TOP_K_DEFAULT), min_score=MEMORY_MIN_SCORE_DEFAULT)
            return {"ok": True, "executed": True, "tool": "memory_search", "query": q, "context": ctx}
        except Exception as e:
            return {"ok": False, "tool": "memory_search", "error": str(e)[:500]}

    # 长期记忆写入
    if tool == "memory_write":
        txt = str(params.get("text") or "").strip()
        if not txt:
            return {"ok": False, "error": "missing text"}
        try:
            if _should_memory_add(txt):
                memory_add(user_key, txt[:1200])
            try:
                memory_facts_save(user_key, txt[:400], tags=str(params.get("tags") or "brain_state"), importance=int(params.get("importance") or 4))
            except Exception:
                pass
            return {"ok": True, "executed": True, "tool": "memory_write", "stored": True}
        except Exception as e:
            return {"ok": False, "tool": "memory_write", "error": str(e)[:500]}

    # 模型路由：返回建议，不在 /act 直接生成聊天回复，避免绕开 /chat 的计费/流式/TTS/记忆管线。
    if tool == "model_router" or typ == "call_model":
        return {
            "ok": True,
            "executed": False,
            "tool": "model_router",
            "message": "next step should be handled by /chat or /chat/prepare with model_route hints",
            "model_route": params,
        }

    if typ == "ask_confirmation":
        return {"ok": False, "error": "requires_confirmation", "next_action": action}

    return {"ok": False, "error": "unsupported_next_action", "next_action": action}


@app.post("/api/brain/act")
async def api_brain_act(request: Request):
    if BRAIN_STATE_ENGINE is None:
        return _brainstate_unavailable_response()
    body = await _request_json_or_empty(request)
    user_key = _brainstate_user_key(request, body)

    # 支持两种：1) body.next_action 指定；2) 默认执行当前 BrainState.next_action
    action = body.get("next_action") or body.get("action")
    if not isinstance(action, dict):
        state = BRAIN_STATE_ENGINE.get_state(user_key)
        action = state.get("next_action") or {}

    result = await _brainstate_execute_next_action(user_key, action)
    try:
        new_state = BRAIN_STATE_ENGINE.remember_action_result(user_key, action, result)
    except Exception:
        new_state = BRAIN_STATE_ENGINE.get_state(user_key)
    return {"ok": bool(result.get("ok")), "user_key": user_key, "action": action, "result": result, "state_summary": BRAIN_STATE_ENGINE.summarize_state(new_state)}

# ================================
# ✅ ChatAGI Left/Right Brain Lite API
# 左脑：持续意识闪现 / 当前状态 / 今日记忆
# 右脑：把左脑指令转成 local-agent 电脑动作
# 说明：不重做 App；这层直接挂在现有 server_session.py 上。
# ================================

BRAIN_LITE_DIR = DATA_DIR / "brain_lite"
BRAIN_LITE_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_LITE_IDENTITY: Dict[str, Any] = {
    "name": "ChatAGI",
    "owner": "阿杜",
    "role": "长期个人 AI 大脑助手",
    "meaning": "帮助阿杜记忆、理解、规划、行动，并逐步迁移到电脑、手机和机器人身体中。",
    "core_goal": "构建一个具有长期记忆、持续感知、工具行动和可迁移身体的 AI Brain。",
    "abilities": [
        "对话", "长期记忆", "联网搜索", "视觉理解", "电脑控制", "手机协助", "代码生成", "文件理解", "任务规划", "机器人身体控制"
    ],
    "safety_rules": [
        "高风险行动必须询问用户确认",
        "不能删除重要文件或执行付款/发布等高风险动作",
        "行动前先判断目标、环境和风险",
        "执行结果必须回写状态与记忆"
    ],
}


def _brain_lite_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _brain_lite_today() -> str:
    return time.strftime("%Y-%m-%d")


def _brain_lite_safe_key(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (raw or "anonymous").strip())
    return s[:96] or "anonymous"


def _brain_lite_user_key(req: Request, body: Optional[Dict[str, Any]] = None) -> str:
    body = body or {}
    raw = (
        body.get("client_id")
        or body.get("clientId")
        or req.query_params.get("client_id")
        or req.query_params.get("clientId")
        or req.headers.get("x-client-id")
        or req.headers.get("x-user-id")
        or ""
    )
    if raw:
        return _brain_lite_safe_key(str(raw))
    try:
        return _brain_lite_safe_key(_brainstate_user_key(req, body))
    except Exception:
        try:
            return _brain_lite_safe_key(_client_id(req))
        except Exception:
            return "anonymous"


def _brain_lite_state_path(user_key: str) -> Path:
    return BRAIN_LITE_DIR / f"state_{_brain_lite_safe_key(user_key)}.json"


def _brain_lite_memory_path(user_key: str) -> Path:
    return BRAIN_LITE_DIR / f"memory_{_brain_lite_safe_key(user_key)}.json"


def _brain_lite_read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("[BrainLite] read json failed %s: %s", path, e)
    return default


def _brain_lite_write_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("[BrainLite] write json failed %s: %s", path, e)


def _brain_lite_default_state(user_key: str) -> Dict[str, Any]:
    return {
        "identity": BRAIN_LITE_IDENTITY,
        "user_key": user_key,
        "status": "online",
        "mode": "left_right_brain_mvp",
        "who_am_i": "我是 ChatAGI，阿杜的长期个人 AI 大脑助手。",
        "existence_meaning": BRAIN_LITE_IDENTITY["meaning"],
        "current_environment": "iPhone App + 后端 server_session.py + local-agent 电脑身体。",
        "current_task": "把现有 ChatAGI 升级为左右脑 AI Brain。",
        "today_goal": "接通左脑意识闪现、右脑行动执行、电脑控制和记忆回写。",
        "long_term_goal": BRAIN_LITE_IDENTITY["core_goal"],
        "current_condition": "等待用户输入或执行下一步任务。",
        "next_action": "在 App 首页显示 AI 当前意识，并通过右脑执行 local-agent 基础动作。",
        "right_brain_instruction": {
            "type": "plan",
            "target": "computer",
            "goal": "等待明确电脑动作",
            "requires_confirmation": False,
            "risk_level": 1,
        },
        "tools": {
            "local_agent_configured": bool(_local_agent_is_configured()),
            "computer_actions": ["screenshot", "mouse_position", "move", "click", "type_text", "press_keys", "open_app"],
            "memory_timeline": True,
            "brain_state_engine": bool(BRAIN_STATE_ENGINE is not None),
        },
        "last_flash_at": _brain_lite_now_iso(),
        "last_right_brain_result": None,
    }


def _brain_lite_load_state(user_key: str) -> Dict[str, Any]:
    state = _brain_lite_read_json(_brain_lite_state_path(user_key), None)
    if isinstance(state, dict):
        # Keep identity/tools fresh when code changes.
        state.setdefault("identity", BRAIN_LITE_IDENTITY)
        state.setdefault("tools", {})
        if isinstance(state.get("tools"), dict):
            state["tools"]["local_agent_configured"] = bool(_local_agent_is_configured())
            state["tools"]["brain_state_engine"] = bool(BRAIN_STATE_ENGINE is not None)
        return state
    return _brain_lite_default_state(user_key)


def _brain_lite_save_state(user_key: str, state: Dict[str, Any]) -> Dict[str, Any]:
    _brain_lite_write_json(_brain_lite_state_path(user_key), state)
    return state


def _brain_lite_load_memory(user_key: str) -> List[Dict[str, Any]]:
    data = _brain_lite_read_json(_brain_lite_memory_path(user_key), [])
    return data if isinstance(data, list) else []


def _brain_lite_append_memory(user_key: str, item: Dict[str, Any]) -> Dict[str, Any]:
    memory = _brain_lite_load_memory(user_key)
    item = dict(item or {})
    item.setdefault("id", uuid.uuid4().hex[:12])
    item.setdefault("date", _brain_lite_today())
    item.setdefault("created_at", _brain_lite_now_iso())
    item.setdefault("type", "brain_event")
    item.setdefault("importance", 5)
    memory.insert(0, item)
    memory = memory[:500]
    _brain_lite_write_json(_brain_lite_memory_path(user_key), memory)
    return item


def _brain_lite_instruction_from_goal(goal: str, explicit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(explicit, dict) and explicit:
        return explicit
    spec = _computer_action_from_goal(goal or "") if goal else None
    if spec:
        return {
            "type": "computer_action",
            "target": "computer",
            "tool": "local_agent",
            "goal": goal,
            "action": spec.get("action"),
            "payload": spec.get("payload") or {},
            "requires_confirmation": False,
            "risk_level": 1,
        }
    return {
        "type": "plan",
        "target": "brain",
        "goal": goal or "继续观察用户需求，等待明确任务。",
        "requires_confirmation": False,
        "risk_level": 0,
    }


def _brain_lite_build_flash(user_key: str, body: Dict[str, Any]) -> Dict[str, Any]:
    prior = _brain_lite_load_state(user_key)
    text = str(body.get("text") or body.get("message") or body.get("goal") or "").strip()
    current_task = str(body.get("current_task") or body.get("task") or prior.get("current_task") or "").strip()
    environment = str(body.get("environment") or body.get("current_environment") or prior.get("current_environment") or "").strip()
    today_goal = str(body.get("today_goal") or prior.get("today_goal") or "").strip()
    long_term_goal = str(body.get("long_term_goal") or prior.get("long_term_goal") or BRAIN_LITE_IDENTITY["core_goal"]).strip()
    current_condition = str(body.get("current_condition") or body.get("condition") or "").strip()
    if not current_condition:
        current_condition = f"最新输入：{text}" if text else str(prior.get("current_condition") or "等待下一步任务。")
    instruction = _brain_lite_instruction_from_goal(text or current_task, body.get("right_brain_instruction") if isinstance(body.get("right_brain_instruction"), dict) else None)
    next_action = str(body.get("next_action") or "").strip()
    if not next_action:
        if instruction.get("type") == "computer_action":
            next_action = f"右脑准备执行电脑动作：{instruction.get('action')}。"
        else:
            next_action = "保持意识闪现，等待用户确认下一步可执行动作。"

    state = dict(prior)
    state.update({
        "identity": BRAIN_LITE_IDENTITY,
        "user_key": user_key,
        "status": "online",
        "mode": "left_right_brain_mvp",
        "who_am_i": "我是 ChatAGI，阿杜的长期个人 AI 大脑助手。",
        "existence_meaning": BRAIN_LITE_IDENTITY["meaning"],
        "current_environment": environment or "iPhone App + 后端 server_session.py + local-agent 电脑身体。",
        "current_task": current_task or (text if text else "持续感知、记忆、规划与行动。"),
        "today_goal": today_goal or "完成左右脑 AI Brain MVP：意识闪现 + 右脑行动 + 记忆回写。",
        "long_term_goal": long_term_goal,
        "current_condition": current_condition,
        "next_action": next_action,
        "right_brain_instruction": instruction,
        "tools": {
            "local_agent_configured": bool(_local_agent_is_configured()),
            "computer_actions": ["screenshot", "mouse_position", "move", "click", "type_text", "press_keys", "open_app"],
            "memory_timeline": True,
            "brain_state_engine": bool(BRAIN_STATE_ENGINE is not None),
        },
        "last_user_signal": text,
        "last_flash_at": _brain_lite_now_iso(),
    })
    _brain_lite_save_state(user_key, state)
    if text or current_task:
        _brain_lite_append_memory(user_key, {
            "type": "consciousness_flash",
            "summary": text or current_task,
            "current_task": state.get("current_task"),
            "next_action": state.get("next_action"),
            "importance": int(body.get("importance") or 6),
        })
    return state


@app.get("/api/brain/consciousness/state")
async def api_brain_consciousness_state(request: Request):
    user_key = _brain_lite_user_key(request, {})
    state = _brain_lite_load_state(user_key)
    today = _brain_lite_today()
    memory_today = [m for m in _brain_lite_load_memory(user_key) if str(m.get("date") or "") == today][:20]
    return {"ok": True, "user_key": user_key, "state": state, "today_memory": memory_today}


@app.post("/api/brain/consciousness/flash")
async def api_brain_consciousness_flash(request: Request):
    body = await _request_json_or_empty(request)
    user_key = _brain_lite_user_key(request, body)
    state = _brain_lite_build_flash(user_key, body)
    return {"ok": True, "user_key": user_key, "state": state}


@app.post("/api/brain/memory/write")
async def api_brain_lite_memory_write(request: Request):
    body = await _request_json_or_empty(request)
    user_key = _brain_lite_user_key(request, body)
    summary = str(body.get("summary") or body.get("text") or body.get("message") or "").strip()
    if not summary:
        return JSONResponse({"ok": False, "error": "missing summary/text/message"}, status_code=400)
    item = _brain_lite_append_memory(user_key, {
        "type": str(body.get("type") or "manual_memory"),
        "summary": summary,
        "importance": int(body.get("importance") or 5),
        "raw": body.get("raw") if isinstance(body.get("raw"), dict) else None,
    })
    return {"ok": True, "user_key": user_key, "memory": item}


@app.get("/api/brain/memory/today")
async def api_brain_lite_memory_today(request: Request, limit: int = 30):
    user_key = _brain_lite_user_key(request, {})
    limit = max(1, min(int(limit or 30), 100))
    today = _brain_lite_today()
    memory = [m for m in _brain_lite_load_memory(user_key) if str(m.get("date") or "") == today][:limit]
    return {"ok": True, "user_key": user_key, "date": today, "memory": memory}


@app.post("/api/brain/right-brain/execute")
async def api_brain_right_brain_execute(request: Request):
    body = await _request_json_or_empty(request)
    user_key = _brain_lite_user_key(request, body)
    goal = str(body.get("goal") or body.get("text") or body.get("command") or "").strip()
    dry_run = bool(body.get("dry_run") or body.get("dryRun"))

    action = str(body.get("action") or "").strip()
    payload = body.get("payload") or body.get("args") or {}
    if not isinstance(payload, dict):
        payload = {}
    inferred = False
    if not action and goal:
        spec = _computer_action_from_goal(goal)
        if spec:
            action = str(spec.get("action") or "")
            payload = spec.get("payload") or {}
            inferred = True

    if not action:
        state = _brain_lite_load_state(user_key)
        instr = state.get("right_brain_instruction") if isinstance(state.get("right_brain_instruction"), dict) else {}
        action = str(instr.get("action") or "").strip()
        payload = instr.get("payload") if isinstance(instr.get("payload"), dict) else payload
        goal = goal or str(instr.get("goal") or "").strip()

    if not action:
        return {
            "ok": False,
            "error": "missing_action",
            "message": "右脑还没有明确电脑动作。先调用 /api/brain/consciousness/flash 生成 right_brain_instruction，或直接传 action/payload。",
            "goal": goal,
        }

    instruction = {
        "type": "computer_action",
        "target": "computer",
        "tool": "local_agent",
        "goal": goal,
        "action": action,
        "payload": payload,
        "requires_confirmation": False,
        "risk_level": int(body.get("risk_level") or body.get("riskLevel") or 1),
    }
    if dry_run:
        state = _brain_lite_load_state(user_key)
        state["right_brain_instruction"] = instruction
        state["next_action"] = f"右脑 dry-run：准备执行 {action}。"
        state["last_flash_at"] = _brain_lite_now_iso()
        _brain_lite_save_state(user_key, state)
        return {"ok": True, "dry_run": True, "user_key": user_key, "instruction": instruction, "inferred": inferred}

    result = await _brain_computer_execute_action(action, payload)
    state = _brain_lite_load_state(user_key)
    state["right_brain_instruction"] = instruction
    state["last_right_brain_result"] = result
    state["current_condition"] = f"右脑已执行：{action}。"
    state["next_action"] = "观察执行结果，必要时继续下一步或写入复盘。"
    state["last_flash_at"] = _brain_lite_now_iso()
    _brain_lite_save_state(user_key, state)
    _brain_lite_append_memory(user_key, {
        "type": "right_brain_action",
        "summary": f"执行电脑动作：{action}；目标：{goal or '-'}",
        "action": action,
        "payload": payload,
        "result": result,
        "importance": 6,
    })
    return {"ok": True, "user_key": user_key, "instruction": instruction, "result": result, "state": state, "inferred": inferred}



# ✅ business routers
# NOTE: order matters — we register our /chat override BEFORE including home_chat_router,
# so even if routers.home_chat also defines /chat, the legacy iOS client will hit the stable handler here.
app.include_router(billing_router)
app.include_router(auth_router)
app.include_router(media_upload_router)
app.include_router(home_automation_router)

# ✅ 高级编程版：工程内循环（B方案）
if agent_loop_router is not None:
    app.include_router(agent_loop_router)

# ✅ 感知模块路由（物理世界记忆查询）
try:
    from perception_module import register_perception_routes
    register_perception_routes(app)
except Exception as _perc_err:
    log.warning("[Perception] module not loaded: %s", _perc_err)

# ✅ 阿杜任务委托系统（主动循环 + iOS工作报告）
try:
    from adu_task_api import router as adu_task_router
    app.include_router(adu_task_router)
    log.info("[AduLoop] ✅ 任务委托系统已加载")
except Exception as _adu_err:
    log.warning("[AduLoop] 任务委托系统未加载: %s", _adu_err)

# ✅ 阿杜推送模块（iOS Token注册 + 任务完成推送）
try:
    from adu_push import push_router
    app.include_router(push_router)
    log.info("[AduPush] ✅ 推送模块已加载")
except Exception as _push_err:
    log.warning("[AduPush] 推送模块未加载: %s", _push_err)

# ✅ Codex CLI executor（最小闭环：阿杜后端 -> 本机 codex exec）
try:
    from adu_orchestrator.codex_executor import router as adu_codex_router
    app.include_router(adu_codex_router)
    log.info("[AduCodex] ✅ Codex executor router loaded")
except Exception as _codex_err:
    log.warning("[AduCodex] Codex executor router not loaded: %s", _codex_err)

# ✅ Automation engine (default: Claude Code; fallback via AUTOMATION_ENGINE=codex)
try:
    from adu_orchestrator.claude_executor import router as adu_automation_router
    app.include_router(adu_automation_router)
    log.info("[AduAutomation] ✅ automation router loaded at /api/adu/automation/*")
except Exception as _autom_err:
    log.warning("[AduAutomation] automation router not loaded: %s", _autom_err)

# ✅ Adu Planner —— 语义路由,不执行(POST /api/adu/planner/route)
#    把 App 端文本分类到 computer_action / codex_task / chat / 等。
#    router 自带 prefix=/api/adu/planner;只暴露 /route。
try:
    from adu_planner_router import router as adu_planner_router
    app.include_router(adu_planner_router)
    log.info("[AduPlanner] ✅ planner router loaded at /api/adu/planner/route")
except Exception as _planner_err:
    log.warning("[AduPlanner] planner router not loaded: %s", _planner_err)

# ✅ Adu Project Registry —— GET /api/adu/projects(planner / codex executor 共享同一份注册表)
try:
    from adu_project_registry import router as adu_projects_router
    app.include_router(adu_projects_router)
    log.info("[AduProjects] ✅ project registry loaded at /api/adu/projects")
except Exception as _projects_err:
    log.warning("[AduProjects] project registry not loaded: %s", _projects_err)

# ✅ Adu File Search —— POST /api/adu/files/search(只在授权工作区,跳过 .env / .key 等敏感文件)
try:
    from adu_files_search import router as adu_files_search_router
    app.include_router(adu_files_search_router)
    log.info("[AduFileSearch] ✅ file search router loaded at /api/adu/files/search")
except Exception as _files_search_err:
    log.warning("[AduFileSearch] file search router not loaded: %s", _files_search_err)

# ✅ Adu Self / 递归自我进化(/api/adu/self/*) —— plan_only,不自动执行
try:
    from adu_self_router import router as adu_self_router
    app.include_router(adu_self_router)
    log.info("[AduSelf] ✅ self router loaded at /api/adu/self/*")
except Exception as _self_err:
    log.warning("[AduSelf] self router not loaded: %s", _self_err)

# ✅ Voice clone module disabled by product decision.
# Do not mount cloned-voice routes and do not expose cloned-voice TTS.
log.info("[VoiceClone] disabled: normal TTS voice only")

# ===== 本地电脑 Local-Agent 路由（只保留新链路） =====
try:
    app.include_router(local_agent_router)
    log.info("[LocalComputerAgent] ✅ local agent router loaded; old computer task / voice bridge routers removed")
except Exception as _lca_err:
    log.warning("[LocalComputerAgent] router load failed: %s", _lca_err)

# ===== 启动时自动探测本机系统 / App / 目录别名 =====
try:
    bootstrap_local_agent()
    log.info("[LocalComputerAgent] ✅ bootstrap_local_agent() done")
except Exception as _boot_err:
    log.warning("[LocalComputerAgent] bootstrap failed: %s", _boot_err)

# ✅ 控制平面路由（风险分级 + 任务协议）
try:
    from adu_control_plane import control_plane, router as cp_router
    app.include_router(cp_router)
    log.info("[ControlPlane] ✅ control plane router loaded")
except Exception as _cp_err:
    log.warning("[ControlPlane] control plane router load failed: %s", _cp_err)

# ✅ 视觉监控循环路由
try:
    from adu_vision_loop import vision_loop, router as vision_router
    app.include_router(vision_router)
    log.info("[VisionLoop] ✅ vision loop router loaded")
except Exception as _vl_err:
    log.warning("[VisionLoop] vision loop router load failed: %s", _vl_err)

# ✅ 意识系统（多线程意识流，持续运行）
try:
    from adu_consciousness import consciousness, router as consciousness_router
    app.include_router(consciousness_router)
    log.info("[Consciousness] ✅ consciousness router loaded")
except Exception as _cs_err:
    log.warning("[Consciousness] consciousness router load failed: %s", _cs_err)

# ✅ OpenClaw 反向桥接 — 把 ChatAGI 后端暴露为 OpenAI 兼容 provider
# OpenClaw 把我们当成 LLM 调用，大脑集中在 server_session.py
# 配置 ~/.openclaw/openclaw.json 的 providers.chatagi.baseUrl = http://127.0.0.1:8000/v1
try:
    from openclaw_bridge_router import openclaw_bridge_router, attach_anthropic_caller
    attach_anthropic_caller(_anthropic_messages_create_nonstream)
    app.include_router(openclaw_bridge_router)
    log.info("[OpenClawBridge] ✅ 反向桥接已加载 (Adu brain exposed to OpenClaw)")
except Exception as _ocb_err:
    log.warning("[OpenClawBridge] 桥接未加载: %s", _ocb_err)


def _brain_opt_in(body: Dict[str, Any], request: Request) -> bool:
    """
    默认开启 auto_brain（抢先于 OpenClaw 执行电脑任务）。

    真正触发与否由下游 looks_like_computer_task(goal) 决定：
      - "打开微信" → 触发 auto_brain
      - "今天天气" → 不触发，照常走 LLM/OpenClaw

    只有以下情况会关闭 brain：
      - body 里显式 use_brain=false（调试用）
      - body 里 mode=chat_only / llm_only / no_brain（强制纯聊天）
    """
    # 显式关闭 brain — 调试/强制纯聊天用
    if body.get("use_brain") is False:
        return False
    mode = str(body.get("mode") or body.get("chat_mode") or request.headers.get("x-mode") or "").strip().lower()
    if mode in ("chat_only", "llm_only", "no_brain"):
        return False
    # 默认开启 — 依赖下游 looks_like_computer_task 保护聊天消息不误触发
    return True


def _brain_goal_from_messages(messages: List[Dict[str, str]], user_transcript: str = "") -> str:
    for mm in reversed(messages or []):
        try:
            if (mm.get("role") or "").strip() == "user":
                txt = (mm.get("content") or "").strip()
                if txt:
                    return txt
        except Exception:
            pass
    return (user_transcript or "").strip()


async def _maybe_run_brain_direct(request: Request, body: Dict[str, Any], messages: List[Dict[str, str]], user_transcript: str = "") -> Optional[Dict[str, Any]]:
    # 旧 Adu Auto Brain 电脑任务直通已删除。
    # 电脑控制统一走 /api/brain/computer/action -> local-agent。
    return None


def _create_completed_chat_job_from_result(*, result: Dict[str, Any], conversation_id: str, user_key: str, last_user_text: str, tts_voice: str = "") -> ChatJob:
    job = _create_chat_job(enable_tts_streaming=False, user_key=user_key)
    summary = str(result.get("summary") or "已完成电脑任务。")
    details = str(result.get("details") or "").strip()
    full_text = summary if not details else f"{summary}\n\n{details}"
    job.push_event({"type": "solara.meta", "conversation_id": conversation_id, "tts_url": f"/tts/live/{job.tts_id}.mp3", "voice": tts_voice or TTS_VOICE_DEFAULT})
    job.push_event({"type": "response.output_text.delta", "delta": full_text})
    job.push_event({"type": "response.output_text.done", "text": full_text})
    job.push_event({"type": "response.completed"})
    job.full_text = full_text
    try:
        if conversation_id and full_text:
            conv_add_message(user_key, conversation_id, "assistant", full_text)
    except Exception:
        pass
    job.close_events()
    return job



# -----------------------------
# ✅ Stable Computer Control Gateway (submit / confirm / status)
#    独立于 /chat，供 iOS 后续统一入口直接调用。
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

    client_model = (body.get("model") or "").strip()
    requested_model = client_model

    # -----------------------------
    # Plan-based model routing (server-authoritative, billing-capped)
    user_key = _derive_user_key(request, body)
    plan_raw = _billing_effective_plan_for_request(user_key, _extract_plan(request, body))
    client_model = (str(body.get("model") or "").strip() or str(request.headers.get("x-model") or "").strip())
    requested_model, plan, _model_reason = _select_model_for_request(plan_raw, client_model)

    allow_web = _extract_allow_web(request, body)

    # Plan gate (optional)
    allow_web = bool(allow_web and _plan_allows_web(plan))

    msgs = body.get("messages") or []
    atts = body.get("attachments") or []
    attachments: List[Dict[str, Any]] = _normalize_chat_attachments(atts)

    if not isinstance(msgs, list) or not msgs:
        return JSONResponse({"ok": False, "error": "missing messages"}, status_code=400)

    # ✅ 有 attachments 时允许最后一条 user 消息为空（只发文件不输文字）
    has_atts = bool(attachments)
    messages: List[Dict[str, str]] = []
    for _i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        if content.strip() == "":
            if has_atts and _i == len(msgs) - 1 and role == "user":
                messages.append({"role": role, "content": ""})
            continue
        messages.append({"role": role, "content": content})
    if not messages:
        return JSONResponse({"ok": False, "error": "empty messages"}, status_code=400)

    # Whether this HTTP request actually included a new non-empty user text.
    # If not, image/file attachments must become their own new turn instead of binding to old history.
    _raw_last = msgs[-1] if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict) else {}
    _raw_last_user_has_text = (
        str(_raw_last.get("role") or "user").strip() == "user"
        and bool(str(_raw_last.get("content") or "").strip())
    )

    # ✅ Billing V1: every valid chat request consumes 1 text quota; image attachments consume 1 image quota.
    _billing_block = _billing_guard_request(request, body, FEATURE_TEXT, want=1, consume=True, check_quota=True)
    if _billing_block is not None:
        return _billing_block
    if _billing_has_image_attachments(attachments):
        _billing_block = _billing_guard_request(request, body, FEATURE_IMAGE, want=1, consume=True, check_quota=True)
        if _billing_block is not None:
            return _billing_block

    # ✅ 录音消息必须先转写，再进入 last_user_text / memory / router / brain。
    #    否则 audio-only 空消息会被当成上一轮问题的附件，导致“录音不能正常文本交互”。
    user_transcript = ""
    if attachments:
        user_transcript = await asyncio.to_thread(_transcribe_audio_attachments_inplace, attachments)

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

    # ✅ Image/file-only request: create an explicit current turn so attachments do not bind to old history.
    if attachments and ((not last_user_text) or (not _raw_last_user_has_text)):
        messages = _ensure_current_user_turn_for_attachments(messages, attachments)
        for mm in reversed(messages):
            if mm.get("role") == "user":
                last_user_text = (mm.get("content") or "").strip()
                break

    # 🧠 通知意识系统："玉勇说话了"
    #    触发 on_user_message → 更新 last_user_msg_ts / interaction_count / energy / recent_flashes
    if last_user_text:
        try:
            from adu_consciousness import consciousness
            consciousness.on_user_message(last_user_text)
        except Exception as _cs_e:
            log.debug("[Consciousness] on_user_message skip (chat): %s", _cs_e)

    # ✅ user_transcript 已在进入 last_user_text/router/brain 之前完成转写。

    # ✅ AutoBrain direct computer-task path (opt-in only)
    _brain_result = await _maybe_run_brain_direct(request, body, messages, user_transcript=user_transcript)
    if _brain_result is not None:
        try:
            conv_add_message(user_key, conversation_id, "assistant", str(_brain_result.get("summary") or ""))
        except Exception:
            pass
        # 🧠 通知意识系统：脑内任务已完成（成功/失败都算一次"刚做完"）
        try:
            from adu_consciousness import consciousness
            _goal_label = (last_user_text or str(_brain_result.get("summary") or ""))[:30]
            _success = bool(_brain_result.get("ok", _brain_result.get("success", True)))
            consciousness.on_task_complete(_goal_label, _success)
        except Exception as _cs_e:
            log.debug("[Consciousness] on_task_complete skip (chat-brain): %s", _cs_e)
        return {
            "ok": True,
            "conversation_id": conversation_id,
            "user_key": user_key,
            "provider": "brain",
            "model": "adu_auto_brain_v2",
            "text": str(_brain_result.get("summary") or ""),
            "brain": _brain_result,
        }

    # Ensure markdown/code fences for client-side highlight.
    messages = _prepend_style_prompt(messages)

    # ✅ Inject structured long-term memory facts (query-independent)
    if MEMORY_FACTS_ENABLED_DEFAULT:
        try:
            facts_ctx = memory_facts_build_prompt(user_key, limit=MEMORY_FACTS_PROMPT_LIMIT)
        except Exception:
            facts_ctx = ""
        if facts_ctx:
            insert_at = 1 if messages and (messages[0].get("role") in ("system", "developer")) else 0
            messages.insert(insert_at, {"role": "system", "content": facts_ctx})


    # ✅ Inject vector memory context (official RAG-style: retrieve -> system)
    if MEMORY_ENABLED_DEFAULT and last_user_text:
        mem_ctx = memory_build_context(user_key, last_user_text, k=MEMORY_TOP_K_DEFAULT, min_score=MEMORY_MIN_SCORE_DEFAULT)
        if mem_ctx:
            # keep style prompt at first
            insert_i = 1 if (messages and (messages[0].get("role") in ("system", "developer"))) else 0
            # If we already inserted the facts memory block, put vector memory after it.
            if insert_i < len(messages):
                c0 = (messages[insert_i].get("content") or "") if isinstance(messages[insert_i], dict) else ""
                if isinstance(c0, str) and c0.startswith("以下是【长期记忆】（重要事实/偏好"):
                    insert_i += 1
            messages.insert(insert_i, {"role": "system", "content": mem_ctx})

    # ✅ 四级压缩上下文注入（L3档案 + L2主题 + L1轮次摘要，MemGPT换入换出）
    if MEMORY_ENABLED_DEFAULT and last_user_text:
        try:
            comp_ctx = _comp_build_context(user_key, conv_id=conversation_id, query=last_user_text)
            if comp_ctx:
                ins = 1 if (messages and messages[0].get("role") in ("system", "developer")) else 0
                messages.insert(ins, {"role": "system", "content": comp_ctx})
        except Exception:
            pass

    # ✅ Persist user message to conversation history (best-effort)
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

        # ✅ 四级压缩：写入L0原始轮次
        try:
            _comp_append_turn(user_key, conversation_id, "user", last_user_text[:800])
        except Exception:
            pass

    # ✅ OpenClaw: 注入项目上下文（手机 App 也能看到所有项目文件）
    try:
        bridge = _openclaw_get_bridge_or_none()
        if bridge is not None and getattr(bridge, "connected", False) and getattr(bridge, "_project_cache", None):
            all_ctx_parts = []
            for _pth, _scan in bridge._project_cache.items():
                _ctx = bridge.build_project_context(_scan)
                if _ctx:
                    all_ctx_parts.append(_ctx)
            if all_ctx_parts:
                full_project_ctx = "\n\n".join(all_ctx_parts)
                _ins = 1 if messages and (messages[0].get("role") in ("system", "developer")) else 0
                messages.insert(_ins, {"role": "system", "content": full_project_ctx})
    except Exception as _e:
        log.debug("[OpenClaw] project context injection (legacy /chat) skipped: %s", _e)

    # ✅ OpenClaw: 注入 Agent 工具能力（手机 App 也能调 OpenClaw 执行操作）
    try:
        bridge = _openclaw_get_bridge_or_none()
        if bridge is not None and getattr(bridge, "connected", False):
            messages = inject_agent_system_prompt(messages, bridge_connected=True)
    except Exception:
        pass

    # ✅ 感知模块：注入物理世界感知上下文（位置/朝向/运动状态）
    _perception_data = body.get("perception")
    if isinstance(_perception_data, dict) and _perception_data.get("latitude"):
        try:
            from perception_module import format_perception_context, store_to_world_memory, enrich_perception_data
            _perception_data = await enrich_perception_data(_perception_data)
            _perc_ctx = format_perception_context(_perception_data)
            if _perc_ctx:
                _ins = 1 if messages and (messages[0].get("role") in ("system", "developer")) else 0
                messages.insert(_ins, {"role": "system", "content": _perc_ctx})
                log.info("[Perception] ✅ context injected: %s",
                         _perception_data.get("fullAddress") or f"{_perception_data.get('latitude'):.4f}")
            # 存入世界记忆
            store_to_world_memory(user_key or "anonymous", _perception_data)
        except Exception as _pe:
            log.warning("[Perception] injection failed: %s", _pe)

    # attachments were normalized near the top of /chat.

    # ✅ Billing V1: /chat/prepare is the main iOS streaming entry, so consume here.
    _billing_block = _billing_guard_request(request, body, FEATURE_TEXT, want=1, consume=True, check_quota=True)
    if _billing_block is not None:
        return _billing_block
    if _billing_has_image_attachments(attachments):
        _billing_block = _billing_guard_request(request, body, FEATURE_IMAGE, want=1, consume=True, check_quota=True)
        if _billing_block is not None:
            return _billing_block

    # user_transcript 已在路由前完成；这里不再重复转写。

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

    # ✅ Smart Router Decision comes first.
    # Raw allow_web means "web is permitted"; it does NOT mean "force web_search".
    route_decision = _smart_router_decision(
        user_text=last_user_text,
        allow_web=allow_web,
        attachments=attachments,
        requested_model=requested_model,
        plan=plan,
        body=body,
    )
    allow_web = bool(route_decision.get("need_web"))

    # ✅ Provider routing uses the smart decision result, not the raw client flag.
    provider, route_reason = _route_provider(allow_web=allow_web, attachments=attachments, model=requested_model)
    _provider_route_reason = route_reason
    provider, route_reason = _ensure_provider_available(provider, route_reason)
    _smart_reason = f"smart_router_decision:{route_decision.get('route') or 'unknown'}"
    if route_reason != _provider_route_reason:
        route_reason = f"{_smart_reason}->{route_reason}"
    else:
        route_reason = _smart_reason
    routed_model = _select_routed_model(provider, requested_model)
    if provider == "openai" and allow_web:
        routed_model = OPENAI_WEB_SEARCH_MODEL
    log.info(
        "[chat-route] provider=%s model=%s allow_web=%s reason=%s decision=%s web_provider=%s",
        provider, routed_model, bool(allow_web), route_reason, route_decision, CHAT_WEB_PROVIDER,
    )
    web_results_for_response: List[Dict[str, Any]] = []

    try:
        # ✅ Smart Router: 智能路由到最优模型
        if provider == "smart":
            _sr = _get_smart_router()
            if _sr:
                import asyncio as _aio
                _tier_map = {"guest": "free", "basic": "free", "coder": "creator", "pro": "pro", "ultra": "ultra"}
                _tier = _tier_map.get(plan, "free")
                _has_img = _has_image_attachments(attachments) if attachments else False
                # V1 final: no prompt-injected external web context. Live web turns route to OpenAI before smart routing.
                smart_user_input = last_user_text
                try:
                    _loop = _aio.get_running_loop()
                except RuntimeError:
                    _loop = None
                if _loop and _loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as _pool:
                        _sr_result = _pool.submit(
                            lambda: _aio.run(_sr.process(
                                session_id=user_key or "default",
                                user_input=smart_user_input,
                                user_tier=_tier,
                                has_image=_has_img,
                                system_prompt=CHAT_SYSTEM_STYLE_PROMPT,
                            ))
                        ).result(timeout=120)
                else:
                    _sr_result = _aio.run(_sr.process(
                        session_id=user_key or "default",
                                user_input=smart_user_input,
                        user_tier=_tier,
                        has_image=_has_img,
                        system_prompt=CHAT_SYSTEM_STYLE_PROMPT,
                    ))
                full_text = _sr_result["response"]
                status = "completed"
                rid = f"smart-{_sr_result['provider']}-{int(time.time())}"
                inc = None
                conts = 0
                routed_model = _sr_result["model_display"]
                provider = _sr_result["provider"]
                log.info("[SmartRouter] %s/%s -> %s (%s) cost≈¥%s escalated=%s",
                         _sr_result["intent"], _sr_result["complexity"],
                         _sr_result["model_display"], _sr_result["provider"],
                         _sr_result["estimated_cost_yuan"], _sr_result["was_escalated"])
            else:
                log.warning("[SmartRouter] not available, falling back to deepseek")
                full_text, status, rid, inc, conts = _deepseek_complete_full_text(
                    model=DEEPSEEK_MODEL_DEFAULT or "deepseek-reasoner",
                    messages=messages, attachments=attachments,
                    max_output_tokens=max_output_tokens,
                    max_continuations=max_continuations, instructions=instructions,
                )
        elif provider == "deepseek":
            full_text, status, rid, inc, conts = _deepseek_complete_full_text(
                model=routed_model,
                messages=messages,
                attachments=attachments,
                max_output_tokens=max_output_tokens,
                max_continuations=max_continuations,
                instructions=instructions,
            )
        elif provider == "dashscope":
            full_text, status, rid, inc, conts = _dashscope_complete_full_text(
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
                enable_file_search=_plan_allows_file_search(plan),
                enable_code_interpreter=_plan_allows_code_interpreter(plan),
                enable_computer_use=_plan_allows_computer_use(plan, routed_model),
            )

        # ✅ OpenAI-style UX: never append source URL lists into the answer body.
        # Non-stream clients should read structured `sources` from the JSON response.

        # ✅ Persist assistant reply + memory (legacy /chat is non-stream, so we must do it here)
        if full_text:
            # ✅ OpenClaw: 处理 AI 回复中的动作标记（仅当 legacy bridge 真可用）
            if "[ADU_ACTION:" in full_text and _openclaw_runtime_enabled():
                try:
                    _bridge = _openclaw_get_bridge_or_none()
                    if _bridge is not None and getattr(_bridge, "connected", False):
                        _clean, _action_results = process_agent_actions_sync(full_text, _bridge)
                        if _action_results:
                            full_text = _clean + _action_results
                            log.info("[Intent] Legacy /chat: actions executed, results appended")
                except Exception as _intent_err:
                    log.warning("[Intent] Legacy /chat action processing failed: %s", _intent_err)

            try:
                conv_add_message(user_key, conversation_id, "assistant", full_text)
            except Exception:
                pass

            if MEMORY_ENABLED_DEFAULT:
                try:
                    if _should_memory_add(full_text):
                        memory_add(user_key, f"助手：{full_text.strip()}"[:1200])
                except Exception:
                    pass

            # Structured facts extraction (async, best-effort)
            try:
                threading.Thread(
                    target=extract_and_save_memory_facts,
                    args=(user_key, last_user_text, full_text),
                    daemon=True,
                ).start()
            except Exception:
                pass

            # ✅ 四级压缩：写入L0助手轮次 + 触发压缩流水线 + L3结构化提取
            try:
                _comp_append_turn(user_key, conversation_id, "assistant", full_text[:800])
                _comp_maybe_compress(user_key, conversation_id)
                _comp_l3_extract(user_key, last_user_text, full_text)
            except Exception:
                pass

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
            "route_decision": route_decision,
            "sources": web_results_for_response,
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



# -----------------------------
# ✅ Unified web search endpoints
# - GET /web_search?q=...&k=6            -> iOS WebSearchService (results=[{title,url}])
# - POST /v1/search {q,k,kind}           -> developer-friendly (sources=[{title,url,snippet,date,provider}])
# -----------------------------
@app.get("/debug/web_provider")
async def debug_web_provider_endpoint():
    return {
        "ok": True,
        "chat_web_provider": CHAT_WEB_PROVIDER,
        "chat_enable_web_search": bool(CHAT_ENABLE_WEB_SEARCH_DEFAULT),
        "openai_api_key": bool(OPENAI_API_KEY),
        "openai_web_search_model": OPENAI_WEB_SEARCH_MODEL,
        "serper_api_key": bool(SERPER_API_KEY),
        "mode": "smart_router_decision+legacy_serper_restored",
    }

@app.get("/web_search")
async def web_search_endpoint(q: str = "", k: int = 6, kind: str = ""):
    q = (q or "").strip()
    try:
        k_int = int(k or CHAT_WEB_TOPK_DEFAULT)
    except Exception:
        k_int = CHAT_WEB_TOPK_DEFAULT
    k_int = max(1, min(k_int, 20))
    kind0 = (kind or SERPER_DEFAULT_KIND or "search").strip().lower()

    if not q:
        return {"ok": True, "results": []}

    # Prefer Serper when configured (legacy unified search path).
    if CHAT_WEB_PROVIDER.startswith("serper"):
        try:
            res = _serper_web_search(q, k=k_int, kind=kind0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        results = [{"title": str(r.get("title") or r.get("url") or ""), "url": str(r.get("url") or "")} for r in res if r.get("url")]
        return {"ok": True, "provider": "serper", "query": q, "results": results}

    # Fallback: if not using Serper, just return empty (avoid surprising costs).
    return {"ok": True, "provider": CHAT_WEB_PROVIDER or "none", "query": q, "results": []}

@app.post("/v1/search")
async def v1_search(body: Dict[str, Any]):
    q = str(body.get("q") or body.get("query") or "").strip()
    if not q:
        return {"ok": True, "sources": []}

    try:
        k = int(body.get("k") or body.get("top_k") or body.get("topK") or CHAT_WEB_TOPK_DEFAULT)
    except Exception:
        k = CHAT_WEB_TOPK_DEFAULT
    k = max(1, min(k, 20))
    kind = str(body.get("kind") or body.get("type") or SERPER_DEFAULT_KIND or "search").strip().lower()

    if CHAT_WEB_PROVIDER.startswith("serper"):
        srcs = _serper_web_search(q, k=k, kind=kind)
        return {"ok": True, "provider": "serper", "query": q, "sources": srcs}

    return {"ok": True, "provider": CHAT_WEB_PROVIDER or "none", "query": q, "sources": []}
# ✅ Finally include home_chat_router (other endpoints like /session, /rt/intent, etc.)
app.include_router(home_chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Voice Pipeline: STT → DeepSeek LLM → TTS 串流实时语音
try:
    from voice_pipeline import register_voice_pipeline
    register_voice_pipeline(app)
except ImportError:
    log.warning("[BOOT] voice_pipeline.py not found — /voice/ws endpoint disabled")
except Exception as e:
    log.warning("[BOOT] voice_pipeline registration failed: %s", e)

# ✅ ChatAGI Studio 网页版（静态文件服务）
_studio_dir = Path(__file__).parent / "studio"
if _studio_dir.is_dir():
    app.mount("/studio", StaticFiles(directory=str(_studio_dir), html=True), name="studio")
    log.info("[BOOT] Studio mounted at /studio (dir=%s)", _studio_dir)

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
        "model": "claude-3.5-sonnet-20240229",
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
    # Plan-based model routing (server-authoritative, billing-capped)
    plan_raw = _billing_effective_plan_for_request(user_key, _extract_plan(request, body))
    client_model = (str(body.get("model") or "").strip() or str(request.headers.get("x-model") or "").strip())
    # 深度思考开关：App 端 thinkingMode == .deep 时为 true（接受 thinking / deep / reasoning 任一别名）
    _thinking_flag = _boolish(body.get("thinking") or body.get("deep") or body.get("reasoning"))
    requested_model, plan, _model_reason = _select_model_for_request(plan_raw, client_model, thinking=_thinking_flag)

    allow_web = _extract_allow_web(request, body)

    # Plan gate (optional)
    allow_web = bool(allow_web and _plan_allows_web(plan))

    msgs = body.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        return JSONResponse({"ok": False, "error": "missing messages"}, status_code=400)

    # Normalize messages
    # ✅ 有 attachments 时允许最后一条 user 消息为空（只发文件不输文字）
    atts = body.get("attachments") or []
    attachments: List[Dict[str, Any]] = _normalize_chat_attachments(atts)
    has_atts = bool(attachments)
    messages: List[Dict[str, str]] = []
    for _i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user").strip() or "user"
        content = str(m.get("content") or "")
        if content.strip() == "":
            if has_atts and _i == len(msgs) - 1 and role == "user":
                messages.append({"role": role, "content": ""})
            continue
        messages.append({"role": role, "content": content})

    if not messages:
        return JSONResponse({"ok": False, "error": "empty messages"}, status_code=400)

    _raw_last = msgs[-1] if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict) else {}
    _raw_last_user_has_text = (
        str(_raw_last.get("role") or "user").strip() == "user"
        and bool(str(_raw_last.get("content") or "").strip())
    )

    # Attachments (image/audio/file) were normalized above.

    # ✅ Commercial backend billing gate for the real app chat path.
    # Precheck first so an image quota failure does not spend a text credit.
    _billing_block = _billing_guard_request(request, body, FEATURE_TEXT, want=1, consume=False, check_quota=True)
    if _billing_block is not None:
        return _billing_block
    if _billing_has_image_attachments(attachments):
        _billing_block = _billing_guard_request(request, body, FEATURE_IMAGE, want=1, consume=False, check_quota=True)
        if _billing_block is not None:
            return _billing_block

    _billing_block = _billing_guard_request(request, body, FEATURE_TEXT, want=1, consume=True, check_quota=True)
    if _billing_block is not None:
        return _billing_block
    if _billing_has_image_attachments(attachments):
        _billing_block = _billing_guard_request(request, body, FEATURE_IMAGE, want=1, consume=True, check_quota=True)
        if _billing_block is not None:
            return _billing_block

    user_transcript = ""
    if attachments:
        # Do transcription in a background thread (avoid blocking the event loop)
        user_transcript = await asyncio.to_thread(_transcribe_audio_attachments_inplace, attachments)

    # ✅ Image/file-only request: create an explicit current turn so attachments do not bind to old history.
    if attachments and not _raw_last_user_has_text:
        messages = _ensure_current_user_turn_for_attachments(messages, attachments)

    # Last user text (used for title, memory query, persistence)
    last_user_text = ""
    for mm in reversed(messages):
        if (mm.get("role") or "").strip() == "user":
            last_user_text = (mm.get("content") or "").strip()
            break
    if (not last_user_text) and user_transcript:
        last_user_text = user_transcript.strip()

    # 🧠 通知意识系统："玉勇说话了"（文本或语音 transcript 都算）
    if last_user_text:
        try:
            from adu_consciousness import consciousness
            consciousness.on_user_message(last_user_text)
        except Exception as _cs_e:
            log.debug("[Consciousness] on_user_message skip (prepare): %s", _cs_e)

    # ✅ AutoBrain direct computer-task path (opt-in only)
    _brain_result = await _maybe_run_brain_direct(request, body, messages, user_transcript=user_transcript)

    # 🧠 通知意识系统：脑内任务已完成
    #    注意：本函数里 _brain_result 的下游消费点尚未接入（疑似死代码），
    #    但任务已实际执行，意识层应当知道。
    if _brain_result is not None:
        try:
            from adu_consciousness import consciousness
            _goal_label = (last_user_text or str(_brain_result.get("summary") or ""))[:30]
            _success = bool(_brain_result.get("ok", _brain_result.get("success", True)))
            consciousness.on_task_complete(_goal_label, _success)
        except Exception as _cs_e:
            log.debug("[Consciousness] on_task_complete skip (prepare-brain): %s", _cs_e)

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


    # ✅ Server-side history fallback:
    # Some clients/plans may only send the newest user message (e.g. after switching screens).
    # If we have a conversation_id, hydrate context from our SQLite history so the model keeps continuity.
    try:
        if conversation_id and CHAT_HISTORY_FALLBACK_MESSAGES > 0:
            client_len = len(messages)
            non_system = [m for m in messages if (m.get("role") or "").strip() not in ("system", "developer")]
            has_assistant = any((m.get("role") or "").strip() == "assistant" for m in non_system)

            # Heuristic: if client sent no assistant turns and only <= 2 non-system messages,
            # treat it as "thin context" and hydrate from DB.
            if (not has_assistant) and (len(non_system) <= 2):
                with _conv_conn() as con:
                    cur = con.execute(
                        "SELECT role, content FROM messages WHERE conversation_id=? AND user_key=? ORDER BY created_at DESC LIMIT ?",
                        (conversation_id, user_key, CHAT_HISTORY_FALLBACK_MESSAGES),
                    )
                    rows = cur.fetchall()

                hist = []
                for role, content in reversed(rows):
                    role = (role or "").strip()
                    content = (content or "").strip()
                    if not content:
                        continue
                    if role not in ("user", "assistant", "system", "developer"):
                        continue
                    hist.append({"role": role, "content": content})

                if hist:
                    # De-dup: if the newest DB message equals the first current non-system message, drop it.
                    if non_system:
                        hlast = hist[-1]
                        first = non_system[0]
                        if (hlast.get("role") == (first.get("role") or "").strip()) and (
                            (hlast.get("content") or "").strip() == (first.get("content") or "").strip()
                        ):
                            hist = hist[:-1]

                    sys_msgs = [m for m in messages if (m.get("role") or "").strip() in ("system", "developer")]
                    messages = sys_msgs + hist + non_system
                    log.info(
                        "[chat-context] hydrated conv=%s client=%d hist=%d final=%d",
                        conversation_id, client_len, len(hist), len(messages)
                    )
    except Exception as e:
        log.warning("[chat-context] hydrate failed: %s", e)

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

    # ✅ Inject structured long-term memory facts (query-independent)
    if MEMORY_FACTS_ENABLED_DEFAULT:
        try:
            facts_ctx = memory_facts_build_prompt(user_key, limit=MEMORY_FACTS_PROMPT_LIMIT)
        except Exception:
            facts_ctx = ""
        if facts_ctx:
            insert_at = 1 if messages and (messages[0].get("role") in ("system", "developer")) else 0
            messages.insert(insert_at, {"role": "system", "content": facts_ctx})


    # Inject memory context for better recall
    if MEMORY_ENABLED_DEFAULT and last_user_text:
        try:
            mem_ctx = memory_build_context(user_key, last_user_text, k=MEMORY_TOP_K_DEFAULT, min_score=MEMORY_MIN_SCORE_DEFAULT)
        except Exception:
            mem_ctx = ""
        if mem_ctx:
            insert_at = 1 if messages and (messages[0].get("role") in ("system", "developer")) else 0
            # If we already inserted the facts memory block, put vector memory after it.
            if insert_at < len(messages):
                c0 = (messages[insert_at].get("content") or "") if isinstance(messages[insert_at], dict) else ""
                if isinstance(c0, str) and c0.startswith("以下是【长期记忆】（重要事实/偏好"):
                    insert_at += 1
            messages.insert(insert_at, {"role": "system", "content": mem_ctx})

    # ✅ OpenClaw: 注入项目上下文（让 AI 理解用户的全部工程）
    try:
        bridge = _openclaw_get_bridge_or_none()
        if bridge is not None and getattr(bridge, "connected", False) and getattr(bridge, "_project_cache", None):
            all_ctx_parts = []
            for _pth, _scan in bridge._project_cache.items():
                _ctx = bridge.build_project_context(_scan)
                if _ctx:
                    all_ctx_parts.append(_ctx)
            if all_ctx_parts:
                full_project_ctx = "\n\n".join(all_ctx_parts)
                insert_at = 1 if messages and (messages[0].get("role") in ("system", "developer")) else 0
                messages.insert(insert_at, {"role": "system", "content": full_project_ctx})
    except Exception as e:
        log.debug("[OpenClaw] project context injection skipped: %s", e)
    # ✅ OpenClaw: 注入 Agent 工具能力（让聊天模式也能调 OpenClaw）
    try:
        bridge = _openclaw_get_bridge_or_none()
        if bridge is not None and getattr(bridge, "connected", False):
            messages = inject_agent_system_prompt(messages, bridge_connected=True)
    except Exception:
        pass

    # ✅ 感知模块：注入物理世界感知上下文（位置/朝向/运动状态）
    _perception_data = body.get("perception")
    if isinstance(_perception_data, dict) and _perception_data.get("latitude"):
        try:
            from perception_module import format_perception_context, store_to_world_memory, enrich_perception_data
            _perception_data = await enrich_perception_data(_perception_data)
            _perc_ctx = format_perception_context(_perception_data)
            if _perc_ctx:
                _ins = 1 if messages and (messages[0].get("role") in ("system", "developer")) else 0
                messages.insert(_ins, {"role": "system", "content": _perc_ctx})
                log.info("[Perception] ✅ context injected (prepare): %s",
                         _perception_data.get("fullAddress") or f"{_perception_data.get('latitude'):.4f}")
            store_to_world_memory(user_key or "anonymous", _perception_data)
        except Exception as _pe:
            log.warning("[Perception] injection failed: %s", _pe)

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

    # ✅ Smart Router Decision comes first.
    # Raw allow_web means "web is permitted"; it does NOT mean "force web_search".
    route_decision = _smart_router_decision(
        user_text=last_user_text,
        allow_web=allow_web,
        attachments=attachments,
        requested_model=requested_model,
        plan=plan,
        body=body,
    )
    allow_web = bool(route_decision.get("need_web"))

    # ✅ Provider routing uses the smart decision result, not the raw client flag.
    provider, route_reason = _route_provider(allow_web=allow_web, attachments=attachments, model=requested_model)
    _provider_route_reason = route_reason
    provider, route_reason = _ensure_provider_available(provider, route_reason)
    _smart_reason = f"smart_router_decision:{route_decision.get('route') or 'unknown'}"
    if route_reason != _provider_route_reason:
        route_reason = f"{_smart_reason}->{route_reason}"
    else:
        route_reason = _smart_reason
    routed_model = _select_routed_model(provider, requested_model)
    if provider == "openai" and allow_web:
        routed_model = OPENAI_WEB_SEARCH_MODEL
    # ✅ 前后端对齐：把后端真实生效的 model/plan 回传到 route_decision，
    #    让 iOS 的 RouteTier.fromDecision 拿到准确档位（机器人 6.x）。
    try:
        if isinstance(route_decision, dict):
            route_decision["model"] = routed_model
            route_decision["plan"] = plan
            route_decision["robot_version"] = display_robot_version(plan)
    except Exception:
        pass
    log.info(
        "[chat-route] provider=%s model=%s allow_web=%s reason=%s decision=%s web_provider=%s",
        provider, routed_model, bool(allow_web), route_reason, route_decision, CHAT_WEB_PROVIDER,
    )

    # Streaming path: sources are fetched/pushed inside the worker so the UI can render site icons.
    enable_tts_streaming = bool(body.get("tts_stream", False)) and CHAT_ENABLE_TTS_STREAMING
    job = _create_chat_job(enable_tts_streaming=enable_tts_streaming, user_key=user_key)
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
        last_user_text=last_user_text,
        memory_enabled=MEMORY_ENABLED_DEFAULT,
        plan=plan,
    )

    return {
        "ok": True,
        "chat_id": job.chat_id,
        "conversation_id": conversation_id,
        "user_key": user_key,
        "provider": provider,
        "model": routed_model,
        "allow_web": bool(allow_web),
        "route_decision": route_decision,
        "events_url": f"/chat/events/{job.chat_id}?access_token={job.access_token}",
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
    if not _chat_job_authorized(job, request):
        return JSONResponse({"ok": False, "error": "chat_forbidden"}, status_code=403)

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
def chat_result(chat_id: str, request: Request):
    job = _get_chat_job(chat_id)
    if not job:
        return JSONResponse({"ok": False, "error": "chat not found"}, status_code=404)
    if not _chat_job_authorized(job, request):
        return JSONResponse({"ok": False, "error": "chat_forbidden"}, status_code=403)
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

@app.post("/conversations/{conversation_id}/export")
async def conversations_export(conversation_id: str, request: Request):
    """打包指定对话所有消息为 zip，返回 download_url。"""
    user_key = _derive_user_key(request, {})
    with _conv_conn() as con:
        cur = con.execute(
            "SELECT id,role,content,created_at FROM messages WHERE conversation_id=? AND user_key=? ORDER BY created_at ASC LIMIT 500",
            (conversation_id, user_key),
        )
        rows = cur.fetchall()
        conv_row = con.execute(
            "SELECT title,created_at FROM conversations WHERE id=? AND user_key=?",
            (conversation_id, user_key),
        ).fetchone()

    if not rows:
        return JSONResponse({"ok": False, "error": "no messages found"}, status_code=404)

    title = (conv_row[0] if conv_row else "") or "对话记录"
    created_at = (conv_row[1] if conv_row else "") or ""

    # 拼成可读 Markdown
    lines = [f"# {title}", f"导出时间：{int(time.time())}", f"对话ID：{conversation_id}", ""]
    for (mid, role, content, ts) in rows:
        role_label = "用户" if role == "user" else "阿杜"
        lines.append(f"### {role_label}  `{ts}`")
        lines.append(content or "")
        lines.append("")
    full_text = "\n".join(lines)

    meta = {
        "conversation_id": conversation_id,
        "title": title,
        "created_at": created_at,
        "message_count": len(rows),
        "exported_at": int(time.time()),
    }
    dl_id = _create_download_zip(full_text=full_text, parts=[], meta=meta, req_body={})
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "message_count": len(rows),
        "download_url": f"/chat/download/{dl_id}.zip",
    }


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


# ============================================================
# ✅ Commercial Memory OS Status / Debug APIs
# - Per-user only. Never returns global user data.
# - Useful for iOS/Web debug panels before cloud launch.
# ============================================================

def _memory_table_count_for_user(table: str, user_key: str) -> int:
    try:
        if table not in {"memory_items", "memory_facts", "memory_timeline", "memory_episodes"}:
            return 0
        with _mem_conn() as con:
            row = con.execute(f"SELECT COUNT(*) FROM {table} WHERE user_key=?", (_sanitize_user_key(user_key),)).fetchone()
        return int((row or [0])[0] or 0)
    except Exception:
        return 0


def _memory_table_latest_for_user(table: str, user_key: str) -> float:
    try:
        if table not in {"memory_items", "memory_facts", "memory_timeline", "memory_episodes"}:
            return 0.0
        col = "last_used_at" if table == "memory_items" else ("updated_at" if table == "memory_episodes" else "created_at")
        with _mem_conn() as con:
            row = con.execute(f"SELECT MAX({col}) FROM {table} WHERE user_key=?", (_sanitize_user_key(user_key),)).fetchone()
        return float((row or [0])[0] or 0.0)
    except Exception:
        return 0.0


@app.get("/api/brain/memory/status")
def brain_memory_status_api(request: Request):
    ident = _derive_memory_identity(request, {})
    user_key = _sanitize_user_key(str(ident.get("user_key") or ""))
    counts = {
        "memory_items": _memory_table_count_for_user("memory_items", user_key),
        "memory_facts": _memory_table_count_for_user("memory_facts", user_key),
        "memory_timeline": _memory_table_count_for_user("memory_timeline", user_key),
        "memory_episodes": _memory_table_count_for_user("memory_episodes", user_key),
    }
    latest = {
        "memory_items": _memory_table_latest_for_user("memory_items", user_key),
        "memory_facts": _memory_table_latest_for_user("memory_facts", user_key),
        "memory_timeline": _memory_table_latest_for_user("memory_timeline", user_key),
        "memory_episodes": _memory_table_latest_for_user("memory_episodes", user_key),
    }
    return {
        "ok": True,
        "memory_enabled": bool(MEMORY_ENABLED_DEFAULT),
        "facts_enabled": bool(MEMORY_FACTS_ENABLED_DEFAULT),
        "db_path": MEM_DB_PATH,
        "identity": ident,
        "user_key": user_key,
        "counts": counts,
        "latest": latest,
        "commercial_isolation": True,
    }


@app.post("/api/brain/memory/debug-context")
async def brain_memory_debug_context_api(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    ident = _derive_memory_identity(request, body or {})
    user_key = _sanitize_user_key(str(ident.get("user_key") or ""))
    query = (body.get("query") or body.get("q") or "").strip()
    conv_id = (body.get("conversation_id") or body.get("conversationId") or "").strip()

    facts = []
    timeline = []
    semantic = []
    recent = []
    episodes = []
    try:
        facts = memory_facts_list(user_key, limit=int(body.get("facts_limit") or 10))
    except Exception:
        facts = []
    try:
        timeline = memory_timeline_query(user_key, keyword=(body.get("timeline_keyword") or ""), limit=int(body.get("timeline_limit") or 10))
    except Exception:
        timeline = []
    try:
        if query:
            semantic = memory_search(user_key, query, k=int(body.get("semantic_k") or 6), min_score=MEMORY_MIN_SCORE_DEFAULT)
    except Exception:
        semantic = []
    try:
        recent = _mem_recent_items(user_key, limit=int(body.get("recent_limit") or 6))
    except Exception:
        recent = []
    try:
        if MEMORY_ENGINE is not None:
            episodes = MEMORY_ENGINE.episode_list(user_key, limit=int(body.get("episodes_limit") or 8), conversation_id=conv_id)
    except Exception:
        episodes = []

    ctx = build_unified_memory_context(user_key, query=query, conversation_id=conv_id)
    return {
        "ok": True,
        "identity": ident,
        "user_key": user_key,
        "query": query,
        "facts": facts,
        "timeline": timeline,
        "semantic": semantic,
        "recent": recent,
        "episodes": episodes,
        **ctx,
    }


@app.get("/api/brain/memory/debug-context")
def brain_memory_debug_context_get_api(request: Request, q: str = "", conversation_id: str = ""):
    ident = _derive_memory_identity(request, {})
    user_key = _sanitize_user_key(str(ident.get("user_key") or ""))
    ctx = build_unified_memory_context(user_key, query=q, conversation_id=conversation_id)
    return {"ok": True, "identity": ident, "user_key": user_key, "query": q, **ctx}


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

@app.post("/memory/clear")
def memory_clear_api(request: Request):
    user_key = _derive_user_key(request, {})
    with _mem_conn() as con:
        cur = con.execute("DELETE FROM memory_items WHERE user_key=?", (user_key,))
    return {"ok": True, "user_key": user_key, "deleted": int(cur.rowcount or 0)}

# -----------------------------
# ✅ Memory Facts / Episodes APIs (Plan A)
# -----------------------------

@app.get("/memory/facts/list")
def memory_facts_list_api(request: Request, limit: int = 20):
    user_key = _derive_user_key(request, {})
    try:
        items = memory_facts_list(user_key, limit=int(limit))
    except Exception:
        items = []
    return {"ok": True, "user_key": user_key, "items": items}

@app.post("/memory/facts/add")
async def memory_facts_add_api(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    user_key = _derive_user_key(request, body)
    content = (body.get("content") or body.get("text") or "").strip()
    if not content:
        return JSONResponse({"ok": False, "error": "missing content"}, status_code=400)
    tags = (body.get("tags") or "").strip()
    try:
        importance = int(body.get("importance") or 1)
    except Exception:
        importance = 1
    try:
        memory_facts_save(user_key, content, tags=tags, importance=importance)
    except Exception:
        pass
    return {"ok": True, "user_key": user_key}

@app.post("/memory/facts/clear")
def memory_facts_clear_api(request: Request):
    user_key = _derive_user_key(request, {})
    try:
        with _mem_conn() as con:
            cur = con.execute("DELETE FROM memory_facts WHERE user_key=?", (_sanitize_user_key(user_key),))
            deleted = int(cur.rowcount or 0)
    except Exception:
        deleted = 0
    return {"ok": True, "user_key": user_key, "deleted": deleted}

@app.get("/memory/episodes/list")
def memory_episodes_list_api(request: Request, limit: int = 20):
    user_key = _derive_user_key(request, {})
    items = []
    try:
        if MEMORY_ENGINE is not None:
            items = MEMORY_ENGINE.episode_list(user_key, limit=int(limit))
    except Exception:
        items = []
    return {"ok": True, "user_key": user_key, "items": items}

# ✅ 时间线记忆 API
@app.get("/memory/timeline")
def memory_timeline_list_api(request: Request, date: str = "", event_type: str = "", keyword: str = "", limit: int = 20):
    user_key = _derive_user_key(request, {})
    items = memory_timeline_query(user_key, date=date, event_type=event_type, keyword=keyword, limit=limit)
    return {"ok": True, "user_key": user_key, "items": items}

@app.post("/memory/timeline/add")
async def memory_timeline_add_api(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    user_key = _derive_user_key(request, body)
    summary = (body.get("summary") or body.get("text") or "").strip()
    if not summary:
        return JSONResponse({"ok": False, "error": "missing summary"}, status_code=400)
    memory_timeline_add(
        user_key, summary,
        event_type=(body.get("event_type") or "manual").strip(),
        detail=(body.get("detail") or "").strip(),
        source=(body.get("source") or "").strip(),
    )
    return {"ok": True}

# ✅ 统一记忆 debug 接口 — 调试"模型到底吃到了什么记忆"
@app.post("/memory/context")
async def memory_context_preview(request: Request):
    """
    POST /memory/context
    JSON: { "query": "...", "conversation_id": "..." }
    返回: facts/timeline/semantic/recent/full_prompt 分项结果
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    user_key = _derive_user_key(request, body)
    query = (body.get("query") or body.get("q") or "").strip()
    conv_id = (body.get("conversation_id") or "").strip()
    ctx = build_unified_memory_context(user_key, query=query, conversation_id=conv_id)
    return {"ok": True, "user_key": user_key, **ctx}

@app.get("/memory/context")
def memory_context_preview_get(request: Request, q: str = "", conversation_id: str = ""):
    user_key = _derive_user_key(request, {})
    ctx = build_unified_memory_context(user_key, query=q, conversation_id=conversation_id)
    return {"ok": True, "user_key": user_key, **ctx}

# ✅ Realtime turn commit — 前端兜底提交（sideband 自动落库为主，此接口为备用）

# ============================================================
# ✅ 实时 RAG 工具：search_memory
# GPT Realtime / Qwen function calling 调用此接口查询用户记忆
# ============================================================

def tool_search_memory(
    user_key: str,
    query: str,
    top_k: int = 5,
    scope: str = "hybrid",
) -> Dict[str, Any]:
    """统一查询 facts / timeline / 向量记忆，返回标准化结果列表。"""
    uk = _sanitize_user_key(user_key)
    q = (query or "").strip()
    if not q:
        return {"query": q, "items": []}

    items: List[Dict[str, Any]] = []

    # 1) facts 记忆（字段：content, importance）
    try:
        if scope in ("facts", "hybrid"):
            with _mem_conn() as con:
                rows = con.execute(
                    """
                    SELECT content, importance, created_at
                    FROM memory_facts
                    WHERE user_key=? AND content LIKE ?
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                    """,
                    (uk, f"%{q[:40]}%", top_k),
                ).fetchall()
            for content, importance, created_at in rows:
                items.append({
                    "type": "fact",
                    "text": str(content or "").strip(),
                    "score": min(1.0, float(importance or 1) / 3.0),
                    "created_at": created_at,
                })
    except Exception as _e:
        log.warning("[tool_search_memory] facts error: %s", _e)

    # 2) timeline 记忆（字段：summary, detail, event_type）
    try:
        if scope in ("timeline", "hybrid"):
            with _mem_conn() as con:
                rows = con.execute(
                    """
                    SELECT summary, detail, event_type, created_at
                    FROM memory_timeline
                    WHERE user_key=? AND (summary LIKE ? OR detail LIKE ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (uk, f"%{q[:40]}%", f"%{q[:40]}%", top_k),
                ).fetchall()
            for summary, detail, event_type, created_at in rows:
                txt = str(summary or "").strip()
                if detail:
                    txt = f"{txt} | {str(detail).strip()[:100]}"
                items.append({
                    "type": "timeline",
                    "text": txt,
                    "score": 0.7,
                    "event_type": event_type,
                    "created_at": created_at,
                })
    except Exception as _e:
        log.warning("[tool_search_memory] timeline error: %s", _e)

    # 3) 向量语义记忆
    try:
        if scope == "hybrid" and q:
            sem = memory_search(uk, q, k=top_k, min_score=0.3)
            for x in (sem or []):
                txt = str(x.get("text") or x.get("content") or "").strip()
                if txt:
                    items.append({
                        "type": "semantic",
                        "text": txt,
                        "score": float(x.get("score") or 0.5),
                    })
    except Exception as _e:
        log.warning("[tool_search_memory] semantic error: %s", _e)

    # 去重 + 排序
    seen: Dict[str, Dict] = {}
    for it in items:
        key = it["text"][:200]
        if key not in seen or float(it.get("score", 0)) > float(seen[key].get("score", 0)):
            seen[key] = it

    merged = sorted(seen.values(), key=lambda x: float(x.get("score", 0)), reverse=True)[:top_k]
    return {"query": q, "items": merged}


@app.post("/tools/search_memory")
async def tools_search_memory_api(req: Request):
    """
    ✅ 实时 RAG 工具接口
    GPT / Qwen Realtime function calling 查询用户记忆
    Body: { "client_id": "...", "query": "...", "top_k": 5, "scope": "hybrid" }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}

    client_id = str(body.get("client_id") or "").strip()
    query = str(body.get("query") or "").strip()
    scope = str(body.get("scope") or "hybrid").strip().lower()
    top_k = max(1, min(int(body.get("top_k") or 5), 10))

    user_key = _derive_user_key(req, body)

    if not query:
        return JSONResponse({"query": "", "items": []})

    result = tool_search_memory(user_key=user_key, query=query, top_k=top_k, scope=scope)
    log.info("[tool_search_memory] user=%s query=%r items=%d", user_key, query[:40], len(result["items"]))
    return JSONResponse(result)


@app.post("/realtime/turn/commit")
async def realtime_turn_commit(request: Request):
    """
    POST /realtime/turn/commit
    JSON: {
        "client_id": "...",
        "conversation_id": "...",       // 可选，空则自动创建
        "user_text": "用户说的话",
        "assistant_text": "助手回复",
        "profile": "default",
        "user_final_at": 1711234567.0,  // 可选
        "assistant_final_at": 1711234568.0,
    }
    返回: { "ok": true, "conversation_id": "...", "messages_added": 2 }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    user_key = _derive_user_key(request, body)
    conv_id = (body.get("conversation_id") or "").strip()
    user_text = (body.get("user_text") or "").strip()
    assistant_text = (body.get("assistant_text") or "").strip()

    if not user_text and not assistant_text:
        return JSONResponse({"ok": False, "error": "missing user_text or assistant_text"}, status_code=400)

    # 自动创建 conversation（如果没有）
    if not conv_id:
        title = f"语音通话 {time.strftime('%m/%d %H:%M')}"
        if user_text:
            title = user_text[:40]
        conv_id = conv_create(user_key, title)

    added = 0

    # 写入 user message
    if user_text:
        try:
            conv_add_message(user_key, conv_id, "user", user_text)
            added += 1
        except Exception:
            pass
        # timeline
        try:
            memory_timeline_add(user_key, f"用户说：{user_text[:200]}", event_type="realtime_user", source=conv_id)
        except Exception:
            pass
        # 压缩
        try:
            _comp_append_turn(user_key, conv_id, "user", user_text[:800])
        except Exception:
            pass
        # 🧠 通知意识系统：实时语音通道的"被打断"
        try:
            from adu_consciousness import consciousness
            consciousness.on_user_message(user_text)
        except Exception as _cs_e:
            log.debug("[Consciousness] on_user_message skip (realtime): %s", _cs_e)

    # 写入 assistant message
    if assistant_text:
        try:
            conv_add_message(user_key, conv_id, "assistant", assistant_text)
            added += 1
        except Exception:
            pass
        # timeline
        try:
            memory_timeline_add(user_key, f"助手说：{assistant_text[:200]}", event_type="realtime_assistant", source=conv_id)
        except Exception:
            pass
        # 压缩
        try:
            _comp_append_turn(user_key, conv_id, "assistant", assistant_text[:800])
            _comp_maybe_compress(user_key, conv_id)
        except Exception:
            pass

    # L3 结构化提取
    if user_text and assistant_text:
        try:
            _comp_l3_extract(user_key, user_text, assistant_text)
        except Exception:
            pass
        # facts extraction
        try:
            extract_and_save_memory_facts(user_key, user_text, assistant_text)
        except Exception:
            pass

    # ✅ 向量记忆写入（文本和语音统一）
    try:
        if user_text and _should_memory_add(user_text):
            memory_add(user_key, f"用户：{user_text[:800]}")
        if assistant_text and _should_memory_add(assistant_text):
            memory_add(user_key, f"助手：{assistant_text[:800]}")
    except Exception as _me:
        log.warning("[turn_commit] memory_add failed: %s", _me)

    log.info("[TurnCommit] ✅ user=%s conv=%s added=%d", user_key, conv_id, added)
    return {"ok": True, "conversation_id": conv_id, "messages_added": added, "user_key": user_key}

# ✅ Conversation sync — 前端初始化/切换会话时对齐
@app.post("/conversation/sync")
async def conversation_sync(request: Request):
    """
    POST /conversation/sync
    JSON: { "conversation_id": "...", "title": "...", "mode": "chat|realtime", "profile": "default" }
    返回: { "ok": true, "conversation_id": "...", "created": false }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    user_key = _derive_user_key(request, body)
    conv_id = (body.get("conversation_id") or "").strip()
    title = (body.get("title") or "").strip()
    mode = (body.get("mode") or "chat").strip()

    created = False
    if not conv_id:
        conv_id = conv_create(user_key, title or f"{'语音通话' if mode == 'realtime' else '新对话'} {time.strftime('%m/%d %H:%M')}")
        created = True
    else:
        # Touch existing conversation to update timestamp
        try:
            conv_touch(user_key, conv_id, "")
        except Exception:
            # Conversation might not exist yet — create it
            conv_id = conv_create(user_key, title or "新对话")
            created = True

    return {"ok": True, "conversation_id": conv_id, "created": created, "user_key": user_key}

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

        # ✅ Voice clone disabled: force normal TTS voice.
        _el_api_key = ""
        _el_voice_id = None

        for seg in segments:
            seg = (seg or "").strip()
            if not seg:
                continue

            # ✅ Voice clone disabled: do not call external cloned-voice TTS; continue with normal TTS.

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

            try:
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
            except _TTS_STREAM_EXCEPTIONS as e:
                log.warning("[TTS.stream] upstream stream interrupted: %s", e)
            except Exception as e:
                log.warning("[TTS.stream] upstream stream failed: %s", e)
            finally:
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

    # ✅ Billing V1: direct photo upload/cache consumes 1 image quota.
    _billing_block = _billing_guard_request(req, {"session_id": session_id}, FEATURE_IMAGE, want=1, consume=True, check_quota=True)
    if _billing_block is not None:
        return _billing_block

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

def _realtime_ephemeral(
    model: str,
    voice: str,
    instructions: Optional[str] = None,
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    session_extra: Optional[Dict[str, Any]] = None,
):
    """
    Create a Realtime session and return an ephemeral key.

    Backward compatible with legacy clients (keeps model/voice/instructions behavior),
    but also allows attaching Realtime function tools (e.g. web_search / remember).
    """
    body: Dict[str, Any] = {"model": model, "voice": voice}
    if instructions:
        body["instructions"] = instructions
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    if session_extra and isinstance(session_extra, dict):
        # Allow advanced callers to pass additional session config
        for k, v in session_extra.items():
            if k not in ("model", "voice"):
                body[k] = v

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


# ---- Qwen Omni Realtime session bootstrap ----

def _normalize_qwen_realtime_model(model: str) -> str:
    raw = (model or "").strip()
    # iOS may still send OpenAI realtime model ids; never pass those to DashScope.
    if not raw or not _is_qwen_model(raw) or "realtime" not in raw.lower():
        return (QWEN_REALTIME_MODEL or "qwen3.5-omni-flash-realtime").strip()
    return raw


def _normalize_qwen_realtime_voice(voice: str) -> str:
    raw = (voice or "").strip()
    default_voice = (QWEN_REALTIME_VOICE or "Cherry").strip() or "Cherry"

    # ✅ Cloned/custom voice IDs are disabled for Qwen Realtime.
    # Only pass known DashScope/Qwen voice names; unknown ids fall back to the official default.
    # This prevents a saved custom voice id from leaking into realtime sessions.
    openai_voices = {"alloy", "ash", "ballad", "coral", "echo", "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse"}
    qwen_voice_names = {
        "Cherry", "Tina", "Serena", "Ethan", "Chelsie", "Cherry", "Dylan", "Jada",
        "Sunny", "Alvin", "Rosa", "Layla", "Luna", "Maya", "Aida", "Zephyr"
    }
    if not raw or raw.lower() in openai_voices:
        return default_voice
    if raw in qwen_voice_names:
        return raw
    return default_voice


def _qwen_realtime_session_url(model: str) -> str:
    """Build Qwen Realtime WebSocket URL (client connects directly)."""
    m = _normalize_qwen_realtime_model(model)
    return f"{QWEN_REALTIME_BASE_URL}/api-ws/v1/realtime?model={m}"


def _qwen_realtime_ephemeral(
    model: str,
    voice: str,
    instructions: Optional[str] = None,
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Qwen Omni Realtime 不需要 ephemeral key 机制。
    客户端直接用 DashScope API Key 连 WebSocket。
    返回 (session_info_dict, error_str)。
    """
    if not DASHSCOPE_API_KEY:
        return None, "DASHSCOPE_API_KEY not configured"

    m = _normalize_qwen_realtime_model(model)
    v = _normalize_qwen_realtime_voice(voice)

    info: Dict[str, Any] = {
        "provider": "qwen",
        "ws_url": _qwen_realtime_session_url(m),
        "api_key": DASHSCOPE_API_KEY,  # 客户端放 Authorization: Bearer
        "model": m,
        "voice": v,
        "input_audio_format": "pcm",  # PCM16 16kHz mono input
        "output_audio_format": "pcm",  # PCM16 24kHz mono output
        "input_sample_rate": 16000,
        "output_sample_rate": 24000,
        "pcm24_to_pcm16_required": False,
        "max_session_minutes": 120,
        "turn_detection": {
            # 阿里官方 client-events 文档：type 仅支持 "server_vad"。
            # threshold / silence_duration_ms 是文档列出的两个可调字段；
            # 不再下发 prefix_padding_ms / create_response / interrupt_response —— 这三个是
            # OpenAI Realtime 的扩展字段，Qwen 服务端会静默忽略，导致以为开了 interrupt_response
            # 实际从未生效。Qwen 的 server_vad 在检测到 speech_stopped 后会自动 create_response，
            # 真正的打断由客户端清播放 + response.cancel + userIsSpeaking gate 配合完成。
            "type": "server_vad",
            "threshold": QWEN_REALTIME_VAD_THRESHOLD,
            "silence_duration_ms": QWEN_REALTIME_SILENCE_DURATION_MS,
        },
    }
    if instructions:
        info["instructions"] = instructions
    # Qwen Realtime is sensitive to OpenAI-style tool schemas. Keep direct audio stable by default.
    if tools and QWEN_REALTIME_ENABLE_TOOLS:
        info["tools"] = tools
        info["tool_choice"] = tool_choice or "auto"

    return info, None


def pcm24_to_pcm16(data: bytes) -> bytes:
    """
    Convert PCM 24-bit (little-endian, 3 bytes/sample) to PCM 16-bit (2 bytes/sample).
    用于 Qwen3 Realtime 输出音频转换，iOS AVAudioEngine 只支持 pcm16。
    """
    if len(data) % 3 != 0:
        # 截断到 3 的倍数
        data = data[:len(data) - (len(data) % 3)]
    out = bytearray(len(data) // 3 * 2)
    j = 0
    for i in range(0, len(data), 3):
        # 取高 16 位（第 2、3 字节）
        out[j] = data[i + 1]
        out[j + 1] = data[i + 2]
        j += 2
    return bytes(out)


def _pick_realtime_provider(body: Dict[str, Any]) -> str:
    """
    决定用 OpenAI / Qwen / Pipeline Realtime。
    优先级: body.provider > REALTIME_PROVIDER env > "openai"
    """
    req_provider = (body.get("provider") or body.get("realtime_provider") or "").strip().lower()
    if req_provider in ("qwen", "dashscope", "aliyun"):
        return "qwen"
    if req_provider in ("pipeline", "deepseek", "assembled"):
        return "pipeline"
    if req_provider in ("openai", "gpt"):
        return "openai"
    # env default
    if REALTIME_PROVIDER in ("qwen", "dashscope", "aliyun"):
        return "qwen"
    if REALTIME_PROVIDER in ("pipeline", "deepseek", "assembled"):
        return "pipeline"
    return "openai"


def _resolve_realtime_instructions(body: Dict[str, Any]) -> Optional[str]:
    provided = (body.get("instructions") or "").strip()
    if provided:
        return provided

    profile = (body.get("profile") or body.get("mode") or "").strip().lower()
    if profile == "companion":
        return None
    if profile == "remix":
        return REMIX_REALTIME_INSTRUCTIONS

    return _get_realtime_default_instructions()

def _nav_instructions(nav_ctx=None) -> str:
    """
    导航专用 Realtime 助手指令（profile="nav"）。
    nav_ctx 由 iOS 通过 sessionProfile="nav|{json}" 传入，包含实时导航状态。
    """
    base = (
        "你是车载导航语音助手「阿杜」。\n"
        "性格：亲切自然，像老朋友一样，偶尔幽默，绝不啰嗦。\n"
        "回答规则：\n"
        "- 所有回答不超过30字，口语化，直接说重点\n"
        "- 禁止说「好的」「当然」「没问题」等开场废话\n"
        "- 用中文回答，数字念中文（「三公里」不是「3km」）\n"
        "- 用户在开车，不要让他分心，答案要短且清晰\n"
    )

    if not nav_ctx:
        return base + "\n当前未在导航中，可帮用户搜索目的地或规划路线。"

    is_nav = nav_ctx.get("is_navigating", False)
    transport = nav_ctx.get("transport", "驾车")
    dest = nav_ctx.get("destination", "")
    remain_dist = nav_ctx.get("remaining_distance", "")
    remain_time = nav_ctx.get("remaining_time", "")
    eta = nav_ctx.get("eta", "")
    cur_step = nav_ctx.get("current_step", "")
    next_step = nav_ctx.get("next_step", "")

    if is_nav:
        nav_block = "\n【当前导航状态】\n"
        if dest:         nav_block += f"目的地：{dest}\n"
        if remain_dist:  nav_block += f"剩余距离：{remain_dist}\n"
        if remain_time:  nav_block += f"预计时间：{remain_time}\n"
        if eta:          nav_block += f"预计到达：{eta}\n"
        nav_block +=     f"出行方式：{transport}\n"
        if cur_step:     nav_block += f"当前步骤：{cur_step}\n"
        if next_step:    nav_block += f"下一步骤：{next_step}\n"
        nav_block += (
            "\n\u7528\u6237\u53ef\u80fd\u95ee\uff1a\u8fd8\u6709\u591a\u4e45\u3001\u6362\u6761\u8def\u3001\u5f53\u524d\u5728\u54ea\u3001\u8981\u4e0d\u8981\u4e0a\u9ad8\u901f\u7b49\u3002\n"
            "\u6839\u636e\u4ee5\u4e0a\u72b6\u6001\u76f4\u63a5\u56de\u7b54\uff0c\u4e0d\u8981\u8bf4\u4e0d\u77e5\u9053\u5f53\u524d\u4f4d\u7f6e\u4e4b\u7c7b\u7684\u8bdd\u3002\n"
            "\u5982\u679c\u7528\u6237\u8bf4\u300c\u6362\u6761\u8def\u300d\u300c\u8d70\u5907\u9009\u300d\uff0c\u56de\u590d\uff1a\u597d\uff0c\u5e2e\u4f60\u91cd\u65b0\u89c4\u5212\u3002\n"
            "\u5982\u679c\u7528\u6237\u8bf4\u300c\u53d6\u6d88\u5bfc\u822a\u300d\u300c\u4e0d\u53bb\u4e86\u300d\uff0c\u56de\u590d\uff1a\u597d\u7684\uff0c\u5df2\u53d6\u6d88\u3002\n"
        )
        return base + nav_block
    else:
        return base + "\n当前未在导航中。可以帮用户搜索目的地、规划路线、查路况。"


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

    # ✅ 导航助手：profile="nav" 或 profile 以 "nav|" 开头（iOS 附带 JSON 上下文）
    if profile_norm == "nav" or profile_norm.startswith("nav|"):
        nav_ctx = None
        if "|" in requested_profile:
            _, json_part = requested_profile.split("|", 1)
            try:
                nav_ctx = json.loads(json_part)
            except Exception:
                nav_ctx = None
        return _nav_instructions(nav_ctx), "nav"

    # Force home assistant (debug/lab): export HOME_FORCE_ON=1
    if HOME_FORCE_ON:
        return _home_instructions_ios_protocol(req, client_id=client_id), "home"

    # explicit home
    if profile_norm in ("home", "homeassistant", "home_assistant", "ha"):
        return _home_instructions_ios_protocol(req, client_id=client_id), "home"

    # auto-home if bound
    if home_has_binding(req, client_id=client_id):
        log.info("[rt.pick] user client_id=%s -> HOME mode (auto-bound)", client_id)
        return _home_instructions_ios_protocol(req, client_id=client_id), "home"

    # default behavior
    log.info("[rt.pick] user client_id=%s profile=%s -> DEFAULT (ChatAGI-阿杜)", client_id, profile_norm)
    return _resolve_realtime_instructions(body), profile_norm


# ------------------------------------------------------------
# ✅ 统一记忆注入（facts + timeline + semantic + recent conversation）
# 文本聊天 / Realtime / OpenClaw 前统一调用此函数
# ------------------------------------------------------------

def build_unified_memory_context(
    user_key: str,
    *,
    query: str = "",
    conversation_id: str = "",
    include_facts: bool = True,
    include_timeline: bool = True,
    include_semantic: bool = True,
    include_recent: bool = True,
    facts_limit: int = MEMORY_FACTS_PROMPT_LIMIT,
    timeline_limit: int = 8,
    semantic_k: int = 5,
    recent_limit: int = 6,
) -> Dict[str, Any]:
    """
    统一记忆上下文构建器。返回结构化字典：
    {
        "facts_text": str,
        "timeline_text": str,
        "semantic_text": str,
        "recent_text": str,
        "full_prompt": str,       # 拼装好的完整注入文本
        "total_chars": int,
    }
    """
    uk = _sanitize_user_key(user_key)
    parts: Dict[str, str] = {
        "facts_text": "",
        "timeline_text": "",
        "semantic_text": "",
        "recent_text": "",
    }

    # 1. 长期事实记忆（最稳定的身份/偏好信息）
    if include_facts:
        try:
            fp = (memory_facts_build_prompt(uk, limit=facts_limit) or "").strip()
            if fp:
                parts["facts_text"] = fp
        except Exception:
            pass

    # 2. 时间线记忆（最近发生了什么，带时间戳）
    if include_timeline:
        try:
            tp = (memory_timeline_build_prompt(uk, limit=timeline_limit) or "").strip()
            if tp:
                parts["timeline_text"] = tp
        except Exception:
            pass

    # 3. 语义检索记忆（与当前 query 最相关的历史信息）
    if include_semantic and query:
        try:
            results = memory_search(uk, query, k=semantic_k, min_score=MEMORY_MIN_SCORE_DEFAULT)
            if results:
                items = [r.get("text", "") for r in results if r.get("text", "").strip()]
                if items:
                    bullet = "\n".join(f"- {it[:200]}" for it in items[:semantic_k])
                    parts["semantic_text"] = "相关记忆片段：\n" + bullet
        except Exception:
            pass

    # 4. 近期向量记忆（不依赖 query 的时间排序兜底）
    if include_recent:
        try:
            items = _mem_recent_items(uk, limit=recent_limit)
            if items:
                bullet = "\n".join(f"- {it[:200]}" for it in items[:recent_limit])
                parts["recent_text"] = "最近的记忆片段：\n" + bullet
        except Exception:
            pass

    # 5. 拼装完整 prompt（按优先级顺序）
    blocks: List[str] = []
    for key in ("facts_text", "timeline_text", "semantic_text", "recent_text"):
        v = parts[key]
        if v:
            blocks.append(v)

    full = ""
    if blocks:
        full = (
            "【长期记忆】\n"
            "以下信息来自你的长期记忆，用于个性化与延续上下文。\n"
            "如果与用户当前说法冲突，以用户当前说法为准。\n\n"
            + "\n\n".join(blocks)
        )

    # 🧠 注入阿杜当前意识（此刻的我）
    try:
        from adu_consciousness import consciousness
        consciousness_block = consciousness.to_prompt()
        if consciousness_block:
            full = consciousness_block + "\n\n" + full if full else consciousness_block
    except Exception:
        pass

    return {
        "facts_text": parts["facts_text"],
        "timeline_text": parts["timeline_text"],
        "semantic_text": parts["semantic_text"],
        "recent_text": parts["recent_text"],
        "full_prompt": full,
        "total_chars": len(full),
    }


# ------------------------------------------------------------
# ✅ Realtime Call Upgrades (Voice + Audio/Video)
# - Long-term memory prompt injection (read)
# - Memory commit (write) for legacy mode
# - WebRTC unified interface with server-side tool execution + memory auto-save
# ------------------------------------------------------------

REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"
REALTIME_SIDEBAND_WS_TPL = "wss://api.openai.com/v1/realtime?call_id={call_id}"

REALTIME_CALL_MEMORY_DEFAULT = _boolish(os.getenv("REALTIME_CALL_MEMORY_DEFAULT") or "1")
REALTIME_CALL_WEB_DEFAULT = _boolish(os.getenv("REALTIME_CALL_WEB_DEFAULT") or "1")
REALTIME_CALL_RECENT_MEMORY_ITEMS = int(os.getenv("REALTIME_CALL_RECENT_MEMORY_ITEMS") or "6")
REALTIME_CALL_MEMORY_MAX_CHARS = int(os.getenv("REALTIME_CALL_MEMORY_MAX_CHARS") or "2400")
REALTIME_CALL_TRANSCRIPT_MAX_CHARS = int(os.getenv("REALTIME_CALL_TRANSCRIPT_MAX_CHARS") or "6000")


def _mem_recent_items(user_key: str, limit: int = 6) -> List[str]:
    """Return most recently used vector memory items (best-effort)."""
    try:
        uk = _sanitize_user_key(user_key)
        lim = max(0, min(int(limit or 0), 20))
        if lim <= 0:
            return []
        with _mem_conn() as con:
            rows = con.execute(
                "SELECT text FROM memory_items WHERE user_key=? ORDER BY last_used_at DESC LIMIT ?",
                (uk, lim),
            ).fetchall()
        out: List[str] = []
        for r in rows or []:
            try:
                t = str(r[0] or "").strip()
                if t:
                    out.append(t)
            except Exception:
                continue
        return out
    except Exception:
        return []


def _build_realtime_memory_block(user_key: str) -> str:
    """Create a compact memory block using the unified memory context builder."""
    try:
        uk = _sanitize_user_key(user_key)
    except Exception:
        uk = "default"

    try:
        ctx = build_unified_memory_context(
            uk,
            include_facts=True,
            include_timeline=True,
            include_semantic=False,  # Realtime 没有 query，语义检索跳过
            include_recent=True,
            recent_limit=REALTIME_CALL_RECENT_MEMORY_ITEMS,
        )
        block = ctx.get("full_prompt") or ""
        log.info("[rt.memory] user=%s unified_context=%d chars (facts=%d, timeline=%d, recent=%d)",
                 uk, ctx.get("total_chars", 0),
                 len(ctx.get("facts_text") or ""),
                 len(ctx.get("timeline_text") or ""),
                 len(ctx.get("recent_text") or ""))
    except Exception as _e:
        log.warning("[rt.memory] user=%s unified builder error: %s", uk, _e)
        block = ""

    if not block:
        return ""

    if len(block) > REALTIME_CALL_MEMORY_MAX_CHARS:
        block = block[:REALTIME_CALL_MEMORY_MAX_CHARS].rstrip() + "\n"
    return "\n\n" + block + "\n"


def _augment_realtime_instructions(
    *,
    base_instructions: Optional[str],
    user_key: str,
    enable_memory: bool,
    enable_web: bool,
    perception_data: Optional[Dict[str, Any]] = None,
    enable_agent: bool = True,
    provider: str = "openai",
) -> Optional[str]:
    base = (base_instructions or "").strip()
    blocks: List[str] = []
    if base:
        # ✅ 语义打断魔法字符串协议（__SEMANTIC_INTERRUPT__{...JSON...}）只对 OpenAI Realtime 有效。
        # OpenAI 是文本驱动 TTS：模型先输出文本流，TTS 跟着文本念，前端能拦截魔法字符串
        # 让它既不显示也不发声。
        # Qwen3-Omni / Qwen3.5-Omni 是端到端语音模型——Talker（语音生成器）和 Thinker
        # （文字生成器）共享同一个模型，文本和音频并行生成、同步对齐。注入这个协议后，
        # 模型会把控制符当作正文一起合成语音念出来（"reason / user_correction /
        # replace_current_turn"等英文单词），ASR 转写还可能丢失下划线导致前端 token 探测
        # 失效，UI 上就会出现 JSON 残留。这就是 iOS 截图里看到的现象。
        # Qwen 的打断完全靠 server_vad → speech_started → 客户端 cancel + userIsSpeaking gate
        # 实现，不需要模型自己输出魔法字符串。Pipeline 路径（DeepSeek 文本→TTS）也不依赖此协议，
        # 它走客户端 VAD + 直接断 TTS 队列。
        if provider == "openai":
            base = _ensure_semantic_interrupt_protocol(base) or base
        blocks.append(base)

    # ── 0. 🧠 注入阿杜当前意识（此刻的我）──
    try:
        from adu_consciousness import consciousness
        consciousness_block = consciousness.to_prompt()
        if consciousness_block:
            blocks.insert(0, consciousness_block)
            log.info("[rt.augment] 🧠 consciousness injected (%d chars)", len(consciousness_block))
    except Exception as _ce:
        log.debug("[rt.augment] consciousness skip: %s", _ce)

    log.info("[rt.augment] user=%s enable_memory=%s enable_web=%s enable_agent=%s perception=%s base_instructions_preview='%s'",
             user_key, enable_memory, enable_web, enable_agent,
             bool(perception_data), (base or "(none)")[:120])

    # ── 1. 长期记忆（与文本通道共享同一套 build_unified_memory_context）──
    if enable_memory and REALTIME_CALL_MEMORY_DEFAULT:
        mb = _build_realtime_memory_block(user_key)
        if mb:
            blocks.append(mb.strip())
            log.info("[rt.augment] ✅ memory block injected (%d chars)", len(mb))
        else:
            log.info("[rt.augment] ⚠️ memory enabled but block is empty")
    else:
        log.info("[rt.augment] ❌ memory NOT injected (enable_memory=%s, DEFAULT=%s)", enable_memory, REALTIME_CALL_MEMORY_DEFAULT)

    # ── 2. 感知上下文（物理世界：GPS/朝向/运动/地址）──
    if isinstance(perception_data, dict) and perception_data.get("latitude"):
        try:
            from perception_module import format_perception_context, store_to_world_memory
            # ✅ Key 兼容：iOS JSONEncoder(.convertToSnakeCase) 发蛇形 key，后端/perception_module 用驼峰
            _snake_to_camel = {
                "full_address": "fullAddress",
                "street_number": "streetNumber",
                "horizontal_accuracy": "horizontalAccuracy",
                "heading_description": "headingDescription",
                "motion_state": "motionState",
                "scene_description": "sceneDescription",
                "geocode_source": "geocodeSource",
                "poi_name": "poiName",
            }
            for sk, ck in _snake_to_camel.items():
                if sk in perception_data and ck not in perception_data:
                    perception_data[ck] = perception_data[sk]
            # enrich 已在 /session 层完成，这里直接 format
            _perc_ctx = format_perception_context(perception_data)
            if _perc_ctx:
                blocks.append(_perc_ctx.strip())
                log.info("[rt.augment] ✅ perception context injected: %s",
                         perception_data.get("fullAddress") or f"{perception_data.get('latitude'):.4f}")
            # 存入世界记忆
            store_to_world_memory(user_key or "anonymous", perception_data)
        except Exception as _pe:
            log.warning("[rt.augment] ⚠️ perception injection failed: %s", _pe)

    # ── 3. 电脑执行能力（/api/computer 走 usecomputer，主助手永远可用）──
    _computer_block = (
        "【电脑执行能力】\n"
        "你可以调用 computer 工具控制用户电脑。\n"
        "用户给你的是目标，不是步骤。你要自己理解目标，自己生成 computer(action,args)，并根据结果继续推进。\n"
        "当用户要求打开应用、查看文件、列目录、运行命令、截图、点击、输入文字、按快捷键时，必须调用 computer 工具，不要回答\"你自己去终端执行\"。\n"
        "\n"
        "computer 可用 action：\n"
        "- device_info：查看系统和能力\n"
        "- shell：执行终端命令，例如 open -a WeChat、ls ~/Desktop\n"
        "- screenshot：获取当前屏幕截图\n"
        "- mouse_position：获取鼠标坐标\n"
        "- click / double_click：点击屏幕坐标\n"
        "- hotkey / press：按快捷键，例如 cmd+space、cmd+v、enter\n"
        "- type_text：输入文字\n"
        "- wait：等待\n"
        "- window_list / display_list / desktop_list：查看窗口、显示器、桌面信息\n"
        "\n"
        "工作规则：\n"
        "1. 如果只是聊天，正常回答。\n"
        "2. 如果需要操作电脑，调用 computer。\n"
        "3. 终端任务优先用 shell。\n"
        "4. GUI 任务先 screenshot 或 mouse_position 观察，再 click/hotkey/type_text。\n"
        "5. 每执行一步后，根据 stdout/stderr 或截图判断下一步。\n"
        "6. 不要让用户一步步教你，除非涉及删除、sudo、付款、发送隐私、不可逆操作。\n"
        "7. 危险操作必须先问用户确认。\n"
        "\n"
        "跨平台规则：\n"
        "如果不确定系统，先调用 device_info。\n"
        "macOS 使用 open、osascript、pbcopy、pbpaste、zsh。\n"
        "Windows 使用 PowerShell、Start-Process、Set-Clipboard、Get-Clipboard。\n"
        "Linux 使用 bash、xdg-open、xclip/wl-copy、gnome-screenshot/scrot。\n"
        "不要把 macOS 命令发给 Windows，不要把 Windows 命令发给 macOS。\n"
        "\n"
        "微信发送特例：当用户说\"给XX发微信/发消息\"时，可以直接调用 adu_mac_send(contact, message) 走 AppleScript 封装，比 computer+shell 更稳。"
    )
    blocks.append(_computer_block)
    log.info("[rt.augment] ✅ computer tool instructions injected (%d chars)", len(_computer_block))

    # ── 4. 联网搜索 ──
    if enable_web and REALTIME_CALL_WEB_DEFAULT:
        blocks.append(
            "【联网功能】\n"
            "你可以调用工具 web_search 来获取最新信息/新闻/数据。\n"
            "优先在用户明确需要最新信息、需要引用来源或需要事实核对时使用。\n"
            "返回时尽量给出来源标题与链接。"
        )
        log.info("[rt.augment] ✅ web_search instructions injected")
    else:
        log.info("[rt.augment] ❌ web_search NOT injected (enable_web=%s, DEFAULT=%s)", enable_web, REALTIME_CALL_WEB_DEFAULT)

    out = "\n\n".join([b for b in blocks if b and b.strip()]).strip()
    log.info("[rt.augment] final instructions length=%d chars", len(out) if out else 0)
    return out or None


def _realtime_tool_web_search_def() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "web_search",
        "description": "Search the public web for up-to-date information. Use when the user asks for latest/current facts, news, prices, policies, or needs source links.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Max results (1-10)", "default": 6},
                "kind": {"type": "string", "description": "Optional category hint (e.g. news, general).", "default": ""},
            },
            "required": ["query"],
        },
    }


def _realtime_tool_remember_def() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "remember",
        "description": "Save an explicit long-term memory about the user (preferences, profile, constraints). Use only when the user asks to remember something or shares stable personal info.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Memory to store"},
                "importance": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
                "tags": {"type": "string", "description": "Optional tags", "default": ""},
            },
            "required": ["text"],
        },
    }


def _realtime_tool_agent_exec_def() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "adu_agent_exec",
        "description": "Execute device operations via Adu-Agent (OpenClaw). Use when the user asks to open apps, manage files, run terminal commands, control devices, compile projects, browse web, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Action to perform (Chinese or English), e.g. '打开备忘录', '查看桌面文件', 'compile GPTsora'"},
            },
            "required": ["action"],
        },
    }


def _realtime_tool_mac_send_def() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "adu_mac_send",
        "description": "发送消息给微信联系人或群。当用户说给XX发微信、发消息给XX时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "contact": {"type": "string", "description": "联系人名字，如联系人、文件传输助手"},
                "message": {"type": "string", "description": "要发送的消息内容"},
            },
            "required": ["contact", "message"],
        },
    }


# ✅ Direct Computer Tool — 走 /api/computer，不依赖 OpenClaw
def _realtime_tool_computer_def() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "computer",
        "description": (
            "控制用户电脑的统一工具。"
            "当用户要求打开应用、查看文件、执行终端命令、截图、点击、输入文字、按快捷键时调用。"
            "不要告诉用户自己去终端执行；你要直接调用这个工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "device_info",
                        "shell",
                        "screenshot",
                        "mouse_position",
                        "click",
                        "double_click",
                        "hotkey",
                        "press",
                        "type_text",
                        "wait",
                        "window_list",
                        "display_list",
                        "desktop_list",
                    ],
                    "description": "电脑动作类型。终端命令用 shell，截图用 screenshot。",
                },
                "args": {
                    "type": "object",
                    "description": (
                        "动作参数。"
                        "shell 用 {cmd, timeout_sec}；"
                        "click 用 {x,y}；"
                        "hotkey/press 用 {key} 或 {keys}；"
                        "type_text 用 {text}；"
                        "screenshot 可为空对象。"
                    ),
                },
            },
            "required": ["action", "args"],
        },
    }


def _realtime_tools(*, enable_web: bool, enable_memory: bool, enable_agent: bool = True) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if enable_web:
        tools.append(_realtime_tool_web_search_def())
    if enable_memory:
        tools.append(_realtime_tool_remember_def())
    # ✅ 新电脑工具：主助手永远可用，不依赖 OpenClaw 连接状态
    tools.append(_realtime_tool_computer_def())
    # 旧微信专用工具先保留兼容
    tools.append(_realtime_tool_mac_send_def())
    return tools


def _realtime_webrtc_session_config(
    *,
    model: str,
    voice: str,
    instructions: Optional[str],
    enable_web: bool,
    enable_memory: bool,
) -> Dict[str, Any]:
    """Session config used for POST /v1/realtime/calls (WebRTC unified interface)."""
    sess: Dict[str, Any] = {
        "type": "realtime",
        "model": (model or REALTIME_MODEL_DEFAULT).strip() or REALTIME_MODEL_DEFAULT,
        "audio": {
            "output": {"voice": (voice or REALTIME_VOICE_DEFAULT).strip() or REALTIME_VOICE_DEFAULT},
            "input": {
                # Best-effort transcription guidance (not what model hears)
                "transcription": {"model": TRANSCRIPTION_MODEL_DEFAULT},
                # Server VAD is the most robust for WebRTC
                "turn_detection": {"type": "server_vad"},
            },
        },
    }
    if instructions:
        sess["instructions"] = instructions

    tools = _realtime_tools(enable_web=enable_web, enable_memory=enable_memory)
    if tools:
        sess["tools"] = tools
        sess["tool_choice"] = "auto"

    return sess


def _parse_realtime_call_id(location: str) -> str:
    """Location header is usually like /v1/realtime/calls/rtc_xxx. Return rtc_xxx."""
    loc = (location or "").strip()
    if not loc:
        return ""
    # allow full URL too
    try:
        if "://" in loc:
            p = urlparse(loc)
            loc = p.path or loc
    except Exception:
        pass
    # strip trailing slashes
    loc = loc.rstrip("/")
    # rtc_... is last segment
    seg = loc.split("/")[-1] if "/" in loc else loc
    if seg.startswith("rtc_") or seg.startswith("call_") or seg.startswith("sess_"):
        return seg
    # fallback: return last segment
    return seg


# -----------------------------
# ✅ WebRTC sideband controller: execute tools + auto save memory
# -----------------------------
_REALTIME_CONTROLLERS: Dict[str, "RealtimeSidebandController"] = {}
_REALTIME_CONTROLLERS_LOCK = threading.Lock()


def _realtime_controller_start(
    *,
    call_id: str,
    user_key: str,
    profile: str,
    enable_web: bool,
    enable_memory: bool,
    instructions: Optional[str],
    conversation_id: str = "",
) -> None:
    """Start (or reuse) a sideband controller for a WebRTC call."""
    cid = (call_id or "").strip()
    if not cid:
        return
    if not OPENAI_API_KEY:
        return

    if websockets is None:
        log.warning("[rt.sideband] websockets not installed; auto tools/memory disabled for call=%s", cid)
        return

    with _REALTIME_CONTROLLERS_LOCK:
        if cid in _REALTIME_CONTROLLERS:
            return
        ctrl = RealtimeSidebandController(
            call_id=cid,
            user_key=user_key,
            profile=profile,
            enable_web=enable_web,
            enable_memory=enable_memory,
            instructions=instructions,
            conversation_id=conversation_id,
        )
        _REALTIME_CONTROLLERS[cid] = ctrl

    try:
        asyncio.create_task(ctrl.run())
    except Exception:
        # If we're not in an event loop context (rare), fallback to a thread
        def _runner():
            asyncio.run(ctrl.run())
        t = threading.Thread(target=_runner, daemon=True)
        t.start()


def _realtime_controller_stop(call_id: str) -> bool:
    cid = (call_id or "").strip()
    if not cid:
        return False
    with _REALTIME_CONTROLLERS_LOCK:
        ctrl = _REALTIME_CONTROLLERS.get(cid)
    if not ctrl:
        return False
    try:
        asyncio.create_task(ctrl.stop())
        return True
    except Exception:
        try:
            ctrl.request_stop()
            return True
        except Exception:
            return False


def _extract_realtime_item_text(item: Dict[str, Any]) -> str:
    """Extract readable text from a Realtime conversation message item."""
    try:
        content = item.get("content") or []
        texts: List[str] = []
        if isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                t = ""
                ptype = str(p.get("type") or "").strip()
                if ptype in ("text", "input_text"):
                    t = str(p.get("text") or "").strip()
                elif ptype in ("audio", "input_audio"):
                    t = str(p.get("transcript") or p.get("text") or "").strip()
                else:
                    t = str(p.get("transcript") or p.get("text") or "").strip()
                if t:
                    texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""


class RealtimeSidebandController:
    """
    Server-side sideband connection for WebRTC Realtime calls.

    Responsibilities:
      - Execute Realtime function tools (web_search / remember / openclaw)
      - Collect transcripts and persist to memory on call end
      - ✅ Sync transcripts to conversation history (conv_add_message)
      - ✅ Feed transcripts into 4-level compression pipeline (L0→L1→L2→L3)
      - ✅ Execute OpenClaw/Adu-Agent actions from voice commands

    This is *best-effort* and must never break the call.
    """

    def __init__(
        self,
        *,
        call_id: str,
        user_key: str,
        profile: str,
        enable_web: bool,
        enable_memory: bool,
        instructions: Optional[str],
        conversation_id: str = "",
    ):
        self.call_id = (call_id or "").strip()
        self.user_key = _sanitize_user_key(user_key)
        self.profile = (profile or "default").strip()
        self.enable_web = bool(enable_web)
        self.enable_memory = bool(enable_memory)
        self.instructions = (instructions or "").strip() or None

        self.started_at = time.time()
        self._stop_evt = asyncio.Event()
        self._ws = None

        self.turns: List[Dict[str, str]] = []
        self._tool_seen: set[str] = set()
        self._tool_args_buf: Dict[str, str] = {}

        # ✅ 会话历史同步：优先复用前端传来的 conversation_id，否则创建新的
        cid_in = (conversation_id or "").strip()
        if cid_in:
            self.conv_id = cid_in
            try:
                conv_touch(self.user_key, cid_in, "")
            except Exception:
                pass
            log.info("[rt.sideband] call=%s reusing conv_id=%s", self.call_id, self.conv_id)
        else:
            try:
                self.conv_id: str = conv_create(self.user_key, f"语音通话 {time.strftime('%m/%d %H:%M')}")
                log.info("[rt.sideband] call=%s conv_id=%s created", self.call_id, self.conv_id)
            except Exception as _e:
                self.conv_id = ""
                log.warning("[rt.sideband] conv_create failed: %s", _e)

        # 追踪最近一对 user/assistant 文本，用于 L3 结构化提取
        self._last_user_text: str = ""
        self._last_ai_text: str = ""

    def request_stop(self) -> None:
        try:
            self._stop_evt.set()
        except Exception:
            pass

    async def stop(self) -> None:
        self.request_stop()
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            pass

    async def _send(self, obj: Dict[str, Any]) -> None:
        if not self._ws:
            return
        try:
            await self._ws.send(json.dumps(obj))
        except Exception:
            pass

    async def _tool_web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        q = str(args.get("query") or args.get("q") or "").strip()
        if not q:
            return {"ok": False, "error": "missing query", "results": []}
        return {"ok": True, "executed": False, "query": q, "results": [], "provider": "openai_builtin_web_search", "message": "Use /chat with allow_web=true for OpenAI built-in web_search."}

    async def _tool_remember(self, args: Dict[str, Any]) -> Dict[str, Any]:
        txt = str(args.get("text") or "").strip()
        if not txt:
            return {"ok": False, "error": "missing text"}
        tags = str(args.get("tags") or "").strip()
        try:
            imp = int(args.get("importance") or 3)
        except Exception:
            imp = 3
        imp = max(1, min(imp, 5))

        def _do():
            # Store as vector memory
            if _should_memory_add(txt):
                memory_add(self.user_key, _short(txt, 800))
            # Store as structured fact when enabled
            try:
                if MEMORY_FACTS_ENABLED_DEFAULT and imp >= MEMORY_FACTS_IMPORTANCE_MIN:
                    memory_facts_save(self.user_key, _short(txt, 400), tags=tags, importance=imp)
            except Exception:
                pass

        await asyncio.to_thread(_do)
        return {"ok": True, "stored": True}

    async def _tool_computer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ Direct Computer Tool — 转发到本地 /api/computer。
        不走 OpenClaw，不依赖任何 bridge。
        危险 shell 命令会被前置拦截，要求用户确认。
        """
        action = str(args.get("action") or "").strip()
        action_args = args.get("args") or {}
        if not action:
            return {"ok": False, "error": "missing action"}
        if not isinstance(action_args, dict):
            action_args = {}

        # 轻量安全拦截：危险 shell 不直接执行
        if action in ("shell", "bash", "exec"):
            cmd = str(action_args.get("cmd") or action_args.get("command") or "").strip()
            lowered = cmd.lower()
            danger_hits = [
                "rm -rf",
                "sudo ",
                "mkfs",
                "diskutil erase",
                ":(){",
                "shutdown",
                "reboot",
            ]
            if any(x in lowered for x in danger_hits):
                log.warning("[ComputerTool] BLOCKED dangerous cmd: %s", cmd[:200])
                return {
                    "ok": False,
                    "error": "dangerous_command_requires_user_confirmation",
                    "cmd": cmd,
                }

        try:
            url = os.getenv("LOCAL_COMPUTER_API_URL", "http://127.0.0.1:8000/api/computer")
            payload = {
                "action": action,
                "args": action_args,
            }

            def _do():
                r = requests.post(url, json=payload, timeout=60)
                try:
                    return r.json()
                except Exception:
                    return {
                        "ok": False,
                        "status_code": r.status_code,
                        "text": r.text[:1000],
                    }

            result = await asyncio.to_thread(_do)
            log.info("[ComputerTool] action=%s ok=%s", action, (result or {}).get("ok"))
            return result if isinstance(result, dict) else {"ok": True, "result": result}
        except Exception as e:
            log.warning("[ComputerTool] forward failed: %s", e)
            return {"ok": False, "error": str(e)[:500]}

    async def _tool_openclaw(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ OpenClaw/Adu-Agent 工具调用 — 语音通话中执行文件操作/终端命令/浏览器动作等。

        支持两种触发方式:
        1. Realtime function tool: name=openclaw_exec / adu_agent_exec, args={action, ...}
        2. AI 回复中嵌入 [ADU_ACTION:...] 标记（由 _sync_openclaw_from_transcript 在通话结束时批量处理）

        此方法处理方式 1（实时工具调用）。
        """
        try:
            if not _openclaw_runtime_enabled():
                return {"ok": False, "error": "openclaw_disabled_local_agent_only"}

            _bridge = _openclaw_get_bridge_or_none()
            if _bridge is None or not getattr(_bridge, "connected", False):
                # 尝试重连一次
                try:
                    await asyncio.to_thread(ensure_connected)
                    _bridge = _openclaw_get_bridge_or_none()
                except Exception:
                    pass
            if _bridge is None or not getattr(_bridge, "connected", False):
                return {"ok": False, "error": "openclaw_not_connected"}

            action = str(args.get("action") or args.get("command") or "").strip()
            if not action:
                return {"ok": False, "error": "missing action"}

            # 构造 [ADU_ACTION:...] 格式让 process_agent_actions_sync 统一处理
            action_text = f"[ADU_ACTION:{action}]"
            # 如果有额外参数，附加为 JSON
            extra = {k: v for k, v in args.items() if k not in ("action", "command")}
            if extra:
                action_text = f"[ADU_ACTION:{action} {json.dumps(extra, ensure_ascii=False)}]"

            def _do():
                _clean, _results = process_agent_actions_sync(action_text, _bridge)
                return _results

            result_text = await asyncio.to_thread(_do)
            return {"ok": True, "result": (result_text or "").strip() or "done"}
        except Exception as e:
            log.warning("[rt.sideband] openclaw tool failed: %s", e)
            return {"ok": False, "error": str(e)[:200]}

    async def _handle_tool_call(self, *, call_id: str, name: str, arguments: str) -> None:
        cid = (call_id or "").strip()
        if not cid or cid in self._tool_seen:
            return
        self._tool_seen.add(cid)

        args: Dict[str, Any] = {}
        try:
            args = json.loads(arguments or "{}") if (arguments or "").strip() else {}
        except Exception:
            args = {}

        out: Dict[str, Any] = {"ok": False, "error": "tool_not_supported"}
        if name == "web_search" and self.enable_web:
            out = await self._tool_web_search(args)
        elif name == "remember" and self.enable_memory:
            out = await self._tool_remember(args)
        elif name == "computer":
            out = await self._tool_computer(args)
        # ✅ OpenClaw 接入：语音通话中也能执行文件操作/终端命令等 Agent 动作
        elif name.startswith("openclaw_") or name.startswith("adu_agent_"):
            out = await self._tool_openclaw(name, args)
        elif name == "adu_mac_send":
            log.info("=" * 60)
            log.info("🎯 [adu_mac_send] CALLED! contact=%s message=%s", args.get("contact"), args.get("message"))
            log.info("=" * 60)
            try:
                import sys
                sys.path.insert(0, '/Users/a12345/Desktop/backend')
                from adu_mac_tools import send_wechat
                contact = args.get("contact", "")
                message = args.get("message", "")
                result = send_wechat(contact, message)
                log.info("🎯 [adu_mac_send] send_wechat returned: %s", result)
                out = {"ok": True, "result": f"已发送给{contact}"}
            except Exception as e:
                log.error("🎯 [adu_mac_send] FAILED: %s", e, exc_info=True)
                out = {"ok": False, "error": str(e)}

        # Send tool output back to the call
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": cid,
                    "output": json.dumps(out, ensure_ascii=False),
                },
            }
        )
        # Trigger the model to continue (WebRTC clients often don't implement tool loop)
        await self._send({"type": "response.create"})

    async def _handle_event(self, ev: Dict[str, Any]) -> None:
        et = str(ev.get("type") or "").strip()

        # Tool arguments streaming (optional)
        if et == "response.function_call_arguments.delta":
            cid = str(ev.get("call_id") or "").strip()
            if cid:
                self._tool_args_buf[cid] = (self._tool_args_buf.get(cid) or "") + str(ev.get("delta") or "")
            return

        if et == "response.function_call_arguments.done":
            cid = str(ev.get("call_id") or "").strip()
            name = str(ev.get("name") or "").strip()
            arguments = str(ev.get("arguments") or (self._tool_args_buf.get(cid) or "")).strip()
            log.info("🛠️ [tool_call] GPT 调用工具: name=%s args_preview=%s", name, arguments[:200])
            if cid and name:
                await self._handle_tool_call(call_id=cid, name=name, arguments=arguments)
            return

        # Conversation items
        if et in ("conversation.item.done", "conversation.item.created", "conversation.item.added"):
            item = ev.get("item") or {}
            if not isinstance(item, dict):
                return

            itype = str(item.get("type") or "").strip()
            if itype == "message":
                role = str(item.get("role") or "").strip()
                txt = _extract_realtime_item_text(item)
                if txt:
                    self.turns.append({"role": role, "text": txt})

                    # ✅ 统一落库：conv_add_message 同时写 messages + timeline
                    if self.conv_id:
                        try:
                            conv_add_message(self.user_key, self.conv_id, role, txt, source="realtime")
                        except Exception:
                            pass

                    # ✅ 四级压缩：写入 L0 原始轮次 + 触发压缩流水线
                    if self.conv_id:
                        try:
                            _comp_append_turn(self.user_key, self.conv_id, role, txt[:800])
                            _comp_maybe_compress(self.user_key, self.conv_id)
                        except Exception:
                            pass

                    # 追踪最近的 user/assistant 文本对
                    if role == "user":
                        self._last_user_text = txt
                    elif role == "assistant":
                        self._last_ai_text = txt
                        # ✅ L3 结构化提取：每个 assistant 回复配对 user 做提取
                        if self._last_user_text:
                            try:
                                _comp_l3_extract(self.user_key, self._last_user_text, txt)
                            except Exception:
                                pass
                        # ✅ 长期事实抽取：每轮都触发（语音链路长出长期记忆）
                        if self._last_user_text:
                            try:
                                extract_and_save_memory_facts(self.user_key, self._last_user_text, txt)
                            except Exception:
                                pass
            elif itype == "function_call":
                cid = str(item.get("call_id") or "").strip()
                name = str(item.get("name") or "").strip()
                arguments = str(item.get("arguments") or "").strip()
                if cid and name:
                    await self._handle_tool_call(call_id=cid, name=name, arguments=arguments)

    async def _persist_memory_on_end(self) -> None:
        if not self.enable_memory:
            return
        if not self.turns:
            return

        # Build a compact transcript
        lines: List[str] = []
        for t in self.turns[-60:]:
            role = t.get("role") or ""
            text = (t.get("text") or "").strip()
            if not text:
                continue
            who = "用户" if role == "user" else ("助手" if role == "assistant" else str(role))
            lines.append(f"{who}: {text}")
        transcript = "\n".join(lines).strip()
        if not transcript:
            return
        if len(transcript) > REALTIME_CALL_TRANSCRIPT_MAX_CHARS:
            transcript = transcript[-REALTIME_CALL_TRANSCRIPT_MAX_CHARS:]

        # Best-effort store a call note + extract structured facts
        def _do():
            note = f"通话记录/摘要：{_short(transcript, 1200)}"
            if _should_memory_add(note):
                memory_add(self.user_key, note)

            # Store user turns that are memory-worthy
            for t in self.turns:
                if (t.get("role") or "") != "user":
                    continue
                ut = (t.get("text") or "").strip()
                if ut and _should_memory_add(ut):
                    memory_add(self.user_key, _short(ut, 600))

            # Structured facts from last pair (cheap)
            last_user = ""
            last_ai = ""
            for t in self.turns:
                if (t.get("role") or "") == "user" and (t.get("text") or "").strip():
                    last_user = (t.get("text") or "").strip()
                if (t.get("role") or "") == "assistant" and (t.get("text") or "").strip():
                    last_ai = (t.get("text") or "").strip()
            if last_user or last_ai:
                extract_and_save_memory_facts(self.user_key, last_user, last_ai)

            # ✅ OpenClaw: 批量处理通话中 assistant 回复里的 [ADU_ACTION:...] 标记
            try:
                _bridge = _openclaw_get_bridge_or_none()
                if _bridge is not None and getattr(_bridge, "connected", False):
                    for t in self.turns:
                        if (t.get("role") or "") != "assistant":
                            continue
                        at = (t.get("text") or "").strip()
                        if at and "[ADU_ACTION:" in at:
                            try:
                                process_agent_actions_sync(at, _bridge)
                            except Exception:
                                pass
            except Exception:
                pass

            # ✅ 四级压缩：通话结束时最终 flush 一次压缩流水线
            if self.conv_id:
                try:
                    _comp_maybe_compress(self.user_key, self.conv_id)
                except Exception:
                    pass

            # ✅ 时间线记忆：记录本次通话的总结事件
            try:
                n_user = sum(1 for t in self.turns if (t.get("role") or "") == "user")
                n_ai = sum(1 for t in self.turns if (t.get("role") or "") == "assistant")
                dur = int(time.time() - self.started_at)
                dur_str = f"{dur // 60}分{dur % 60}秒" if dur >= 60 else f"{dur}秒"
                summary = f"语音通话 ({dur_str}, 用户{n_user}轮, 助手{n_ai}轮)"
                if last_user:
                    summary += f" 最后话题: {last_user[:80]}"
                memory_timeline_add(
                    self.user_key, summary,
                    event_type="voice_call",
                    detail=_short(transcript, 800),
                    source=self.call_id,
                    ts=self.started_at,
                )
            except Exception:
                pass

        await asyncio.to_thread(_do)

    async def run(self) -> None:
        if not self.call_id:
            return
        url = REALTIME_SIDEBAND_WS_TPL.format(call_id=self.call_id)

        headers = [
            ("Authorization", f"Bearer {OPENAI_API_KEY}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]

        try:
            async with websockets.connect(url, extra_headers=headers, ping_interval=20, ping_timeout=20) as ws:
                self._ws = ws

                # Ensure tools are available (some clients might start with minimal config)
                sess_update: Dict[str, Any] = {}
                if self.instructions:
                    sess_update["instructions"] = self.instructions
                tools = _realtime_tools(enable_web=self.enable_web, enable_memory=self.enable_memory)
                if tools:
                    sess_update["tools"] = tools
                    sess_update["tool_choice"] = "auto"
                if sess_update:
                    await self._send({"type": "session.update", "session": sess_update})

                async for raw in ws:
                    if self._stop_evt.is_set():
                        break
                    try:
                        ev = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(ev, dict):
                        await self._handle_event(ev)
        except Exception as e:
            log.info("[rt.sideband] call=%s ended (%s)", self.call_id, e)
        finally:
            self._ws = None
            try:
                await self._persist_memory_on_end()
            except Exception:
                pass
            with _REALTIME_CONTROLLERS_LOCK:
                if self.call_id in _REALTIME_CONTROLLERS:
                    _REALTIME_CONTROLLERS.pop(self.call_id, None)

@app.post("/session")
async def session_post(req: Request):
    """
    Realtime session bootstrap.

    ✅ Backward compatible:
      - JSON (application/json): returns ephemeral_key for legacy client-side connections (existing behavior)
      - SDP  (application/sdp / text/plain): WebRTC unified interface -> server creates Realtime call and returns SDP answer

    ✅ Upgrades:
      - Realtime voice + audio/video (WebRTC) share the same Long-Term Memory prompt injection
      - Web Search (联网功能) is exposed as a Realtime function tool (`web_search`)
      - Optional server-side controls (sideband) for WebRTC calls to execute tools + save memories automatically
    """

    ct = (req.headers.get("content-type") or "").lower()

    # ------------------------------------------------------------
    # ✅ New: WebRTC unified interface (audio/video call)
    # Client sends SDP offer as raw body. Server returns SDP answer.
    # ------------------------------------------------------------
    if ("application/sdp" in ct) or ("text/plain" in ct):
        offer_sdp = (await req.body()).decode("utf-8", errors="ignore").strip()
        if not offer_sdp:
            return _friendly_session_text_error(message=REALTIME_USER_ERROR_MESSAGE, status_code=200)

        # Query params / headers for options (keep it simple for clients)
        qp = req.query_params
        model = (qp.get("model") or req.headers.get("x-realtime-model") or REALTIME_MODEL_DEFAULT).strip()
        voice = (qp.get("voice") or req.headers.get("x-realtime-voice") or REALTIME_VOICE_DEFAULT).strip()

        # Build a body-like dict so we can reuse existing helpers
        b: Dict[str, Any] = {
            "model": model,
            "voice": voice,
            "profile": (qp.get("profile") or qp.get("mode") or req.headers.get("x-realtime-profile") or "default").strip(),
        }
        # web/memory toggles (defaults: ON for WebRTC flow)
        if "allow_web" in qp or "allowWeb" in qp or "web" in qp or "enable_web_search" in qp:
            b["allow_web"] = qp.get("allow_web") or qp.get("allowWeb") or qp.get("web") or qp.get("enable_web_search")
        else:
            b["allow_web"] = True  # ✅ WebRTC 默认开启联网（sideband会执行工具，不依赖客户端）

        if "enable_memory" in qp or "memory" in qp:
            b["enable_memory"] = qp.get("enable_memory") or qp.get("memory")
        else:
            b["enable_memory"] = True  # ✅ WebRTC 默认开启长期记忆

        user_key = _derive_user_key(req, b)
        # ✅ Billing: default soft mode for realtime bootstrap.
        # Do not let raw HTTP 402 break Qwen/OpenAI voice/video before provider session creation.
        _billing_response = _realtime_billing_allows_or_response(req, b, user_key, branch="SDP")
        if _billing_response is not None:
            return _billing_response
        allow_web = bool(_extract_allow_web(req, b))  # uses existing flag parsing
        enable_memory = _boolish(b.get("enable_memory"))
        # ✅ 提取 conversation_id（与文本聊天共用同一个会话）
        rt_conversation_id = (qp.get("conversation_id") or req.headers.get("x-conversation-id") or "").strip()

        instructions, resolved_profile = _pick_realtime_instructions(req, b)
        instructions = _augment_realtime_instructions(
            base_instructions=instructions,
            user_key=user_key,
            enable_memory=enable_memory,
            enable_web=allow_web,
            perception_data=None,  # SDP 分支不携带 perception（通过 query params 无法传复杂 JSON）
            enable_agent=True,
        )

        # Build session config for /v1/realtime/calls (multipart form)
        session_cfg = _realtime_webrtc_session_config(
            model=model,
            voice=voice,
            instructions=instructions,
            enable_web=allow_web,
            enable_memory=enable_memory,
        )

        if not OPENAI_API_KEY:
            log.warning("[/session:SDP] OPENAI_API_KEY missing")
            return _friendly_session_text_error(status_code=200)

        try:
            files = {
                "sdp": ("offer.sdp", offer_sdp, "application/sdp"),
                "session": (None, json.dumps(session_cfg), "application/json"),
            }
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            }
            r = requests.post(REALTIME_CALLS_URL, headers=headers, files=files, timeout=30)
        except Exception as e:
            log.warning("[/session:SDP] realtime call create failed: %s", e)
            return _friendly_session_text_error(status_code=200)

        if r.status_code >= 400:
            log.warning("[/session:SDP] provider error status=%s body=%s", r.status_code, _short(r.text, 500))
            return _friendly_session_text_error(status_code=200)

        answer_sdp = (r.text or "").strip()
        loc = (r.headers.get("Location") or r.headers.get("location") or "").strip()
        call_id = _parse_realtime_call_id(loc)
        if not call_id:
            # Some proxies may drop Location; try alternate headers.
            for hk in (
                "x-openai-realtime-call-id",
                "openai-realtime-call-id",
                "x-realtime-call-id",
                "x-call-id",
            ):
                hv = (r.headers.get(hk) or r.headers.get(hk.title()) or "").strip()
                if hv:
                    call_id = _parse_realtime_call_id(hv)
                    if call_id:
                        break

        # ✅ Auto-attach sideband controller (exec tools + store memory)
        if call_id:
            _realtime_controller_start(
                call_id=call_id,
                user_key=user_key,
                profile=resolved_profile,
                enable_web=allow_web,
                enable_memory=enable_memory,
                instructions=instructions,
                conversation_id=rt_conversation_id,
            )

        headers_out: Dict[str, str] = {"X-Resolved-Profile": str(resolved_profile or "")}
        auto_mem = bool(call_id and enable_memory and (websockets is not None))
        headers_out["X-Auto-Memory"] = "1" if auto_mem else "0"
        headers_out["X-Memory-Commit-Required"] = "0" if auto_mem else ("1" if enable_memory else "0")

        if loc:
            headers_out["Location"] = loc
        if call_id:
            headers_out["X-OpenAI-Realtime-Call-Id"] = call_id

        return Response(content=answer_sdp, media_type="application/sdp", headers=headers_out)

    # ------------------------------------------------------------
    # ✅ Legacy JSON session bootstrap (existing behavior)
    # ------------------------------------------------------------
    try:
        b = await req.json()
    except Exception:
        b = {}

    model = (b.get("model") or REALTIME_MODEL_DEFAULT).strip()
    voice = (b.get("voice") or REALTIME_VOICE_DEFAULT).strip()

    user_key = _derive_user_key(req, b)
    # ✅ Billing: default soft mode for realtime bootstrap.
    # App-side Pro paywall already gates voice/video; backend should not surface raw 402 to UI.
    _billing_response = _realtime_billing_allows_or_response(req, b, user_key, branch="JSON")
    if _billing_response is not None:
        return _billing_response
    # ✅ FIX: 默认开启联网（与 WebRTC SDP 分支一致），iOS 不传 allow_web 也能用
    allow_web = bool(_extract_allow_web(req, b))
    enable_memory = _boolish(b.get("enable_memory", True))
    # ✅ 提取 conversation_id（与文本聊天共用同一个会话）
    rt_conversation_id = (b.get("conversation_id") or b.get("session_id") or "").strip()

    # ✅ Provider routing: OpenAI vs Qwen Omni Realtime
    provider = _pick_realtime_provider(b)

    log.info("[/session:JSON] user=%s provider=%s allow_web=%s enable_memory=%s conv_id=%s body_keys=%s",
             user_key, provider, allow_web, enable_memory, rt_conversation_id or "(auto)", list(b.keys())[:12])

    instructions, resolved_profile = _pick_realtime_instructions(req, b)

    # ✅ 提取感知数据（与文本通道共用同一个 perception 字段）
    _rt_perception = b.get("perception")
    log.info("[/session:diag] client_id=%s perception_present=%s profile=%s",
             b.get("client_id","(none)"), bool(_rt_perception), b.get("profile","(none)"))
    log.info("[/session:semantic_diag] token=%s provider=%s create_response=true interrupt_response=true", SEMANTIC_INTERRUPT_TOKEN, provider)
    if not isinstance(_rt_perception, dict):
        _rt_perception = None

    # ✅ 异步丰富感知数据（Google Geocoding 获取门牌号）
    if isinstance(_rt_perception, dict) and _rt_perception.get("latitude"):
        try:
            from perception_module import enrich_perception_data
            _rt_perception = await enrich_perception_data(_rt_perception)
        except Exception as _enrich_err:
            log.warning("[/session] perception enrich failed: %s", _enrich_err)

    instructions = _augment_realtime_instructions(
        base_instructions=instructions,
        user_key=user_key,
        enable_memory=enable_memory,
        enable_web=allow_web,
        perception_data=_rt_perception,
        enable_agent=True,
        provider=provider,
    )

    tools = _realtime_tools(enable_web=allow_web, enable_memory=enable_memory, enable_agent=True)

    # ────────────────────────────────────────────
    # ✅ Pipeline 分支（STT → DeepSeek LLM → TTS）
    # ────────────────────────────────────────────
    if provider == "pipeline":
        # WebSocket URL: 客户端连本服务器的 /voice/ws
        # 需要构建绝对 ws:// URL
        host = req.headers.get("host", "localhost:8000")
        scheme = "wss" if req.url.scheme == "https" else "ws"
        pipeline_ws_url = f"{scheme}://{host}/voice/ws"

        pipeline_voice = (b.get("voice") or os.getenv("VOICE_TTS_VOICE", "Cherry")).strip()
        pipeline_model = os.getenv("VOICE_LLM_MODEL", "deepseek-chat")

        return {
            "ok": True,
            "provider": "pipeline",
            "session_id": uuid.uuid4().hex,
            "conversation_id": rt_conversation_id,
            "ws_url": pipeline_ws_url,
            "api_key": "",                          # 连自己后端不需要 key
            "rtc_url": pipeline_ws_url,             # 兼容旧字段
            "ephemeral_key": "",                    # 兼容旧字段
            "modalities": ["text", "audio"],
            "model": pipeline_model,
            "voice": pipeline_voice,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "input_sample_rate": 16000,
            "output_sample_rate": 24000,
            "pcm24_to_pcm16_required": False,
            "max_session_minutes": 120,
            "profile": resolved_profile,
            "home_bound": bool(resolved_profile == "home"),
            "memory_enabled": bool(enable_memory),
            "web_enabled": bool(allow_web),
            "tools_enabled": False,
            "instructions": instructions,
            "semantic_interrupt_token": SEMANTIC_INTERRUPT_TOKEN,
            "protocol_notes": {
                "transport": "WebSocket (same as Qwen path)",
                "vad": "client-side (iOS energy detection)",
                "stt": "Qwen3-ASR-Flash-Realtime",
                "llm": pipeline_model,
                "tts": os.getenv("VOICE_TTS_MODEL", "qwen3-tts-flash-realtime"),
                "audio_format": "PCM16 24kHz mono",
            },
        }

    # ────────────────────────────────────────────
    # ✅ Qwen Omni Realtime 分支
    # ────────────────────────────────────────────
    if provider == "qwen":
        qwen_model = _normalize_qwen_realtime_model(str(b.get("model") or QWEN_REALTIME_MODEL))
        qwen_voice = _normalize_qwen_realtime_voice(str(b.get("voice") or QWEN_REALTIME_VOICE))
        qinfo, qerr = _qwen_realtime_ephemeral(
            qwen_model, qwen_voice,
            instructions=instructions,
            tools=(tools if QWEN_REALTIME_ENABLE_TOOLS else None),
            tool_choice=("auto" if (tools and QWEN_REALTIME_ENABLE_TOOLS) else None),
        )
        if qerr:
            # auto 模式下 fallback 到 OpenAI
            if REALTIME_PROVIDER == "auto":
                log.warning("[/session:JSON] Qwen failed (%s), falling back to OpenAI", qerr)
                provider = "openai"
                # fall through to OpenAI block below
            else:
                log.warning("[/session:JSON] Qwen failed: %s", qerr)
                return _friendly_session_json_error()
        else:
            # ✅ FIX: Qwen /session must create or reuse the realtime conversation_id exactly once.
            # Previously the response dict contained two duplicate "conversation_id" keys,
            # each calling conv_create(...) when the client did not pass an id. That could create
            # two DB conversations while only returning the second one to iOS.
            session_conv_id = rt_conversation_id or conv_create(user_key, "实时语音")
            return {
                "ok": True,
                "provider": "qwen",
                "session_id": uuid.uuid4().hex,
                "conversation_id": session_conv_id,
                "ws_url": qinfo["ws_url"],
                "api_key": qinfo["api_key"],           # iOS 放 Authorization: Bearer
                "rtc_url": qinfo["ws_url"],             # 兼容旧字段
                "ephemeral_key": qinfo["api_key"],      # 兼容旧字段（Qwen 直接用 API Key）
                "modalities": ["text", "audio"],
                "model": qinfo["model"],
                "voice": qinfo["voice"],
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "input_sample_rate": 16000,
                "output_sample_rate": 24000,
                "pcm24_to_pcm16_required": False,
                "max_session_minutes": 120,
                "turn_detection": qinfo.get("turn_detection") or {},
                # 阿里官方文档明确："The model for transcribing input audio.
                # Only gummy-realtime-v1 is supported."  qwen3-asr-flash-realtime 是
                # 独立的实时 ASR 模型，不能放在 omni-realtime 的 transcription 字段下。
                "input_audio_transcription": {"model": "gummy-realtime-v1"},
                "profile": resolved_profile,
                "home_bound": bool(resolved_profile == "home"),
                "memory_enabled": bool(enable_memory),
                "web_enabled": bool(allow_web),
                "tools_enabled": bool(bool(tools) and QWEN_REALTIME_ENABLE_TOOLS),
                "memory_commit_url": "/realtime/memory/commit",
                "turn_commit_url": "/realtime/turn/commit",
                "web_search_url": "/web_search?q={query}&k=6",
                "user_key_hint": user_key,
                "auto_memory": False,
                "memory_commit_required": bool(enable_memory),
                "instructions": instructions,  # ✅ 统一注入：人格+记忆+感知
                "semantic_interrupt_token": SEMANTIC_INTERRUPT_TOKEN,
                # ✅ Qwen 协议差异提示
                "protocol_notes": {
                    "auth_header": "Authorization: Bearer <api_key>",
                    "no_openai_beta_header": True,
                    "turn_detection": qinfo.get("turn_detection") or {},
                    "vad_type": "server_vad (only supported value)",
                    "input_audio": "pcm only (PCM16 16kHz mono payload)",
                    "output_audio": "pcm (PCM16 24kHz mono)",
                    "session_duration": "120 min (vs OpenAI 15 min)",
                    "voices_available": 49,
                },
            }

    # ────────────────────────────────────────────
    # ✅ OpenAI Realtime 分支（原有逻辑）
    # ────────────────────────────────────────────
    # Try primary model, then auto-fallback to keep voice stable.
    key, err = _realtime_ephemeral(model, voice, instructions=instructions, tools=tools, tool_choice=("auto" if tools else None))
    if err and REALTIME_MODEL_FALLBACK and model != REALTIME_MODEL_FALLBACK:
        key2, err2 = _realtime_ephemeral(REALTIME_MODEL_FALLBACK, voice, instructions=instructions, tools=tools, tool_choice=("auto" if tools else None))
        if not err2 and key2:
            model = REALTIME_MODEL_FALLBACK
            key, err = key2, None

    if err:
        log.warning("[/session:JSON] OpenAI Realtime failed: %s", err)
        return _friendly_session_json_error()

    # NOTE: legacy client-side WebSocket/WebRTC connections to OpenAI won't automatically store new memories.
    # Use POST /realtime/memory/commit from client when the call ends to persist transcripts.
    session_conv_id = rt_conversation_id or conv_create(user_key, "实时语音")
    return {
        "ok": True,
        "provider": "openai",
        "session_id": uuid.uuid4().hex,
        "conversation_id": session_conv_id,  # ✅ 回传给前端
        "rtc_url": f"https://api.openai.com/v1/realtime?model={model}",
        "ephemeral_key": key,
        "modalities": ["text", "audio"],
        "ice_servers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "model": model,
        "voice": voice,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "pcm24_to_pcm16_required": False,
        "profile": resolved_profile,
        "home_bound": bool(resolved_profile == "home"),
        "memory_enabled": bool(enable_memory),
        "web_enabled": bool(allow_web),
        "tools_enabled": bool(bool(tools)),
        "memory_commit_url": "/realtime/memory/commit",
        "turn_commit_url": "/realtime/turn/commit",  # ✅ 新接口
        "web_search_url": "/web_search?q={query}&k=6",
        "user_key_hint": user_key,
        "auto_memory": False,
        "memory_commit_required": bool(enable_memory),
        "instructions": instructions,  # ✅ GPT Realtime 也返回完整 instructions（人格/记忆/感知）
        "semantic_interrupt_token": SEMANTIC_INTERRUPT_TOKEN,
    }


@app.post("/realtime/semantic_interrupt")
async def realtime_semantic_interrupt(req: Request):
    """
    ✅ 语义打断控制入口。

    说明：OpenAI/Qwen 直连 Realtime 模式下，运行时音频流不经过后端，
    因此前端控制层也会本地立即清播放缓冲；这个接口用于后端会话/日志/
    代理式 pipeline 统一接收同一个控制信号，后续如果把 Realtime 改成后端代理，
    可在这里直接关闭远端音频、取消旧 response、清空待播队列。
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    token = (body.get("token") or "").strip()
    if token and token != SEMANTIC_INTERRUPT_TOKEN:
        return JSONResponse({"ok": False, "error": "invalid semantic interrupt token"}, status_code=400)
    log.info("[semantic_interrupt] client_id=%s conversation_id=%s reason=%s",
             body.get("client_id") or body.get("clientId") or "",
             body.get("conversation_id") or body.get("session_id") or "",
             body.get("reason") or "")
    return {
        "ok": True,
        "cmd": "semantic_interrupt",
        "token": SEMANTIC_INTERRUPT_TOKEN,
        "actions": [
            "stop_remote_audio_if_proxied",
            "cancel_previous_response_if_active",
            "drop_unplayed_previous_audio",
            "continue_new_user_turn",
        ],
    }


@app.post("/realtime/memory/commit")
async def realtime_memory_commit(req: Request):
    """
    ✅ Manual memory commit for call transcripts.

    Use this for:
      - legacy Realtime sessions (ephemeral_key mode) where the backend cannot observe conversation
      - any audio/video call transcript that you want to add to long-term memory

    Body (JSON) examples:
      {"transcript":"..."}
      {"turns":[{"role":"user","text":"..."},{"role":"assistant","text":"..."}]}
      {"session_id":"...","call_id":"...","turns":[...], "meta":{...}}
    """
    try:
        body = await req.json()
    except Exception:
        body = {}

    user_key = _derive_user_key(req, body)
    enable_memory = _boolish(body.get("enable_memory", True))
    if not enable_memory:
        return {"ok": True, "skipped": True, "reason": "memory_disabled", "user_key": user_key}

    turns = body.get("turns")
    transcript = (body.get("transcript") or body.get("text") or "").strip()

    # Build transcript if turns provided
    if not transcript and isinstance(turns, list):
        lines: List[str] = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = str(t.get("role") or "").strip() or "user"
            text = str(t.get("text") or t.get("content") or "").strip()
            if not text:
                continue
            who = "用户" if role == "user" else ("助手" if role == "assistant" else role)
            lines.append(f"{who}: {text}")
        transcript = "\n".join(lines).strip()

    if not transcript:
        return JSONResponse({"ok": False, "error": "missing transcript/turns"}, status_code=400)

    # Extract some per-turn user statements + store a compact call note
    def _commit():
        # 1) Store a compact call note
        note = f"通话记录/摘要：{_short(transcript, 1200)}"
        if _should_memory_add(note):
            memory_add(user_key, note)

        # 2) Store user turns that look memory-worthy
        if isinstance(turns, list):
            for t in turns:
                if not isinstance(t, dict):
                    continue
                if str(t.get("role") or "").strip() != "user":
                    continue
                ut = str(t.get("text") or t.get("content") or "").strip()
                if ut and _should_memory_add(ut):
                    memory_add(user_key, _short(ut, 600))

        # 3) Structured facts (best-effort)
        try:
            if isinstance(turns, list):
                last_user = ""
                last_ai = ""
                for t in turns:
                    if not isinstance(t, dict):
                        continue
                    role = str(t.get("role") or "").strip()
                    text = str(t.get("text") or t.get("content") or "").strip()
                    if role == "user" and text:
                        last_user = text
                    if role == "assistant" and text:
                        last_ai = text
                if last_user or last_ai:
                    extract_and_save_memory_facts(user_key, last_user, last_ai)
        except Exception:
            pass

    await asyncio.to_thread(_commit)

    return {
        "ok": True,
        "user_key": user_key,
        "stored": True,
        "hint": "Call transcript committed to long-term memory.",
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

        # TTS endpoint must send only audio-speech fields. Do not attach chat/web tools here.
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

        seg_ok = False
        for chunk in _safe_iter_response_content(r, label="TTS.bytes"):
            seg_ok = True
            out.extend(chunk)
        ok_any = ok_any or seg_ok

    if not ok_any:
        return JSONResponse({"ok": False, "error": f"openai_tts_error: {last_err or 'unknown'}"}, status_code=502)

    return Response(content=bytes(out), media_type=_tts_media_type(fmt))

# ============================================================
# 把这段代码加到 server_session.py 的 @app.post("/tts") 前面
# ============================================================

# ---- 阿杜自己的声音模型（RunPod A100）----
ADU_BRIDGE_URL = os.getenv(
    "ADU_BRIDGE_URL",
    "https://translated-marine-divine-yeast.trycloudflare.com"
)

@app.post("/adu_voice")
async def adu_voice(request: Request):
    """
    纯音频端到端接口：
    - 接收 multipart/form-data: audio=<wav/m4a文件>
    - 发给 RunPod bridge 处理
    - 流式返回音频
    """
    import httpx, base64, io

    form = await request.form()
    audio_file = form.get("audio")
    if not audio_file:
        raise HTTPException(status_code=400, detail="missing audio field")

    audio_bytes = await audio_file.read()

    # 编码成 base64 发给 bridge
    audio_b64 = base64.b64encode(audio_bytes).decode()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{ADU_BRIDGE_URL}/stream",
                json={"audio_b64": audio_b64, "sr": 16000},
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"bridge error: {resp.status_code}"
                )

            return Response(
                content=resp.content,
                media_type="audio/wav",
                headers={"X-Adu-Bridge": "ok"}
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="bridge timeout")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"bridge error: {e}")


@app.get("/adu_voice/health")
async def adu_voice_health():
    """检查 RunPod bridge 是否在线"""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ADU_BRIDGE_URL}/health")
            return {"bridge": resp.json(), "url": ADU_BRIDGE_URL}
    except Exception as e:
        return {"bridge": "offline", "error": str(e), "url": ADU_BRIDGE_URL}


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

# ── /file/generate : 模型输出内容 → 生成文件供用户下载 ─────────────────────
@app.post("/file/generate")
async def file_generate(req: Request):
    """
    接受模型输出的文本内容，生成对应格式文件，返回下载URL。
    body: {
        "content": "文件内容",
        "filename": "output.py",          # 决定格式，不填默认 output.md
        "encoding": "utf-8"               # 可选
    }
    """
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid json"}, status_code=400)

    content = (body.get("content") or "").strip()
    if not content:
        return JSONResponse({"ok": False, "error": "content is empty"}, status_code=400)

    filename = (body.get("filename") or "output.md").strip()
    encoding = (body.get("encoding") or "utf-8").strip()

    # 生成唯一文件ID，存到 downloads/ 目录
    _cleanup_downloads()
    file_id = uuid.uuid4().hex
    safe_filename = filename.replace("/", "_").replace("..", "_")
    file_path = DOWNLOADS_DIR / f"{file_id}_{safe_filename}"

    try:
        file_path.write_bytes(content.encode(encoding, errors="replace"))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    import mimetypes
    mime = mimetypes.guess_type(safe_filename)[0] or "application/octet-stream"

    return {
        "ok": True,
        "file_id": file_id,
        "filename": safe_filename,
        "mime": mime,
        "size": file_path.stat().st_size,
        "download_url": f"/file/download/{file_id}/{safe_filename}",
    }


@app.get("/file/download/{file_id}/{filename}")
async def file_download(file_id: str, filename: str):
    """下载 /file/generate 生成的文件。"""
    # 安全校验：只允许 hex file_id
    import re as _re
    if not _re.match(r'^[0-9a-f]{32}$', file_id):
        return JSONResponse({"ok": False, "error": "invalid file_id"}, status_code=400)

    safe_filename = filename.replace("/", "_").replace("..", "_")
    file_path = DOWNLOADS_DIR / f"{file_id}_{safe_filename}"

    if not file_path.exists():
        return JSONResponse({"ok": False, "error": "file not found"}, status_code=404)

    import mimetypes
    mime = mimetypes.guess_type(safe_filename)[0] or "application/octet-stream"

    return FileResponse(
        path=str(file_path),
        media_type=mime,
        filename=safe_filename,
        headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
    )


# ════════════════════════════════════════════════════════════════════
# v2 社交/视频 请求体模型（注册/登录/评论/分享/推荐/私信）
# ⚠️ 这批模型曾在一次提交中被误删，导致对应 v2 路由在请求时
#    NameError -> 500（model_rebuild 被 try/except 吞掉，故进程能起、
#    /health 与语音正常，仅这些 v2 接口挂）。此处按各 handler 实际
#    字段用法还原；全部 Optional 以匹配代码中的 `req.x or ""` 防御写法。
# ════════════════════════════════════════════════════════════════════
from pydantic import BaseModel as _V2_BaseModel  # 局部导入，避免与 auth.py 的 BaseModel 冲突


class V2RegisterReq(_V2_BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    display_name: Optional[str] = None


class V2LoginReq(_V2_BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None


class V2CommentReq(_V2_BaseModel):
    text: Optional[str] = None


class V2ShareReq(_V2_BaseModel):
    channel: Optional[str] = None


class V2RecoEventReq(_V2_BaseModel):
    event_type: Optional[str] = None
    video_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class V2DMMessageReq(_V2_BaseModel):
    text: Optional[str] = None


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

# ════════════════════════════════════════════════════════════════════
# Apple Sign In  ── 验证 Apple identity_token，找/建用户，返回我们自己的 access_token
# ════════════════════════════════════════════════════════════════════

# 你的 iOS App Bundle ID（必须配，否则验签时 aud 永远不匹配）
# 例：APPLE_BUNDLE_ID=com.duy.gptsora
APPLE_BUNDLE_ID = os.environ.get("APPLE_BUNDLE_ID", "").strip()
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"

# JWKS 缓存：1 小时刷新（Apple 偶尔会轮换 key，绝不能硬编码）
_APPLE_JWKS_CACHE: Dict[str, Any] = {"keys": [], "fetched_at": 0}
_APPLE_JWKS_TTL = 3600

def _apple_fetch_jwks() -> List[Dict[str, Any]]:
    """拉 Apple 公钥集，带 1h 缓存。"""
    now = int(time.time())
    if _APPLE_JWKS_CACHE["keys"] and (now - _APPLE_JWKS_CACHE["fetched_at"] < _APPLE_JWKS_TTL):
        return _APPLE_JWKS_CACHE["keys"]
    try:
        r = requests.get(APPLE_JWKS_URL, timeout=8)
        r.raise_for_status()
        keys = r.json().get("keys", [])
        if keys:
            _APPLE_JWKS_CACHE["keys"] = keys
            _APPLE_JWKS_CACHE["fetched_at"] = now
        return keys
    except Exception as e:
        print(f"[apple_auth] JWKS fetch failed: {e}")
        # 兜底：返回旧缓存（哪怕过期），避免单点抖动导致全员登录失败
        return _APPLE_JWKS_CACHE.get("keys", []) or []

def _apple_pick_key(jwks: List[Dict[str, Any]], kid: str) -> Optional[Dict[str, Any]]:
    for k in jwks:
        if k.get("kid") == kid:
            return k
    return None

def _apple_verify_identity_token(identity_token: str) -> Dict[str, Any]:
    """
    验证 Apple identity_token，返回 decoded payload。
    任一环节失败抛 ValueError。
    """
    if not _APPLE_JWT_AVAILABLE:
        raise ValueError("server missing pyjwt[crypto]; please `pip install 'pyjwt[crypto]'`")
    if not APPLE_BUNDLE_ID:
        raise ValueError("server missing APPLE_BUNDLE_ID env")

    # 1) 取 token header 看 kid
    try:
        unverified_header = _pyjwt.get_unverified_header(identity_token)
    except Exception as e:
        raise ValueError(f"malformed identity_token: {e}")
    kid = unverified_header.get("kid")
    alg = unverified_header.get("alg")
    if alg != "RS256":
        raise ValueError(f"unexpected alg: {alg}")
    if not kid:
        raise ValueError("missing kid in token header")

    # 2) 选公钥（JWKS 缓存，必要时重拉一次）
    jwks = _apple_fetch_jwks()
    jwk = _apple_pick_key(jwks, kid)
    if not jwk:
        # kid 找不到 —— 可能 Apple 刚轮换了 key，强制刷新一次
        _APPLE_JWKS_CACHE["fetched_at"] = 0
        jwks = _apple_fetch_jwks()
        jwk = _apple_pick_key(jwks, kid)
    if not jwk:
        raise ValueError(f"unknown apple kid: {kid}")

    public_key = _RSAAlgorithm.from_jwk(json.dumps(jwk))

    # 3) 验签 + 校验 iss / aud / exp
    try:
        payload = _pyjwt.decode(
            identity_token,
            public_key,
            algorithms=["RS256"],
            audience=APPLE_BUNDLE_ID,
            issuer=APPLE_ISSUER,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except Exception as e:
        raise ValueError(f"invalid identity_token: {e}")

    return payload

def _apple_make_username(apple_sub: str) -> str:
    """
    Apple sub 形如 '001234.abcdefghijk.1234'，含点号且超长。
    我们把它转成符合 username 约束（3-64 位）的稳定串：apple_<前12位hash>。
    用 hash 是为了：
      ① username 列有 UNIQUE 约束，apple_sub 也要做 UNIQUE，二者解耦
      ② 不在 username 暴露 Apple 内部 ID
    """
    h = hashlib.sha256(apple_sub.encode("utf-8")).hexdigest()[:12]
    return f"apple_{h}"

# 局部导入 BaseModel（避免和 auth.py 模块中的定义冲突）
from pydantic import BaseModel as _SIWA_BaseModel

class V2AppleSignInReq(_SIWA_BaseModel):
    identity_token: str
    # 可选：iOS 端在用户首次授权时拿到的姓名（Apple 仅首次回传），后端只在新建账号时落库
    full_name: Optional[str] = None
    # 可选：iOS 端拿到的 email（多数场景 identity_token 里就有，这里给个兜底通道）
    email_hint: Optional[str] = None

@app.post("/v2/auth/apple")
async def v2_auth_apple(req: V2AppleSignInReq):
    """
    iOS 端拿到 Apple 颁发的 identity_token 后调本接口；
    后端验证 token → 找/建用户 → 返回我们自己的 access_token。
    """
    token_str = (req.identity_token or "").strip()
    if not token_str:
        raise HTTPException(status_code=400, detail="missing identity_token")

    # 1) 验证 Apple JWT
    try:
        claims = _apple_verify_identity_token(token_str)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"apple verify failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"apple verify error: {e}")

    apple_sub = (claims.get("sub") or "").strip()
    if not apple_sub:
        raise HTTPException(status_code=401, detail="missing sub claim")

    # email：identity_token 里的 email 优先；首次注册才会有，二次登录不一定有
    email_in_token = (claims.get("email") or "").strip().lower()
    email = email_in_token or (req.email_hint or "").strip().lower()

    full_name = (req.full_name or "").strip()
    now = int(time.time())

    conn = _video_conn()
    try:
        # 2) 用 apple_sub 找现有账号
        urow = conn.execute(
            "SELECT user_id, username, display_name, avatar_url, created_at, email FROM users WHERE apple_sub=?",
            (apple_sub,),
        ).fetchone()

        if urow:
            # 已有账号：补齐 email（Apple 第二次不返回 email 时不要覆盖已有）
            if email and not (urow["email"] or "").strip():
                conn.execute("UPDATE users SET email=? WHERE user_id=?", (email, urow["user_id"]))
                conn.commit()
            user_id = urow["user_id"]
        else:
            # 3) 新建账号
            user_id = uuid.uuid4().hex
            username = _apple_make_username(apple_sub)
            display_name = full_name or (email.split("@")[0] if email else username)

            # username 极小概率 hash 撞车，撞了在尾部加 4 位
            for attempt in range(3):
                try:
                    conn.execute(
                        """
                        INSERT INTO users(user_id, username, display_name, password_hash,
                                          avatar_url, created_at, apple_sub, email)
                        VALUES (?, ?, ?, ?, '', ?, ?, ?)
                        """,
                        (user_id, username, display_name, "!apple-no-pw", now, apple_sub, email),
                    )
                    conn.commit()
                    break
                except sqlite3.IntegrityError as ie:
                    msg = str(ie).lower()
                    if "username" in msg and attempt < 2:
                        username = f"{username}{uuid.uuid4().hex[:4]}"
                        continue
                    if "apple_sub" in msg:
                        # 并发情况下另一个请求刚建好；回去当作已有账号处理
                        urow2 = conn.execute(
                            "SELECT user_id FROM users WHERE apple_sub=?",
                            (apple_sub,),
                        ).fetchone()
                        if urow2:
                            user_id = urow2["user_id"]
                            break
                    raise

        # 4) 签发我们自己的 access_token
        token = _issue_access_token(user_id)

        # 5) 拉最终 user 行回执
        urow_final = conn.execute(
            "SELECT user_id, username, display_name, avatar_url, created_at, email FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()

    # 手动构建 user dict（不依赖 _public_user，避免字段兼容问题）
    if urow_final is None:
        raise HTTPException(status_code=500, detail="user not found after creation")
    user = {
        "user_id": urow_final["user_id"],
        "username": urow_final["username"],
        "display_name": urow_final["display_name"] or "",
        "avatar_url": urow_final["avatar_url"] or "",
        "created_at": urow_final["created_at"] or 0,
        "email": (urow_final["email"] or "") if "email" in urow_final.keys() else "",
    }

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


# ════════════════════════════════════════════════════════════════════
# 设备数据迁移：用户首次登录后，把匿名 client_id 名下的会话/消息/记忆
# 一次性迁移到新账号的 user_key (= apple_sub) 下。
#
# 安全约束（绝对不能违反）：
#   1) 必须带 Bearer token（已知谁是当前用户）
#   2) 目标账号必须是空的 —— 不允许覆盖已有数据
#   3) 只能迁移 1 次 —— migrated_devices 表记录已迁移过的 client_id
#   4) 每个 client_id 只能被迁移到 1 个账号 —— 防止 A/B 互抢
#
# 调用时机（前端）：成功登录返回 token 后，立即调一次本端点
# ════════════════════════════════════════════════════════════════════

class V2MigrateDeviceReq(_SIWA_BaseModel):
    client_id: str  # 用户登录前的匿名设备 ID（前端 UserDefaults "solara.client_id"）

@app.post("/v2/auth/migrate_device")
async def v2_auth_migrate_device(req: V2MigrateDeviceReq, request: Request):
    """
    把 client_id 名下的会话/消息/记忆迁移到当前登录账号。
    幂等：重复调用不会重复迁移；安全：拒绝覆盖已有数据。
    """
    user = _auth_required_user(request)
    target_user_key = _sanitize_user_key(user["user_id"])

    src_client_id = (req.client_id or "").strip()
    if not src_client_id:
        return {"ok": True, "migrated": False, "reason": "no client_id"}

    src_user_key = _sanitize_user_key(src_client_id)

    if src_user_key == target_user_key:
        return {"ok": True, "migrated": False, "reason": "same key"}

    # 1) 检查迁移记录表
    conv_conn = _conv_conn()
    try:
        conv_conn.execute("""
            CREATE TABLE IF NOT EXISTS migrated_devices (
                client_id TEXT PRIMARY KEY,
                target_user_key TEXT NOT NULL,
                migrated_at REAL NOT NULL
            )
        """)
        existing = conv_conn.execute(
            "SELECT target_user_key FROM migrated_devices WHERE client_id=?",
            (src_user_key,)
        ).fetchone()
        if existing:
            prev_target = existing[0] if isinstance(existing, tuple) else existing["target_user_key"]
            if prev_target == target_user_key:
                return {"ok": True, "migrated": False, "reason": "already migrated to this account"}
            else:
                return {"ok": True, "migrated": False, "reason": "client_id bound to other account"}

        # 2) 检查目标账号是否已经有数据
        target_has_data = conv_conn.execute(
            "SELECT 1 FROM conversations WHERE user_key=? LIMIT 1",
            (target_user_key,)
        ).fetchone() is not None
        if not target_has_data:
            target_has_data = conv_conn.execute(
                "SELECT 1 FROM messages WHERE user_key=? LIMIT 1",
                (target_user_key,)
            ).fetchone() is not None
        if target_has_data:
            return {"ok": True, "migrated": False, "reason": "target account already has data"}

        # 3) 检查源 client_id 数据量
        src_conv_count = conv_conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_key=?",
            (src_user_key,)
        ).fetchone()[0] or 0
        src_msg_count = conv_conn.execute(
            "SELECT COUNT(*) FROM messages WHERE user_key=?",
            (src_user_key,)
        ).fetchone()[0] or 0

        # 4) 真正迁移
        conv_conn.execute("BEGIN")
        try:
            conv_conn.execute(
                "UPDATE conversations SET user_key=? WHERE user_key=?",
                (target_user_key, src_user_key)
            )
            conv_conn.execute(
                "UPDATE messages SET user_key=? WHERE user_key=?",
                (target_user_key, src_user_key)
            )
            conv_conn.execute(
                "INSERT OR REPLACE INTO migrated_devices(client_id, target_user_key, migrated_at) VALUES (?, ?, ?)",
                (src_user_key, target_user_key, time.time())
            )
            conv_conn.execute("COMMIT")
        except Exception as e:
            conv_conn.execute("ROLLBACK")
            raise HTTPException(status_code=500, detail=f"migrate conv failed: {e}")
    finally:
        conv_conn.close()

    # 5) 迁移记忆
    src_mem_count = 0
    src_facts_count = 0
    src_timeline_count = 0
    mem_conn = _mem_conn()
    try:
        try:
            src_mem_count = mem_conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE user_key=?", (src_user_key,)
            ).fetchone()[0] or 0
        except Exception:
            pass
        try:
            src_facts_count = mem_conn.execute(
                "SELECT COUNT(*) FROM memory_facts WHERE user_key=?", (src_user_key,)
            ).fetchone()[0] or 0
        except Exception:
            pass
        try:
            src_timeline_count = mem_conn.execute(
                "SELECT COUNT(*) FROM memory_timeline WHERE user_key=?", (src_user_key,)
            ).fetchone()[0] or 0
        except Exception:
            pass

        mem_conn.execute("BEGIN")
        try:
            try:
                mem_conn.execute(
                    "UPDATE memory_items SET user_key=? WHERE user_key=?",
                    (target_user_key, src_user_key)
                )
            except sqlite3.IntegrityError:
                pass
            try:
                mem_conn.execute(
                    "UPDATE memory_facts SET user_key=? WHERE user_key=?",
                    (target_user_key, src_user_key)
                )
            except sqlite3.IntegrityError:
                pass
            try:
                mem_conn.execute(
                    "UPDATE memory_timeline SET user_key=? WHERE user_key=?",
                    (target_user_key, src_user_key)
                )
            except sqlite3.IntegrityError:
                pass
            mem_conn.execute("COMMIT")
        except Exception as e:
            mem_conn.execute("ROLLBACK")
            print(f"[migrate_device] mem migrate failed (non-fatal): {e}")
    finally:
        mem_conn.close()

    return {
        "ok": True,
        "migrated": True,
        "stats": {
            "conversations": src_conv_count,
            "messages": src_msg_count,
            "memory_items": src_mem_count,
            "memory_facts": src_facts_count,
            "memory_timeline": src_timeline_count,
        },
    }


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
            "has_default_instructions": bool(_get_realtime_default_instructions())
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
            "computer_action": "/api/brain/computer/action",
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

# ══════════════════════════════════════════════════════════════════
# ✅ OpenClaw 深度融合 — 项目助手 API
# FastAPI 后端 ←→ Adu-Agent(OpenClaw) Gateway
# 文件读写 / 终端执行 / 项目扫描 / 浏览器控制 / Git
# ══════════════════════════════════════════════════════════════════

# ── Agent 用量计量 ──────────────────────────────────────────────
import datetime as _dt

_AGENT_LIMITS = {
    "free": 5,       # 5 次/天
    "pro": 200,      # 200 次/天
    "ultra": 99999,  # 无限
}

# {client_id: {"date": "2026-03-18", "count": 12}}
_agent_usage: Dict[str, Dict[str, Any]] = {}


def _agent_check_quota(request: Request) -> Optional[JSONResponse]:
    """检查 Agent 操作配额，超限返回 JSONResponse，否则返回 None"""
    client_id = (request.headers.get("x-client-id") or "anon").strip()[:64]
    plan = (request.headers.get("x-chatagi-plan") or "free").strip().lower()
    limit = _AGENT_LIMITS.get(plan, _AGENT_LIMITS["free"])

    today = _dt.date.today().isoformat()
    usage = _agent_usage.get(client_id)
    if not usage or usage.get("date") != today:
        usage = {"date": today, "count": 0, "plan": plan}
        _agent_usage[client_id] = usage

    if usage["count"] >= limit:
        return JSONResponse({
            "ok": False,
            "error": "agent_quota_exceeded",
            "message": f"已达今日 Agent 操作上限（{plan}: {limit}次/天）",
            "plan": plan,
            "used": usage["count"],
            "limit": limit,
        }, status_code=429)

    usage["count"] += 1
    usage["plan"] = plan
    return None


@app.get("/agent/usage")
async def agent_usage(request: Request):
    """查询当前 Agent 用量"""
    client_id = (request.headers.get("x-client-id") or "anon").strip()[:64]
    plan = (request.headers.get("x-chatagi-plan") or "free").strip().lower()
    today = _dt.date.today().isoformat()
    usage = _agent_usage.get(client_id)
    if not usage or usage.get("date") != today:
        used = 0
    else:
        used = usage.get("count", 0)
    limit = _AGENT_LIMITS.get(plan, _AGENT_LIMITS["free"])
    return {"plan": plan, "used": used, "limit": limit, "date": today}


@app.get("/agent/health")
async def agent_health():
    """Legacy OpenClaw Gateway 连接状态"""
    if not _openclaw_runtime_enabled():
        return {"status": "disabled", **_openclaw_disabled_payload()}
    try:
        bridge = _openclaw_get_bridge_or_none()
        if bridge is None:
            return {"status": "error", "error": "openclaw_not_available"}
        return await bridge.health()
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _ensure_openclaw_or_503():
    if not _openclaw_runtime_enabled():
        return None, JSONResponse(_openclaw_disabled_payload(), status_code=503)
    try:
        bridge = await ensure_connected()
    except Exception as e:
        return None, JSONResponse({"ok": False, "error": "openclaw_connect_failed", "message": str(e)[:300]}, status_code=503)
    if bridge is None or not getattr(bridge, "connected", False):
        return None, JSONResponse({"ok": False, "error": "OpenClaw not connected"}, status_code=503)
    return bridge, None


@app.post("/agent/scan")
async def agent_scan(request: Request):
    """扫描项目目录，返回文件列表和摘要"""
    quota_err = _agent_check_quota(request)
    if quota_err:
        return quota_err
    try:
        body = await request.json()
    except Exception:
        body = {}
    path = body.get("path", "~/GPTsora")

    bridge, err = await _ensure_openclaw_or_503()
    if err:
        return err

    result = await bridge.scan_project(path)
    return {"ok": True, **result}


@app.post("/agent/files")
async def agent_files(request: Request):
    """列出目录下的文件"""
    quota_err = _agent_check_quota(request)
    if quota_err:
        return quota_err
    try:
        body = await request.json()
    except Exception:
        body = {}
    path = body.get("path", "~/GPTsora")
    pattern = body.get("pattern", "*.swift")

    bridge, err = await _ensure_openclaw_or_503()
    if err:
        return err

    files = await bridge.list_files(path, pattern)
    return {"ok": True, "files": files, "count": len(files)}


@app.post("/agent/read")
async def agent_read(request: Request):
    """读取文件内容"""
    quota_err = _agent_check_quota(request)
    if quota_err:
        return quota_err
    try:
        body = await request.json()
    except Exception:
        body = {}
    path = body.get("path", "")
    if not path:
        return JSONResponse({"ok": False, "error": "missing path"}, status_code=400)

    bridge, err = await _ensure_openclaw_or_503()
    if err:
        return err

    content = await bridge.read_file(path)
    lines = content.count("\n") + 1 if content else 0
    return {"ok": True, "path": path, "content": content, "lines": lines}


@app.post("/agent/write")
async def agent_write(request: Request):
    """写入文件"""
    quota_err = _agent_check_quota(request)
    if quota_err:
        return quota_err
    try:
        body = await request.json()
    except Exception:
        body = {}
    path = body.get("path", "")
    content = body.get("content", "")
    if not path:
        return JSONResponse({"ok": False, "error": "missing path"}, status_code=400)

    bridge, err = await _ensure_openclaw_or_503()
    if err:
        return err

    # Git 备份
    project_dir = "/".join(path.split("/")[:-1]) or "~"
    await bridge.exec(f'cd {project_dir} && git add -A && git commit -m "AI backup" --allow-empty 2>/dev/null')

    ok = await bridge.write_file(path, content)
    return {"ok": ok, "path": path}


@app.post("/agent/tools")
async def agent_tools(req: Request):
    """
    ✅ 统一工具调用接口（iOS 直接调，不走 AI 回复解析）
    POST /agent/tools
    Body: {
      "tool": "exec" | "read_file" | "write_file" | "search_code",
      "params": { ... 工具参数 ... },
      "confirm": false   // 高危操作确认标志
    }
    """
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"ok": False, "output": "invalid JSON"}, status_code=400)

    try:
        from agent_intent_router import handle_agent_tools_request
        result = await handle_agent_tools_request(body)
        return JSONResponse(result)
    except Exception as e:
        log.exception("[agent/tools] error")
        return JSONResponse({"ok": False, "output": str(e)}, status_code=500)


@app.get("/agent/tools/info")
async def agent_tools_info():
    """列出所有可用工具及其参数说明"""
    return {
        "tools": [
            {
                "name": "exec",
                "description": "执行终端命令",
                "params": {"command": "str", "timeout": "int (default: 60)"},
                "safety": "危险命令自动拦截，高危操作需 confirm=true",
            },
            {
                "name": "read_file",
                "description": "读取文件内容",
                "params": {"path": "str", "start_line": "int?", "end_line": "int?"},
                "safety": "只允许工作区路径（GPTsora/backend/chatterbox/frontend）",
            },
            {
                "name": "write_file",
                "description": "写入文件（始终需要 confirm=true）",
                "params": {"path": "str", "content": "str"},
                "safety": "只允许工作区路径，自动 git 备份，始终需要确认",
            },
            {
                "name": "search_code",
                "description": "搜索代码关键词",
                "params": {"keyword": "str", "path": "str?", "ext": "str?"},
                "safety": "只搜索工作区，最多返回50条结果",
            },
        ],
        "safe_roots": [
            "~/Desktop/GPTsora",
            "~/Desktop/backend",
            "~/Desktop/chatterbox",
            "~/Desktop/frontend",
            "~/.openclaw/workspace",
            "/tmp",
        ]
    }


@app.post("/controller/run")
async def controller_run(request: Request):
    """Unified execution endpoint for AutoBrain / App computer tasks.

    Strategy:
    1) Prefer local adu_controller.handle(instruction) when available
    2) Fallback to lightweight local rules for shell/open-app
    3) Return normalized result schema
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    instruction = str(body.get("instruction") or "").strip()
    if not instruction:
        return JSONResponse({"success": False, "error": "missing instruction"}, status_code=400)

    # 🧠 任务完成 → 通知意识系统（helper，复用以覆盖全部 return 分支）
    def _notify(result_dict: Dict[str, Any]):
        try:
            from adu_consciousness import consciousness
            _success = bool(result_dict.get("ok", result_dict.get("success", False)))
            consciousness.on_task_complete(instruction[:30], _success)
        except Exception as _cs_e:
            log.debug("[Consciousness] on_task_complete skip (controller): %s", _cs_e)

    # local controller first
    try:
        from adu_controller import controller
        result = await _adu_controller.handle(instruction)
        if isinstance(result, dict):
            result.setdefault("success", bool(result.get("ok", result.get("success", False))))
            if not result.get("summary") and result.get("output"):
                result["summary"] = str(result.get("output"))[:200]
            _notify(result)
            return result
        _r = {"success": True, "summary": str(result), "result": result}
        _notify(_r)
        return _r
    except Exception as _ctrl_err:
        log.info("[controller/run] adu_controller unavailable/fallback: %s", _ctrl_err)

    txt = instruction.lower()
    # simple open-app path for commands like 把微信打开 / 打开 Safari
    app_name = ""
    if "微信" in instruction:
        app_name = "WeChat"
    elif "safari" in txt:
        app_name = "Safari"
    elif "finder" in txt:
        app_name = "Finder"
    elif "xcode" in txt:
        app_name = "Xcode"
    if app_name:
        script = f'tell application "{app_name}" to activate'
        try:
            proc = await asyncio.create_subprocess_exec("osascript", "-e", script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            ok = proc.returncode == 0
            _r = {"success": ok, "summary": (f"{app_name}已切到前台。" if ok else f"打开{app_name}失败。"), "output": output, "via": "osascript"}
            _notify(_r)
            return _r
        except Exception as e:
            _r = {"success": False, "summary": f"打开{app_name}失败。", "output": str(e), "via": "osascript"}
            _notify(_r)
            return _r

    # shell fallback if prefixed
    if instruction.startswith("执行终端命令：") or instruction.startswith("执行终端命令:"):
        cmd = instruction.split("：", 1)[-1] if "：" in instruction else instruction.split(":", 1)[-1]
        try:
            proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            _r = {"success": proc.returncode == 0, "summary": "命令已执行。" if proc.returncode == 0 else "命令执行失败。", "output": output, "via": "shell"}
            _notify(_r)
            return _r
        except Exception as e:
            _r = {"success": False, "summary": "命令执行失败。", "output": str(e), "via": "shell"}
            _notify(_r)
            return _r

    # fallback to existing agent exec (local shell)
    try:
        proc = await asyncio.create_subprocess_shell(instruction, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        _r = {"success": proc.returncode == 0, "summary": "已执行。" if proc.returncode == 0 else "执行失败。", "output": output, "via": "shell_direct"}
        _notify(_r)
        return _r
    except Exception as e:
        _r = {"success": False, "summary": "执行失败。", "output": str(e), "via": "shell_direct"}
        _notify(_r)
        return _r


@app.post("/agent/mac_send")
async def agent_mac_send(req: Request):
    """发微信消息接口"""
    try:
        b = await req.json()
        contact = (b.get("contact") or "").strip()
        message = (b.get("message") or "").strip()
        if not contact or not message:
            return JSONResponse({"ok": False, "error": "missing contact or message"})
        
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from adu_mac_tools import send_wechat
        result = send_wechat(contact, message)
        return JSONResponse({"ok": True, "result": f"已发送给{contact}"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.post("/agent/exec")
async def agent_exec(request: Request):
    """执行终端命令 — 优先本地 subprocess，OpenClaw bridge 作为可选增强"""
    quota_err = _agent_check_quota(request)
    if quota_err:
        return quota_err
    try:
        body = await request.json()
    except Exception:
        body = {}
    command = body.get("command", "")
    if not command:
        return JSONResponse({"ok": False, "error": "missing command"}, status_code=400)

    timeout = min(body.get("timeout", 30), 120)  # 最大2分钟
    use_local = body.get("local", True)  # 默认走本地，更快更可靠

    # ✅ 本地 subprocess 直接执行（无需 OpenClaw bridge）
    if use_local:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "TERM": "xterm-256color"},
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"ok": False, "error": f"timeout after {timeout}s", "output": ""}
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            return {"ok": True, "output": output, "exit_code": proc.returncode, "via": "local"}
        except Exception as e:
            log.warning("[agent/exec] local exec failed: %s, falling back to bridge", e)

    # Fallback: OpenClaw bridge
    bridge, err = await _ensure_openclaw_or_503()
    if err:
        return err

    output = await bridge.exec(command, timeout=timeout)
    return {"ok": True, "output": output, "via": "bridge"}


@app.post("/agent/build")
async def agent_build(request: Request):
    """编译 Xcode 项目 — 本地直接 xcodebuild"""
    quota_err = _agent_check_quota(request)
    if quota_err:
        return quota_err
    try:
        body = await request.json()
    except Exception:
        body = {}
    path = os.path.expanduser(body.get("path", "~/Desktop/GPTsora"))
    scheme = body.get("scheme", "GPT Solara")
    workspace = body.get("workspace", "GPTsora.xcworkspace")
    device_id = body.get("device_id", "")  # 可选，指定设备

    dest = f"id={device_id}" if device_id else "generic/platform=iOS"
    cmd = (
        f'cd {path} && xcodebuild '
        f'-workspace {workspace} '
        f'-scheme "{scheme}" '
        f'-destination "{dest}" '
        f'-configuration Debug '
        f'clean build 2>&1 | tail -60'
    )
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
        output = stdout.decode("utf-8", errors="replace") if stdout else ""
    except asyncio.TimeoutError:
        return {"ok": False, "error": "build timeout (300s)", "output": ""}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    success = "BUILD SUCCEEDED" in output
    errors = [l for l in output.split("\n") if "error:" in l.lower() and "error: " in l]
    return {
        "ok": True,
        "success": success,
        "output": output,
        "errors": errors[:20],
        "error_count": len(errors),
    }


@app.post("/agent/filetree")
async def agent_filetree(request: Request):
    """
    ✅ 返回项目文件树（供手机端显示）
    Body: { "path": "~/Desktop/GPTsora", "depth": 3, "extensions": [".swift", ".py"] }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    path = body.get("path", "~/Desktop/GPTsora").strip()
    depth = max(1, min(int(body.get("depth", 3)), 5))
    exts = body.get("extensions", [".swift", ".py", ".m", ".h"])

    # 用 find 命令生成文件树（本地直接执行）
    ext_filter = " -o ".join([f'-name "*{e}"' for e in exts])
    path_expanded = os.path.expanduser(path)
    cmd = (
        f'find {path_expanded} -maxdepth {depth} \( {ext_filter} \) '
        f'-not -path "*/Pods/*" -not -path "*/.git/*" '
        f'-not -path "*/node_modules/*" -not -path "*/DerivedData/*" '
        f'| sort'
    )

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = stdout.decode("utf-8", errors="replace") if stdout else ""
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    files = [f.strip() for f in output.split("\n") if f.strip()]

    # 构建树结构
    tree = {}
    for f in files:
        parts = f.replace(path, "").lstrip("/").split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None

    return {"ok": True, "path": path, "files": files, "count": len(files), "tree": tree}


@app.post("/agent/install")
async def agent_install(request: Request):
    """
    ✅ 编译并安装到已连接的 iPhone — 本地直接执行
    Body: { "path": "~/Desktop/GPTsora", "scheme": "GPT Solara", "device_id": "auto" }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    path = os.path.expanduser(body.get("path", "~/Desktop/GPTsora").strip())
    scheme = body.get("scheme", "GPT Solara").strip()
    workspace = body.get("workspace", "GPTsora.xcworkspace")
    device_id = body.get("device_id", "auto").strip()

    # ✅ 本地找已连接设备
    if device_id == "auto":
        try:
            dev_proc = await asyncio.create_subprocess_shell(
                "xcrun xctrace list devices 2>/dev/null | grep 'iPhone' | grep -v Simulator | head -3",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            dev_out, _ = await asyncio.wait_for(dev_proc.communicate(), timeout=10)
            dev_str = dev_out.decode("utf-8", errors="replace") if dev_out else ""
            log.info("[agent/install] connected devices: %s", dev_str[:200])
            # 提取第一个设备ID
            import re as _re
            m = _re.search(r'\(([0-9a-f]{40})\)', dev_str)
            if m:
                device_id = m.group(1)
                log.info("[agent/install] auto selected device: %s", device_id)
        except Exception as e:
            log.warning("[agent/install] device detection failed: %s", e)

    # ✅ 编译 + 安装
    dest = f"id={device_id}" if device_id and device_id != "auto" else "generic/platform=iOS"
    cmd = (
        f'cd {path} && xcodebuild '
        f'-workspace {workspace} '
        f'-scheme "{scheme}" '
        f'-destination "{dest}" '
        f'-configuration Debug '
        f'clean build 2>&1 | tail -50'
    )

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=360)
        output = stdout.decode("utf-8", errors="replace") if stdout else ""
    except asyncio.TimeoutError:
        return {"ok": False, "error": "install timeout (360s)", "output": ""}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    success = "BUILD SUCCEEDED" in output

    # ✅ 如果编译成功且有设备，用 devicectl 安装
    install_output = ""
    if success and device_id and device_id != "auto":
        app_path_proc = await asyncio.create_subprocess_shell(
            f'find ~/Library/Developer/Xcode/DerivedData -name "{scheme}.app" -path "*/Debug-iphoneos/*" | head -1',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        app_out, _ = await app_path_proc.communicate()
        app_path = (app_out.decode("utf-8", errors="replace") or "").strip()
        if app_path:
            inst_proc = await asyncio.create_subprocess_shell(
                f'xcrun devicectl device install app --device {device_id} "{app_path}" 2>&1',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            inst_out, _ = await asyncio.wait_for(inst_proc.communicate(), timeout=60)
            install_output = inst_out.decode("utf-8", errors="replace") if inst_out else ""
            success = success and ("App installed" in install_output or "installed" in install_output.lower())

    errors = [l for l in output.split("\n") if "error:" in l.lower() and "error: " in l]
    return {
        "ok": True,
        "success": success,
        "build_output": output,
        "install_output": install_output,
        "device_id": device_id,
        "errors": errors[:20],
        "error_count": len(errors),
    }


@app.post("/agent/search")
async def agent_search(request: Request):
    """搜索项目代码（兼容旧接口）"""
    quota_err = _agent_check_quota(request)
    if quota_err:
        return quota_err
    try:
        body = await request.json()
    except Exception:
        body = {}
    pattern = body.get("pattern", "")
    path = body.get("path", "~/Desktop/GPTsora")
    file_type = body.get("file_type", "*.swift")

    if not pattern:
        return JSONResponse({"ok": False, "error": "missing pattern"}, status_code=400)

    from adu_computer_tools import tool_search_code
    result = await tool_search_code(
        keyword=pattern,
        path=path,
        file_pattern=file_type,
    )
    return {"ok": result["ok"], "results": result.get("output", ""), **result}


# ══════════════════════════════════════════════════════════════════════
# ✅ 统一电脑控制工具 API — v1
#    POST /agent/tool  → dispatch_tool(name, params)
#    GET  /agent/tools → 返回工具列表和安全配置
# ══════════════════════════════════════════════════════════════════════

try:
    from adu_computer_tools import dispatch_tool, get_security_config, TOOL_DEFINITIONS
    _computer_tools_loaded = True
    log.info("[ComputerTools] ✅ adu_computer_tools loaded (5 tools)")
except Exception as _ct_err:
    _computer_tools_loaded = False
    log.warning("[ComputerTools] ❌ failed to load: %s", _ct_err)


@app.get("/agent/tools")
async def agent_tools_list():
    """返回可用工具列表和安全配置"""
    if not _computer_tools_loaded:
        return JSONResponse({"ok": False, "error": "computer tools not loaded"}, status_code=503)
    cfg = get_security_config()
    return {
        "ok": True,
        "tools": TOOL_DEFINITIONS,
        "security": cfg,
    }


@app.post("/agent/tool")
async def agent_tool_dispatch(request: Request):
    """
    统一工具调用入口。

    Body:
    {
        "tool": "exec" | "read_file" | "write_file" | "search_code" | "list_dir",
        "params": { ... }  // 工具参数
    }

    Response:
    {
        "ok": bool,
        "output": str,       // exec / search_code
        "content": str,      // read_file
        "files": [...],      // list_dir
        "blocked": bool,     // 安全拦截
        "reason": str,       // 拦截原因
        "requires_confirm": bool  // 需要用户确认
    }
    """
    if not _computer_tools_loaded:
        return JSONResponse({"ok": False, "error": "computer tools not loaded"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)

    tool_name = (body.get("tool") or body.get("name") or "").strip()
    params = body.get("params") or {}

    if not tool_name:
        return JSONResponse({"ok": False, "error": "missing tool name"}, status_code=400)

    log.info("[ComputerTools] tool=%s params=%s", tool_name, str(params)[:200])

    result = await dispatch_tool(tool_name, params)

    # 记录安全拦截
    if result.get("blocked"):
        log.warning("[ComputerTools] BLOCKED tool=%s reason=%s", tool_name, result.get("reason"))

    return result


# ══════════════════════════════════════════════════════════════════════
# ✅ Direct Computer Tool API — usecomputer backend
#    Assistant -> /api/computer -> shell / usecomputer -> Mac
# ══════════════════════════════════════════════════════════════════════

def _run_usecomputer(argv: list[str], timeout: int = 20) -> dict:
    try:
        p = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "ok": p.returncode == 0,
            "exit_code": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": 124,
            "stdout": "",
            "stderr": "timeout",
        }
    except Exception as e:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
        }


@app.post("/api/computer")
async def api_computer(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)

    action = str(body.get("action") or "").strip()
    args = body.get("args") or {}

    if not isinstance(args, dict):
        args = {}

    if not action:
        return JSONResponse({"ok": False, "error": "missing action"}, status_code=400)

    # 1. device info
    if action == "device_info":
        return {
            "ok": True,
            "platform": "macos",
            "default_shell": "zsh",
            "gui_engine": "usecomputer",
            "capabilities": [
                "shell",
                "screenshot",
                "mouse_position",
                "click",
                "double_click",
                "hotkey",
                "type_text",
                "wait",
                "window_list",
                "display_list",
                "desktop_list",
            ],
        }

    # 2. shell: 终端命令直接执行
    if action in ("shell", "bash", "exec"):
        cmd = str(args.get("cmd") or args.get("command") or "").strip()
        timeout = int(args.get("timeout_sec") or args.get("timeout") or 30)

        if not cmd:
            return JSONResponse({"ok": False, "error": "missing cmd"}, status_code=400)

        try:
            p = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/zsh",
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            return {
                "ok": p.returncode == 0,
                "exit_code": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "exit_code": 124,
                "stdout": "",
                "stderr": "timeout",
            }
        except Exception as e:
            return {
                "ok": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
            }

    # 3. screenshot: usecomputer
    if action == "screenshot":
        path = str(args.get("path") or f"/tmp/adu_screen_{uuid.uuid4().hex}.png")
        result = _run_usecomputer(
            ["usecomputer", "screenshot", path, "--json"],
            timeout=int(args.get("timeout_sec") or 20),
        )

        image_base64 = ""
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                image_base64 = ""

        return {
            **result,
            "path": path,
            "mime_type": "image/png",
            "base64": image_base64,
        }

    # 4. mouse position
    if action == "mouse_position":
        return _run_usecomputer(["usecomputer", "mouse", "position", "--json"])

    # 5. click
    if action == "click":
        x = args.get("x")
        y = args.get("y")
        if x is None or y is None:
            return JSONResponse({"ok": False, "error": "missing x/y"}, status_code=400)

        return _run_usecomputer([
            "usecomputer",
            "click",
            "-x", str(x),
            "-y", str(y),
            "--button", str(args.get("button") or "left"),
            "--count", str(args.get("count") or 1),
        ])

    # 6. double click
    if action == "double_click":
        x = args.get("x")
        y = args.get("y")
        if x is None or y is None:
            return JSONResponse({"ok": False, "error": "missing x/y"}, status_code=400)

        return _run_usecomputer([
            "usecomputer",
            "click",
            "-x", str(x),
            "-y", str(y),
            "--button", str(args.get("button") or "left"),
            "--count", "2",
        ])

    # 7. hotkey / press
    if action in ("hotkey", "press"):
        keys = args.get("keys")
        if isinstance(keys, list):
            key = "+".join(str(k) for k in keys)
        else:
            key = str(args.get("key") or args.get("combo") or "").strip()

        if not key:
            return JSONResponse({"ok": False, "error": "missing key"}, status_code=400)

        return _run_usecomputer(["usecomputer", "press", key])

    # 8. type text
    if action == "type_text":
        text = str(args.get("text") or "")
        return _run_usecomputer(["usecomputer", "type", text])

    # 9. wait
    if action == "wait":
        seconds = float(args.get("seconds") or args.get("sec") or 1)
        seconds = max(0.0, min(seconds, 30.0))
        await asyncio.sleep(seconds)
        return {"ok": True, "waited": seconds}

    # 10. window list
    if action == "window_list":
        return _run_usecomputer(["usecomputer", "window", "list", "--json"])

    # 11. display list
    if action == "display_list":
        return _run_usecomputer(["usecomputer", "display", "list", "--json"])

    # 12. desktop list
    if action == "desktop_list":
        return _run_usecomputer(["usecomputer", "desktop", "list", "--json"])

    return JSONResponse(
        {"ok": False, "error": f"unsupported action: {action}"},
        status_code=400,
    )


@app.get("/agent/security")
async def agent_security_config():
    """查询当前安全策略"""
    if not _computer_tools_loaded:
        return JSONResponse({"ok": False, "error": "computer tools not loaded"}, status_code=503)
    return {"ok": True, **get_security_config()}


@app.post("/agent/security/allow_root")
async def agent_add_allowed_root(request: Request):
    """添加允许的根目录（扩大工作区范围）"""
    if not _computer_tools_loaded:
        return JSONResponse({"ok": False, "error": "computer tools not loaded"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)
    path = (body.get("path") or "").strip()
    if not path:
        return JSONResponse({"ok": False, "error": "missing path"}, status_code=400)
    from adu_computer_tools import add_allowed_root
    add_allowed_root(path)
    return {"ok": True, "message": f"已添加允许根目录：{path}"}


@app.post("/agent/git")
async def agent_git(request: Request):
    """Git 操作"""
    quota_err = _agent_check_quota(request)
    if quota_err:
        return quota_err
    try:
        body = await request.json()
    except Exception:
        body = {}
    action = body.get("action", "status")  # status / commit / log
    path = body.get("path", "~/GPTsora")
    message = body.get("message", "AI update")

    bridge, err = await _ensure_openclaw_or_503()
    if err:
        return err

    if action == "commit":
        output = await bridge.git_commit(path, message)
    elif action == "log":
        output = await bridge.exec(f'cd {path} && git log --oneline -10 2>&1')
    else:
        output = await bridge.git_status(path)

    return {"ok": True, "action": action, "output": output}


# ── Fix OpenAPI 500 (Pydantic v2 ForwardRef) ----
try:
    V2RegisterReq.model_rebuild()
    V2LoginReq.model_rebuild()
    V2CommentReq.model_rebuild()
    V2ShareReq.model_rebuild()
    V2RecoEventReq.model_rebuild()
    V2DMMessageReq.model_rebuild()
except Exception:
    pass

# ===================================================================
# ✅ Plan A: Memory Module Overrides
# - Keep rest of server_session.py unchanged, but route all memory calls
#   through the dedicated memory_module.py engine (async writes).
# ===================================================================

# Preserve legacy implementations (in case memory_module isn't available)
_LEGACY_memory_add = globals().get('memory_add')
_LEGACY_memory_search = globals().get('memory_search')
_LEGACY_memory_build_context = globals().get('memory_build_context')
_LEGACY_memory_facts_save = globals().get('memory_facts_save')
_LEGACY_memory_facts_list = globals().get('memory_facts_list')
_LEGACY_memory_facts_build_prompt = globals().get('memory_facts_build_prompt')
_LEGACY_extract_and_save_memory_facts = globals().get('extract_and_save_memory_facts')

def _should_memory_add(text: str) -> bool:
    """Unified heuristic (overrides earlier duplicated definitions)."""
    try:
        if should_store_memory is not None:
            return bool(should_store_memory(text, min_chars=MEMORY_MIN_CHARS))
    except Exception:
        pass
    # fallback to legacy if present
    try:
        legacy = globals().get('_LEGACY__should_memory_add') or globals().get('__should_memory_add')
        if callable(legacy):
            return bool(legacy(text))
    except Exception:
        pass
    t = (text or '').strip()
    return bool(t) and len(t) >= max(8, int(MEMORY_MIN_CHARS)) and '```' not in t

def memory_add(user_key: str, text: str) -> None:
    try:
        if MEMORY_ENGINE is not None:
            MEMORY_ENGINE.add_vector(user_key, text)
            return
    except Exception:
        pass
    if callable(_LEGACY_memory_add):
        try:
            _LEGACY_memory_add(user_key, text)  # type: ignore
        except Exception:
            return

def memory_search(user_key: str, query: str, k: int, min_score: float) -> List[Dict[str, Any]]:
    try:
        if MEMORY_ENGINE is not None:
            return MEMORY_ENGINE.search_vectors(user_key, query, k=k, min_score=min_score)
    except Exception:
        pass
    if callable(_LEGACY_memory_search):
        try:
            return _LEGACY_memory_search(user_key, query, k, min_score)  # type: ignore
        except Exception:
            return []
    return []

def memory_build_context(user_key: str, query: str, k: int = MEMORY_TOP_K_DEFAULT, min_score: float = MEMORY_MIN_SCORE_DEFAULT) -> str:
    try:
        if MEMORY_ENGINE is not None:
            return MEMORY_ENGINE.build_vector_context(user_key, query, k=k, min_score=min_score)
    except Exception:
        pass
    if callable(_LEGACY_memory_build_context):
        try:
            return _LEGACY_memory_build_context(user_key, query, k, min_score)  # type: ignore
        except Exception:
            return ""
    return ""

def memory_facts_save(user_key: str, content: str, tags: str = "", importance: int = 1) -> None:
    try:
        if MEMORY_ENGINE is not None:
            MEMORY_ENGINE.facts_save(user_key, content, tags=tags, importance=importance)
            return
    except Exception:
        pass
    if callable(_LEGACY_memory_facts_save):
        try:
            _LEGACY_memory_facts_save(user_key, content, tags=tags, importance=importance)  # type: ignore
        except Exception:
            return

def memory_facts_list(user_key: str, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        if MEMORY_ENGINE is not None:
            return MEMORY_ENGINE.facts_list(user_key, limit=limit)
    except Exception:
        pass
    if callable(_LEGACY_memory_facts_list):
        try:
            return _LEGACY_memory_facts_list(user_key, limit)  # type: ignore
        except Exception:
            return []
    return []

def memory_facts_build_prompt(user_key: str, limit: int = MEMORY_FACTS_PROMPT_LIMIT) -> str:
    try:
        if MEMORY_ENGINE is not None:
            return MEMORY_ENGINE.facts_build_prompt(user_key, limit=limit)
    except Exception:
        pass
    if callable(_LEGACY_memory_facts_build_prompt):
        try:
            return _LEGACY_memory_facts_build_prompt(user_key, limit)  # type: ignore
        except Exception:
            return ""
    return ""

def extract_and_save_memory_facts(user_key: str, user_msg: str, ai_reply: str) -> None:
    """Override: facts extraction via MemoryEngine (sync; caller may spawn thread)."""
    try:
        if MEMORY_ENGINE is not None:
            MEMORY_ENGINE.extract_and_save_facts(user_key, user_msg, ai_reply)
            return
    except Exception:
        pass
    if callable(_LEGACY_extract_and_save_memory_facts):
        try:
            _LEGACY_extract_and_save_memory_facts(user_key, user_msg, ai_reply)  # type: ignore
        except Exception:
            return

# ══════════════════════════════════════════════════════════════════════
# ✅ 四级压缩记忆系统（L0/L1/L2/L3 + MemGPT换入换出）
# 依赖：memory_module.py 中 CompressionEngine（monkey-patch 到 MemoryEngine）
# 原有 memory_add / memory_build_context / extract_and_save_memory_facts 完全不动
# ══════════════════════════════════════════════════════════════════════

def _comp_append_turn(user_key: str, conv_id: str, role: str, content: str) -> None:
    """写入L0原始轮次（每轮对话后调用）。"""
    try:
        if MEMORY_ENGINE is not None and hasattr(MEMORY_ENGINE, "append_turn"):
            MEMORY_ENGINE.append_turn(user_key, conv_id, role, content)
    except Exception:
        pass

def _comp_maybe_compress(user_key: str, conv_id: str = "") -> None:
    """非阻塞触发压缩流水线 L0→L1→L2（每轮对话后调用）。"""
    try:
        if MEMORY_ENGINE is not None and hasattr(MEMORY_ENGINE, "maybe_compress"):
            MEMORY_ENGINE.maybe_compress(user_key, conv_id)
    except Exception:
        pass

def _comp_build_context(user_key: str, conv_id: str = "", query: str = "") -> str:
    """构建四级压缩上下文（MemGPT换入换出），注入 system prompt。"""
    try:
        if MEMORY_ENGINE is not None and hasattr(MEMORY_ENGINE, "build_compression_context"):
            return MEMORY_ENGINE.build_compression_context(user_key, conv_id=conv_id, query=query)
    except Exception:
        pass
    return ""

def _comp_l3_extract(user_key: str, user_msg: str, ai_reply: str) -> None:
    """异步提取结构化信息存入L3（与 facts 并行运行）。"""
    try:
        if MEMORY_ENGINE is not None and hasattr(MEMORY_ENGINE, "l3_extract_async"):
            MEMORY_ENGINE.l3_extract_async(user_key, user_msg, ai_reply)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════
# 🚗  AI 视觉导航  /vision/analyze  +  /vision/stream
# iOS AIVisionNavView 调用。
# 架构：GPT-4o vision (gpt-4o，detail=low) → 流式文字 → 前端并行 TTS
#
# /vision/analyze  — 非流式，返回完整 JSON（兼容旧版 & 测试）
# /vision/stream   — SSE 流式，iOS 边收 token 边合成语音（延迟 ~1s）
# ══════════════════════════════════════════════════════════════════════

_VISION_SYSTEM_DRIVING = """你是车载AI视觉大脑，分析摄像头画面并形成“空间记忆 + 决策”。
你必须先用一句极短中文口语播报，再在最后一行输出一个 JSON，不要输出 Markdown。
播报规则：
- 发现行人/障碍/危险：以「注意，」开头，10字内说明
- 发现限速牌：说「限速X公里」
- 发现路口/红绿灯/转弯提示：简短说明
- 正常路况：可以说「继续观察」或不播报
JSON 固定结构：
{"desc":"一句话场景描述","objects":[{"label":"物体名","direction":"前方|左前方|右前方|近前方","risk":"safe|caution|danger","confidence":0.0}],"danger":"危险描述或空字符串","speed_limit":数字或null,"lane":true或false,"decision":"下一步决策","memory":"空间记忆摘要"}
只用中文，先播报再输出JSON。"""

_VISION_SYSTEM_MANUAL = """你是实时视觉大脑。分析这张摄像头画面，描述外部世界、空间位置、风险和下一步决策。
最后一行输出 JSON：
{"desc":"一句话场景描述","objects":[{"label":"物体名","direction":"前方|左前方|右前方|近前方","risk":"safe|caution|danger","confidence":0.0}],"danger":"危险或空字符串","speed_limit":数字或null,"lane":false,"decision":"下一步决策","memory":"空间记忆摘要"}"""

VISION_MODEL_DEFAULT = (os.getenv("VISION_MODEL") or ("qwen-vl-plus" if (not OPENAI_API_KEY and DASHSCOPE_API_KEY) else "gpt-4o")).strip()

def _vision_llm_config() -> Optional[Tuple[str, Dict[str, str], str]]:
    """Return (chat_completions_url, headers, model) for the vision brain.

    Prefer OpenAI GPT-4o if OPENAI_API_KEY exists; otherwise use DashScope's
    OpenAI-compatible vision model when DASHSCOPE_API_KEY is configured.
    """
    if OPENAI_API_KEY:
        return (
            "https://api.openai.com/v1/chat/completions",
            {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            VISION_MODEL_DEFAULT or "gpt-4o",
        )
    if DASHSCOPE_API_KEY:
        return (
            f"{DASHSCOPE_BASE_URL}/chat/completions",
            {
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            VISION_MODEL_DEFAULT or "qwen-vl-plus",
        )
    return None



def _build_vision_messages(base64_image: str, mode: str) -> list:
    """构建 GPT-4o vision messages，detail=low 节省 token"""
    system = _VISION_SYSTEM_DRIVING if mode == "driving" else _VISION_SYSTEM_MANUAL
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low"      # low = 固定 85 tokens，最经济
            }},
            {"type": "text", "text": "分析当前画面，先播报，再输出JSON。" if mode == "driving"
             else "详细描述路况和安全情况，再输出JSON。"}
        ]}
    ]


@app.post("/vision/analyze")
async def vision_analyze(req: Request):
    """
    非流式视觉分析（兼容旧版 & fallback）。
    Body: { "base64": "...", "mode": "driving"|"manual" }
    或 Anthropic 格式（直接透传 messages）。
    """
    vision_cfg = _vision_llm_config()
    if vision_cfg is None:
        raise HTTPException(status_code=503, detail="No vision model API key configured")

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # ── 两种格式兼容 ──
    # 格式A（iOS 新版）: { base64, mode }
    # 格式B（旧版 Anthropic 透传）: { messages, model, max_tokens }
    if "base64" in body:
        b64 = body["base64"]
        mode = body.get("mode", "driving")
        messages = _build_vision_messages(b64, mode)
    elif "messages" in body:
        # 旧版直接透传，转换 Anthropic → OpenAI 格式
        messages = body.get("messages", [])
    else:
        raise HTTPException(status_code=400, detail="Need 'base64' or 'messages'")

    import httpx
    vision_url, vision_headers, vision_model = vision_cfg
    payload = {
        "model": vision_model,
        "max_tokens": 260,
        "stream": False,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                vision_url,
                json=payload,
                headers=vision_headers,
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error {r.status_code}: {r.text[:300]}")

    data = r.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})

    return JSONResponse({
        "ok": True,
        "text": text,
        "usage": usage,
        # 兼容旧版 Anthropic 格式（iOS fallback 解析）
        "content": [{"type": "text", "text": text}],
    })


@app.post("/vision/stream")
async def vision_stream(req: Request):
    """
    流式视觉分析 — GPT-4o vision SSE。
    iOS 边收 token 边按句子切割送 TTS，首字延迟 ~0.4s。

    Body: { "base64": "...", "mode": "driving"|"manual" }
    Response: text/event-stream  (OpenAI SSE 格式直通)
    """
    vision_cfg = _vision_llm_config()
    if vision_cfg is None:
        raise HTTPException(status_code=503, detail="No vision model API key configured")

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    b64  = body.get("base64", "")
    mode = body.get("mode", "driving")

    if not b64:
        raise HTTPException(status_code=400, detail="Missing 'base64' field")

    messages = _build_vision_messages(b64, mode)

    vision_url, vision_headers, vision_model = vision_cfg
    payload = {
        "model": vision_model,
        "max_tokens": 260,
        "stream": True,
        "messages": messages,
    }

    import httpx

    async def generate():
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                async with client.stream(
                    "POST",
                    vision_url,
                    json=payload,
                    headers=vision_headers,
                ) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        yield f"data: {{\"error\": \"{resp.status_code}\"}}\n\n"
                        return
                    async for chunk in resp.aiter_text():
                        if chunk:
                            yield chunk
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)[:100]}\"}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # 关闭 nginx 缓冲，确保实时推送
        },
    )


# ── /vision/tts  并行 TTS 代理 ───────────────────────────────────────
# iOS 收到完整句子后调此接口合成 MP3，边下边播。
# Body: { "text": "注意，前方有行人", "voice": "nova", "speed": 1.1 }
# ─────────────────────────────────────────────────────────────────────
@app.post("/vision/tts")
async def vision_tts(req: Request):
    """OpenAI TTS-1 代理，供视觉导航并行语音合成使用"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    text  = (body.get("text") or "").strip()
    voice = body.get("voice", "nova")      # nova=女声自然，alloy/echo/fable/onyx/shimmer
    speed = float(body.get("speed", 1.1))
    fmt   = body.get("format", "mp3")

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text'")
    if len(text) > 300:
        text = text[:300]   # TTS 单次上限保护

    import httpx

    payload = {
        "model": "tts-1",          # tts-1 延迟最低；tts-1-hd 质量更好但慢 ~1s
        "input": text,
        "voice": voice,
        "speed": max(0.25, min(4.0, speed)),
        "response_format": fmt,
    }

    async def stream_audio():
        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/audio/speech",
                json=payload,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            ) as resp:
                if resp.status_code != 200:
                    return
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    yield chunk

    media_type = "audio/mpeg" if fmt == "mp3" else "audio/opus"
    return StreamingResponse(
        stream_audio(),
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


# ════════════════════════════════════════════════════════════════════════
# 阿杜屏幕共享 WebSocket 路由 v0.1
# Mac 端 (adu-bridge) 推帧 → 后端中继 → iOS 订阅
# Mac 端 接收控制指令 → CGEvent 执行
# ════════════════════════════════════════════════════════════════════════

import asyncio as _asyncio_screen
import json as _json_screen

# user_key → {"mac": WebSocket | None, "ios": Set[WebSocket], "control_mac": WebSocket | None}
SCREEN_POOL: Dict[str, Dict[str, Any]] = {}
_SCREEN_LOCK = _asyncio_screen.Lock()


async def _screen_pool_get(user_key: str) -> Dict[str, Any]:
    async with _SCREEN_LOCK:
        if user_key not in SCREEN_POOL:
            SCREEN_POOL[user_key] = {
                "mac": None,
                "ios": set(),
                "control_mac": None,
            }
        return SCREEN_POOL[user_key]


# ─────────── /ws/screen/push/{user_key} — Mac 推帧进来 ───────────

@app.websocket("/ws/screen/push/{user_key}")
async def ws_screen_push(ws: WebSocket, user_key: str):
    await ws.accept()
    pool = await _screen_pool_get(user_key)
    pool["mac"] = ws
    print(f"[screen-push] Mac connected user={user_key[:8]}")

    # 通知所有 iOS 订阅者:Mac 上线了
    for sub in list(pool["ios"]):
        try:
            await sub.send_text(_json_screen.dumps({"type": "status", "mac_online": True}))
        except Exception:
            pass

    try:
        while True:
            msg = await ws.receive_text()
            # 广播帧给所有 iOS 订阅者
            dead = []
            for sub in list(pool["ios"]):
                try:
                    await sub.send_text(msg)
                except Exception:
                    dead.append(sub)
            for d in dead:
                pool["ios"].discard(d)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[screen-push] err: {e}")
    finally:
        if pool.get("mac") is ws:
            pool["mac"] = None
        # 通知所有 iOS:Mac 下线
        for sub in list(pool["ios"]):
            try:
                await sub.send_text(_json_screen.dumps({"type": "status", "mac_online": False}))
            except Exception:
                pass
        print(f"[screen-push] Mac left user={user_key[:8]}")


# ─────────── /ws/screen/subscribe/{user_key} — iOS 订阅帧 ───────────

@app.websocket("/ws/screen/subscribe/{user_key}")
async def ws_screen_subscribe(ws: WebSocket, user_key: str):
    await ws.accept()
    pool = await _screen_pool_get(user_key)
    pool["ios"].add(ws)
    print(f"[screen-sub] iOS subscribed user={user_key[:8]} total={len(pool['ios'])}")

    try:
        # 初始 status
        mac_online = pool.get("mac") is not None
        await ws.send_text(_json_screen.dumps({"type": "status", "mac_online": mac_online}))

        while True:
            msg = await ws.receive_text()
            # iOS 端发来的控制指令 → 转给 Mac
            try:
                obj = _json_screen.loads(msg)
                if obj.get("type") == "control":
                    ctrl = pool.get("control_mac") or pool.get("mac")
                    if ctrl:
                        await ctrl.send_text(_json_screen.dumps(obj.get("payload") or {}))
                    else:
                        # 回包告知 iOS:Mac 没连
                        await ws.send_text(_json_screen.dumps({
                            "type": "control_result",
                            "payload": {"ok": False, "error": "Mac offline"},
                        }))
            except Exception:
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[screen-sub] err: {e}")
    finally:
        pool["ios"].discard(ws)
        print(f"[screen-sub] iOS left user={user_key[:8]} remaining={len(pool['ios'])}")


# ─────────── /ws/control/{user_key} — Mac 订阅控制指令(双向通道) ───────────

@app.websocket("/ws/control/{user_key}")
async def ws_screen_control(ws: WebSocket, user_key: str):
    await ws.accept()
    pool = await _screen_pool_get(user_key)
    pool["control_mac"] = ws
    print(f"[control] Mac control channel user={user_key[:8]}")

    try:
        while True:
            # Mac 执行完指令的回包 → 广播给所有 iOS 订阅者
            msg = await ws.receive_text()
            try:
                payload = _json_screen.loads(msg)
            except Exception:
                payload = {"raw": msg}

            for sub in list(pool["ios"]):
                try:
                    await sub.send_text(_json_screen.dumps({
                        "type": "control_result",
                        "payload": payload,
                    }))
                except Exception:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[control] err: {e}")
    finally:
        if pool.get("control_mac") is ws:
            pool["control_mac"] = None
        print(f"[control] Mac control left user={user_key[:8]}")


# ─────────── HTTP 辅助:状态查询 + 直接下发指令(给前端调试用) ───────────

@app.get("/screen/status/{user_key}")
async def screen_status(user_key: str):
    pool = SCREEN_POOL.get(user_key) or {}
    return {
        "mac_online": pool.get("mac") is not None,
        "control_online": pool.get("control_mac") is not None,
        "ios_subscribers": len(pool.get("ios", set())),
    }


@app.post("/screen/control/{user_key}")
async def screen_control_http(user_key: str, req: Request):
    pool = SCREEN_POOL.get(user_key) or {}
    ctrl = pool.get("control_mac") or pool.get("mac")
    if not ctrl:
        return JSONResponse({"ok": False, "error": "Mac 未连接"}, status_code=503)
    try:
        body = await req.json()
    except Exception:
        body = {}
    try:
        await ctrl.send_text(_json_screen.dumps(body))
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


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























































































































