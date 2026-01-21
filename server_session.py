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
import logging
import threading
import subprocess
import queue
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Iterator, List
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, UploadFile, File, Form, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse, Response
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
    REMIX_REALTIME_INSTRUCTIONS = "You are Solara Remix Assistant. Output a stable English remix prompt."

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

# ---- Chat + TTS streaming ----
CHAT_MODEL_DEFAULT = (os.getenv("CHAT_MODEL") or "gpt-5").strip()
CHAT_STREAM_TIMEOUT_SEC = int(os.getenv("CHAT_STREAM_TIMEOUT_SEC") or "180")
CHAT_STREAM_CHUNK_TIMEOUT_SEC = float(os.getenv("CHAT_STREAM_CHUNK_TIMEOUT_SEC") or "25")
CHAT_JOB_TTL_SEC = int(os.getenv("CHAT_JOB_TTL_SEC") or "1800")  # 30min

# TTS for chat streaming (sentence-by-sentence)
TTS_SPEECH_URL = "https://api.openai.com/v1/audio/speech"
TTS_MODEL_DEFAULT = (os.getenv("TTS_MODEL") or "gpt-4o-mini-tts").strip()
TTS_VOICE_DEFAULT = (os.getenv("TTS_VOICE") or REALTIME_VOICE_DEFAULT).strip()
TTS_INSTRUCTIONS_DEFAULT = (os.getenv("TTS_INSTRUCTIONS") or "Speak in a clear, friendly tone.").strip()
TTS_LIVE_TTL_SEC = int(os.getenv("TTS_LIVE_TTL_SEC") or "1800")  # 30min

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


class ChatJob:
    def __init__(self, chat_id: str, tts_id: str) -> None:
        self.chat_id = chat_id
        self.tts_id = tts_id
        self.created = time.time()
        self.last_touch = self.created

        self.events: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self.done_evt = threading.Event()
        self.full_text = ""
        self.error: Optional[str] = None

    def touch(self) -> None:
        self.last_touch = time.time()

    def push_event(self, typ: str, data: Dict[str, Any]) -> None:
        self.touch()
        self.events.put({"event": typ, "data": data})

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

def _create_chat_job() -> ChatJob:
    _cleanup_chat_jobs()
    chat_id = uuid.uuid4().hex
    tts_id = _create_live_tts_session()
    job = ChatJob(chat_id=chat_id, tts_id=tts_id)
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

def bg_sora_worker(job_id: str, timeout_sec: int = 1800) -> None:
    job = SORA_JOBS.get(job_id)
    if not job:
        return

    try:
        mode = (job.get("mode") or "").strip().lower()
        if not mode:
            mode = "remix" if job.get("remix_base_video_id") else "create"

        ref_path = job.get("ref_path")
        ref_mime = job.get("ref_mime")

        if mode == "remix":
            base_id = _normalize_video_id(job.get("remix_base_video_id") or "")
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
                job["progress"] = max(int(job.get("progress") or 0), prog)
            except Exception:
                prog = None

            if prog is not None and prog >= 99 and not finishing_grace_used:
                deadline = max(deadline, time.time() + 600)
                finishing_grace_used = True

            if status in ("completed", "succeeded", "done", "success"):
                job["url"] = f"/video/stream/{job_id}"
                job["status"] = "done"
                job["progress"] = 100
                return

            if status in ("failed", "error", "cancelled", "canceled"):
                raise RuntimeError(f"sora failed: {status}")

            if time.time() > deadline:
                raise TimeoutError("sora timeout")

            time.sleep(2)

    except Exception as e:
        job["error"] = str(e)
        job["status"] = "failed"
        log.exception("[SORA] job=%s FAILED: %s", job_id, e)

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
    _ENDERS = set(list("。！？!?"))
    _HARD_NEWLINE = True

    def __init__(self, min_chars: int = 18, max_chars: int = 220) -> None:
        self.buf = ""
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)

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

    def _find_cut(self) -> int:
        # 1) newline cut (fast response for multi-line)
        if self._HARD_NEWLINE and "\n" in self.buf:
            idx = self.buf.find("\n")
            if idx >= self.min_chars:
                return idx + 1

        # 2) punctuation-based cut
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


def _tts_stream_fulltext_to_live(tts_id: str, text: str, voice: str, fmt: str = "mp3") -> None:
    """
    Used by /tts_prepare: call OpenAI once, stream bytes to LiveMP3Stream.
    """
    s = _get_live_tts(tts_id)
    if not s:
        return
    try:
        voice = _normalize_tts_voice(voice)
        fmt = (fmt or "mp3").strip().lower()
        payload: Dict[str, Any] = {"model": TTS_MODEL_DEFAULT, "voice": voice, "input": (text or "").strip()}
        inst = (TTS_INSTRUCTIONS_DEFAULT or "").strip()
        if inst:
            payload["instructions"] = inst
        if fmt and fmt != "mp3":
            payload["response_format"] = fmt

        r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=payload, stream=True, timeout=120)
        rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
        log.info("[TTS.prepare] voice=%s fmt=%s -> %s rid=%s", voice, fmt, r.status_code, rid)

        if r.status_code >= 400:
            try:
                s.push(b"")  # no-op
            finally:
                s.close()
            return

        for chunk in r.iter_content(chunk_size=64 * 1024):
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


def _stream_openai_text(model: str, messages: List[Dict[str, str]]) -> Iterator[str]:
    """
    Try Responses API first, then Chat Completions as fallback.
    """
    model = (model or CHAT_MODEL_DEFAULT).strip() or CHAT_MODEL_DEFAULT

    # 1) Responses API
    try:
        url = "https://api.openai.com/v1/responses"
        inp = []
        for m in messages:
            role = (m.get("role") or "user").strip()
            content = (m.get("content") or "").strip()
            inp.append({"role": role, "content": [{"type": "input_text", "text": content}]})
        body = {"model": model, "input": inp, "stream": True}
        r = requests.post(url, headers=_openai_headers(stream=True), json=body, stream=True, timeout=CHAT_STREAM_TIMEOUT_SEC)
        if r.status_code >= 400:
            raise RuntimeError(f"responses error {r.status_code}")

        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                delta = _extract_delta_from_sse_json(obj)
                if delta:
                    yield delta
        try:
            r.close()
        except Exception:
            pass
        return
    except Exception:
        pass

    # 2) Chat Completions
    url = "https://api.openai.com/v1/chat/completions"
    body = {"model": model, "messages": messages, "stream": True}
    r = requests.post(url, headers=_openai_headers(stream=True), json=body, stream=True, timeout=CHAT_STREAM_TIMEOUT_SEC)
    if r.status_code >= 400:
        raise RuntimeError(f"openai chat error {r.status_code}: {_short(r.text, 800)}")

    for raw in r.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
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
            yield delta

    try:
        r.close()
    except Exception:
        pass


def _spawn_chat_worker(job: ChatJob, model: str, messages: List[Dict[str, str]], tts_voice: str) -> None:
    """
    Background worker:
      - stream GPT text deltas -> job.events (SSE)
      - segment sentences -> call TTS in parallel -> push mp3 bytes into LiveMP3Stream
    """
    def run():
        live = _get_live_tts(job.tts_id)
        if not live:
            job.error = "live_tts_not_found"
            job.push_event("error", {"message": "live_tts_not_found"})
            job.close_events()
            return

        seg_q: "queue.Queue[Optional[str]]" = queue.Queue()
        segmenter = SentenceSegmenter()

        # TTS segment consumer thread
        def tts_consumer():
            try:
                while True:
                    seg = seg_q.get()
                    if seg is None:
                        break
                    seg = (seg or "").strip()
                    if not seg:
                        continue
                    try:
                        mp3 = _tts_bytes(seg, voice=tts_voice, fmt="mp3")
                        if mp3:
                            live.push(mp3)
                    except Exception as e:
                        log.warning("[chat-tts] segment tts failed: %s", e)
                        continue
            finally:
                try:
                    live.close()
                except Exception:
                    pass

        threading.Thread(target=tts_consumer, daemon=True).start()

        # meta event first (client uses this to start audio)
        job.push_event("meta", {
            "chat_id": job.chat_id,
            "tts_id": job.tts_id,
            "tts_url": f"/tts/live/{job.tts_id}.mp3",
            "model": model,
        })

        try:
            full = []
            for delta in _stream_openai_text(model=model, messages=messages):
                job.push_event("delta", {"delta": delta})
                full.append(delta)

                for seg in segmenter.feed(delta):
                    seg_q.put(seg)

            for seg in segmenter.flush():
                seg_q.put(seg)

            seg_q.put(None)

            job.full_text = "".join(full).strip()
            job.push_event("done", {"text": job.full_text})

        except Exception as e:
            job.error = str(e)
            job.push_event("error", {"message": str(e)})
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


def _sse_pack(event: str, data_obj: Dict[str, Any]) -> str:
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

app = FastAPI(title="Solara Backend", lifespan=lifespan)

# ✅ business routers
app.include_router(billing_router)
app.include_router(auth_router)
app.include_router(home_chat_router)
app.include_router(media_upload_router)
app.include_router(home_automation_router)

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
    if req.url.path in ("/video", "/video/remix", "/rt/intent", "/session", "/solara/photo", "/chat/prepare"):
        ip = req.client.host if req.client else "-"
        log.info("[AUDIT] ip=%s %s %s -> %s in %dms",
                 ip, req.method, req.url.path, resp.status_code, int((time.time()-t0)*1000))
    return resp

# -----------------------------
# ✅ Chat prepare (server-side TTS, streaming text via SSE)
# -----------------------------
@app.post("/chat/prepare")
async def chat_prepare(request: Request):
    """
    ✅ 新用法（推荐）：
      POST /chat/prepare
      JSON:
      {
        "model": "gpt-5",
        "messages": [{"role":"user","content":"..."}],
        "tts_voice": "realtime"   // optional
      }

    返回：
      {
        "ok": true,
        "chat_id": "...",
        "events_url": "/chat/events/<chat_id>",
        "tts_url": "/tts/live/<tts_id>.mp3"
      }

    客户端：
      - 先 AVPlayer 播放 tts_url
      - 再用 EventSource/SSE 订阅 events_url，边收 delta 边渲染文本
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    model = (body.get("model") or CHAT_MODEL_DEFAULT).strip() or CHAT_MODEL_DEFAULT
    msgs = body.get("messages") or []
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

    tts_voice = (body.get("tts_voice") or body.get("voice") or "realtime").strip()

    job = _create_chat_job()
    _spawn_chat_worker(job=job, model=model, messages=messages, tts_voice=tts_voice)

    return {
        "ok": True,
        "chat_id": job.chat_id,
        "events_url": f"/chat/events/{job.chat_id}",
        "tts_url": f"/tts/live/{job.tts_id}.mp3",
    }

@app.get("/chat/events/{chat_id}")
async def chat_events(chat_id: str, request: Request):
    """
    SSE stream:
      event: meta  -> {chat_id, tts_url, model}
      event: delta -> {delta:"..."}
      event: done  -> {text:"..."}
      event: error -> {message:"..."}
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
                yield ": ping\n\n"

            try:
                item = job.events.get(timeout=0.5)
            except Exception:
                item = None

            if item is None:
                if job.done_evt.is_set():
                    break
                continue

            ev = item.get("event")
            data = item.get("data") or {}
            yield _sse_pack(ev, data)

        yield ": done\n\n"

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
    if len(text) > 8000:
        return JSONResponse({"ok": False, "error": "text too long"}, status_code=413)

    voice = (body.get("voice") or "realtime").strip()
    fmt = (body.get("format") or body.get("response_format") or "mp3")
    fmt = str(fmt).strip().lower()

    tts_id = _create_live_tts_session()
    threading.Thread(
        target=_tts_stream_fulltext_to_live,
        args=(tts_id, text, voice, fmt),
        daemon=True
    ).start()

    return {"ok": True, "tts_id": tts_id, "url": f"/tts/live/{tts_id}.mp3", "expires_in": TTS_LIVE_TTL_SEC}

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

    base_video_id = _normalize_video_id((body.get("base_video_id") or body.get("video_id") or "").strip())
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

    _cleanup_recent()
    prompt_idem = f"REMIX|{base_video_id}|{instruction}"
    idem = _idem_key(ip, prompt_idem, "", "")
    rec = RECENT_KEYS.get(idem)
    if rec:
        old_job_id = rec["job_id"]
        old_job = SORA_JOBS.get(old_job_id)
        if _job_is_active(old_job):
            return {
                "ok": True,
                "job_id": old_job_id,
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

    _spawn_sora_job(job_id, timeout_sec=1800)

    return {
        "ok": True,
        "job_id": job_id,
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
    image_file: UploadFile = File(None),
    video_file: UploadFile = File(None),
):
    ip = request.client.host if request.client else "unknown"

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
        remix_instruction = (remix.get("instruction", "") or "").strip()
        prompt_effective = remix_instruction or "Make subtle improvements. Keep everything else the same."
        prompt_idem = f"REMIX|{remix_base}|{prompt_effective}"

    if not prompt_effective.strip():
        prompt_effective = "Generate a video based on the reference media." if (raw_img or tmp_video_path) else "Generate a video."
        prompt_idem = prompt_effective

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
        "provider": "sora",

        "ref_path": None,
        "ref_mime": None,

        "mode": mode,
        "remix_base_video_id": remix_base or None,
        "remix_instruction": remix_instruction or None,
        "openai_status": None,
    }
    RECENT_KEYS[idem] = {"job_id": job_id, "ts": time.time()}

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

    return {"ok": True, "job_id": job_id, "status_url": f"/video/status/{job_id}", "status": "queued"}

# -----------------------------
# SORA: status / stream / content
# -----------------------------
@app.get("/video/status/{job_id}")
def video_status(job_id: str):
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)

    vid = _normalize_video_id(job.get("video_id") or "")

    return {
        "ok": True,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "url": job.get("url"),
        "error": job.get("error"),

        "video_id": vid,
        "openai_video_id": vid,
        "remixed_from_video_id": job.get("remix_base_video_id") if (job.get("mode") == "remix") else None,

        "provider": "sora",
        "seconds": SORA_SECONDS_DEFAULT,
        "size": SORA_SIZE_DEFAULT,
        "mode": job.get("mode"),
        "remix_base_video_id": job.get("remix_base_video_id"),
        "remix_instruction": job.get("remix_instruction"),
        "openai_status": job.get("openai_status"),
    }

@app.get("/video/stream/{job_id}")
def video_stream(job_id: str, range_header: Optional[str] = Header(None, alias="Range")):
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)

    vid = _normalize_video_id(job.get("video_id") or "")
    if not vid:
        return JSONResponse({"ok": False, "error": "video not ready"}, status_code=409)

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

    SORA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": prompt,
        "prompt_raw": prompt,

        "url": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "provider": "sora",

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

    text = (body.get("text") or body.get("input") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "missing text"}, status_code=400)
    if len(text) > 8000:
        return JSONResponse({"ok": False, "error": "text too long"}, status_code=413)

    voice = _normalize_tts_voice((body.get("voice") or "").strip() or TTS_VOICE_DEFAULT)
    model = (body.get("model") or TTS_MODEL_DEFAULT).strip()
    fmt = "mp3" if force_mp3 else (body.get("format") or body.get("response_format") or "mp3")
    fmt = str(fmt).strip().lower()

    instructions = (body.get("instructions") or TTS_INSTRUCTIONS_DEFAULT or "").strip()
    speed = body.get("speed")

    payload: Dict[str, Any] = {"model": model, "voice": voice, "input": text}
    if instructions:
        payload["instructions"] = instructions
    if fmt and fmt != "mp3":
        payload["response_format"] = fmt
    if isinstance(speed, (int, float)) and 0.25 <= float(speed) <= 4.0:
        payload["speed"] = float(speed)

    try:
        r = requests.post(TTS_SPEECH_URL, headers=_tts_headers(), json=payload, timeout=120)
        rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
        log.info("[TTS.bytes] model=%s voice=%s fmt=%s -> %s rid=%s bytes=%s",
                 model, voice, fmt, r.status_code, rid, len(r.content or b""))
    except Exception as e:
        return JSONResponse({"ok": False, "error": "tts_request_failed", "detail": str(e)}, status_code=502)

    if r.status_code >= 400:
        detail: Any
        try:
            detail = r.json()
        except Exception:
            detail = (r.text or "")[:1200]
        return JSONResponse({"ok": False, "error": "openai_tts_error", "status": r.status_code, "detail": detail}, status_code=502)

    data = r.content or b""
    if len(data) < 128:
        return JSONResponse({"ok": False, "error": "tts_empty_audio", "bytes": len(data)}, status_code=502)

    return Response(content=data, media_type=_tts_media_type(fmt))

@app.post("/tts_stream.mp3")
async def tts_stream_mp3(request: Request):
    return await _tts_bytes_impl(request, force_mp3=True)

@app.post("/tts_stream")
async def tts_stream(request: Request):
    return await _tts_bytes_impl(request, force_mp3=False)

@app.post("/tts")
async def tts(request: Request):
    return await _tts_bytes_impl(request, force_mp3=False)

# -----------------------------
# Health
# -----------------------------
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_session:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
        access_log=False,
    )


