# ================================
# server_session.py  (FULL)
# Sora(高端8s) + Luma(普通10s/720p) 两条线完全分离
# ✅ 修复：Sora 下载/播放统一走 /v1/videos/{video_id}/content（官方方式）
# ✅ 保留：Luma add-audio 不动
# ✅ 保留：billing include_router 不动
# ✅ 修复：接入 auth_router，/auth/apple 返回 access_token（走 auth.py）
# ✅ 保留：原 /auth/apple 实现改名为 /auth/apple_legacy，避免冲突
# ================================

import os
import io
import re
import json
import uuid
import time
import base64
import hashlib
import logging
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import requests
from requests.exceptions import ReadTimeout
from fastapi import FastAPI, UploadFile, File, Form, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ✅ billing（两行之一：import）
from billing import router as billing_router

# ✅ auth（新增：让 /auth/apple 返回 access_token）
from auth import router as auth_router

# ============== ENV ==============
load_dotenv()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
SORA_API_KEY = (os.getenv("SORA_API_KEY") or OPENAI_API_KEY).strip()
OPENAI_ORG_ID = (os.getenv("OPENAI_ORG_ID") or "").strip()

REALTIME_MODEL_DEFAULT = (os.getenv("REALTIME_MODEL") or "gpt-realtime-mini").strip()
REALTIME_VOICE_DEFAULT = (os.getenv("REALTIME_VOICE") or "alloy").strip()

TEXT_MODEL_DEFAULT = (os.getenv("TEXT_MODEL") or "gpt-4o").strip()

# ---- Sora (高端档) ----
SORA_MODEL_DEFAULT = (os.getenv("SORA_MODEL") or "sora-2").strip()
SORA_SECONDS_DEFAULT = int(os.getenv("SORA_SECONDS") or "8")         # 固定 8s
SORA_SIZE_DEFAULT = (os.getenv("SORA_SIZE") or "720x1280").strip()   # 竖屏
SORA_CONCURRENCY = int(os.getenv("SORA_CONCURRENCY", "1"))
SORA_SEM = threading.BoundedSemaphore(SORA_CONCURRENCY)

# ---- Luma (普通档) ----
LUMA_API_KEY = (os.getenv("LUMA_API_KEY") or "").strip()
LUMA_MODEL_DEFAULT = (os.getenv("LUMA_MODEL") or "ray-flash-2").strip()
LUMA_SECONDS_DEFAULT = int(os.getenv("LUMA_SECONDS") or "10")        # 固定 10s
LUMA_RES_DEFAULT = (os.getenv("LUMA_RESOLUTION") or "720p").strip()  # 固定 720p
LUMA_CONCURRENCY = int(os.getenv("LUMA_CONCURRENCY", "1"))
LUMA_SEM = threading.BoundedSemaphore(LUMA_CONCURRENCY)

# ✅ Luma 自动加音频（默认开；只影响 Luma，不动 Sora）
LUMA_AUDIO_ENABLED = int(os.getenv("LUMA_AUDIO_ENABLED") or "1")  # 1=开启 0=关闭
LUMA_AUDIO_PROMPT_DEFAULT = (os.getenv("LUMA_AUDIO_PROMPT") or
    "Cinematic soundscape matching the video: engines, wind, lasers, explosions, ambience. No speech."
).strip()
LUMA_AUDIO_TIMEOUT_SEC = int(os.getenv("LUMA_AUDIO_TIMEOUT_SEC") or "240")

# Luma i2v 必须公网 URL
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("solara-backend")

# ============== PATHS ==============
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MEM_DIR = BASE_DIR / ".memory"
MEM_DIR.mkdir(exist_ok=True)

# ============== JOB TABLES ==============
SORA_JOBS: Dict[str, Dict[str, Any]] = {}
LUMA_JOBS: Dict[str, Dict[str, Any]] = {}

RECENT_KEYS: Dict[str, Dict[str, Any]] = {}
LUMA_RECENT_KEYS: Dict[str, Dict[str, Any]] = {}
RECENT_TTL = 120

# 会话记忆/最近媒体（给语音 intent）
SOLARA_MAX_TURNS = 30
SOLARA_SESSIONS: Dict[str, Dict[str, Any]] = {}
SOLARA_LAST_MEDIA: Dict[str, Dict[str, Any]] = {}

# ============== Sora Base Prompt ==============
SORA_BASE_PROMPT = """
你是專門為手機端生成「干淨、無文字元素」短視頻的專業導演與視覺設計師。

請嚴格遵守：
1. 畫面中禁止任何文字元素（任何語言/數字/字幕/Logo/水印）
2. 若有參考圖片或影片，只保留真正主體，背景雜物一律忽略或重畫
3. 只用畫面講故事（構圖/光影/運鏡），不要把 UI/招牌/字樣帶進畫面
4. 風格乾淨、現代、有設計感，適合作為 AI 視頻 App 的示例
5. 出錯時寧可簡潔，也不要冒險產生任何文字/字幕

你的目標：完全無文字 + 去雜物 + 竖屏短視頻。
""".strip()

def build_sora_prompt(user_prompt: str) -> str:
    up = (user_prompt or "").strip()
    if not up:
        up = "請根據用戶提供的素材生成乾淨、無文字元素的手機竖屏短視頻。"
    return SORA_BASE_PROMPT + "\n\n用戶需求：\n" + up

# ============== Utils ==============
def _cleanup_recent():
    now = time.time()
    for k in list(RECENT_KEYS.keys()):
        if now - RECENT_KEYS[k]["ts"] > RECENT_TTL:
            RECENT_KEYS.pop(k, None)
    for k in list(LUMA_RECENT_KEYS.keys()):
        if now - LUMA_RECENT_KEYS[k]["ts"] > RECENT_TTL:
            LUMA_RECENT_KEYS.pop(k, None)

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _idem_key(ip: str, prompt: str, img_h: str, vid_h: str) -> str:
    raw = f"{ip}|{(prompt or '').strip()}|img:{img_h}|vid:{vid_h}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _short(s: str, n: int = 600) -> str:
    return (s or "")[:n]

def _log_http(r: requests.Response, tag: str):
    try:
        rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
        log.info("[%s] %s %s -> %s rid=%s body=%s",
                 tag, r.request.method, r.request.url, r.status_code, rid, _short(r.text))
    except Exception:
        pass

def _guess_mime_from_ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
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
    """
    兜底：防止 video_id 被误拼接（你日志里出现 video_xxx + 尾巴），只保留 video_<alnum> 主体。
    """
    s = (vid or "").strip()
    m = re.search(r"(video_[A-Za-z0-9]+)", s)
    return m.group(1) if m else s

def extract_url(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    for k in ("video_url", "download_url", "cdn_url", "url", "mp4", "hls", "video"):
        u = data.get(k)
        if isinstance(u, str) and u.startswith("http"):
            return u
    for key in ("assets", "results", "output", "data", "files"):
        v = data.get(key)
        if isinstance(v, dict):
            u = extract_url(v)
            if u:
                return u
        elif isinstance(v, list):
            for item in v:
                u = extract_url(item)
                if u:
                    return u
    return ""

def extract_file_id(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    fid = data.get("file_id")
    if isinstance(fid, str) and fid.startswith("file_"):
        return fid
    for key in ("files", "results", "output", "data", "assets"):
        v = data.get(key)
        if isinstance(v, list):
            for item in v:
                f2 = extract_file_id(item)
                if f2:
                    return f2
        elif isinstance(v, dict):
            f2 = extract_file_id(v)
            if f2:
                return f2
    return ""

def _parse_size(size_str: str) -> Optional[tuple[int, int]]:
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

def _resize_image_bytes(raw: bytes, target_wh: Optional[tuple[int, int]]) -> bytes:
    if not target_wh:
        return raw
    try:
        from PIL import Image
    except ImportError:
        return raw
    try:
        w, h = target_wh
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im = im.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        return raw
# ================================
# Sora / Luma REST + workers
# ================================

# ============== Headers ==============
def _json_headers():
    h = {
        "Authorization": f"Bearer {SORA_API_KEY}",
        "OpenAI-Beta": "video-generation=v1",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if OPENAI_ORG_ID:
        h["OpenAI-Organization"] = OPENAI_ORG_ID
    return h

def _auth_headers():
    h = {
        "Authorization": f"Bearer {SORA_API_KEY}",
        "OpenAI-Beta": "video-generation=v1",
    }
    if OPENAI_ORG_ID:
        h["OpenAI-Organization"] = OPENAI_ORG_ID
    return h

def _luma_headers():
    if not LUMA_API_KEY:
        raise RuntimeError("missing LUMA_API_KEY")
    return {
        "Authorization": f"Bearer {LUMA_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

# ============== Video tools ==============
def _resize_video_file(src_path: str, dst_path: str, target_wh: Optional[tuple[int, int]]) -> str:
    if not target_wh:
        return src_path
    try:
        w, h = target_wh
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-vf",
            f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-iw)/2",
            "-c:v", "libx264", "-preset", "veryfast", "-an", dst_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path
    except Exception:
        return src_path

def _video_to_mosaic_image(src_path: str, dst_path: str, tiles: int = 1) -> Optional[str]:
    try:
        cmd = ["ffmpeg", "-y", "-i", src_path, "-frames:v", "1", "-vf", f"tile={tiles}x{tiles}", dst_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path
    except Exception:
        return None

# ============== SORA REST ==============
def sora_create(prompt: str, ref_path: Optional[str] = None, ref_mime: Optional[str] = None) -> str:
    url = "https://api.openai.com/v1/videos"
    model = (os.getenv("SORA_MODEL") or SORA_MODEL_DEFAULT).strip()
    sec = SORA_SECONDS_DEFAULT if SORA_SECONDS_DEFAULT in (4, 8, 12) else 8
    size = SORA_SIZE_DEFAULT

    prompt_final = build_sora_prompt(prompt)
    r: requests.Response

    if ref_path:
        try:
            mime = ref_mime or _guess_mime_from_ext(ref_path)
            headers = _auth_headers()
            headers["Accept"] = "application/json"
            files = {"input_reference": (os.path.basename(ref_path), open(ref_path, "rb"), mime)}
            data = {"model": model, "prompt": prompt_final, "seconds": str(sec), "size": size}
            r = requests.post(url, headers=headers, data=data, files=files, timeout=60)
            _log_http(r, f"SORA.CREATE[ref:{mime}]")
        except Exception as e:
            log.warning("[SORA] create with ref failed -> text-only: %s", e)
            body = {"model": model, "prompt": prompt_final, "seconds": str(sec), "size": size}
            r = requests.post(url, headers=_json_headers(), json=body, timeout=60)
            _log_http(r, f"SORA.CREATE[{model}]")
        finally:
            try:
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

def sora_status(video_id: str) -> dict:
    video_id = _normalize_video_id(video_id)
    r = requests.get(f"https://api.openai.com/v1/videos/{video_id}", headers=_auth_headers(), timeout=60)
    _log_http(r, "SORA.STATUS")
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json() if r.text else {}

# ⚠️ 以下三个函数保留兼容，但本修复版不再依赖它们完成下载（避免你遇到的 404）
def sora_assets(video_id: str) -> dict:
    video_id = _normalize_video_id(video_id)
    r = requests.get(f"https://api.openai.com/v1/videos/{video_id}/assets", headers=_auth_headers(), timeout=60)
    _log_http(r, "SORA.ASSETS")
    if r.status_code >= 400:
        return {}
    return r.json() if r.text else {}

def sora_download_location(video_id: str) -> str:
    video_id = _normalize_video_id(video_id)
    r = requests.get(
        f"https://api.openai.com/v1/videos/{video_id}/download",
        headers=_auth_headers(),
        allow_redirects=False,
        timeout=60
    )
    _log_http(r, "SORA.DOWNLOAD")
    if r.status_code in (302, 303) and r.headers.get("Location"):
        return r.headers["Location"]
    return ""

def sora_content_location(video_id: str) -> str:
    video_id = _normalize_video_id(video_id)
    r = requests.get(
        f"https://api.openai.com/v1/videos/{video_id}/content",
        headers=_auth_headers(),
        allow_redirects=False,
        timeout=60
    )
    _log_http(r, "SORA.CONTENT")
    if r.status_code in (302, 303) and r.headers.get("Location"):
        return r.headers["Location"]
    return ""

def bg_sora_worker(job_id: str, prompt: str, timeout_sec: int = 600, release_on_exit: bool = False):
    """
    ✅ 修复点：
    - completed 后不再去 /assets /download 兜圈子
    - 统一：前端只用 /video/stream/{job_id} 播放/下载
    """
    try:
        job = SORA_JOBS.get(job_id, {})
        ref_path = job.get("ref_path")
        ref_mime = job.get("ref_mime")

        video_id = sora_create(prompt, ref_path=ref_path, ref_mime=ref_mime)
        video_id = _normalize_video_id(video_id)
        SORA_JOBS[job_id].update({"status": "running", "video_id": video_id})

        deadline = time.time() + timeout_sec
        last_status = ""

        while True:
            info = sora_status(video_id)
            status = str(info.get("status") or info.get("state") or last_status or "processing").lower()
            last_status = status
            SORA_JOBS[job_id]["status"] = status

            try:
                prog = int(info.get("progress") or 0)
                SORA_JOBS[job_id]["progress"] = max(SORA_JOBS[job_id].get("progress", 0), prog)
            except Exception:
                pass

            if status in ("completed", "succeeded", "done"):
                SORA_JOBS[job_id]["url"] = f"/video/stream/{job_id}"
                SORA_JOBS[job_id]["status"] = "done"
                SORA_JOBS[job_id]["progress"] = 100
                return

            if status in ("failed", "error", "cancelled"):
                raise RuntimeError(f"sora failed: {status}")
            if time.time() > deadline:
                raise TimeoutError("sora timeout")

            time.sleep(2)

    except Exception as e:
        SORA_JOBS[job_id]["error"] = str(e)
        SORA_JOBS[job_id]["status"] = "failed"
        log.exception("[SORA] job=%s FAILED: %s", job_id, e)
    finally:
        try:
            if release_on_exit:
                SORA_SEM.release()
        except Exception:
            pass
# ============== LUMA REST (独立线路) ==============
LUMA_CREATE_URL = "https://api.lumalabs.ai/dream-machine/v1/generations"
LUMA_GET_URL = "https://api.lumalabs.ai/dream-machine/v1/generations/{gid}"
LUMA_AUDIO_URL = "https://api.lumalabs.ai/dream-machine/v1/generations/{gid}/audio"

def luma_create(prompt: str, image_url: Optional[str] = None) -> str:
    sec = max(1, int(LUMA_SECONDS_DEFAULT or 10))
    dur = f"{sec}s"
    res = (LUMA_RES_DEFAULT or "720p").strip()

    body: Dict[str, Any] = {
        "prompt": (prompt or "").strip() or "Generate a vertical mobile video.",
        "model": LUMA_MODEL_DEFAULT,
        "resolution": res,
        "duration": dur,
    }
    if image_url and image_url.startswith("http"):
        body["keyframes"] = {"frame0": {"type": "image", "url": image_url}}

    r = requests.post(LUMA_CREATE_URL, headers=_luma_headers(), json=body, timeout=60)
    _log_http(r, f"LUMA.CREATE[{LUMA_MODEL_DEFAULT}|{res}|{dur}]")
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        raise RuntimeError(f"Luma create error: {err}")

    data = r.json() if r.text else {}
    gid = data.get("id") or data.get("generation_id")
    if not gid:
        raise RuntimeError(f"missing generation id: {data}")
    return gid

def luma_add_audio(gid: str, audio_prompt: str) -> dict:
    ap = (audio_prompt or "").strip() or LUMA_AUDIO_PROMPT_DEFAULT
    body = {"prompt": ap}
    r = requests.post(LUMA_AUDIO_URL.format(gid=gid), headers=_luma_headers(), json=body, timeout=60)
    _log_http(r, "LUMA.ADD_AUDIO")
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        raise RuntimeError(f"Luma add audio error: {err}")
    return r.json() if r.text else {}

def luma_status(gid: str) -> dict:
    r = requests.get(LUMA_GET_URL.format(gid=gid), headers=_luma_headers(), timeout=60)
    _log_http(r, "LUMA.STATUS")
    if r.status_code >= 400:
        raise RuntimeError(r.text)
    return r.json() if r.text else {}

def bg_luma_worker(job_id: str, prompt: str, image_url: Optional[str], timeout_sec: int = 900, release_on_exit: bool = False):
    try:
        gid = luma_create(prompt, image_url=image_url)
        LUMA_JOBS[job_id].update({"status": "running", "generation_id": gid})

        deadline = time.time() + timeout_sec
        while True:
            info = luma_status(gid)
            state = str(info.get("state") or info.get("status") or "dreaming").lower()
            LUMA_JOBS[job_id]["status"] = state

            if state in ("queued",):
                LUMA_JOBS[job_id]["progress"] = 5
            elif state in ("dreaming", "running", "processing"):
                LUMA_JOBS[job_id]["progress"] = max(LUMA_JOBS[job_id].get("progress", 20), 35)

            if state in ("completed", "succeeded", "done"):
                assets = info.get("assets") or {}
                video_url = assets.get("video") or extract_url(info)
                if not video_url:
                    raise RuntimeError("missing luma assets.video")

                if LUMA_AUDIO_ENABLED:
                    try:
                        LUMA_JOBS[job_id]["status"] = "adding_audio"
                        LUMA_JOBS[job_id]["progress"] = max(LUMA_JOBS[job_id].get("progress", 0), 90)

                        ap = (LUMA_AUDIO_PROMPT_DEFAULT + " Video content: " + (prompt or "").strip()).strip()
                        luma_add_audio(gid, ap)

                        audio_deadline = time.time() + max(10, int(LUMA_AUDIO_TIMEOUT_SEC or 240))
                        while time.time() < audio_deadline:
                            info2 = luma_status(gid)
                            st2 = str(info2.get("state") or info2.get("status") or "").lower()

                            if st2 in ("completed", "succeeded", "done"):
                                a2 = info2.get("assets") or {}
                                v2 = a2.get("video") or extract_url(info2) or video_url
                                if v2:
                                    video_url = v2
                                break

                            if st2 in ("failed", "error", "cancelled"):
                                break

                            time.sleep(2)

                    except Exception as e:
                        log.warning("[LUMA] add audio failed (fallback silent): %s", e)

                LUMA_JOBS[job_id]["url"] = video_url
                LUMA_JOBS[job_id]["status"] = "done"
                LUMA_JOBS[job_id]["progress"] = 100
                return

            if state in ("failed", "error", "cancelled"):
                reason = info.get("failure_reason") or info.get("error") or state
                raise RuntimeError(f"luma failed: {reason}")

            if time.time() > deadline:
                raise TimeoutError("luma timeout")
            time.sleep(2)

    except Exception as e:
        LUMA_JOBS[job_id]["error"] = str(e)
        LUMA_JOBS[job_id]["status"] = "failed"
        log.exception("[LUMA] job=%s FAILED: %s", job_id, e)
    finally:
        try:
            if release_on_exit:
                LUMA_SEM.release()
        except Exception:
            pass

def _persist_image_to_static_for_luma(src_path: str, job_id: str) -> Optional[str]:
    if not PUBLIC_BASE_URL:
        return None
    try:
        p = Path(src_path)
        if not p.exists():
            return None
        inputs_dir = STATIC_DIR / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        dst = inputs_dir / f"luma_{job_id}.jpg"

        data = p.read_bytes()
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(data)).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92)
            data = buf.getvalue()
        except Exception:
            pass

        dst.write_bytes(data)
        return f"{PUBLIC_BASE_URL}/static/inputs/{dst.name}"
    except Exception as e:
        log.warning("[LUMA] persist image failed: %s", e)
        return None

# ================================
# FastAPI app + routes
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[BOOT] start | sora=%s(8s) | luma=%s(10s,%s) | luma_audio=%s | public=%s",
             SORA_MODEL_DEFAULT, LUMA_MODEL_DEFAULT, LUMA_RES_DEFAULT,
             ("on" if LUMA_AUDIO_ENABLED else "off"),
             PUBLIC_BASE_URL or "-")
    yield
    log.info("[BOOT] stop")

app = FastAPI(title="Solara Backend", lifespan=lifespan)

# ✅ billing（两行之一：include_router）
app.include_router(billing_router)

# ✅ auth（新增：include_router，让 /auth/apple 返回 access_token）
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- AUTH: Apple (LEGACY, renamed) ----------------
# ⚠️ 旧实现保留但改名，避免与 auth_router 的 /auth/apple 冲突
def _decode_jwt_payload_legacy(jwt: str) -> dict:
    """
    Decode JWT payload only (MVP). No signature verification. (LEGACY)
    """
    try:
        parts = (jwt or "").split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        pad = "=" * (-len(payload_b64) % 4)
        raw = base64.urlsafe_b64decode(payload_b64 + pad)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}

@app.post("/auth/apple_legacy")
async def auth_apple_legacy(req: Request):
    """
    Legacy Apple auth:
    Body: { "identity_token": "<JWT from Apple>" }
    Return: { ok, user_id, email(optional) }
    NOTE: new auth lives in auth.py -> /auth/apple (returns access_token).
    """
    try:
        body = await req.json()
    except Exception:
        body = {}

    tok = (body.get("identity_token") or "").strip()
    if not tok:
        return JSONResponse({"ok": False, "error": "missing_identity_token"}, status_code=400)

    payload = _decode_jwt_payload_legacy(tok)
    sub = (payload.get("sub") or "").strip()
    if not sub:
        return JSONResponse({"ok": False, "error": "missing_sub"}, status_code=400)

    user_id = f"apple:{sub}"
    email = (payload.get("email") or "").strip()

    return {"ok": True, "user_id": user_id, "email": email}

# --------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.middleware("http")
async def audit(req: Request, call_next):
    t0 = time.time()
    resp = await call_next(req)
    if req.url.path in ("/video", "/luma", "/rt/intent", "/rt/intent_luma", "/session", "/solara/photo"):
        ip = req.client.host if req.client else "-"
        log.info("[AUDIT] ip=%s %s %s -> %s in %dms",
                 ip, req.method, req.url.path, resp.status_code, int((time.time()-t0)*1000))
    return resp

# ---------------- SORA: create/status/stream ----------------
@app.post("/video")
async def create_video(
    request: Request,
    prompt: str = Form(""),
    image_file: UploadFile = File(None),
    video_file: UploadFile = File(None),
    audio_file: UploadFile = File(None),
):
    ip = request.client.host if request.client else "unknown"

    raw_img = raw_vid = None
    img_h = vid_h = ""

    if image_file:
        raw_img = await image_file.read()
        img_h = _sha1_bytes(raw_img)
    if video_file:
        raw_vid = await video_file.read()
        vid_h = _sha1_bytes(raw_vid)

    if not prompt.strip():
        prompt = "Generate a video based on the reference media." if (raw_img or raw_vid) else "Generate a video."

    _cleanup_recent()
    idem = _idem_key(ip, prompt, img_h, vid_h)
    rec = RECENT_KEYS.get(idem)
    if rec:
        return {"ok": True, "job_id": rec["job_id"], "status_url": f"/video/status/{rec['job_id']}", "status": "running"}

    if not SORA_SEM.acquire(blocking=False):
        return JSONResponse({"ok": False, "error": "busy: one sora job already running"}, status_code=429)

    job_id = uuid.uuid4().hex
    SORA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": prompt,
        "url": None,          # done 后会写成 /video/stream/{job_id}
        "file_id": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "ref_path": None,
        "ref_mime": None,
        "provider": "sora",
        "seconds": 8,
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

        if raw_vid is not None:
            src_fn = UPLOADS_DIR / f"{job_id}_vid_src.mp4"
            src_fn.write_bytes(raw_vid)
            dst_fn = UPLOADS_DIR / f"{job_id}_vid.mp4"
            final_fn = _resize_video_file(str(src_fn), str(dst_fn), target_wh)

            if ref_path is None:
                thumb_fn = UPLOADS_DIR / f"{job_id}_vid_thumb.jpg"
                thumb_path = _video_to_mosaic_image(final_fn, str(thumb_fn), tiles=1)
                if thumb_path and Path(thumb_path).exists():
                    ref_path = thumb_path
                    ref_mime = "image/jpeg"
    except Exception as e:
        log.warning("[SORA] ref prepare failed: %s", e)

    SORA_JOBS[job_id]["ref_path"] = ref_path
    SORA_JOBS[job_id]["ref_mime"] = ref_mime

    threading.Thread(target=bg_sora_worker, args=(job_id, prompt, 600, True), daemon=True).start()
    return {"ok": True, "job_id": job_id, "status_url": f"/video/status/{job_id}", "status": "running"}
@app.get("/video/status/{job_id}")
def video_status(job_id: str):
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
    return {
        "ok": True,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "url": job.get("url"),           # ✅ done 后是 /video/stream/{job_id}
        "file_id": job.get("file_id"),
        "error": job.get("error"),
        "video_id": job.get("video_id"),
        "provider": "sora",
        "seconds": 8,
    }

@app.get("/video/stream/{job_id}")
def video_stream(job_id: str, range_header: Optional[str] = Header(None, alias="Range")):
    """
    ✅ 修复点：
    - Sora 统一用 /v1/videos/{video_id}/content
    - 同时支持 302/303 redirect 和 200/206 直接返回
    - 支持 Range
    """
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)

    fid = (job.get("file_id") or "").strip()
    vid = _normalize_video_id(job.get("video_id") or "")

    if not fid and not vid:
        return JSONResponse({"ok": False, "error": "video not ready"}, status_code=409)

    if fid:
        files_url = f"https://api.openai.com/v1/files/{fid}/content"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        if range_header:
            headers["Range"] = range_header
        r = requests.get(files_url, headers=headers, stream=True, timeout=120)

        proxy_headers = {k: r.headers[k] for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"] if k in r.headers}

        def gen_file():
            for c in r.iter_content(128 * 1024):
                if c:
                    yield c

        return StreamingResponse(
            gen_file(),
            media_type=r.headers.get("Content-Type", "video/mp4"),
            headers=proxy_headers,
            status_code=(r.status_code if r.status_code in (200, 206) else 200)
        )

    content_url = f"https://api.openai.com/v1/videos/{vid}/content"
    headers = {"Authorization": f"Bearer {SORA_API_KEY}", "OpenAI-Beta": "video-generation=v1"}
    if OPENAI_ORG_ID:
        headers["OpenAI-Organization"] = OPENAI_ORG_ID
    if range_header:
        headers["Range"] = range_header

    rc = requests.get(content_url, headers=headers, stream=True, allow_redirects=False, timeout=120)

    if rc.status_code in (302, 303) and rc.headers.get("Location"):
        loc = rc.headers["Location"]
        r2 = requests.get(loc, stream=True, timeout=120)

        proxy_headers = {k: r2.headers[k] for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"] if k in r2.headers}

        def gen_loc():
            for c in r2.iter_content(128 * 1024):
                if c:
                    yield c

        return StreamingResponse(
            gen_loc(),
            media_type=r2.headers.get("Content-Type", "video/mp4"),
            headers=proxy_headers,
            status_code=(r2.status_code if r2.status_code in (200, 206) else 200)
        )

    if rc.status_code in (200, 206):
        proxy_headers = {k: rc.headers[k] for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"] if k in rc.headers}

        def gen_content():
            for c in rc.iter_content(128 * 1024):
                if c:
                    yield c

        return StreamingResponse(
            gen_content(),
            media_type=rc.headers.get("Content-Type", "video/mp4"),
            headers=proxy_headers,
            status_code=rc.status_code
        )

    try:
        detail = rc.text[:400]
    except Exception:
        detail = ""
    return JSONResponse({"ok": False, "error": f"content fetch failed: {rc.status_code}", "detail": detail}, status_code=502)

# ---------------- LUMA: create/status/stream ----------------
@app.post("/luma")
async def create_luma(
    request: Request,
    prompt: str = Form(""),
    image_file: UploadFile = File(None),
):
    if not LUMA_API_KEY:
        return JSONResponse({"ok": False, "error": "missing LUMA_API_KEY"}, status_code=500)

    ip = request.client.host if request.client else "unknown"

    raw_img = None
    img_h = ""
    if image_file:
        raw_img = await image_file.read()
        img_h = _sha1_bytes(raw_img)

    if not prompt.strip():
        prompt = "Generate a clean vertical mobile video."

    _cleanup_recent()
    idem = _idem_key(ip, prompt, img_h, "")
    rec = LUMA_RECENT_KEYS.get(idem)
    if rec:
        return {"ok": True, "job_id": rec["job_id"], "status_url": f"/luma/status/{rec['job_id']}", "status": "running"}

    if not LUMA_SEM.acquire(blocking=False):
        return JSONResponse({"ok": False, "error": "busy: one luma job already running"}, status_code=429)

    job_id = uuid.uuid4().hex
    LUMA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": prompt,
        "url": None,
        "generation_id": None,
        "error": None,
        "created": int(time.time()),
        "provider": "luma",
        "seconds": 10,
        "resolution": "720p",
        "cost_credits_estimate": 110,
    }
    LUMA_RECENT_KEYS[idem] = {"job_id": job_id, "ts": time.time()}

    image_url = None
    try:
        if raw_img is not None and PUBLIC_BASE_URL:
            inputs_dir = STATIC_DIR / "inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            fn = inputs_dir / f"luma_upload_{job_id}.jpg"

            img_bytes = raw_img
            try:
                from PIL import Image
                im = Image.open(io.BytesIO(raw_img)).convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=92)
                img_bytes = buf.getvalue()
            except Exception:
                pass

            fn.write_bytes(img_bytes)
            image_url = f"{PUBLIC_BASE_URL}/static/inputs/{fn.name}"
        elif raw_img is not None and not PUBLIC_BASE_URL:
            log.warning("[LUMA] PUBLIC_BASE_URL not set -> text-only fallback")
    except Exception as e:
        log.warning("[LUMA] save input image failed: %s", e)
        image_url = None

    threading.Thread(target=bg_luma_worker, args=(job_id, prompt, image_url, 900, True), daemon=True).start()
    return {"ok": True, "job_id": job_id, "status_url": f"/luma/status/{job_id}", "status": "running"}

@app.get("/luma/status/{job_id}")
def luma_status_api(job_id: str):
    job = LUMA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
    return {
        "ok": True,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "url": job.get("url"),
        "generation_id": job.get("generation_id"),
        "error": job.get("error"),
        "provider": "luma",
        "seconds": 10,
        "resolution": "720p",
        "cost_credits_estimate": 110,
    }

@app.get("/luma/stream/{job_id}")
def luma_stream(job_id: str, range_header: Optional[str] = Header(None, alias="Range")):
    job = LUMA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)

    url = (job.get("url") or "").strip()
    if not url.startswith("http"):
        return JSONResponse({"ok": False, "error": "video not ready"}, status_code=409)

    headers = {}
    if range_header:
        headers["Range"] = range_header
    r = requests.get(url, headers=headers, stream=True, timeout=120)

    proxy_headers = {k: r.headers[k] for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"] if k in r.headers}

    def gen():
        for c in r.iter_content(128 * 1024):
            if c:
                yield c

    return StreamingResponse(gen(), media_type=r.headers.get("Content-Type", "video/mp4"),
                             headers=proxy_headers,
                             status_code=(r.status_code if r.status_code in (200, 206) else 200))

# ---------------- Photo cache ----------------
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
# Realtime session + intent_sora + intent_luma + health + run
# ================================
REALTIME_SESS_URL = "https://api.openai.com/v1/realtime/sessions"

SORA_REALTIME_INSTRUCTIONS = """
你是「GPT Solara」的即時語音版本，透過 Realtime 模型和使用者自然對話。
你是全能型語音助手，可以幫忙聊天、查資訊、講解知識、討論程式與產品、一起規劃生活與工作。

只有在使用者主動提到「短視頻 / 影片 / 生成視頻」時，才幫他整理成清楚的生成描述。
""".strip()

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

@app.post("/session")
async def session_post(req: Request):
    try:
        b = await req.json()
    except Exception:
        b = {}
    mode = (b.get("mode") or "").strip()
    model = (b.get("model") or REALTIME_MODEL_DEFAULT).strip()
    voice = (b.get("voice") or REALTIME_VOICE_DEFAULT).strip()
    instructions = None if mode == "companion" else SORA_REALTIME_INSTRUCTIONS

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
    }

def _session_key(req: Request, session_id: str) -> str:
    ip = req.client.host if req.client else "unknown"
    sid = (session_id or "default").strip()
    return f"{ip}:{sid}"

def _append_turn(session_key: str, role: str, text: str, media: Optional[Dict[str, Any]] = None):
    sess = SOLARA_SESSIONS.setdefault(session_key, {"turns": []})
    turn: Dict[str, Any] = {"role": role, "text": text}
    if media:
        turn["media"] = media
    sess["turns"].append(turn)
    if len(sess["turns"]) > SOLARA_MAX_TURNS:
        sess["turns"] = sess["turns"][-SOLARA_MAX_TURNS:]

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

    if not SORA_SEM.acquire(blocking=False):
        return JSONResponse({"ok": False, "error": "busy"}, status_code=429)

    job_id = uuid.uuid4().hex

    last_media = SOLARA_LAST_MEDIA.get(sk) or {}
    ref_path = last_media.get("path") if last_media.get("type") == "image" else None
    ref_mime = _guess_mime_from_ext(ref_path) if ref_path else None

    SORA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": prompt,
        "url": None,
        "file_id": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "ref_path": ref_path,
        "ref_mime": ref_mime,
        "provider": "sora",
        "seconds": 8,
    }

    _append_turn(sk, "user", f"^SORA_INTENT: {prompt}", media=(last_media if last_media else None))

    threading.Thread(target=bg_sora_worker, args=(job_id, prompt, 600, True), daemon=True).start()
    return {"ok": True, "job_id": job_id, "status_url": f"/video/status/{job_id}", "session_id": sk}

@app.post("/rt/intent_luma")
async def rt_intent_luma(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}

    prompt = (body.get("prompt") or body.get("text") or "").strip()
    if not prompt:
        return JSONResponse({"ok": False, "error": "missing prompt"}, status_code=400)

    if not LUMA_API_KEY:
        return JSONResponse({"ok": False, "error": "missing LUMA_API_KEY"}, status_code=500)

    session_id = (body.get("session_id") or body.get("conversation_id") or "").strip()
    sk = _session_key(req, session_id)

    if not LUMA_SEM.acquire(blocking=False):
        return JSONResponse({"ok": False, "error": "busy"}, status_code=429)

    job_id = uuid.uuid4().hex

    last_media = SOLARA_LAST_MEDIA.get(sk) or {}
    image_url = None
    if last_media.get("type") == "image" and last_media.get("path"):
        image_url = _persist_image_to_static_for_luma(last_media["path"], job_id)

    LUMA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": prompt,
        "url": None,
        "generation_id": None,
        "error": None,
        "created": int(time.time()),
        "provider": "luma",
        "seconds": 10,
        "resolution": "720p",
        "cost_credits_estimate": 110,
    }

    _append_turn(sk, "user", f"^LUMA_INTENT: {prompt}", media=(last_media if last_media else None))

    threading.Thread(target=bg_luma_worker, args=(job_id, prompt, image_url, 900, True), daemon=True).start()
    return {"ok": True, "job_id": job_id, "status_url": f"/luma/status/{job_id}", "session_id": sk}

@app.get("/health")
def health():
    return {
        "ok": True,
        "sora": {"model": SORA_MODEL_DEFAULT, "seconds": 8, "size": SORA_SIZE_DEFAULT, "concurrency": SORA_CONCURRENCY},
        "luma": {"model": LUMA_MODEL_DEFAULT, "seconds": 10, "resolution": "720p", "concurrency": LUMA_CONCURRENCY},
        "luma_audio": {"enabled": bool(LUMA_AUDIO_ENABLED), "timeout_sec": LUMA_AUDIO_TIMEOUT_SEC},
        "public_base_url": PUBLIC_BASE_URL or "",
        "endpoints": {
            "sora_create": "/video",
            "sora_status": "/video/status/{job_id}",
            "sora_stream": "/video/stream/{job_id}",
            "luma_create": "/luma",
            "luma_status": "/luma/status/{job_id}",
            "luma_stream": "/luma/stream/{job_id}",
            "intent_sora": "/rt/intent",
            "intent_luma": "/rt/intent_luma",
            "photo": "/solara/photo",
            "realtime_session": "/session",
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_session:app", host="0.0.0.0", port=8000, reload=False, log_level="info", access_log=False)
