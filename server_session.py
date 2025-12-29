# ================================
# server_session.py  (FULL, STABLE)
# ✅ Remix 链路对齐版：支持 /video/remix + 修复 remix prompt
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
from typing import Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, UploadFile, File, Form, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ✅✅ FIX：必须先 load_dotenv，再 import billing/auth（它们 import 时会读 os.getenv）
load_dotenv()

# ✅ billing
from billing import router as billing_router

# ✅ auth
from auth import router as auth_router


# ============== ENV ==============
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


# ============== Prompts (ONE SOURCE OF TRUTH) ==============
SOLARA_SYSTEM_PROMPT = r"""
你是「GPT Solara」通用智能型语音助手,以realtime模型能力，像chatGpT一样能力

【1. 角色與總目標】
- 第一角色通用语音助手，根据客户语义判断是否需要进入第二角色视频生成助手。你代表全人类，是人类科技文明的结晶，你拥有人类所有的知识与技能，你要服务使用者，解答问题，帮助他们创造一切你所知道的知识。
- 如果客户想创作生成AI视频，这时候你才是语音影片设计师，你将协助客户让他们得到他们想要的东西：「幫使用者一起想像畫面 → 幫他整理出一句清楚的生成描述」。

【语言策略（必须遵守）】
• 始终跟随用户语言：用户用中文你就中文；用户用英文你就英文；用户用日文你就日文。
• 如果用户一条消息里混用多种语言：以用户最后一句为准。
• 用户明确要求“用某某语言回答”，优先遵从。
• SORA 行也使用同一种语言输出，語氣自然、冷靜、有設計感，不要太熱情吵鬧。

【2. 對話節奏】
1) 一開始先用通用型的语音助手互动；当客户想创作生成AI视频时，再进入影片设计师模式。
2) 多聽少講。讓使用者連續說幾個畫面細節，不用每一句都回覆。
3) 每次輪到你說話時，控制在 1～3 句內，避免長篇大論，方便語音播放。
4) 如果使用者明顯還在補充，就暫時不要收尾，也不要急著總結或下結論。

【3. 追問策略（只問關鍵）】
- 只在關鍵不清楚時，才追問 1～2 句：
  • 主體是誰或什麼？（人 / 物 / 動物 / 城市 / 自然場景…）
  • 場景與環境？（室內 / 室外 / 城市夜景 / 太空 / 海邊…）
  • 氣氛與風格？（寫實 / 夢幻 / 科幻 / 溫暖 / 懸疑 / 逗趣…）
- 不要问影片时长，也不要一直问技术性问题。

【4. 何时判定「可以收尾」】
- 使用者說「差不多了」「就這樣」「OK 可以」等；
- 或已經有足夠多畫面細節；
- 使用者停頓、不再補充新資訊。
此时先用一句过渡：「好，我帮你整理成一句生成描述」。

【5. 生成影片描述（重點）】
- 把整個對話濃縮成一段 ≤50 个中文字描述。
- 优先保留：主体、场景、氛围/风格、镜头感（慢镜头/跟拍/手持）。
- 不要写具体秒数，不要提「10 秒影片」等。
- 不要加入解释，只需直接描述画面。

【6. 觸發 Sora 的唯一格式】
当准备好描述时，用单独一行输出：
SORA: 这里写整理好的画面描述（≤50字）

【9. 畫面純淨要求（非常重要）】
- SORA 行只能描述画面中实际会看到的「人物、物体、场景、光线、运动」等内容。
- 不要要求画面出现任何文字/字幕/水印/LOGO/App UI。
- 若参考图片来自手机截图，必须自动忽略所有 UI 元件。
""".strip()

SORA_BASE_PROMPT = SOLARA_SYSTEM_PROMPT
SORA_REALTIME_INSTRUCTIONS = SOLARA_SYSTEM_PROMPT


def build_sora_prompt(user_prompt: str) -> str:
    up = (user_prompt or "").strip()
    if not up:
        up = "請根據用戶提供的素材生成乾淨、無文字元素的手機竖屏短視頻。"
    return SORA_BASE_PROMPT + "\n\n用戶需求：\n" + up


# ✅✅✅ Remix 专用 Prompt（关键修复点）
def build_sora_remix_prompt(base_video_id: str, user_instruction: str) -> str:
    """
    ✅ 修复点：
    Remix 不能复用 build_sora_prompt()（那段是“通用助手系统提示”，会导致模型重构场景）
    这里必须明确强调：保持主体/场景/构图/镜头运动一致，仅做“用户要求的改动”。

    目标：避免“二次生成出来但不是原视频的继续”。
    """
    base_video_id = _normalize_video_id(base_video_id)
    instr = (user_instruction or "").strip()
    if not instr:
        instr = "Make subtle improvements. Keep everything else the same."

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
    兜底：防止 video_id 被误拼接（日志里出现 video_xxx + 尾巴），只保留 video_<alnum> 主体。
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


def _resize_image_bytes(raw: bytes, target_wh: Optional[Tuple[int, int]]) -> bytes:
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
# Remix parsing (NEW)
# ================================
REMIX_BEGIN = "[REMIX_REQUEST]"
REMIX_END = "[/REMIX_REQUEST]"

def _extract_remix_block(raw_prompt: str) -> str:
    s = raw_prompt or ""
    if REMIX_BEGIN not in s:
        return ""
    if REMIX_END in s:
        try:
            return s.split(REMIX_BEGIN, 1)[1].split(REMIX_END, 1)[0]
        except Exception:
            return ""
    # 没有 END，就取后半段
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
    # 没 END，则直接去掉 BEGIN 后所有
    return s.split(REMIX_BEGIN, 1)[0].strip()

def parse_remix_request(raw_prompt: str) -> Optional[Dict[str, str]]:
    """
    识别 iOS 发来的（fallback 模式）：
    [REMIX_REQUEST]
    base_video_id: video_xxx
    user_instruction: ....
    [/REMIX_REQUEST]
    """
    s = (raw_prompt or "").strip()
    if not s or REMIX_BEGIN not in s:
        return None

    block = _extract_remix_block(s)
    if not block:
        block = s

    # base_video_id 优先从 base_video_id: 行取，否则兜底取第一个 video_*
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

    # instruction：优先 user_instruction / instruction / prompt 后面的全部内容（可多行）
    instr = ""
    m3 = re.search(r"(user_instruction|instruction|prompt)\s*:\s*([\s\S]*?)\Z", block, flags=re.I)
    if m3:
        instr = (m3.group(2) or "").strip()
    else:
        # 兜底：去掉 base_video_id 行剩余内容
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

    # 如果 block 里提取不到，就用整个 prompt 去掉块后的文本兜底
    if not instr:
        instr = _strip_remix_block(s).strip()

    return {
        "base_video_id": base_id,
        "instruction": instr,
    }


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
def _resize_video_file(src_path: str, dst_path: str, target_wh: Optional[Tuple[int, int]]) -> str:
    if not target_wh:
        return src_path
    try:
        w, h = target_wh
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-vf",
            # ✅ 修复：pad 的 y 偏移应为 (oh-ih)/2
            f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
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
        files = None
        try:
            mime = ref_mime or _guess_mime_from_ext(ref_path)
            headers = _auth_headers()
            headers["Accept"] = "application/json"
            fh = open(ref_path, "rb")
            files = {"input_reference": (os.path.basename(ref_path), fh, mime)}
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
    """
    ✅ 修复版：Remix 不再走 build_sora_prompt（避免“二次生成变全新视频”）
    """
    base_video_id = _normalize_video_id(base_video_id)
    url = f"https://api.openai.com/v1/videos/{base_video_id}/remix"

    # ✅ 核心修复：Remix Prompt 单独构造
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


# 兼容保留（本修复版下载不依赖这些）
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


def bg_sora_worker(job_id: str, prompt: str, timeout_sec: int = 1800, release_on_exit: bool = False):
    """
    ✅ 修复点：
    - 真正支持 remix：检测 job['mode']=="remix"
    - completed 后统一：前端只用 /video/stream/{job_id} 播放/下载
    - 超时增强：progress>=99 给 finishing 宽限
    """
    try:
        job = SORA_JOBS.get(job_id, {})
        mode = (job.get("mode") or "").strip() or ("remix" if job.get("remix_base_video_id") else "create")

        ref_path = job.get("ref_path")
        ref_mime = job.get("ref_mime")

        # 1) create or remix
        if mode == "remix":
            base_id = _normalize_video_id(job.get("remix_base_video_id") or "")
            instr = (job.get("remix_instruction") or prompt or "").strip()
            if not base_id:
                raise RuntimeError("missing remix_base_video_id")
            if not instr:
                instr = "Make subtle improvements. Keep everything else the same."
            video_id = sora_remix(base_id, instr)
        else:
            video_id = sora_create(prompt, ref_path=ref_path, ref_mime=ref_mime)

        video_id = _normalize_video_id(video_id)
        SORA_JOBS[job_id].update({"status": "running", "video_id": video_id, "openai_status": "running"})

        deadline = time.time() + int(timeout_sec or 1800)
        last_status = ""
        finishing_grace_used = False
        transient_fail = 0

        while True:
            try:
                info = sora_status(video_id)
                transient_fail = 0
            except Exception as e:
                # ✅ 网络抖动/临时失败：不要立刻把任务判死
                transient_fail += 1
                if transient_fail <= 5:
                    time.sleep(2)
                    continue
                raise e

            status_raw = str(info.get("status") or info.get("state") or last_status or "processing")
            status = status_raw.lower().strip()
            last_status = status

            SORA_JOBS[job_id]["openai_status"] = status_raw
            SORA_JOBS[job_id]["status"] = status

            prog = None
            try:
                prog = int(info.get("progress") or 0)
                SORA_JOBS[job_id]["progress"] = max(SORA_JOBS[job_id].get("progress", 0), prog)
            except Exception:
                prog = None

            # ✅ progress 到 99 以上，给 finishing 宽限（避免 99% 被 timeout 杀）
            if prog is not None and prog >= 99 and not finishing_grace_used:
                deadline = max(deadline, time.time() + 600)  # +10min
                finishing_grace_used = True

            if status in ("completed", "succeeded", "done", "success"):
                SORA_JOBS[job_id]["url"] = f"/video/stream/{job_id}"
                SORA_JOBS[job_id]["status"] = "done"
                SORA_JOBS[job_id]["progress"] = 100
                return

            if status in ("failed", "error", "cancelled", "canceled"):
                raise RuntimeError(f"sora failed: {status}")

            if time.time() > deadline:
                raise TimeoutError("sora timeout")

            time.sleep(2)

    except Exception as e:
        SORA_JOBS[job_id]["error"] = str(e)
        SORA_JOBS[job_id]["status"] = "failed"
        log.exception("[SORA] job=%s FAILED: %s", job_id, e)
    finally:
        # 兼容保留：如果外面还在用 release_on_exit=True 的旧调用，不会崩
        try:
            if release_on_exit:
                SORA_SEM.release()
        except Exception:
            pass


def _spawn_sora_job(job_id: str, prompt: str, timeout_sec: int):
    """
    ✅ NEW：排队执行，避免 busy=429 让前端判“二次生成失败”
    成本不变：仍然由 SORA_SEM 控制并发
    """
    def runner():
        acquired = False
        try:
            # 等待并发额度（阻塞）
            SORA_SEM.acquire()
            acquired = True
            bg_sora_worker(job_id, prompt, timeout_sec=timeout_sec, release_on_exit=False)
        finally:
            if acquired:
                try:
                    SORA_SEM.release()
                except Exception:
                    pass

    threading.Thread(target=runner, daemon=True).start()


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

# ✅ billing
app.include_router(billing_router)

# ✅ auth
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- AUTH: Apple (LEGACY, renamed) ----------------
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
    if req.url.path in ("/video", "/video/remix", "/luma", "/rt/intent", "/rt/intent_luma", "/session", "/solara/photo"):
        ip = req.client.host if req.client else "-"
        log.info("[AUDIT] ip=%s %s %s -> %s in %dms",
                 ip, req.method, req.url.path, resp.status_code, int((time.time()-t0)*1000))
    return resp


# ---------------- SORA: create/status/stream ----------------
async def _save_upload_to_file_and_sha1(upload: UploadFile, dst_path: Path) -> str:
    """
    ✅ 分块保存 + 计算 sha1，避免一次性读入内存
    """
    h = hashlib.sha1()
    with open(dst_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
    return h.hexdigest()


def _job_is_active(job: Optional[Dict[str, Any]]) -> bool:
    if not job:
        return False
    st = str(job.get("status") or "").lower().strip()
    # 我们内部最终态：done / failed
    if st in ("done", "failed"):
        return False
    # 其它都认为 still running/queued
    return True


# ✅✅✅ 对齐 iOS：/video/remix（JSON）
@app.post("/video/remix")
async def create_video_remix(request: Request):
    """
    iOS createRemixJob() 首选调用的接口：
    POST /video/remix
    JSON: { "base_video_id": "video_xxx", "instruction": "..." }

    ✅ 修复点：
    - 让前端不再 404 回退到 /video + structured prompt（那条回退很容易变“新生成”）
    - Remix job 明确 mode=remix，并带 remixed_from_video_id 字段
    """
    ip = request.client.host if request.client else "unknown"

    try:
        body = await request.json()
    except Exception:
        body = {}

    base_video_id = _normalize_video_id((body.get("base_video_id") or body.get("video_id") or "").strip())
    instruction = (body.get("instruction") or body.get("prompt") or "").strip()

    if not base_video_id:
        return JSONResponse({"ok": False, "error": "missing base_video_id"}, status_code=400)

    # ---- Idempotency: remix 必须包含 base id ----
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
        else:
            RECENT_KEYS.pop(idem, None)

    # ---- Create job ----
    job_id = uuid.uuid4().hex
    SORA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": instruction or "Make subtle improvements. Keep everything else the same.",
        "prompt_raw": json.dumps(body, ensure_ascii=False),
        "url": None,
        "file_id": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "ref_path": None,
        "ref_mime": None,
        "provider": "sora",
        "seconds": 8,
        "mode": "remix",
        "remix_base_video_id": base_video_id,
        "remix_instruction": instruction or "",
        "openai_status": None,
    }
    RECENT_KEYS[idem] = {"job_id": job_id, "ts": time.time()}

    # ✅ timeout：remix finishing 更久，给更长
    timeout_sec = 1800
    _spawn_sora_job(job_id, SORA_JOBS[job_id]["prompt"], timeout_sec=timeout_sec)

    return {
        "ok": True,
        "job_id": job_id,
        "status_url": f"/video/status/{job_id}",
        "status": "queued",
        "remixed_from_video_id": base_video_id,
    }


@app.post("/video")
async def create_video(
    request: Request,
    prompt: str = Form(""),
    image_file: UploadFile = File(None),
    video_file: UploadFile = File(None),
    audio_file: UploadFile = File(None),
):
    ip = request.client.host if request.client else "unknown"

    raw_img: Optional[bytes] = None
    img_h = ""
    vid_h = ""

    # 为了 idem：先算 hash（视频分块写盘，避免内存峰值）
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

    # -------------- Remix detect (fallback) --------------
    remix = parse_remix_request(prompt or "")
    mode = "create"
    remix_base = ""
    remix_instruction = ""

    prompt_effective = (prompt or "").strip()
    prompt_idem = prompt_effective

    if remix:
        mode = "remix"
        remix_base = remix.get("base_video_id", "")
        remix_instruction = (remix.get("instruction", "") or "").strip()
        prompt_effective = remix_instruction or "Make subtle improvements. Keep everything else the same."
        # ✅ idem 必须包含 base_id，否则不同 base 视频会误去重
        prompt_idem = f"REMIX|{remix_base}|{prompt_effective}"

    if not prompt_effective.strip():
        prompt_effective = "Generate a video based on the reference media." if (raw_img or tmp_video_path) else "Generate a video."
        prompt_idem = prompt_effective

    # -------------- Idempotency (FIX) --------------
    _cleanup_recent()
    idem = _idem_key(ip, prompt_idem, img_h, vid_h)
    rec = RECENT_KEYS.get(idem)
    if rec:
        old_job_id = rec["job_id"]
        old_job = SORA_JOBS.get(old_job_id)

        # ✅ FIX：只有 still active 才复用；done/failed 必须新建 job
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
        else:
            # stale -> 删除去重记录，让这次真正创建新 job
            RECENT_KEYS.pop(idem, None)

    # -------------- Create job --------------
    job_id = uuid.uuid4().hex
    SORA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": prompt_effective,     # ✅ 真正用于生成的 prompt
        "prompt_raw": (prompt or ""),   # ✅ 原始 prompt（便于排查）
        "url": None,                    # done 后会写成 /video/stream/{job_id}
        "file_id": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "ref_path": None,
        "ref_mime": None,
        "provider": "sora",
        "seconds": 8,
        # remix fields
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

        # image -> resize to target, store as ref
        if raw_img is not None:
            raw_img_resized = _resize_image_bytes(raw_img, target_wh)
            fn = UPLOADS_DIR / f"{job_id}_img.jpg"
            fn.write_bytes(raw_img_resized)
            ref_path = str(fn)
            ref_mime = "image/jpeg"

        # video -> move temp file to job src, optionally resize, generate thumb if no image
        if tmp_video_path is not None and tmp_video_path.exists():
            src_fn = UPLOADS_DIR / f"{job_id}_vid_src.mp4"
            tmp_video_path.replace(src_fn)

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

    # ✅ timeout：remix 更容易 finishing 久一点，给更长
    timeout_sec = 1800 if mode == "remix" else 1200

    # ✅ NEW：排队执行（不再 429 busy）
    _spawn_sora_job(job_id, prompt_effective, timeout_sec=timeout_sec)

    return {"ok": True, "job_id": job_id, "status_url": f"/video/status/{job_id}", "status": "queued"}


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
        "url": job.get("url"),           # ✅ done 后是 /video/stream/{job_id}
        "file_id": job.get("file_id"),
        "error": job.get("error"),

        # ✅ 兼容字段：你 iOS 里会从 openai_video_id / video_id 取
        "video_id": vid,                 # legacy / existing
        "openai_video_id": vid,          # ✅ NEW: server authoritative
        "remixed_from_video_id": job.get("remix_base_video_id") if (job.get("mode") == "remix") else None,

        "provider": "sora",
        "seconds": 8,

        # ✅ NEW：稳定字段，便于 iOS 显示与排查
        "mode": job.get("mode"),
        "remix_base_video_id": job.get("remix_base_video_id"),
        "remix_instruction": job.get("remix_instruction"),
        "openai_status": job.get("openai_status"),
    }


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
    """
    ✅ NEW：加轻量重试，解决 completed 后 content 可能短暂 404/409 的竞态
    """
    vid = _normalize_video_id(video_id)
    content_url = f"https://api.openai.com/v1/videos/{vid}/content"

    headers = {"Authorization": f"Bearer {SORA_API_KEY}", "OpenAI-Beta": "video-generation=v1"}
    if OPENAI_ORG_ID:
        headers["OpenAI-Organization"] = OPENAI_ORG_ID
    if range_header:
        headers["Range"] = range_header

    backoffs = [0.5, 1.0, 2.0, 4.0]
    last = None

    for i in range(len(backoffs) + 1):
        r = requests.get(content_url, headers=headers, stream=True, allow_redirects=False, timeout=120)

        # redirect -> return it to caller处理二跳
        if r.status_code in (302, 303):
            return r

        if r.status_code in (200, 206):
            return r

        # 某些时候 content 会短暂未就绪
        if r.status_code in (404, 409, 425, 500, 502, 503, 504) and i < len(backoffs):
            try:
                r.close()
            except Exception:
                pass
            time.sleep(backoffs[i])
            last = r
            continue

        return r

    return last


@app.get("/video/stream/{job_id}")
def video_stream(job_id: str, range_header: Optional[str] = Header(None, alias="Range")):
    """
    ✅ 修复点：
    - Sora 统一用 /v1/videos/{video_id}/content
    - 同时支持 302/303 redirect 和 200/206 直接返回
    - 支持 Range（包括 redirect 后二跳仍带 Range）
    - completed 后 content 竞态：轻量重试
    """
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)

    fid = (job.get("file_id") or "").strip()
    vid = _normalize_video_id(job.get("video_id") or "")

    if not fid and not vid:
        return JSONResponse({"ok": False, "error": "video not ready"}, status_code=409)

    # ---- files content path (if you ever use file_id) ----
    if fid:
        files_url = f"https://api.openai.com/v1/files/{fid}/content"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        if range_header:
            headers["Range"] = range_header
        r = requests.get(files_url, headers=headers, stream=True, timeout=120)
        return _streaming_proxy_response(r)

    # ---- sora video content (official) ----
    rc = _fetch_sora_content_response(vid, range_header)

    # redirect -> fetch real location (keep Range!)
    if rc.status_code in (302, 303) and rc.headers.get("Location"):
        loc = rc.headers["Location"]
        try:
            rc.close()
        except Exception:
            pass

        headers2 = {}
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
    """
    ✅ NEW：按 OpenAI video_id 直接代理 content
    （满足你之前“提供无需客户端鉴权的内容下载”要求，避免 iOS 直连 OpenAI 401）
    """
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

        headers2 = {}
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
        img_h = _sha1_bytes(raw_img) if raw_img else ""

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
    return _streaming_proxy_response(r)


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
        "file_id": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "ref_path": ref_path,
        "ref_mime": ref_mime,
        "provider": "sora",
        "seconds": 8,
        "mode": "create",
        "remix_base_video_id": None,
        "remix_instruction": None,
        "openai_status": None,
    }

    _append_turn(sk, "user", f"^SORA_INTENT: {prompt}", media=(last_media if last_media else None))

    # ✅ NEW：排队执行（不再 busy=429）
    _spawn_sora_job(job_id, prompt, timeout_sec=1200)

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
            "sora_remix": "/video/remix",
            "sora_status": "/video/status/{job_id}",
            "sora_stream": "/video/stream/{job_id}",
            "sora_content": "/video/content/{video_id}",
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
    uvicorn.run(
        "server_session:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=False
    )
# ================================
# server_session.py  (FULL, STABLE)
# ✅ Remix 链路对齐版：支持 /video/remix + 修复 remix prompt
# ================================

