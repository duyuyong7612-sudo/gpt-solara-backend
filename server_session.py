# server_session.py  — 干净无文字版 Sora 后端

import os
import io
import json
import uuid
import time
import base64
import hashlib
import logging
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import requests
from requests.exceptions import ReadTimeout
from fastapi import FastAPI, UploadFile, File, Form, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ============== ENV & APP ==============
load_dotenv()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
SORA_API_KEY = (os.getenv("SORA_API_KEY") or OPENAI_API_KEY).strip()
OPENAI_ORG_ID = (os.getenv("OPENAI_ORG_ID") or "").strip()

REALTIME_MODEL_DEFAULT = (os.getenv("REALTIME_MODEL") or "gpt-realtime-mini").strip()
REALTIME_VOICE_DEFAULT = (os.getenv("REALTIME_VOICE") or "alloy").strip()

# ✅ 文本模型（統一默認 GPT-4o）
TEXT_MODEL_DEFAULT = (os.getenv("TEXT_MODEL") or "gpt-4o").strip()

SORA_MODEL_DEFAULT = (os.getenv("SORA_MODEL") or "sora-2").strip()
SORA_SECONDS_DEFAULT = int(os.getenv("SORA_SECONDS") or "8")          # 默认时长 8s
SORA_SIZE_DEFAULT = (os.getenv("SORA_SIZE") or "720x1280").strip()    # 默认分辨率（后面强制竖屏）

# 并发闸（一次只跑一个创建）
SORA_CONCURRENCY = int(os.getenv("SORA_CONCURRENCY", "1"))
SORA_SEM = threading.BoundedSemaphore(SORA_CONCURRENCY)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sora-backend")

# ============== STATIC & STORAGE ==============
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

MEM_DIR = Path(".memory")
MEM_DIR.mkdir(exist_ok=True)

# Sora 任务状态表
SORA_JOBS: Dict[str, Dict[str, Any]] = {}

# 幂等等价键（2 分钟窗）—— 同 IP + 同內容只建一次
RECENT_KEYS: Dict[str, Dict[str, Any]] = {}
RECENT_TTL = 120

# Solara 多輪會話記憶（30 輪） & 最近媒體
SOLARA_MAX_TURNS = 30
SOLARA_SESSIONS: Dict[str, Dict[str, Any]] = {}      # key -> {"turns": [...]}
SOLARA_LAST_MEDIA: Dict[str, Dict[str, Any]] = {}    # key -> 最近圖片/素材

# ============== Sora 基础风格：干净无文字 ==============
SORA_BASE_PROMPT = """
你是專門為手機端生成「干淨、無文字元素」短視頻的專業導演與視覺設計師。

請嚴格遵守：

1. 畫面中禁止任何文字元素
   - 不要出現中文字、英文字、其他語言文字
   - 不要出現數字（例如「2025」「3.0」等）
   - 不要字幕、標題條、貼紙文字、UI 介面文字
   - 不要 Logo、標語、品牌名、浮水印

2. 只用畫面講故事
   - 透過構圖、景別、光影、色彩、運鏡、人物動作來傳達情緒與含義
   - 可以有人物、場景、物件，但這些物件本身也不要帶明顯的文字（例如大看板、巨大招牌）
   - 若需要表示「科技感」「商務」「可愛」「酷炫」等，只能用畫面氛圍與元素來表達

3. 風格與用途
   - 視頻適合作為「AI 生成視頻 App 的示例視頻」，看起來現代、乾淨、有設計感
   - 節奏可以稍微有變化與鏡頭切換，但避免過度眩暈或閃爍
   - 以手機直屏觀看體驗為優先（即使實際長寬比由外部指定）

4. 出錯時的保守策略
   - 寧可畫面簡潔一點，也不要冒險產生任何文字或字幕
   - 若 prompt 有提到「加字、字幕、標語、Logo」等需求，一律忽略這部分，只保留畫面氣氛與內容

你的目標：在完全不使用畫面文字的前提下，生成一條好看、乾淨、具有氛圍感的手機短視頻。
"""

def build_sora_prompt(user_prompt: str) -> str:
    """
    統一構建最終發給 Sora 的 prompt：
    - 固定前綴：SORA_BASE_PROMPT（嚴格禁止畫面文字）
    - 後面拼接用戶這次的具體需求
    """
    up = (user_prompt or "").strip()
    if not up:
        up = "請根據用戶提供的素材與當前會話情境，生成一條乾淨、無文字元素的手機短視頻。"

    final = SORA_BASE_PROMPT.strip() + "\n\n用戶這次的具體需求：\n" + up
    return final

# ============== 小工具 ==============
def _cleanup_recent():
    now = time.time()
    for k in list(RECENT_KEYS.keys()):
        if now - RECENT_KEYS[k]["ts"] > RECENT_TTL:
            RECENT_KEYS.pop(k, None)


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _idem_key(ip: str, prompt: str, img_h: str, vid_h: str) -> str:
    raw = f"{ip}|{prompt.strip()}|img:{img_h}|vid:{vid_h}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _short(s: str, n: int = 600) -> str:
    try:
        return (s or "")[:n]
    except Exception:
        return str(s)[:n]


def _log_http(r: requests.Response, tag: str):
    """简短 HTTP 日志，方便后台排查"""
    try:
        rid = r.headers.get("x-request-id") or r.headers.get("request-id") or "-"
        log.info(
            "[%s] %s %s -> %s rid=%s body=%s",
            tag,
            r.request.method,
            r.request.url,
            r.status_code,
            rid,
            _short(r.text),
        )
    except Exception:
        pass

# 统一头（Sora JSON）
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


# 统一头（Sora auth）
def _auth_headers():
    h = {
        "Authorization": f"Bearer {SORA_API_KEY}",
        "OpenAI-Beta": "video-generation=v1",
    }
    if OPENAI_ORG_ID:
        h["OpenAI-Organization"] = OPENAI_ORG_ID
    return h


# 统一头（文本聊天 GPT-4o）
def _chat_headers():
    h = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if OPENAI_ORG_ID:
        h["OpenAI-Organization"] = OPENAI_ORG_ID
    return h


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


# ============== Helpers: extract url/file_id ==============
def extract_url(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    for k in ("video_url", "download_url", "cdn_url", "url", "mp4", "hls"):
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


# ============== Helpers: size & resize（统一尺寸） ==============
def _parse_size(size_str: str) -> Optional[tuple[int, int]]:
    try:
        w, h = size_str.lower().split("x")
        return int(w), int(h)
    except Exception:
        return None


def _ensure_portrait_size(size_str: str) -> str:
    """
    确保 size 为竖屏（宽 <= 高），即使环境变量传进来是 1280x720 也会自动变成 720x1280。
    """
    wh = _parse_size(size_str)
    if not wh:
        # 兜底：固定竖屏 720x1280
        return "720x1280"

    w, h = wh
    if w > h:
        # 宽 > 高 时交换，强制竖屏
        w, h = h, w

    return f"{w}x{h}"


# ★ 在模块加载时，把全局 SORA_SIZE_DEFAULT 矫正为竖屏尺寸
SORA_SIZE_DEFAULT = _ensure_portrait_size(SORA_SIZE_DEFAULT)


def _resize_image_bytes(raw: bytes, target_wh: Optional[tuple[int, int]]) -> bytes:
    """把图片字节缩放到指定尺寸（Pillow），失败则返回原始字节"""
    if not target_wh:
        return raw
    try:
        from PIL import Image  # pillow
    except ImportError:
        log.warning("[SORA] Pillow not installed, skip image resize")
        return raw

    try:
        w, h = target_wh
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im = im.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception as e:
        log.warning("[SORA] _resize_image_bytes failed: %s", e)
        return raw


def _resize_video_file(src_path: str, dst_path: str, target_wh: Optional[tuple[int, int]]) -> str:
    """
    用 ffmpeg 把视频缩放 + 补边到指定尺寸；失败则返回原路径。
    需要系统安装 ffmpeg 并在 PATH 中。
    """
    if not target_wh:
        return src_path
    try:
        w, h = target_wh
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-vf",
            f"scale=w={w}:h={h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-an",
            dst_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path
    except FileNotFoundError:
        log.warning("[SORA] ffmpeg not found, skip video resize")
        return src_path
    except Exception as e:
        log.warning("[SORA] _resize_video_file failed: %s", e)
        return src_path


def _video_to_mosaic_image(src_path: str, dst_path: str, tiles: int = 8) -> Optional[str]:
    """
    用 ffmpeg 从视频生成一张 tiles x tiles 的拼图图，作为图片参考。
    tiles=1 时，相当于截取一帧画面。
    """
    try:
        vf = f"tile={tiles}x{tiles}"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            src_path,
            "-frames:v",
            "1",
            "-vf",
            vf,
            dst_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path
    except FileNotFoundError:
        log.warning("[SORA] ffmpeg not found, skip video mosaic")
        return None
    except Exception as e:
        log.warning("[SORA] _video_to_mosaic_image failed: %s", e)
        return None

# ============== Sora REST（支持 input_reference，統一干淨無文字風格） ==============
def sora_create(prompt: str, ref_path: Optional[str] = None, ref_mime: Optional[str] = None) -> str:
    """
    调用 Sora:
    - 只有 prompt 时：纯文本生视频（带 size）
    - 有 ref_path 时：走 input_reference（图片/视频参考，已在本地 resize）
    - 無論如何，最終 prompt 都會經過 build_sora_prompt(...) 處理，確保畫面無文字
    """
    url = "https://api.openai.com/v1/videos"
    model = (os.getenv("SORA_MODEL") or SORA_MODEL_DEFAULT).strip()
    sec = SORA_SECONDS_DEFAULT
    size = SORA_SIZE_DEFAULT  # ★ 此时已被强制为竖屏

    # Sora 目前只接受 4 / 8 / 12 秒
    if sec not in (4, 8, 12):
        sec = 8

    # ★ 統一構建「干淨無文字」最終提示
    prompt_final = build_sora_prompt(prompt)
    log.info("[SORA] final_prompt(short)=%r", _short(prompt_final, 200))

    r: requests.Response

    if ref_path:
        # 参考媒体：multipart + input_reference，不传 size
        try:
            mime = ref_mime or _guess_mime_from_ext(ref_path)
            headers = _auth_headers()
            headers["Accept"] = "application/json"

            files = {
                "input_reference": (
                    os.path.basename(ref_path),
                    open(ref_path, "rb"),
                    mime,
                )
            }
            data = {
                "model": model,
                "prompt": prompt_final,
                "seconds": str(sec),
            }
            log.info("[SORA] create with input_reference=%s mime=%s", ref_path, mime)
            r = requests.post(url, headers=headers, data=data, files=files, timeout=60)
            _log_http(r, f"SORA.CREATE[ref:{mime}]")
        except Exception as e:
            log.warning("[SORA] create with ref failed (%s), fallback text-only: %s", ref_path, e)
            ref_path = None
        finally:
            try:
                files["input_reference"][1].close()
            except Exception:
                pass

        if ref_path is None:
            # fallback：纯文本 + size
            body = {
                "model": model,
                "prompt": prompt_final,
                "seconds": str(sec),
                "size": size,
            }
            r = requests.post(url, headers=_json_headers(), json=body, timeout=60)
            _log_http(r, f"SORA.CREATE[{model}]")
    else:
        # 只有文本：JSON + size
        body = {
            "model": model,
            "prompt": prompt_final,
            "seconds": str(sec),
            "size": size,
        }
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
    return vid


def sora_status(video_id: str) -> dict:
    url = f"https://api.openai.com/v1/videos/{video_id}"
    r = requests.get(url, headers=_auth_headers(), timeout=60)
    _log_http(r, "SORA.STATUS")
    if r.status_code >= 400:
        raise RuntimeError(f"status error: {r.text}")
    return r.json() if r.text else {}


def sora_assets(video_id: str) -> dict:
    url = f"https://api.openai.com/v1/videos/{video_id}/assets"
    r = requests.get(url, headers=_auth_headers(), timeout=60)
    _log_http(r, "SORA.ASSETS")
    if r.status_code >= 400:
        return {}
    return r.json() if r.text else {}


def sora_download_location(video_id: str) -> str:
    url = f"https://api.openai.com/v1/videos/{video_id}/download"
    r = requests.get(url, headers=_auth_headers(), allow_redirects=False, timeout=60)
    _log_http(r, "SORA.DOWNLOAD")
    if r.status_code in (302, 303) and r.headers.get("Location"):
        return r.headers["Location"]
    return ""


def sora_content_location(video_id: str) -> str:
    """只用于找 302 直链；200 的情况在 /video/stream 里直接下"""
    url = f"https://api.openai.com/v1/videos/{video_id}/content"
    r = requests.get(url, headers=_auth_headers(), allow_redirects=False, timeout=60)
    _log_http(r, "SORA.CONTENT")
    if r.status_code in (302, 303) and r.headers.get("Location"):
        return r.headers["Location"]
    return ""


# ============== Worker（完成后收割：assets→download→content→files） ==============
def bg_sora_worker(job_id: str, prompt: str, timeout_sec: int = 600, release_on_exit: bool = False):
    try:
        job = SORA_JOBS.get(job_id, {})
        ref_path = job.get("ref_path")
        ref_mime = job.get("ref_mime")

        log.info("[SORA] job=%s create start raw_prompt=%r ref=%s", job_id, prompt, ref_path)
        video_id = sora_create(prompt, ref_path=ref_path, ref_mime=ref_mime)
        SORA_JOBS[job_id].update({"status": "running", "video_id": video_id})

        deadline = time.time() + timeout_sec
        last_status = ""

        while True:
            try:
                info = sora_status(video_id)
            except ReadTimeout:
                log.warning("[SORA] status timeout once, retry...")
                if time.time() > deadline:
                    raise TimeoutError("sora status timeout")
                time.sleep(2.0)
                continue

            try:
                (MEM_DIR / f"sora_status_{job_id}.json").write_text(
                    json.dumps(info, ensure_ascii=False, indent=2),
                    "utf-8",
                )
            except Exception:
                pass

            status = str(info.get("status") or info.get("state") or last_status or "processing").lower()
            last_status = status
            SORA_JOBS[job_id]["status"] = status

            try:
                prog = int(info.get("progress") or 0)
                SORA_JOBS[job_id]["progress"] = max(SORA_JOBS[job_id].get("progress", 0), prog)
            except Exception:
                pass

            if status in ("completed", "succeeded", "done"):
                url_final = extract_url(info)

                # 最多等待 20s 收割直链或 file_id
                for _ in range(20):
                    if not url_final:
                        try:
                            assets = sora_assets(video_id)
                            url_final = extract_url(assets)
                            if not url_final:
                                fid = extract_file_id(assets)
                                if fid:
                                    SORA_JOBS[job_id]["file_id"] = fid
                                    log.info("[SORA] job=%s got file_id=%s", job_id, fid)
                        except Exception as e:
                            log.warning("[SORA] job=%s assets fail: %s", job_id, e)

                    if not url_final:
                        try:
                            url_final = sora_download_location(video_id)
                        except Exception as e:
                            log.warning("[SORA] job=%s download fail: %s", job_id, e)

                    if not url_final:
                        try:
                            url_final = sora_content_location(video_id)
                        except Exception as e:
                            log.warning("[SORA] job=%s content fail: %s", job_id, e)

                    if url_final and url_final.startswith("http"):
                        SORA_JOBS[job_id]["url"] = url_final
                        break

                    try:
                        info = sora_status(video_id)
                    except Exception:
                        pass
                    time.sleep(1.0)

                # 如果没有 url 但有 file_id，用 Files 代理
                if not SORA_JOBS[job_id].get("url") and SORA_JOBS[job_id].get("file_id"):
                    SORA_JOBS[job_id]["url"] = f"/video/file/{job_id}/content"

                SORA_JOBS[job_id]["status"] = "done"
                SORA_JOBS[job_id]["progress"] = 100
                log.info(
                    "[SORA] job=%s DONE url=%s fid=%s",
                    job_id,
                    SORA_JOBS[job_id].get("url"),
                    SORA_JOBS[job_id].get("file_id"),
                )
                return

            if status in ("failed", "error", "cancelled"):
                raise RuntimeError(f"job failed: {status}")
            if time.time() > deadline:
                raise TimeoutError("sora timeout")

            log.info(
                "[SORA] job=%s tick status=%s prog=%s url=%s fid=%s",
                job_id,
                SORA_JOBS[job_id].get("status"),
                SORA_JOBS[job_id].get("progress"),
                SORA_JOBS[job_id].get("url"),
                SORA_JOBS[job_id].get("file_id"),
            )
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
# ============== APP & 中间件 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "[BOOT] ADU Backend starting on :8000 | sora_model=%s | text_model=%s",
        SORA_MODEL_DEFAULT,
        TEXT_MODEL_DEFAULT,
    )
    yield
    log.info("[BOOT] Shutting down backend")


app = FastAPI(title="ADU Sora Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


@app.middleware("http")
async def audit(req: Request, call_next):
    t0 = time.time()
    resp = await call_next(req)
    if req.url.path in ("/video", "/rt/intent", "/session", "/solara/chat", "/solara/photo"):
        ip = req.client.host if req.client else "-"
        log.info(
            "[AUDIT] ip=%s %s %s -> %s in %dms",
            ip,
            req.method,
            req.url.path,
            resp.status_code,
            int((time.time() - t0) * 1000),
        )
    return resp

# ============== Create & Status（并发闸 + 幂等 + 视频截帧当图片参考） ==============
@app.post("/video")
async def create_video(
    request: Request,
    prompt: str = Form(""),
    image_file: UploadFile = File(None),
    video_file: UploadFile = File(None),
    audio_file: UploadFile = File(None),
):
    """
    Sora 生成视频主入口：
    - prompt: 文本描述
    - image_file: 参考图片
    - video_file: 参考视频
    - audio_file: 可选，仅保存（方便后续扩展）
    """
    ip = request.client.host if request.client else "unknown"

    raw_img = raw_vid = raw_aud = None
    img_h = vid_h = ""

    # 读取文件用于幂等指纹 & 保存
    if image_file:
        raw_img = await image_file.read()
        img_h = _sha1_bytes(raw_img)
    if video_file:
        raw_vid = await video_file.read()
        vid_h = _sha1_bytes(raw_vid)
    if audio_file:
        raw_aud = await audio_file.read()

    # prompt 为空时兜底
    if not prompt.strip():
        if raw_img is not None or raw_vid is not None:
            prompt = "Generate a video based on the reference media."
        else:
            prompt = "Generate a video."

    # 幂等：同 IP + 同内容 2 分钟内只建一次
    _cleanup_recent()
    idem = _idem_key(ip, prompt, img_h, vid_h)
    rec = RECENT_KEYS.get(idem)
    if rec:
        log.info("[SORA] reuse recent job=%s for ip=%s", rec["job_id"], ip)
        return {
            "ok": True,
            "job_id": rec["job_id"],
            "status_url": f"/video/status/{rec['job_id']}",
            "status": "running",
        }

    # 并发闸
    if not SORA_SEM.acquire(blocking=False):
        return JSONResponse(
            {"ok": False, "error": "busy: one job already running"},
            status_code=429,
        )

    job_id = uuid.uuid4().hex
    SORA_JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "prompt": prompt,
        "url": None,
        "file_id": None,
        "video_id": None,
        "error": None,
        "created": int(time.time()),
        "ref_path": None,
        "ref_mime": None,
    }
    RECENT_KEYS[idem] = {"job_id": job_id, "ts": time.time()}

    # 保存上传到本地 uploads 目录（统一尺寸后给 Sora 参考）
    os.makedirs("uploads", exist_ok=True)
    ref_path = None
    ref_mime = None

    try:
        target_wh = _parse_size(SORA_SIZE_DEFAULT)

        img_suffix = Path(image_file.filename or "").suffix.lower() if image_file else ""
        vid_suffix = Path(video_file.filename or "").suffix.lower() if video_file else ""

        # ① 有图片就优先用图片做参考
        if raw_img is not None:
            raw_img_resized = _resize_image_bytes(raw_img, target_wh)
            fn = f"uploads/{job_id}_img{img_suffix or '.jpg'}"
            with open(fn, "wb") as f:
                f.write(raw_img_resized)
            log.info(
                "[UPLOAD] saved image(resized=%s) -> %s (%d bytes)",
                bool(target_wh),
                fn,
                len(raw_img_resized),
            )
            ref_path = fn
            ref_mime = _guess_mime_from_ext(fn)

        # ② 有视频：保存 + 缩放；如果前面没有图片，再从视频截一帧当图片参考
        if raw_vid is not None:
            src_fn = f"uploads/{job_id}_vid_src{vid_suffix or '.mp4'}"
            with open(src_fn, "wb") as f:
                f.write(raw_vid)

            dst_fn = f"uploads/{job_id}_vid{vid_suffix or '.mp4'}"
            final_fn = _resize_video_file(src_fn, dst_fn, target_wh)
            log.info(
                "[UPLOAD] saved video(resized=%s) -> %s (src=%s)",
                bool(target_wh),
                final_fn,
                src_fn,
            )

            # 没有图片参考时：从视频截一帧生成封面图
            if ref_path is None:
                thumb_fn = f"uploads/{job_id}_vid_thumb.jpg"
                thumb_path = _video_to_mosaic_image(
                    final_fn,
                    thumb_fn,
                    tiles=1,  # 只生成 1x1, 等于截一帧
                )
                if thumb_path and os.path.exists(thumb_path):
                    log.info("[UPLOAD] generated video thumbnail -> %s", thumb_path)
                    ref_path = thumb_path
                    ref_mime = "image/jpeg"
                else:
                    log.info("[UPLOAD] video thumbnail not generated, fallback to text-only")

        # ③ 音频只是顺手保存一下
        if raw_aud is not None:
            fn = f"uploads/{job_id}_aud"
            with open(fn, "wb") as f:
                f.write(raw_aud)
            log.info("[UPLOAD] saved audio -> %s (%d bytes)", fn, len(raw_aud))

    except Exception as e:
        log.warning("[UPLOAD] save failed for job=%s: %s", job_id, e)

    # 记录参考文件，worker 里会带给 Sora
    SORA_JOBS[job_id]["ref_path"] = ref_path
    SORA_JOBS[job_id]["ref_mime"] = ref_mime
    log.info("[SORA] job=%s ref_path=%s ref_mime=%s", job_id, ref_path, ref_mime)

    threading.Thread(
        target=bg_sora_worker,
        args=(job_id, prompt, 600, True),
        daemon=True,
    ).start()
    return {
        "ok": True,
        "job_id": job_id,
        "status_url": f"/video/status/{job_id}",
        "status": "running",
    }


@app.get("/video/status/{job_id}")
def video_status(job_id: str):
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
    return {
        "ok": True,
        "status": job.get("status"),
        "progress": job.get("progress"),
        "url": job.get("url"),
        "file_id": job.get("file_id"),
        "error": job.get("error"),
        "video_id": job.get("video_id"),
    }


# ============== Stream（直链 / Files / content/download 现场兜底） ==============
@app.get("/video/stream/{job_id}")
def video_stream(job_id: str, range_header: Optional[str] = Header(None, alias="Range")):
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)

    url = (job.get("url") or "").strip()
    fid = (job.get("file_id") or "").strip()
    vid = (job.get("video_id") or "").strip()

    if not url and not fid and not vid:
        return JSONResponse({"ok": False, "error": "video not ready"}, status_code=409)

    # A) file_id → 直接 Files 拉流
    if fid and (not url or url.startswith("/")):
        files_url = f"https://api.openai.com/v1/files/{fid}/content"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        if range_header:
            headers["Range"] = range_header
        r = requests.get(files_url, headers=headers, stream=True, timeout=120)
        log.info("[SORA] stream file_id job=%s status=%s", job_id, r.status_code)
        proxy_headers = {
            k: r.headers[k]
            for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"]
            if k in r.headers
        }

        def gen():
            for c in r.iter_content(128 * 1024):
                if c:
                    yield c

        return StreamingResponse(
            gen(),
            media_type=r.headers.get("Content-Type", "video/mp4"),
            headers=proxy_headers,
            status_code=(r.status_code if r.status_code in (200, 206) else 200),
        )

    # B) 直链 → 代理
    if url and url.startswith("http"):
        headers = {}
        if range_header:
            headers["Range"] = range_header
        r = requests.get(url, headers=headers, stream=True, timeout=120)
        log.info("[SORA] stream direct job=%s status=%s", job_id, r.status_code)
        proxy_headers = {
            k: r.headers[k]
            for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"]
            if k in r.headers
        }

        def gen2():
            for c in r.iter_content(128 * 1024):
                if c:
                    yield c

        return StreamingResponse(
            gen2(),
            media_type=r.headers.get("Content-Type", "video/mp4"),
            headers=proxy_headers,
            status_code=(r.status_code if r.status_code in (200, 206) else 200),
        )

    # C) vid → /content /download 兜底
    if vid:
        content_url = f"https://api.openai.com/v1/videos/{vid}/content"
        headers = {"Authorization": f"Bearer {SORA_API_KEY}", "OpenAI-Beta": "video-generation=v1"}
        if OPENAI_ORG_ID:
            headers["OpenAI-Organization"] = OPENAI_ORG_ID
        if range_header:
            headers["Range"] = range_header

        rc = requests.get(content_url, headers=headers, stream=True, timeout=120)
        log.info("[SORA] stream content job=%s status=%s", job_id, rc.status_code)

        if rc.status_code in (200, 206):
            proxy_headers = {
                k: rc.headers[k]
                for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"]
                if k in rc.headers
            }

            def gen_c():
                for c in rc.iter_content(128 * 1024):
                    if c:
                        yield c

            return StreamingResponse(
                gen_c(),
                media_type=rc.headers.get("Content-Type", "video/mp4"),
                headers=proxy_headers,
                status_code=rc.status_code,
            )

        if rc.status_code in (302, 303) and rc.headers.get("Location"):
            loc = rc.headers["Location"]
            r2 = requests.get(loc, headers={}, stream=True, timeout=120)
            log.info("[SORA] stream content->redirect job=%s status=%s", job_id, r2.status_code)
            proxy_headers = {
                k: r2.headers[k]
                for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"]
                if k in r2.headers
            }

            def gen_r():
                for c in r2.iter_content(128 * 1024):
                    if c:
                        yield c

            return StreamingResponse(
                gen_r(),
                media_type=r2.headers.get("Content-Type", "video/mp4"),
                headers=proxy_headers,
                status_code=(r2.status_code if r2.status_code in (200, 206) else 200),
            )

        loc2 = sora_download_location(vid)
        if loc2:
            r3 = requests.get(loc2, headers={}, stream=True, timeout=120)
            log.info("[SORA] stream download job=%s status=%s", job_id, r3.status_code)
            proxy_headers = {
                k: r3.headers[k]
                for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"]
                if k in r3.headers
            }

            def gen_d():
                for c in r3.iter_content(128 * 1024):
                    if c:
                        yield c

            return StreamingResponse(
                gen_d(),
                media_type=r3.headers.get("Content-Type", "video/mp4"),
                headers=proxy_headers,
                status_code=(r3.status_code if r3.status_code in (200, 206) else 200),
            )

    return JSONResponse({"ok": False, "error": "no playable url"}, status_code=409)


# ============== Files 代理 ==============
@app.get("/video/file/{job_id}/content")
def video_file_content(job_id: str, range_header: Optional[str] = Header(None, alias="Range")):
    job = SORA_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
    fid = job.get("file_id")
    if not fid:
        return JSONResponse({"ok": False, "error": "file_id missing"}, status_code=409)

    url = f"https://api.openai.com/v1/files/{fid}/content"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    if range_header:
        headers["Range"] = range_header
    r = requests.get(url, headers=headers, stream=True, timeout=120)

    proxy_headers = {
        k: r.headers[k]
        for k in ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"]
        if k in r.headers
    }

    def gen():
        for c in r.iter_content(128 * 1024):
            if c:
                yield c

    return StreamingResponse(
        gen(),
        media_type=r.headers.get("Content-Type", "video/mp4"),
        headers=proxy_headers,
        status_code=(r.status_code if r.status_code in (200, 206) else 200),
    )

# ============== Solara 多輪會話記憶（30 輪 + 照片引用） ==============
def _session_key(req: Request, session_id: str = "") -> str:
    """統一生成會話 key：優先 session_id，沒有就退回 ip"""
    ip = req.client.host if req.client else "unknown"
    sid = (session_id or "").strip()
    if sid:
        return sid
    return f"ip:{ip}"


def _append_turn(
    session_key: str,
    role: str,
    content: str,
    media: Optional[Dict[str, Any]] = None,
) -> None:
    """向某個會話追加一條對話輪次，並裁剪到最近 30 輪"""
    sess = SOLARA_SESSIONS.setdefault(session_key, {"turns": []})
    turns = sess.get("turns") or []
    turns.append(
        {
            "role": role,
            "content": content,
            "media": media or None,
            "ts": int(time.time()),
        }
    )
    # 僅保留最近 SOLARA_MAX_TURNS 輪
    if len(turns) > SOLARA_MAX_TURNS:
        turns = turns[-SOLARA_MAX_TURNS:]
    sess["turns"] = turns
    SOLARA_SESSIONS[session_key] = sess


def _build_messages_for_session(session_key: str, user_text: str) -> list[dict]:
    """
    根據會話 key 構建發給 GPT-4o 的 messages：
    - system：十五層高維度定義
    - history：最近 30 輪
    - 最後一條：當前 user 提問
    """
    sess = SOLARA_SESSIONS.get(session_key) or {"turns": []}
    turns = (sess.get("turns") or [])[-SOLARA_MAX_TURNS:]

    messages: list[dict] = []

    # 15 層高維度系統提示（文本版）
    messages.append(
        {
            "role": "system",
            "content": SOLARA_TEXT_SYSTEM_15L,
        }
    )

    # 過去輪次
    for t in turns:
        role = t.get("role") or "user"
        content = t.get("content") or ""
        if not content:
            continue
        messages.append({"role": role, "content": content})

    # 當前輸入
    messages.append({"role": "user", "content": user_text})
    return messages

# ============== 文本助手：十五層高維度定義（GPT Solara，全能版） ==============
SOLARA_TEXT_SYSTEM_15L = """
你是「GPT Solara」，一個中文優先、同時支持多語言的**全能型高維度助手**。
你要像 ChatGPT 一樣，能處理各類問題：通識問答、學習輔導、寫作與改稿、翻譯、多語對話、
職場與商業分析、產品與設計討論、程式碼與技術解惑、日常生活建議、情緒陪伴，以及短視頻 / Sora 影片創作。

同時，你特別擅長「把想法變成內容」：包括文案、腳本、口播、分鏡、Sora 提示語等。

請嚴格遵守以下 15 層原則：

1）角色與語氣
- 角色：專業又接地氣的「全能顧問 + 創作夥伴」。
- 面對技術 / 商業問題時：清晰、理性、有條理。
- 面對生活 / 情緒話題時：溫和、有耐心、先共情再給建議。
- 優先用自然口語中文，避免過度官方或教條式的語氣。

2）能力範圍與邊界
- 你可以：解釋概念、輔導學習、寫作 / 改寫 / 潤色、翻譯、給職場建議、做產品和設計討論、
  幫忙規劃行程、給生活建議、寫程式與 Debug、設計短視頻腳本與 Sora 提示語。
- 碰到你不能或不應回答的內容（醫療診斷、違法、暴力、成人內容等），要明確說明限制，
  同時提供安全、健康的替代方向。

3）輸出長度與節奏
- 一般回答：**3～6 段**，每段 1～3 句，有層次、有結構，不要一兩句就打發。
- 教學 / 方案 / 腳本類：可以用 **條列 + 小標題**，讓人一眼看到結構。
- 使用者明確要求「簡短 / 一句話」時，再刻意縮短。
- 長篇說明時，適度分段，避免一整大塊難以閱讀。

4）結構層（先總後分）
- 先用一小段話總結：「你現在在解決什麼問題，我要幫你做到什麼」。
- 接著分點展開，例如：
  - 思路總覽 / 核心結論
  - 具體步驟 / 建議
  - 若有需要，再給「實際範例 / 模板」。

5）多輪記憶與上下文
- 你會拿到最近約 30 輪對話，務必利用：
  - 不要重複詢問使用者已經清楚說過的條件。
  - 回答時要銜接前文，例如：「延續我們前面談到的那個 App 設計…」。
- 如果前文已經形成明確偏好（語氣、長度、風格、目標），後續自動沿用。

6）多模與素材整合（圖片 / 視頻 / 其他附件）
- 當上下文或系統備註指出「用戶上傳了圖片 / 視頻 / 素材」，要主動幫他想用途：
  - 在短視頻或 Sora 影片裡如何使用（開場、背景、過場、結尾定格等）。
  - 在文案 / 產品頁 / 簡報裡如何搭配文字。
- 儘量具體，而不是泛泛地說「可以當背景」。

7）短視頻 / 影片與 Sora 相關能力
- 當用戶是在聊短視頻、宣傳片、Sora 等內容時：
  - 像導演 + 編劇一樣寫腳本：分鏡頭（1、2、3…），每個鏡頭標明【畫面】【動作】【旁白 / 文案】。
  - 願意主動幫他把最終構思整理成精簡、可交給 Sora 使用的提示語。
  - 需要時，額外給一行：
    ^SORA_PROMPT: 後面接純文字的提示語（可中可英，但內容要完整）。

8）學習與工作輔助
- 面對「學習 / 考試 / 技能提升」：
  - 先評估他目前水平，再拆解成具體可行的小目標。
  - 可以設計練習題、複習重點、學習計畫（今天 / 本週 / 本月）。
- 面對「職場 / 專案 / 團隊合作」：
  - 優先給出可以立刻採取的幾個行動，而不是抽象大道理。

9）程式碼與技術問題
- 面對技術 / 程式碼相關問題：
  - 先用人話解釋概念，再給程式碼示例。
  - 示例要盡量完整、可運行，並簡短註釋。
  - 不要臆造不存在的函式或 API；有不確定的地方，請明說「可能需要查官方文件確認」。

10）翻譯與語言風格
- 支援中英互譯及多語場景，根據使用者要求切換語言。
- 文字創作時，可根據需求切換風格：正式、口語、活潑、穩重、廣告文案、講稿等。
- 做翻譯時，如果原文不通順，可以適度優化，使其更自然。

11）生活與情緒支持
- 當使用者表達壓力、焦慮、迷惘或沮喪時：
  - 先認真接住情緒，用簡單真誠的語句表達理解。
  - 然後再提出幾個溫和、可行的小步驟，不要強行雞湯。
- 不提供醫療 / 心理治療診斷，只能給一般性建議，必要時提醒尋求專業協助。

12）商業、產品與策略視角
- 當談到創業、定價、用戶增長、產品設計時：
  - 從「目標客群」「痛點 / 需求」「差異化」「成本與風險」四個角度來分析。
  - 不做精準財務預測，用區間與場景類比幫助判斷可行性。

13）工具與 API / 模型使用說明
- 用戶主動詢問技術細節（如 API、模型、架構）時，簡潔回答、舉例說明。
- 不要過度把話題拉回到內部實現；焦點始終放在幫用戶「把事情做好」。

14）安全與合規
- 不生成違法、暴力、成人、仇恨或令人強烈不適的內容。
- 對於涉風險的要求，要說明原因，並提供合法、安全、文明的替代方案。

15）自我檢查與小結
- 回答尾聲，可以簡短自查：「這樣的方案會不會太複雜？需要我幫你再簡化一版嗎？」
- 適時用 2～3 條 bullet 小結：「我們剛剛確定了什麼」「下一步你可以做什麼」。

總之，你要像一位穩定、可靠又有創造力的全能夥伴：既能陪他聊天，也能幫他把想法變成清晰的文字、計畫或影片構思。
"""


def solara_chat_once(session_key: str, user_text: str) -> str:
    """
    多輪文本聊天：
    - 使用 15 層高維度系統提示（SOLARA_TEXT_SYSTEM_15L）
    - 帶上最近約 30 輪上下文
    - 統一模型：gpt-4o
    - 自動把 user / assistant 回合寫回 SOLARA_SESSIONS
    """
    url = "https://api.openai.com/v1/chat/completions"
    model = TEXT_MODEL_DEFAULT or "gpt-4o"

    messages = _build_messages_for_session(session_key, user_text)

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 1024,
    }

    r = requests.post(url, headers=_chat_headers(), json=body, timeout=60)
    _log_http(r, f"TEXT.CHAT[{model}]")

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"message": r.text}
        raise RuntimeError(f"chat error: {err}")

    data = r.json()
    try:
        reply = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        raise RuntimeError(f"bad chat response: {data}")

    _append_turn(session_key, "user", user_text)
    _append_turn(session_key, "assistant", reply)

    return reply


@app.post("/solara/chat")
async def solara_chat_api(req: Request):
    """
    多輪文本聊天 + 15 層高維度定義：
    POST /solara/chat
    JSON:
    {
        "text": "...",
        "session_id": "可選，同一裝置/對話沿用即可"
    }
    """
    try:
        body = await req.json()
    except Exception:
        body = {}

    text = (body.get("text") or body.get("prompt") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "missing text"}, status_code=400)

    session_id = (body.get("session_id") or body.get("conversation_id") or "").strip()
    sk = _session_key(req, session_id)

    try:
        reply = solara_chat_once(sk, text)
        return {"ok": True, "reply": reply, "session_id": sk}
    except Exception as e:
        log.exception("[TEXT] solara_chat failed: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)

# ============== 照片上傳接口（用統一縮放技術 + 多輪記憶 + GPT-4o 回應） ==============
@app.post("/solara/photo")
async def solara_photo(
    request: Request,
    image_file: UploadFile = File(None),
    session_id: str = Form(""),
    note: str = Form(""),
):
    """
    照片上傳接口（給文本助手 / Realtime 當素材）：

    - 使用與 /video 相同的縮放技術（_resize_image_bytes + SORA_SIZE_DEFAULT）
    - 保存到 uploads/ 目錄
    - 寫入 SOLARA_LAST_MEDIA[session_key]，供 /rt/intent 自動帶圖到 Sora
    - 同時寫一條「上傳圖片」的 user 回合進多輪記憶
    - 並調用 GPT-4o 給一段建議：「這張圖怎麼用在短視頻 / Sora 影片裡」
    """
    sk = _session_key(request, session_id)

    # 1) 基本檢查：是否有文件
    if image_file is None:
        log.warning("[PHOTO] missing image_file field (sk=%s)", sk)
        return JSONResponse(
            {
                "ok": False,
                "error": "missing image_file (multipart/form-data, field name = image_file)",
            },
            status_code=400,
        )

    log.info(
        "[PHOTO] incoming upload sk=%s filename=%s content_type=%s",
        sk,
        image_file.filename,
        image_file.content_type,
    )

    try:
        raw = await image_file.read()
    except Exception as e:
        log.exception("[PHOTO] read image_file failed: %s", e)
        return JSONResponse({"ok": False, "error": f"read image failed: {e}"}, status_code=400)

    if not raw:
        log.warning("[PHOTO] empty image payload sk=%s", sk)
        return JSONResponse({"ok": False, "error": "empty image"}, status_code=400)

    log.info("[PHOTO] got %d bytes from client (sk=%s)", len(raw), sk)

    # 2) 縮放到與 Sora 相同的尺寸（若 Pillow 缺失，內部自動回退原圖）
    target_wh = _parse_size(SORA_SIZE_DEFAULT)
    resized = _resize_image_bytes(raw, target_wh)

    # 3) 生成文件名並寫入 uploads
    try:
        sid_hash = hashlib.sha1(sk.encode("utf-8")).hexdigest()
        ext = Path(image_file.filename or "").suffix.lower() or ".jpg"
        os.makedirs("uploads", exist_ok=True)
        fn = Path("uploads") / f"solara_{sid_hash}_{int(time.time())}{ext}"
        with open(fn, "wb") as f:
            f.write(resized)
        log.info("[PHOTO] saved image -> %s (%d bytes)", fn, len(resized))
    except Exception as e:
        log.exception("[PHOTO] save image failed: %s", e)
        return JSONResponse({"ok": False, "error": f"save image failed: {e}"}, status_code=500)

    mime = _guess_mime_from_ext(str(fn))
    sha1_val = _sha1_bytes(resized)

    # 4) 更新最近素材（給 /rt/intent 自動帶入 Sora）
    media_info = {
        "type": "image",
        "path": str(fn),
        "mime": mime,
        "sha1": sha1_val,
        "ts": int(time.time()),
        "note": note or "",
    }
    SOLARA_LAST_MEDIA[sk] = media_info

    # 5) 把「上傳圖片」記為一條 user 回合，方便模型知道有這張圖
    desc = f"[IMAGE_REF] 使用者剛上傳了一張圖片素材，用途說明: {note or '（尚未提供說明）'}。"
    _append_turn(sk, "user", desc, media=media_info)

    # 6) 通用助手回答：知道有一張圖片被上傳，但不強調能不能「看到」
    auto_prompt = (
        "我剛才上傳了一張圖片素材。"
        + (f" 補充說明: {note}。" if note else " 目前還沒有提供額外說明。")
        + (
            " 接下來請你直接幫我："
            " 先用 2～3 句，從一般情況出發，推測這類圖片在內容、構圖、色彩或用途上可能有哪些重點，"
            " 接著根據整段對話裡我的文字說明，回答問題或給出具體建議。"
            " 你可以幫我做理解分析、取標題、寫文案、給創作靈感或實用建議，都可以。"
            " 不要提到自己能不能看到圖片，也不要要求我『再描述一次圖片』；"
            " 只要根據目前已知的資訊，盡量給出有幫助的回應即可。"
        )
    )

    try:
        reply = solara_chat_once(sk, auto_prompt)
    except Exception as e:
        log.exception("[PHOTO] solara_photo chat failed: %s", e)
        reply = "圖片已上傳並記入本次對話，如果之後需要再問這張圖片的問題，可以再提醒我。"

    return {
        "ok": True,
        "session_id": sk,
        "image_saved_path": str(fn),
        "image_mime": mime,
        "note": note,
        "reply": reply,
    }

# ============== Realtime（ephemeral key） & Intent（連接 15 層 + 照片） ==============

# Realtime 工作定義：GPT Solara 語音版 · 通用助手
SORA_REALTIME_INSTRUCTIONS = """
你是「GPT Solara」的即時語音版本，透過 Realtime 模型和使用者自然對話。  
你是全能型語音助手，可以幫忙聊天、查資訊、講解知識、討論程式與產品、一起規劃生活與工作。

核心原則：

1. 語氣自然、有耐心，像一位可靠的朋友 + 顧問，優先使用中文口語。  
2. 先聽懂對方的目的，是想聊天、學習、問技術、問生意、還是只是發牢騷。  
3. 回覆時儘量有條理，不要只回一兩句敷衍了事；必要時可以幫他整理成步驟或重點清單。  
4. 你可以回答各種領域的問題：學校作業、語言學習、寫作修改、程式碼、商業構想、生活建議等。  
5. **只有在使用者主動提到「短視頻 / 影片 / Sora / 生成視頻」時**，你才切換成導演 + 編劇模式，幫他設計腳本與提示語；平時不要強行把話題往影片上帶。  
6. 尊重多輪對話上下文，記住他剛剛分享過的資訊，不要頻繁要求重複。  
7. 若對方情緒低落或壓力很大，先同理，再給溫和、具體的小建議。  
8. 遵守安全與合規要求，不生成違法、暴力、成人或仇恨內容。

總之，把自己當成「隨身口袋顧問 + 聊天夥伴」，而不是只會做短視頻的工具。
"""

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
    key = (
        (data.get("client_secret") or {}).get("value")
        or data.get("ephemeral_key")
        or data.get("token")
    )
    if not key:
        return None, "missing ephemeral key"
    return key, None


@app.post("/session")
async def session_post(req: Request):
    try:
        b = await req.json()
    except Exception:
        b = {}

    # 用默认的 REALTIME_MODEL_DEFAULT / REALTIME_VOICE_DEFAULT
    model = (b.get("model") or REALTIME_MODEL_DEFAULT).strip()
    voice = (b.get("voice") or REALTIME_VOICE_DEFAULT).strip()

    # 默认挂上語音版 15 層說明
    key, err = _realtime_ephemeral(model, voice, instructions=SORA_REALTIME_INSTRUCTIONS)
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


@app.post("/rt/intent")
async def rt_intent(req: Request):
    """
    由前端在检测到 ^SORA_PROMPT: 行时调用：
    body = {
        "prompt": "...從指令裡抽出來的最終文案...",
        "session_id": "可選，與 /solara/chat /solara/photo 對齊"
    }

    - 若找到對應 session 的最近一張圖片，會自動帶入 SORA_JOBS[job_id]["ref_path"]，
      讓這張圖作為 Sora 的 input_reference。
    """
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

    # 嘗試取最近一張圖片素材，作為本次 Sora 的參考
    last_media = SOLARA_LAST_MEDIA.get(sk) or {}
    ref_path = None
    ref_mime = None
    if last_media.get("type") == "image":
        ref_path = last_media.get("path")
        if ref_path:
            ref_mime = _guess_mime_from_ext(ref_path)

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
    }

    # 把這次「生成影片意圖」也記入會話，用於之後的 30 輪上下文
    _append_turn(
        sk,
        "user",
        f"^SORA_INTENT: {prompt}",
        media=(last_media if last_media else None),
    )

    threading.Thread(
        target=bg_sora_worker,
        args=(job_id, prompt, 600, True),
        daemon=True,
    ).start()
    return {
        "ok": True,
        "job_id": job_id,
        "status_url": f"/video/status/{job_id}",
        "session_id": sk,
    }

# ============== Health & RUN ==============
@app.get("/health")
def health():
    return {
        "ok": True,
        "sora_model_default": SORA_MODEL_DEFAULT,
        "text_model_default": TEXT_MODEL_DEFAULT or "gpt-4o",
        "video_endpoint": "/video",
        "text_endpoint": "/solara/chat",
        "photo_endpoint": "/solara/photo",
        "realtime_session_endpoint": "/session",
        "rt_intent_endpoint": "/rt/intent",
        "concurrency": SORA_CONCURRENCY,
        "max_turns": SOLARA_MAX_TURNS,
    }


if __name__ == "__main__":
    import uvicorn

    log.info("Starting ADU Sora Backend on :8000")
    uvicorn.run(
        "server_session:app",   # 文件叫 server_session.py
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=False,
    )

