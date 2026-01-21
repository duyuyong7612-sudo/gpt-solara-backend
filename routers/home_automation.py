from __future__ import annotations

import json
import time
import uuid
import re
import os
import threading
from typing import Any, Dict, Optional, List

import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

router = APIRouter(prefix="/home", tags=["home-automation"])

# -----------------------------
# Storage (in-memory)
# -----------------------------

_HOME_BINDINGS: Dict[str, Dict[str, Any]] = {}
_HOME_LOCK = threading.Lock()
HOME_BIND_TTL_SEC = int(os.getenv("HOME_BIND_TTL_SEC") or "86400")  # 24h default

# ✅ Dev fallback (to support legacy clients that DO NOT send client_id):
# - HOME_DEV_FALLBACK_UNIQUE=1 : if exactly one CID binding exists, use it as fallback
# - HOME_DEV_FALLBACK_LATEST=1 : if multiple CID bindings exist, pick latest updated_at (less safe; dev only)
HOME_DEV_FALLBACK_UNIQUE = (os.getenv("HOME_DEV_FALLBACK_UNIQUE") or "").strip().lower() in ("1", "true", "yes", "on")
HOME_DEV_FALLBACK_LATEST = (os.getenv("HOME_DEV_FALLBACK_LATEST") or "").strip().lower() in ("1", "true", "yes", "on")


def _now() -> float:
    return time.time()


def _cleanup() -> None:
    now = _now()
    with _HOME_LOCK:
        for k in list(_HOME_BINDINGS.keys()):
            b = _HOME_BINDINGS.get(k) or {}
            ts = float(b.get("updated_at") or 0)
            if ts and (now - ts) > HOME_BIND_TTL_SEC:
                _HOME_BINDINGS.pop(k, None)


def _client_ip(req: Request) -> str:
    """Prefer X-Forwarded-For if behind proxy, else req.client.host."""
    try:
        xff = (req.headers.get("x-forwarded-for") or "").strip()
        if xff:
            return xff.split(",")[0].strip()
    except Exception:
        pass

    try:
        if req.client and req.client.host:
            return req.client.host
    except Exception:
        pass
    return "unknown"


def _client_id_from(req: Request, client_id: Optional[str]) -> Optional[str]:
    if client_id and str(client_id).strip():
        return str(client_id).strip()
    h = (req.headers.get("x-client-id") or req.headers.get("x-device-id") or "").strip()
    return h or None


def _binding_key(req: Request, client_id: Optional[str]) -> str:
    cid = _client_id_from(req, client_id)
    if cid:
        return f"cid:{cid}"
    return f"ip:{_client_ip(req)}"


def _norm_url(u: str) -> str:
    s = (u or "").strip()
    while s.endswith("/"):
        s = s[:-1]
    return s


_HOSTPORT_RE = re.compile(r"^[A-Za-z0-9.\-]+(:\d+)?$")


def _looks_like_http_url(s: str) -> bool:
    ss = (s or "").strip().lower()
    return ss.startswith("http://") or ss.startswith("https://")


def _safe_url_or_none(s: str) -> Optional[str]:
    """
    ✅ 允许用户/二维码不给 scheme：
      - "192.168.0.206:8008" -> "http://192.168.0.206:8008"
      - "raspberrypi.local:8008" -> "http://raspberrypi.local:8008"
    """
    u = _norm_url(s)
    if not u:
        return None

    if _looks_like_http_url(u):
        return u

    if _HOSTPORT_RE.match(u):
        return "http://" + u

    return None


def _extract_gateway_from_profile(profile_json: str) -> Optional[str]:
    try:
        obj = json.loads(profile_json)
        if isinstance(obj, dict):
            for k in ("gateway", "gateway_url", "gatewayBaseURL", "gateway_base_url", "pi_base_url", "base_url"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    uu = _safe_url_or_none(v)
                    if uu:
                        return _norm_url(uu)
    except Exception:
        pass
    return None


def _extract_commands_from_profile(profile_json: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not profile_json:
        return out
    try:
        obj = json.loads(profile_json)
        if not isinstance(obj, dict):
            return out
        cmds = obj.get("commands")
        if not isinstance(cmds, list):
            return out
        for c in cmds:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            method = str(c.get("method") or "POST").strip().upper()
            path = str(c.get("path") or "").strip()
            if not name or not path:
                continue
            if not path.startswith("/"):
                path = "/" + path
            if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                method = "POST"
            out[name] = {"method": method, "path": path}
    except Exception:
        pass
    return out


def _extract_device_name(profile_json: str) -> Optional[str]:
    try:
        obj = json.loads(profile_json)
        if isinstance(obj, dict):
            v = obj.get("device") or obj.get("name")
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass
    return None


def _extract_examples(profile_json: str) -> Optional[list]:
    try:
        obj = json.loads(profile_json)
        if isinstance(obj, dict):
            ex = obj.get("examples")
            if isinstance(ex, list) and ex:
                return ex
    except Exception:
        pass
    return None


# -----------------------------
# Dev fallback helpers
# -----------------------------

def _unique_cid_bindings_snapshot() -> List[Dict[str, Any]]:
    """
    Return a de-duplicated list of bindings for keys that start with 'cid:'.
    The in-memory table also stores ip aliases pointing to the same record,
    so we dedupe by bind_id if present, else by gateway+updated_at.
    """
    _cleanup()
    items: List[Dict[str, Any]] = []
    seen: set = set()

    with _HOME_LOCK:
        for k, b in _HOME_BINDINGS.items():
            if not isinstance(k, str) or not k.startswith("cid:"):
                continue
            if not isinstance(b, dict):
                continue
            bid = str(b.get("bind_id") or "")
            if bid:
                if bid in seen:
                    continue
                seen.add(bid)
                items.append(b)
            else:
                sig = f"{b.get('gateway')}|{b.get('updated_at')}"
                if sig in seen:
                    continue
                seen.add(sig)
                items.append(b)

    return items


def _dev_fallback_binding_if_any(req: Request, client_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Use only when NO client_id is provided (legacy clients).
    - UNIQUE: if exactly one CID binding exists, return it.
    - LATEST: if multiple CID bindings exist, return latest updated_at (dev-only).
    """
    cid = _client_id_from(req, client_id)
    if cid:
        return None

    if not (HOME_DEV_FALLBACK_UNIQUE or HOME_DEV_FALLBACK_LATEST):
        return None

    lst = _unique_cid_bindings_snapshot()
    if not lst:
        return None

    if HOME_DEV_FALLBACK_UNIQUE:
        if len(lst) == 1:
            return lst[0]
        # multiple -> do not guess
        return None

    # HOME_DEV_FALLBACK_LATEST
    best = None
    best_ts = -1.0
    for b in lst:
        try:
            ts = float(b.get("updated_at") or 0)
        except Exception:
            ts = 0
        if ts >= best_ts:
            best_ts = ts
            best = b
    return best


def home_has_binding(req: Request, client_id: Optional[str] = None) -> bool:
    _cleanup()
    key = _binding_key(req, client_id)
    with _HOME_LOCK:
        ok = key in _HOME_BINDINGS
    if ok:
        return True

    # ✅ dev fallback for legacy clients
    fb = _dev_fallback_binding_if_any(req, client_id)
    return fb is not None


def _get_binding(req: Request, client_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    _cleanup()
    key = _binding_key(req, client_id)
    with _HOME_LOCK:
        b = _HOME_BINDINGS.get(key)

    if b:
        return b

    # ✅ dev fallback for legacy clients
    fb = _dev_fallback_binding_if_any(req, client_id)
    return fb


# -----------------------------
# Instruction builder (Realtime system prompt)
# -----------------------------

def home_instructions_for_request(req: Request, client_id: Optional[str] = None) -> Optional[str]:
    """
    ✅ 对齐前端 HOME_CMD 协议（iOS parser 期待字段）
    - 只输出一行：HOME_CMD: + JSON
    - JSON 字段固定：cmd, device_id, confidence, reason
    - 不确定用 ask_clarify（不要用 noop）
    - 助手名字：阿杜
    """
    b = _get_binding(req, client_id)
    if not b:
        return None

    device_display = (b.get("display_name") or b.get("device_name") or b.get("device") or "家居设备").strip()
    commands: Dict[str, Dict[str, str]] = b.get("commands") or {}
    cmd_list = list(commands.keys()) if commands else ["light_on", "light_off", "light_toggle"]
    examples = b.get("examples") or ["打开灯", "关灯", "切换灯"]

    device_id = "200"
    allowed = cmd_list[:]
    if "ask_clarify" not in allowed:
        allowed.append("ask_clarify")

    return (
        "你叫“阿杜”，是【家居控制专用助手】。\n"
        "你的唯一任务：把用户话语的控制意图转换为可执行命令。\n"
        "\n"
        f"【已绑定设备】{device_display}\n"
        f"【允许 cmd】{', '.join(allowed)}\n"
        f"【示例】{', '.join([str(x) for x in examples])}\n"
        "\n"
        "【输出协议（必须严格遵守）】\n"
        "1) 你每次回复只能输出【一行】。\n"
        "2) 必须以 HOME_CMD: 开头，后面紧跟严格 JSON。\n"
        "3) JSON 必须包含字段：cmd, device_id, confidence, reason。\n"
        f"4) device_id 固定为 \"{device_id}\"。\n"
        "5) 除这一行外，禁止输出任何其它文字（不解释、不寒暄、不换行）。\n"
        "\n"
        "【JSON 模板】\n"
        f'HOME_CMD: {{"cmd":"<cmd>","device_id":"{device_id}","confidence":0.0,"reason":"..."}}\n'
        "\n"
        "【判定规则】\n"
        "- 意图明确：cmd=light_on/light_off/light_toggle，confidence 给 0.6~1.0，reason 简短说明。\n"
        "- 意图不明确/矛盾/不是控制：cmd=ask_clarify，confidence=0.2，reason 写一句你要问的问题。\n"
        "\n"
        "【语义映射】\n"
        "- 开灯/打开灯/亮一点/太暗了 => light_on\n"
        "- 关灯/关闭灯/暗一点/太亮了 => light_off\n"
        "- 切换/开关一下/翻转 => light_toggle\n"
    )


# -----------------------------
# Models
# -----------------------------

class GatewayProfileIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    client_id: Optional[str] = Field(default=None, description="Stable client identifier. Prefer passing this in production.")
    session_id: Optional[str] = None

    gateway_base_url: Optional[str] = None
    gatewayBaseURL: Optional[str] = None
    gateway_url: Optional[str] = None
    gatewayUrl: Optional[str] = None
    gateway: Optional[str] = None
    pi_base_url: Optional[str] = None
    base_url: Optional[str] = None

    gateway_token: Optional[str] = None
    bearer: Optional[str] = None
    token: Optional[str] = None

    profile_json: Optional[str] = None
    profileJSON: Optional[str] = None
    profile: Optional[str] = None
    device_profile: Optional[Dict[str, Any]] = None

    profile_media_url: Optional[str] = None
    profileMediaURL: Optional[str] = None

    display_name: Optional[str] = None
    device_name: Optional[str] = None
    device: Optional[str] = None
    name: Optional[str] = None


class DispatchIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    client_id: Optional[str] = None
    session_id: Optional[str] = None
    cmd: Optional[str] = None
    command: Optional[str] = None
    action: Optional[str] = None
    text: Optional[str] = None
    utterance: Optional[str] = None
    prompt: Optional[str] = None


class HomeSessionIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    client_id: Optional[str] = None
    model: Optional[str] = None
    modalities: Optional[List[str]] = None
    voice: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------

def _extract_gateway_from_body(raw: Dict[str, Any], profile_json: str) -> Optional[str]:
    for k in ("gateway_base_url", "gatewayBaseURL", "gateway_url", "gatewayUrl", "gateway", "pi_base_url", "base_url"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            u = _safe_url_or_none(v)
            if u:
                return _norm_url(u)

    g2 = _extract_gateway_from_profile(profile_json)
    if g2:
        return _norm_url(g2)
    return None


def _extract_profile_json(raw: Dict[str, Any]) -> str:
    if isinstance(raw.get("device_profile"), dict):
        try:
            return json.dumps(raw["device_profile"], ensure_ascii=False)
        except Exception:
            pass

    for k in ("profile_json", "profileJSON", "profile"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_token(raw: Dict[str, Any]) -> Optional[str]:
    for k in ("gateway_token", "token", "bearer"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _headers_for_gateway(token: Optional[str]) -> Dict[str, str]:
    t = (token or "").strip()
    if not t:
        return {}
    if t.lower().startswith("bearer "):
        return {"Authorization": t}
    return {"Authorization": f"Bearer {t}"}


# -----------------------------
# Bind endpoints
# -----------------------------

@router.post("/gateway_profile")
async def set_gateway_profile(req: Request, body: GatewayProfileIn):
    raw = body.model_dump(exclude_none=True)

    key = _binding_key(req, raw.get("client_id"))
    ip_key = f"ip:{_client_ip(req)}"

    profile_json = _extract_profile_json(raw)
    token = _extract_token(raw)

    display_name = str(raw.get("display_name") or "").strip() or None

    raw_device_name = (
        str(raw.get("device_name") or raw.get("device") or raw.get("name") or "").strip()
        or None
    )

    gateway = _extract_gateway_from_body(raw, profile_json)
    if not gateway:
        return JSONResponse(
            {"ok": False, "error": "missing gateway url", "hint": "请提供 gateway_base_url/gateway 或在 profile_json 里包含 gateway 字段"},
            status_code=400,
        )

    commands = _extract_commands_from_profile(profile_json)
    device_name = _extract_device_name(profile_json) or raw_device_name
    examples = _extract_examples(profile_json)

    bind_id = uuid.uuid4().hex
    updated_at = _now()

    health_ok = None
    try:
        r = requests.get(gateway + "/health", headers=_headers_for_gateway(token), timeout=2.5)
        health_ok = 200 <= r.status_code < 300
    except Exception:
        health_ok = None

    rec = {
        "bind_id": bind_id,
        "key": key,
        "ip_key": ip_key,
        "gateway": gateway,
        "token": token,
        "profile_json": profile_json,
        "profile_media_url": str(raw.get("profile_media_url") or raw.get("profileMediaURL") or "").strip() or None,
        "commands": commands,
        "device_name": device_name,
        "display_name": display_name,
        "examples": examples,
        "health_ok": health_ok,
        "updated_at": updated_at,
    }

    _cleanup()
    with _HOME_LOCK:
        _HOME_BINDINGS[key] = rec
        _HOME_BINDINGS[ip_key] = rec  # IP alias

    return {
        "ok": True,
        "bind_id": bind_id,
        "key": key,
        "gateway": gateway,
        "device_name": device_name,
        "display_name": display_name,
        "commands": list(commands.keys()) if commands else [],
        "health_ok": health_ok,
        "updated_at": updated_at,
    }


@router.post("/bind")
async def bind_alias(req: Request, body: GatewayProfileIn):
    return await set_gateway_profile(req, body)


@router.post("/device_profile")
async def device_profile_alias(req: Request, body: GatewayProfileIn):
    return await set_gateway_profile(req, body)


@router.get("/gateway_profile")
async def get_gateway_profile(req: Request, client_id: Optional[str] = None):
    key = _binding_key(req, client_id)
    b = _get_binding(req, client_id)
    if not b:
        return {"ok": True, "bound": False, "key": key, "binding": None}

    b2 = dict(b)
    if "token" in b2:
        b2["token"] = "***"
    return {"ok": True, "bound": True, "key": key, "binding": b2}


@router.get("/instructions")
async def get_instructions(req: Request, client_id: Optional[str] = None):
    inst = home_instructions_for_request(req, client_id=client_id)
    if not inst:
        return {"ok": True, "bound": False, "instructions": None}
    return {"ok": True, "bound": True, "instructions": inst}


# -----------------------------
# Home Realtime Session
# -----------------------------

@router.post("/session")
async def home_session(req: Request, body: HomeSessionIn):
    raw = body.model_dump(exclude_none=True)
    client_id = raw.get("client_id")

    inst = home_instructions_for_request(req, client_id=client_id)
    if not inst:
        return JSONResponse(
            {"ok": False, "error": "not_bound", "hint": "请先调用 /home/gateway_profile 绑定网关与设备参数"},
            status_code=409,
        )

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return JSONResponse({"ok": False, "error": "missing_OPENAI_API_KEY"}, status_code=500)

    model = (raw.get("model") or os.getenv("HOME_REALTIME_MODEL") or os.getenv("REALTIME_MODEL") or "gpt-realtime").strip()

    modalities = raw.get("modalities") or ["audio", "text"]
    if not isinstance(modalities, list) or not modalities:
        modalities = ["audio", "text"]

    payload: Dict[str, Any] = {
        "model": model,
        "modalities": modalities,
        "instructions": inst,
        "turn_detection": {"type": "server_vad"},
        "temperature": 0.1,
        "max_response_output_tokens": 120,
    }

    voice = (raw.get("voice") or os.getenv("REALTIME_VOICE") or "").strip()
    if voice and ("audio" in modalities):
        payload["voice"] = voice

    sessions_url = os.getenv("OPENAI_REALTIME_SESSIONS_URL", "https://api.openai.com/v1/realtime/sessions")

    try:
        r = requests.post(
            sessions_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": "realtime_session_request_failed", "detail": str(e)}, status_code=500)

    if r.status_code >= 400:
        return JSONResponse(
            {"ok": False, "error": "realtime_session_create_failed", "status": r.status_code, "detail": r.text},
            status_code=500,
        )

    data = r.json()

    rtc_url = data.get("rtc_url") or data.get("url") or data.get("webrtc_url")
    client_secret = data.get("client_secret") or {}
    ephemeral_key = data.get("ephemeral_key") or (client_secret.get("value") if isinstance(client_secret, dict) else None)

    data["ok"] = True
    data["profile"] = "home"
    data["binding_key"] = _binding_key(req, client_id)
    data["client_id"] = _client_id_from(req, client_id)
    if rtc_url:
        data["rtc_url"] = rtc_url
    if ephemeral_key:
        data["ephemeral_key"] = ephemeral_key

    return JSONResponse(data)


# -----------------------------
# Dispatch endpoint
# -----------------------------

_CMD_RE = re.compile(r"HOME_CMD:\s*(\{.*?\})", re.I)


def _infer_cmd_from_text(t: str) -> Optional[str]:
    s = (t or "").strip().lower()
    if not s:
        return None
    if any(k in s for k in ("开灯", "打开灯", "turn on", "light on", "switch on", "on the light")):
        return "light_on"
    if any(k in s for k in ("关灯", "关闭灯", "turn off", "light off", "switch off", "off the light")):
        return "light_off"
    if any(k in s for k in ("切换", "toggle", "翻转")):
        return "light_toggle"
    return None


def _extract_cmd_from_any(raw: Dict[str, Any]) -> Optional[str]:
    for k in ("cmd", "command", "action"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for k in ("text", "utterance", "prompt"):
        v = raw.get(k)
        if isinstance(v, str) and v.strip():
            m = _CMD_RE.search(v)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    if isinstance(obj, dict) and isinstance(obj.get("cmd"), str):
                        return obj["cmd"].strip()
                except Exception:
                    pass
            c = _infer_cmd_from_text(v)
            if c:
                return c

    return None


@router.post("/dispatch")
async def dispatch(req: Request, body: DispatchIn):
    raw = body.model_dump(exclude_none=True)

    b = _get_binding(req, raw.get("client_id"))
    if not b:
        return JSONResponse(
            {"ok": False, "error": "not_bound", "hint": "请先调用 /home/gateway_profile 绑定网关与设备参数"},
            status_code=409,
        )

    cmd = _extract_cmd_from_any(raw)
    if not cmd:
        return JSONResponse({"ok": False, "error": "missing cmd/action/text"}, status_code=400)

    cmd = str(cmd).strip()

    commands: Dict[str, Dict[str, str]] = b.get("commands") or {}
    if not commands:
        commands = {
            "light_on": {"method": "POST", "path": "/light/on"},
            "light_off": {"method": "POST", "path": "/light/off"},
            "light_toggle": {"method": "POST", "path": "/light/toggle"},
        }

    if cmd in ("noop", "ask_clarify"):
        return {"ok": True, "cmd": cmd, "skipped": True, "reason": cmd}

    spec = commands.get(cmd)
    if not spec:
        return JSONResponse({"ok": False, "error": "unknown_cmd", "cmd": cmd, "known": list(commands.keys())}, status_code=400)

    method = (spec.get("method") or "POST").upper()
    path = (spec.get("path") or "").strip()
    if not path.startswith("/"):
        path = "/" + path if path else ""

    gateway = _safe_url_or_none(str(b.get("gateway") or ""))
    if not gateway:
        return JSONResponse({"ok": False, "error": "invalid_gateway"}, status_code=500)

    url = gateway + path
    headers = _headers_for_gateway(b.get("token"))

    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=4.5)
        else:
            r = requests.post(url, headers=headers, timeout=4.5)

        ok = 200 <= r.status_code < 300
        try:
            data = r.json()
        except Exception:
            data = (r.text or "")[:800]

        return {"ok": ok, "cmd": cmd, "url": url, "status": r.status_code, "resp": data}
    except Exception as e:
        return JSONResponse({"ok": False, "error": "gateway_request_failed", "detail": str(e), "cmd": cmd, "url": url}, status_code=502)

