# billing.py (DEV / SAFE ROUTER)
# -------------------------------------------------------------------
# 目标：
# 1) 让 server_session.py 的 `from billing import router as billing_router` 正常工作
# 2) 提供 /billing/me /billing/credits /billing/ingest（购买入库）
# 3) 兼容前端 VoiceBillingGuard：/billing/voice/start|ping|end（也做了若干别名路由）
#
# 说明：
# - 这是「开发环境」最小可用版本：内存记账，重启会丢
# - 生产环境请替换成 DB/Redis + 真实 IAP/RevenueCat 校验逻辑
# -------------------------------------------------------------------

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# pydantic v1/v2 兼容：extra=ignore
try:
    from pydantic import BaseModel, ConfigDict  # type: ignore

    class _BaseModel(BaseModel):
        model_config = ConfigDict(extra="ignore")
except Exception:  # pragma: no cover
    from pydantic import BaseModel  # type: ignore

    class _BaseModel(BaseModel):
        class Config:  # type: ignore
            extra = "ignore"


router = APIRouter(prefix="/billing", tags=["billing"])

# -----------------------------
# In-memory store (DEV only)
# -----------------------------
_USERS: Dict[str, Dict[str, Any]] = {}

def _now() -> int:
    return int(time.time())

def _user_key(req: Request) -> str:
    # 优先使用前端持久化的 Client-ID（强烈建议你在 iOS 里加一个 UUID 并带上 x-client-id）
    cid = (req.headers.get("x-client-id") or req.headers.get("X-Client-Id") or "").strip()
    if cid:
        return cid
    # 否则退化为 IP（Wi-Fi 变更会导致换号）
    return (req.client.host if req.client else "unknown")

def _defaults() -> Dict[str, int]:
    return {
        "text": int(os.getenv("DEFAULT_TEXT_CREDITS", "20")),
        "voice": int(os.getenv("DEFAULT_VOICE_CREDITS", "10")),
        "video": int(os.getenv("DEFAULT_VIDEO_CREDITS", "3")),
    }

def _get_or_create_user(req: Request) -> Dict[str, Any]:
    k = _user_key(req)
    u = _USERS.get(k)
    if u:
        return u

    d = _defaults()
    u = {
        "user_key": k,
        "created_at": _now(),
        "plan": "free",
        "is_subscribed": False,
        "credits_text": d["text"],
        "credits_voice": d["voice"],
        "credits_video": d["video"],
        # 购买去重
        "tx_ids": set(),          # type: ignore
        # voice session
        "voice_calls": {},        # call_id -> {"last": ts}
    }
    _USERS[k] = u
    return u

def _resp_me(u: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "user_key": u["user_key"],
        "plan": u.get("plan", "free"),
        "is_subscribed": bool(u.get("is_subscribed")),
        "credits": {
            "text": int(u.get("credits_text", 0)),
            "voice": int(u.get("credits_voice", 0)),
            "video": int(u.get("credits_video", 0)),
        },
        "ts": _now(),
    }

def _bypass_gates() -> bool:
    return os.getenv("BILLING_BYPASS_GATES", "0") in ("1", "true", "TRUE", "yes", "YES")

def _insufficient(kind: str, left: int, need: int = 1) -> JSONResponse:
    return JSONResponse(
        status_code=402,
        content={
            "ok": False,
            "error": "insufficient_credits",
            "kind": kind,
            "left": int(left),
            "need": int(need),
            "ts": _now(),
        },
    )

# -----------------------------
# Public APIs
# -----------------------------

@router.get("/health")
def billing_health():
    return {"ok": True, "ts": _now()}

@router.get("/me")
def billing_me(req: Request):
    u = _get_or_create_user(req)
    return _resp_me(u)

@router.get("/credits")
def billing_credits(req: Request):
    u = _get_or_create_user(req)
    return {
        "ok": True,
        "text": int(u.get("credits_text", 0)),
        "voice": int(u.get("credits_voice", 0)),
        "video": int(u.get("credits_video", 0)),
        "ts": _now(),
    }

# -----------------------------
# Purchase ingest
# -----------------------------

class IngestPurchaseReq(_BaseModel):
    productId: str
    txId: str
    expiresMs: Optional[int] = None

@router.post("/ingest")
def billing_ingest(req: Request, body: IngestPurchaseReq):
    """购买成功后回传入库（DEV 版：只做本地记账 + 发放 credits）"""
    u = _get_or_create_user(req)

    pid = (body.productId or "").strip()
    tx = (body.txId or "").strip()
    if not pid or not tx:
        return JSONResponse(status_code=400, content={"ok": False, "error": "missing productId/txId"})

    txs = u.setdefault("tx_ids", set())
    # 兼容：如果旧数据是 list
    if isinstance(txs, list):
        txs = set(txs)
        u["tx_ids"] = txs

    if tx in txs:
        return {
            "ok": True,
            "dedup": True,
            "user_key": u["user_key"],
            "productId": pid,
            "txId": tx,
            "credits": _resp_me(u)["credits"],
            "ts": _now(),
        }

    add_video = 0
    add_voice = 0
    add_text = 0
    subscribed = False

    pid_lower = pid.lower()

    # —— 视频包（按你当前商品 id：solara_video_pack_5 / solara_video_pack_10）——
    if "video_pack_10" in pid_lower or pid_lower.endswith("_10"):
        add_video = 10
    elif "video_pack_5" in pid_lower or pid_lower.endswith("_5"):
        add_video = 5

    # —— 订阅（如果你后续有 pro/weekly 等）——
    if any(k in pid_lower for k in ("pro", "sub", "weekly", "monthly", "year")):
        subscribed = True

    u["credits_video"] = int(u.get("credits_video", 0)) + add_video
    u["credits_voice"] = int(u.get("credits_voice", 0)) + add_voice
    u["credits_text"] = int(u.get("credits_text", 0)) + add_text

    if subscribed:
        u["is_subscribed"] = True
        u["plan"] = "pro"
        if body.expiresMs is not None:
            u["sub_expires_ms"] = int(body.expiresMs)

    txs.add(tx)

    return {
        "ok": True,
        "user_key": u["user_key"],
        "productId": pid,
        "txId": tx,
        "added": {"text": add_text, "voice": add_voice, "video": add_video},
        "plan": u.get("plan", "free"),
        "is_subscribed": bool(u.get("is_subscribed")),
        "credits": _resp_me(u)["credits"],
        "ts": _now(),
    }

# -----------------------------
# Voice gating endpoints
# - 兼容前端 VoiceBillingGuard / SolaraPaywallCenter.requestStartVoice
# -----------------------------

class VoiceStartReq(_BaseModel):
    # 允许前端不传任何字段
    pass

class VoicePingReq(_BaseModel):
    callId: str

class VoiceEndReq(_BaseModel):
    callId: str

def _voice_start_impl(req: Request):
    u = _get_or_create_user(req)

    if _bypass_gates() or u.get("is_subscribed"):
        call_id = uuid.uuid4().hex
        u.setdefault("voice_calls", {})[call_id] = {"last": _now()}
        return {"ok": True, "callId": call_id, "credits": _resp_me(u)["credits"], "ts": _now()}

    left = int(u.get("credits_voice", 0))
    if left <= 0:
        return _insufficient("voice", left, 1)

    # start 即扣 1 次（你也可以改为 ping 扣）
    u["credits_voice"] = left - 1
    call_id = uuid.uuid4().hex
    u.setdefault("voice_calls", {})[call_id] = {"last": _now()}
    return {"ok": True, "callId": call_id, "credits": _resp_me(u)["credits"], "ts": _now()}

@router.post("/voice/start")
@router.post("/start_voice")
def billing_voice_start(req: Request):
    return _voice_start_impl(req)

def _voice_ping_impl(req: Request, call_id: str):
    u = _get_or_create_user(req)
    calls = u.setdefault("voice_calls", {})
    if call_id not in calls:
        # 未知 call：当作已结束/已重启
        return JSONResponse(status_code=404, content={"ok": False, "error": "unknown_call", "callId": call_id})

    calls[call_id]["last"] = _now()

    if _bypass_gates() or u.get("is_subscribed"):
        return {"ok": True, "callId": call_id, "credits": _resp_me(u)["credits"], "ts": _now()}

    left = int(u.get("credits_voice", 0))
    if left <= 0:
        return _insufficient("voice", left, 1)

    # ping 扣 1 次（你可以改成每 N 秒扣一次）
    u["credits_voice"] = left - 1
    return {"ok": True, "callId": call_id, "credits": _resp_me(u)["credits"], "ts": _now()}

@router.post("/voice/ping")
@router.post("/ping_voice")
def billing_voice_ping(req: Request, body: VoicePingReq):
    return _voice_ping_impl(req, body.callId)

def _voice_end_impl(req: Request, call_id: str):
    u = _get_or_create_user(req)
    calls = u.setdefault("voice_calls", {})
    calls.pop(call_id, None)
    return {"ok": True, "callId": call_id, "credits": _resp_me(u)["credits"], "ts": _now()}

@router.post("/voice/end")
@router.post("/end_voice")
def billing_voice_end(req: Request, body: VoiceEndReq):
    return _voice_end_impl(req, body.callId)

# -----------------------------
# Optional server-side credit check helper
# -----------------------------

def require_credit(req: Request, kind: str = "text", cost: int = 1) -> Optional[Dict[str, Any]]:
    """返回 None 表示放行；返回 dict 表示拒绝原因。"""
    u = _get_or_create_user(req)

    if _bypass_gates() or u.get("is_subscribed"):
        return None

    field = {
        "text": "credits_text",
        "voice": "credits_voice",
        "video": "credits_video",
    }.get(kind, "credits_text")

    left = int(u.get(field, 0))
    if left < cost:
        return {"ok": False, "error": "insufficient_credits", "kind": kind, "left": left, "need": cost}

    u[field] = left - cost
    return None

