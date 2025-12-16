# billing.py
# MVP Billing for launch:
# - Pro weekly subscription: 100 min voice/week (6000 sec)
# - Voice topups: add seconds
# - Video packs: add credits; consume 1 per generation
#
# NOTE (launch MVP):
# - We decode StoreKit2 transaction JWS payload for productId/transactionId/expiresDate.
# - We DO NOT verify JWS signature here (add App Store Server API v2 later).
# - We keep idempotency by transactionId.

import os
import json
import time
import uuid
import base64
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/billing", tags=["billing"])

# -----------------------------
# Product IDs (match App Store Connect)
# -----------------------------
PRO_WEEKLY_PRODUCT = os.getenv("PRO_WEEKLY_PRODUCT", "solara_pro_weekly_899")

VOICE_PACKS = {
    os.getenv("VOICE_PACK_30M", "solara_voice_30m"): 1800,   # 30m
    os.getenv("VOICE_PACK_60M", "solara_voice_60m"): 3600,   # 60m
    os.getenv("VOICE_PACK_180M", "solara_voice_180m"): 10800 # 180m
}

VIDEO_PACKS = {
    os.getenv("VIDEO_PACK_5", "solara_video_pack_5"): 5,
    os.getenv("VIDEO_PACK_10", "solara_video_pack_10"): 10
}

# weekly voice allowance
WEEKLY_VOICE_ALLOW_SEC = int(os.getenv("WEEKLY_VOICE_ALLOW_SEC", "6000"))  # 100 min

# storage file
_DEFAULT_STORE = Path(__file__).with_name("billing_store.json")
BILLING_STORE_PATH = Path(os.getenv("BILLING_STORE_PATH", str(_DEFAULT_STORE)))

# lock
_LOCK = threading.Lock()

# -----------------------------
# Helpers
# -----------------------------
def _now() -> int:
    return int(time.time())

def _load_db() -> Dict[str, Any]:
    try:
        if BILLING_STORE_PATH.exists():
            return json.loads(BILLING_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"users": {}}

def _save_db(db: Dict[str, Any]) -> None:
    BILLING_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(BILLING_STORE_PATH) + ".tmp"
    Path(tmp).write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, str(BILLING_STORE_PATH))

def _user_key(req: Request, user_id: str) -> str:
    """
    Prefer real user_id (from auth). If empty, fallback to guest:ip
    You can extend to include device_id later.
    """
    ip = req.client.host if req.client else "unknown"
    uid = (user_id or "").strip()
    return uid if uid else f"guest:{ip}"

def _ensure_user(u: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(u, dict) and u:
        # ensure keys
        u.setdefault("pro_until", 0)
        u.setdefault("week_reset_at", 0)
        u.setdefault("weekly_voice_used_sec", 0)
        u.setdefault("voice_topup_sec", 0)
        u.setdefault("video_credits", 0)
        u.setdefault("seen_tx", {})       # tx_id -> {product_id, ts}
        u.setdefault("voice_calls", {})   # call_id -> {start, rem_at_start}
        return u
    return {
        "pro_until": 0,
        "week_reset_at": 0,
        "weekly_voice_used_sec": 0,
        "voice_topup_sec": 0,
        "video_credits": 0,
        "seen_tx": {},
        "voice_calls": {}
    }

def _ensure_week_reset(u: Dict[str, Any]) -> None:
    now = _now()
    reset_at = int(u.get("week_reset_at") or 0)
    if reset_at == 0:
        u["week_reset_at"] = now + 7 * 24 * 3600
        u["weekly_voice_used_sec"] = 0
        return
    if now >= reset_at:
        u["week_reset_at"] = now + 7 * 24 * 3600
        u["weekly_voice_used_sec"] = 0

def _is_pro(u: Dict[str, Any]) -> bool:
    return _now() < int(u.get("pro_until") or 0)

def _voice_remaining(u: Dict[str, Any]) -> int:
    _ensure_week_reset(u)
    weekly_used = int(u.get("weekly_voice_used_sec") or 0)
    weekly_rem = max(0, WEEKLY_VOICE_ALLOW_SEC - weekly_used)
    topup = int(u.get("voice_topup_sec") or 0)
    return weekly_rem + topup

def _decode_jws_payload(jws: str) -> Dict[str, Any]:
    """
    JWS format: header.payload.signature
    We only decode payload JSON for MVP:
    - productId
    - transactionId
    - expiresDate (ms)
    """
    try:
        parts = (jws or "").split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        pad = "=" * (-len(payload_b64) % 4)
        raw = base64.urlsafe_b64decode(payload_b64 + pad)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}

# -----------------------------
# Routes
# -----------------------------
@router.get("/me")
def billing_me(req: Request, user_id: str = ""):
    with _LOCK:
        db = _load_db()
        uid = _user_key(req, user_id)
        u = _ensure_user(db["users"].get(uid))
        _ensure_week_reset(u)

        out = {
            "ok": True,
            "user_id": uid,
            "is_pro": _is_pro(u),
            "pro_until": int(u.get("pro_until") or 0),
            "week_reset_at": int(u.get("week_reset_at") or 0),
            "voice_remaining_sec": _voice_remaining(u) if _is_pro(u) else 0,
            "video_credits": int(u.get("video_credits") or 0),
        }
        db["users"][uid] = u
        _save_db(db)
        return out

@router.post("/ingest")
async def billing_ingest(req: Request):
    """
    Ingest StoreKit2 transaction JWS and grant entitlements.
    Body:
      { "user_id": "...", "jws": "..." }

    Product mapping:
      - PRO_WEEKLY_PRODUCT => pro_until update (+ week window init)
      - VOICE_PACKS => voice_topup_sec += seconds
      - VIDEO_PACKS => video_credits += N
    """
    body = await req.json()
    user_id = (body.get("user_id") or "").strip()
    jws = (body.get("jws") or "").strip()
    if not jws:
        return JSONResponse({"ok": False, "error": "missing_jws"}, status_code=400)

    payload = _decode_jws_payload(jws)
    product_id = (payload.get("productId") or "").strip() or (body.get("product_id") or "").strip()
    tx_id = str(payload.get("transactionId") or body.get("tx_id") or "")
    expires_ms = payload.get("expiresDate")  # ms epoch for subscription

    if not product_id:
        return JSONResponse({"ok": False, "error": "missing_product_id"}, status_code=400)

    with _LOCK:
        db = _load_db()
        uid = _user_key(req, user_id)
        u = _ensure_user(db["users"].get(uid))
        _ensure_week_reset(u)

        seen_tx = u.get("seen_tx") or {}
        if tx_id and tx_id in seen_tx:
            return {"ok": True, "skipped": True, "product_id": product_id}

        now = _now()

        if product_id == PRO_WEEKLY_PRODUCT:
            # subscription
            if isinstance(expires_ms, (int, float)) and expires_ms > 0:
                pro_until = int(expires_ms / 1000)
            else:
                pro_until = now + 7 * 24 * 3600
            u["pro_until"] = max(int(u.get("pro_until") or 0), pro_until)
            if int(u.get("week_reset_at") or 0) == 0:
                u["week_reset_at"] = now + 7 * 24 * 3600
                u["weekly_voice_used_sec"] = 0

        elif product_id in VOICE_PACKS:
            u["voice_topup_sec"] = int(u.get("voice_topup_sec") or 0) + int(VOICE_PACKS[product_id])

        elif product_id in VIDEO_PACKS:
            u["video_credits"] = int(u.get("video_credits") or 0) + int(VIDEO_PACKS[product_id])

        else:
            return JSONResponse({"ok": False, "error": f"unknown_product:{product_id}"}, status_code=400)

        if tx_id:
            seen_tx[tx_id] = {"product_id": product_id, "ts": now}
            u["seen_tx"] = seen_tx

        db["users"][uid] = u
        _save_db(db)

    return {"ok": True, "product_id": product_id}

@router.post("/voice/start")
async def billing_voice_start(req: Request):
    """
    Start a voice call (quota check + create call_id).
    Body: { "user_id": "..." }
    """
    body = await req.json()
    user_id = (body.get("user_id") or "").strip()

    with _LOCK:
        db = _load_db()
        uid = _user_key(req, user_id)
        u = _ensure_user(db["users"].get(uid))

        if not _is_pro(u):
            return JSONResponse({"ok": False, "error": "not_pro"}, status_code=402)

        rem = _voice_remaining(u)
        if rem < 30:
            return JSONResponse({"ok": False, "error": "voice_quota_low", "voice_remaining_sec": rem}, status_code=402)

        call_id = uuid.uuid4().hex
        calls = u.get("voice_calls") or {}
        calls[call_id] = {"start": _now(), "rem_at_start": rem}
        u["voice_calls"] = calls

        db["users"][uid] = u
        _save_db(db)

    return {"ok": True, "call_id": call_id, "voice_remaining_sec": rem}

@router.post("/voice/end")
async def billing_voice_end(req: Request):
    """
    End a voice call and charge duration by server time.
    Body: { "user_id": "...", "call_id": "..." }
    """
    body = await req.json()
    user_id = (body.get("user_id") or "").strip()
    call_id = (body.get("call_id") or "").strip()
    if not call_id:
        return JSONResponse({"ok": False, "error": "missing_call_id"}, status_code=400)

    with _LOCK:
        db = _load_db()
        uid = _user_key(req, user_id)
        u = _ensure_user(db["users"].get(uid))

        calls = u.get("voice_calls") or {}
        c = calls.pop(call_id, None)
        if not c:
            # idempotent
            db["users"][uid] = u
            _save_db(db)
            return {"ok": True, "skipped": True, "voice_remaining_sec": _voice_remaining(u) if _is_pro(u) else 0}

        now = _now()
        start = int(c.get("start") or now)
        dur = max(0, now - start)

        _ensure_week_reset(u)

        # charge weekly first, then topup
        weekly_used = int(u.get("weekly_voice_used_sec") or 0)
        weekly_rem = max(0, WEEKLY_VOICE_ALLOW_SEC - weekly_used)
        take_weekly = min(weekly_rem, dur)
        u["weekly_voice_used_sec"] = weekly_used + take_weekly
        left = dur - take_weekly
        if left > 0:
            topup = int(u.get("voice_topup_sec") or 0)
            u["voice_topup_sec"] = max(0, topup - left)

        u["voice_calls"] = calls
        db["users"][uid] = u
        _save_db(db)

    return {"ok": True, "charged_sec": dur, "voice_remaining_sec": _voice_remaining(u)}

@router.post("/video/consume")
async def billing_video_consume(req: Request):
    """
    Consume 1 video credit before generation.
    Body: { "user_id": "..." }
    """
    body = await req.json()
    user_id = (body.get("user_id") or "").strip()

    with _LOCK:
        db = _load_db()
        uid = _user_key(req, user_id)
        u = _ensure_user(db["users"].get(uid))

        if not _is_pro(u):
            return JSONResponse({"ok": False, "error": "not_pro"}, status_code=402)

        credits = int(u.get("video_credits") or 0)
        if credits <= 0:
            return JSONResponse({"ok": False, "error": "no_video_credits"}, status_code=402)

        u["video_credits"] = credits - 1
        db["users"][uid] = u
        _save_db(db)

    return {"ok": True, "video_credits": int(u["video_credits"])}