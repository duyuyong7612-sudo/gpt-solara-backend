# billing.py
# MVP Billing for launch (B方案兼容版，支持 iOS “snapshot” 入账):
# - Pro weekly subscription: $7.99/week => 100 min/week (6000 sec)
# - Video packs: 5 / 10 credits; consume 1 per generation
# - Prefer Authorization: Bearer access_token to identify user
# - Backward-compatible:
#   * accepts user_id query/body
#   * accepts old jws (header.payload.sig) style
# - NEW (recommended): accepts transaction snapshot:
#   { product_id, tx_id, expires_ms? }

import os
import json
import time
import uuid
import base64
import threading
import hmac
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/billing", tags=["billing"])

# -----------------------------
# Product IDs (match App Store Connect)
# -----------------------------
PRO_WEEKLY_PRODUCT = os.getenv("PRO_WEEKLY_PRODUCT", "solara_pro_weekly_799")

VIDEO_PACKS = {
    os.getenv("VIDEO_PACK_5", "solara_video_pack_5"): 5,
    os.getenv("VIDEO_PACK_10", "solara_video_pack_10"): 10,
}

# weekly voice allowance
WEEKLY_VOICE_ALLOW_SEC = int(os.getenv("WEEKLY_VOICE_ALLOW_SEC", "6000"))  # 100 min

# storage file
_DEFAULT_STORE = Path(__file__).with_name("billing_store.json")
BILLING_STORE_PATH = Path(os.getenv("BILLING_STORE_PATH", str(_DEFAULT_STORE)))

# lock
_LOCK = threading.Lock()

# -----------------------------
# JWT (HS256) — must match auth.py
# -----------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")


def _now() -> int:
    return int(time.time())


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _verify_and_get_sub(token: str) -> Optional[str]:
    """
    Verify HS256 token signature + exp, return sub (user_id).
    """
    try:
        parts = (token or "").split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        msg = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = _b64url_decode(sig_b64)
        expected = hmac.new(JWT_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        exp = int(payload.get("exp") or 0)
        if exp and _now() > exp:
            return None
        sub = (payload.get("sub") or "").strip()
        return sub or None
    except Exception:
        return None


def _auth_user_id(req: Request) -> Optional[str]:
    """
    Prefer Authorization: Bearer <token>.
    Return user_id (sub).
    """
    auth = req.headers.get("authorization") or req.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return _verify_and_get_sub(token)
    return None


# -----------------------------
# DB helpers
# -----------------------------
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


def _fallback_user_key(req: Request, user_id: str) -> str:
    """
    Backward compatible: prefer user_id param, else guest:ip.
    """
    ip = req.client.host if req.client else "unknown"
    uid = (user_id or "").strip()
    return uid if uid else f"guest:{ip}"


def _resolve_user_key(req: Request, user_id: str = "") -> Tuple[str, bool]:
    """
    Resolve user key:
    - If Bearer token exists -> use token sub, and mark authed=True
    - Else fallback to query/body user_id -> authed=False
    """
    sub = _auth_user_id(req)
    if sub:
        return sub, True
    return _fallback_user_key(req, user_id), False


def _ensure_user(u: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(u, dict) and u:
        u.setdefault("pro_until", 0)
        u.setdefault("week_reset_at", 0)
        u.setdefault("weekly_voice_used_sec", 0)
        u.setdefault("video_credits", 0)
        u.setdefault("seen_tx", {})       # tx_id -> {product_id, ts}
        u.setdefault("voice_calls", {})   # call_id -> {start, rem_at_start}
        return u
    return {
        "pro_until": 0,
        "week_reset_at": 0,
        "weekly_voice_used_sec": 0,
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


def _weekly_remaining(u: Dict[str, Any]) -> int:
    _ensure_week_reset(u)
    used = int(u.get("weekly_voice_used_sec") or 0)
    return max(0, WEEKLY_VOICE_ALLOW_SEC - used)


def _voice_remaining(u: Dict[str, Any]) -> int:
    # 订阅型：只给每周额度（先不做 topup）
    return _weekly_remaining(u)


def _decode_jws_payload(jws: str) -> Dict[str, Any]:
    """
    Legacy: JWS format header.payload.signature (payload JSON may include productId/transactionId/expiresDate).
    """
    try:
        parts = (jws or "").split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        raw = _b64url_decode(payload_b64)
        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# -----------------------------
# Routes
# -----------------------------
@router.get("/me")
def billing_me(req: Request, user_id: str = ""):
    with _LOCK:
        db = _load_db()
        uid, authed = _resolve_user_key(req, user_id)
        u = _ensure_user(db["users"].get(uid))
        _ensure_week_reset(u)

        out = {
            "ok": True,
            "user_id": uid,
            "authed": authed,
            "is_pro": _is_pro(u),
            "pro_until": int(u.get("pro_until") or 0),
            "week_reset_at": int(u.get("week_reset_at") or 0),
            "voice_weekly_cap_sec": WEEKLY_VOICE_ALLOW_SEC,
            "voice_weekly_used_sec": int(u.get("weekly_voice_used_sec") or 0),
            "voice_remaining_sec": _voice_remaining(u) if _is_pro(u) else 0,
            "video_credits": int(u.get("video_credits") or 0),
        }
        db["users"][uid] = u
        _save_db(db)
        return out


@router.get("/status")
def billing_status(req: Request, user_id: str = ""):
    return billing_me(req, user_id=user_id)


@router.post("/ingest")
async def billing_ingest(req: Request):
    """
    Ingest purchase and grant entitlements.

    Preferred (NEW, for your current iOS):
      { "product_id": "...", "tx_id": "...", "expires_ms": 123... , "user_id": "(optional if bearer)" }

    Legacy:
      { "jws": "header.payload.sig", "user_id": "(optional if bearer)" }

    Product mapping:
      - PRO_WEEKLY_PRODUCT => pro_until update
      - VIDEO_PACKS => video_credits += N
    """
    body = await req.json()
    user_id = (body.get("user_id") or "").strip()

    # NEW snapshot fields
    product_id = (body.get("product_id") or "").strip()
    tx_id = str(body.get("tx_id") or "")
    expires_ms = body.get("expires_ms")  # ms epoch for subscription

    # Legacy JWS (optional now)
    jws = (body.get("jws") or "").strip()

    # If snapshot not provided, fallback to legacy jws decode
    if not product_id:
        if not jws:
            return JSONResponse({"ok": False, "error": "missing_product_id_or_jws"}, status_code=400)

        payload = _decode_jws_payload(jws)
        product_id = (payload.get("productId") or "").strip() or (body.get("product_id") or "").strip()
        tx_id = str(payload.get("transactionId") or body.get("tx_id") or "")
        expires_ms = payload.get("expiresDate")

    if not product_id:
        return JSONResponse({"ok": False, "error": "missing_product_id"}, status_code=400)

    if not tx_id:
        # 必须要 tx_id 才能幂等；StoreKit2 Transaction.id 一定有
        return JSONResponse({"ok": False, "error": "missing_tx_id"}, status_code=400)

    # normalize expires_ms
    if isinstance(expires_ms, str):
        try:
            expires_ms = int(expires_ms)
        except Exception:
            expires_ms = None

    with _LOCK:
        db = _load_db()
        uid, authed = _resolve_user_key(req, user_id)
        u = _ensure_user(db["users"].get(uid))
        _ensure_week_reset(u)

        seen_tx = u.get("seen_tx") or {}
        if tx_id in seen_tx:
            return {"ok": True, "skipped": True, "user_id": uid, "product_id": product_id}

        now = _now()

        if product_id == PRO_WEEKLY_PRODUCT:
            # subscription (weekly)
            if isinstance(expires_ms, (int, float)) and expires_ms > 0:
                pro_until = int(int(expires_ms) / 1000)
            else:
                pro_until = now + 7 * 24 * 3600

            u["pro_until"] = max(int(u.get("pro_until") or 0), pro_until)

            if int(u.get("week_reset_at") or 0) == 0:
                u["week_reset_at"] = now + 7 * 24 * 3600
                u["weekly_voice_used_sec"] = 0

        elif product_id in VIDEO_PACKS:
            u["video_credits"] = int(u.get("video_credits") or 0) + int(VIDEO_PACKS[product_id])

        else:
            return JSONResponse({"ok": False, "error": f"unknown_product:{product_id}"}, status_code=400)

        seen_tx[tx_id] = {"product_id": product_id, "ts": now}
        u["seen_tx"] = seen_tx

        db["users"][uid] = u
        _save_db(db)

    return {"ok": True, "user_id": uid, "authed": authed, "product_id": product_id}


@router.post("/voice/start")
async def billing_voice_start(req: Request):
    """
    Start a voice call (quota check + create call_id).
    Body: { "user_id": "...(optional if Bearer)" }
    """
    body = await req.json()
    user_id = (body.get("user_id") or "").strip()

    with _LOCK:
        db = _load_db()
        uid, _ = _resolve_user_key(req, user_id)
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
    Body: { "user_id": "...(optional if Bearer)", "call_id": "..." }
    """
    body = await req.json()
    user_id = (body.get("user_id") or "").strip()
    call_id = (body.get("call_id") or "").strip()
    if not call_id:
        return JSONResponse({"ok": False, "error": "missing_call_id"}, status_code=400)

    with _LOCK:
        db = _load_db()
        uid, _ = _resolve_user_key(req, user_id)
        u = _ensure_user(db["users"].get(uid))

        calls = u.get("voice_calls") or {}
        c = calls.pop(call_id, None)
        if not c:
            db["users"][uid] = u
            _save_db(db)
            return {"ok": True, "skipped": True, "voice_remaining_sec": _voice_remaining(u) if _is_pro(u) else 0}

        now = _now()
        start = int(c.get("start") or now)
        dur = max(0, now - start)

        _ensure_week_reset(u)

        used = int(u.get("weekly_voice_used_sec") or 0)
        new_used = min(WEEKLY_VOICE_ALLOW_SEC, used + dur)
        u["weekly_voice_used_sec"] = new_used

        u["voice_calls"] = calls
        db["users"][uid] = u
        _save_db(db)

    return {"ok": True, "charged_sec": dur, "voice_remaining_sec": _voice_remaining(u)}


@router.post("/video/consume")
async def billing_video_consume(req: Request):
    """
    Consume 1 video credit before generation.
    Body: { "user_id": "...(optional if Bearer)" }
    """
    body = await req.json()
    user_id = (body.get("user_id") or "").strip()

    with _LOCK:
        db = _load_db()
        uid, _ = _resolve_user_key(req, user_id)
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
