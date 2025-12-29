# billing.py
# Production-minded Billing (JSON store) — Launch-ready (Upgraded + RevenueCat Webhook)
#
# ✅ Pro weekly subscription: $7.99/week => 100 min/week (6000 sec)
# ✅ Video packs: +5 / +10 credits; consume 1 per generation (server authoritative)
# ✅ Prefer Authorization: Bearer access_token to identify user
# ✅ Backward-compatible: accepts user_id query/body; accepts legacy jws payload
#
# ✅ Hard stop when credits missing: 402 + action=open_paywall + billing snapshot
# ✅ Idempotency for ingest via tx_id (and optional X-Idempotency-Key)
# ✅ Video consume + refund for failure compensation (idempotent)
# ✅ Voice ping + hard stop + stale settlement (prevents infinite cost)
# ✅ Cleanup stale voice calls / video consumes / seen_tx
# ✅ Purchase history stored per-user (for customer support / receipts / audit)
# ✅ Auto-merge guest user_id into authed user (fixes “已订阅但后端仍 402” most common root cause)
#
# ✅ RevenueCat Webhook integrated:
#   POST /billing/webhook/revenuecat
#   - Verify Authorization header (REVENUECAT_WEBHOOK_AUTH)
#   - Apply subscription extend/expire and video pack grants
#   - Idempotent via event_id/tx_id hash key
#
# ✅ Logs (added):
#   - [VOICE_START] call_id, rem_at_start, hard_stop_at
#   - [VOICE_PING] elapsed, remaining_call, remaining_rt, will_hangup
#   - [VOICE_END] charged_sec, weekly_used_before/after
#
# NOTE:
# - JSON store is single-process safe. For multi-worker/multi-instance, replace with DB.
# - Receipt verification should be via RevenueCat webhooks or StoreKit server verification.

import os
import json
import time
import uuid
import base64
import threading
import hmac
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from fastapi import APIRouter, Request, Header
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/billing", tags=["billing"])

# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("BILLING_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("billing")


def _rid(req: Request) -> str:
    return (
        req.headers.get("x-request-id")
        or req.headers.get("X-Request-Id")
        or str(uuid.uuid4())
    )


# -----------------------------
# RevenueCat webhook auth
# -----------------------------
REVENUECAT_WEBHOOK_AUTH = os.getenv("REVENUECAT_WEBHOOK_AUTH", "").strip()
# 在 RevenueCat Dashboard -> Integrations -> Webhooks 里配置 Authorization header value
# 例如: "Bearer xxx_random_secret_xxx"


def _auth_revenuecat_webhook(req: Request) -> bool:
    if not REVENUECAT_WEBHOOK_AUTH:
        # 未配置则放行（不建议生产）
        return True
    got = (req.headers.get("authorization") or req.headers.get("Authorization") or "").strip()
    return got == REVENUECAT_WEBHOOK_AUTH


# -----------------------------
# Product IDs (match App Store Connect)
# -----------------------------
PRO_WEEKLY_PRODUCT = os.getenv("PRO_WEEKLY_PRODUCT", "solara_pro_weekly_799")

# allow suffix/variant match by enabling PRODUCT_PREFIX_MATCH=1
PRODUCT_PREFIX_MATCH = os.getenv("PRODUCT_PREFIX_MATCH", "1").strip() in (
    "1",
    "true",
    "True",
    "YES",
    "yes",
)

VIDEO_PACKS = {
    os.getenv("VIDEO_PACK_5", "solara_video_pack_5"): 5,
    os.getenv("VIDEO_PACK_10", "solara_video_pack_10"): 10,
}

# weekly voice allowance
WEEKLY_VOICE_ALLOW_SEC = int(os.getenv("WEEKLY_VOICE_ALLOW_SEC", "6000"))  # 100 min

# Optional policy: whether video generation requires Pro (default: NO)
VIDEO_REQUIRE_PRO = os.getenv("VIDEO_REQUIRE_PRO", "0").strip() in (
    "1",
    "true",
    "True",
    "YES",
    "yes",
)

# Optional: minimum voice remaining seconds required to start call
VOICE_START_MIN_SEC = int(os.getenv("VOICE_START_MIN_SEC", "30"))

# Safety: maximum voice call duration charged in one settlement/end
MAX_SINGLE_VOICE_CHARGE_SEC = int(
    os.getenv("MAX_SINGLE_VOICE_CHARGE_SEC", str(3 * 3600))
)  # 3h cap

# Voice ping grace
VOICE_PING_GRACE_SEC = int(os.getenv("VOICE_PING_GRACE_SEC", "90"))

# Video consume: max amount per request
MAX_VIDEO_CONSUME_AMOUNT = int(os.getenv("MAX_VIDEO_CONSUME_AMOUNT", "5"))

# Purchase history keep length (per user)
PURCHASE_HISTORY_LIMIT = int(os.getenv("PURCHASE_HISTORY_LIMIT", "60"))

# store file
_DEFAULT_STORE = Path(__file__).with_name("billing_store.json")
BILLING_STORE_PATH = Path(os.getenv("BILLING_STORE_PATH", str(_DEFAULT_STORE)))

# cleanup TTL
VOICE_CALL_TTL_SEC = int(os.getenv("VOICE_CALL_TTL_SEC", str(6 * 3600)))  # 6 hours
VIDEO_CONSUME_TTL_SEC = int(os.getenv("VIDEO_CONSUME_TTL_SEC", str(48 * 3600)))  # 48 hours
SEEN_TX_TTL_SEC = int(os.getenv("SEEN_TX_TTL_SEC", str(30 * 24 * 3600)))  # 30 days

# lock (single-process safety)
_LOCK = threading.Lock()

# -----------------------------
# JWT (HS256) — must match auth.py
# -----------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")


def _now() -> int:
    return int(time.time())


def _now_ms() -> int:
    return int(time.time() * 1000)


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _verify_and_get_sub(token: str) -> Optional[str]:
    """Verify HS256 token signature + exp, return sub (user_id)."""
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
    """Prefer Authorization: Bearer <token>. Return user_id (sub)."""
    auth = req.headers.get("authorization") or req.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return _verify_and_get_sub(token)
    return None


# -----------------------------
# DB helpers (JSON store)
# -----------------------------
def _load_db() -> Dict[str, Any]:
    try:
        if BILLING_STORE_PATH.exists():
            return json.loads(BILLING_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"users": {}}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _save_db(db: Dict[str, Any]) -> None:
    _atomic_write_text(BILLING_STORE_PATH, json.dumps(db, ensure_ascii=False))


def _stable_guest_key(req: Request) -> str:
    """Fallback guest key if client forgot to send user_id."""
    ip = req.client.host if req.client else "unknown"
    ua = (req.headers.get("user-agent") or "unknown")[:200]
    raw = f"{ip}|{ua}".encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:16]
    return f"guest:{h}"


def _fallback_user_key(req: Request, user_id: str) -> str:
    uid = (user_id or "").strip()
    if uid:
        return uid
    return _stable_guest_key(req)


def _resolve_user_key(req: Request, user_id: str = "") -> Tuple[str, bool, str]:
    """
    Return: (uid, authed, provided_user_id)
    """
    provided = (user_id or "").strip()
    sub = _auth_user_id(req)
    if sub:
        return sub, True, provided
    return _fallback_user_key(req, provided), False, provided


def _ensure_user(u: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(u, dict) and u is not None:
        u.setdefault("pro_until", 0)  # epoch sec
        u.setdefault("week_reset_at", 0)  # epoch sec
        u.setdefault("weekly_voice_used_sec", 0)
        u.setdefault("video_credits", 0)

        u.setdefault("seen_tx", {})  # idempotency keys
        u.setdefault("voice_calls", {})  # call_id -> data
        u.setdefault("video_consumes", {})  # consume_id -> data

        u.setdefault("purchases", [])  # history
        u.setdefault("last_purchase_at", 0)

        return u

    return {
        "pro_until": 0,
        "week_reset_at": 0,
        "weekly_voice_used_sec": 0,
        "video_credits": 0,
        "seen_tx": {},
        "voice_calls": {},
        "video_consumes": {},
        "purchases": [],
        "last_purchase_at": 0,
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


def _decode_jws_payload(jws: str) -> Dict[str, Any]:
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


def _normalize_expires_ms(expires_ms) -> Optional[int]:
    if expires_ms is None:
        return None
    try:
        if isinstance(expires_ms, str):
            expires_ms = int(expires_ms.strip())
        if not isinstance(expires_ms, (int, float)):
            return None
        x = int(expires_ms)
        if x <= 0:
            return None
        if x < 10_000_000_000:
            return x * 1000
        return x
    except Exception:
        return None


def _ms_to_sec(ms: Optional[int]) -> int:
    try:
        if ms is None:
            return 0
        x = int(ms)
        if x <= 0:
            return 0
        if x < 10_000_000_000:
            return x
        return int(x / 1000)
    except Exception:
        return 0


def _is_pro(u: Dict[str, Any]) -> bool:
    return _now() < int(u.get("pro_until") or 0)


def _weekly_remaining_base(u: Dict[str, Any]) -> int:
    _ensure_week_reset(u)
    used = int(u.get("weekly_voice_used_sec") or 0)
    return max(0, WEEKLY_VOICE_ALLOW_SEC - used)


def _voice_active_call(u: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    calls = u.get("voice_calls") or {}
    if not isinstance(calls, dict) or not calls:
        return None, None
    now = _now()
    best_id = None
    best = None
    best_last_seen = -1
    for cid, c in calls.items():
        if not isinstance(c, dict):
            continue
        start = int(c.get("start") or 0)
        last_seen = int(c.get("last_seen") or start or 0)
        if not start:
            continue
        # ignore expired by ping grace, will be settled in prune later
        if last_seen and (now - last_seen) > VOICE_PING_GRACE_SEC:
            continue
        if last_seen > best_last_seen:
            best_last_seen = last_seen
            best_id = cid
            best = c
    return best_id, best


def _weekly_remaining_realtime(u: Dict[str, Any]) -> Tuple[int, bool, Optional[str], Optional[int], Optional[int]]:
    """
    Return:
      remaining_sec_realtime,
      in_call,
      active_call_id,
      active_call_remaining_sec,
      active_call_hard_stop_at_ms
    """
    base = _weekly_remaining_base(u)
    if not _is_pro(u):
        return 0, False, None, None, None

    cid, c = _voice_active_call(u)
    if not cid or not isinstance(c, dict):
        return base, False, None, None, None

    now = _now()
    start = int(c.get("start") or now)
    rem_at_start = int(c.get("rem_at_start") or 0)
    hard_stop_at = int(c.get("hard_stop_at") or (start + rem_at_start))

    elapsed = max(0, now - start)
    # remaining within this call (cap by rem_at_start)
    call_remaining = max(0, rem_at_start - elapsed) if rem_at_start > 0 else 0

    # realtime remaining should be conservative:
    realtime_remaining = max(0, min(base, call_remaining))

    return realtime_remaining, True, cid, call_remaining, int(hard_stop_at * 1000)


def _tier(authed: bool, u: Dict[str, Any]) -> str:
    if not authed:
        return "guest"
    return "pro" if _is_pro(u) else "free"


def _prune_seen_tx(u: Dict[str, Any], now: int) -> None:
    seen = u.get("seen_tx") or {}
    if not isinstance(seen, dict):
        u["seen_tx"] = {}
        return
    keep = {}
    for tx, info in seen.items():
        try:
            ts = int((info or {}).get("ts") or 0)
        except Exception:
            ts = 0
        if ts and (now - ts) <= SEEN_TX_TTL_SEC:
            keep[tx] = info
    u["seen_tx"] = keep


def _prune_video_consumes(u: Dict[str, Any], now: int) -> None:
    consumes = u.get("video_consumes") or {}
    if not isinstance(consumes, dict):
        u["video_consumes"] = {}
        return
    keep = {}
    for cid, c in consumes.items():
        try:
            ts = int((c or {}).get("ts") or 0)
        except Exception:
            ts = 0
        if ts and (now - ts) <= VIDEO_CONSUME_TTL_SEC:
            keep[cid] = c
    u["video_consumes"] = keep


def _settle_stale_voice_calls(u: Dict[str, Any], now: int) -> None:
    calls = u.get("voice_calls") or {}
    if not isinstance(calls, dict) or not calls:
        u["voice_calls"] = {}
        return

    new_calls = {}
    for call_id, c in calls.items():
        if not isinstance(c, dict):
            continue
        start = int(c.get("start") or 0)
        last_seen = int(c.get("last_seen") or start or 0)
        rem_at_start = int(c.get("rem_at_start") or 0)
        if not start:
            continue

        # total ttl
        if (now - start) > VOICE_CALL_TTL_SEC:
            _ensure_week_reset(u)
            dur = max(0, min(last_seen, now) - start)
            dur = min(dur, MAX_SINGLE_VOICE_CHARGE_SEC)
            if rem_at_start > 0:
                dur = min(dur, rem_at_start)
            used = int(u.get("weekly_voice_used_sec") or 0)
            u["weekly_voice_used_sec"] = min(WEEKLY_VOICE_ALLOW_SEC, used + dur)
            continue

        # ping grace
        if last_seen and (now - last_seen) > VOICE_PING_GRACE_SEC:
            _ensure_week_reset(u)
            dur = max(0, last_seen - start)
            dur = min(dur, MAX_SINGLE_VOICE_CHARGE_SEC)
            if rem_at_start > 0:
                dur = min(dur, rem_at_start)
            used = int(u.get("weekly_voice_used_sec") or 0)
            u["weekly_voice_used_sec"] = min(WEEKLY_VOICE_ALLOW_SEC, used + dur)
            continue

        new_calls[call_id] = c

    u["voice_calls"] = new_calls


def _prune_state(u: Dict[str, Any]) -> None:
    now = _now()
    _prune_seen_tx(u, now)
    _prune_video_consumes(u, now)
    _settle_stale_voice_calls(u, now)


def _match_product_id(pid: str, target: str) -> bool:
    if pid == target:
        return True
    if PRODUCT_PREFIX_MATCH:
        return pid.startswith(target)
    return False


def _video_pack_amount(pid: str) -> Optional[int]:
    for k, v in VIDEO_PACKS.items():
        if _match_product_id(pid, k):
            return int(v)
    return None


def _add_purchase_record(
    u: Dict[str, Any],
    tx_id: str,
    product_id: str,
    expires_ms: Optional[int],
    source: str,
) -> None:
    ts = _now()
    rec = {
        "tx_id": tx_id,
        "product_id": product_id,
        "ts": ts,
        "expires_ms": int(expires_ms) if isinstance(expires_ms, int) and expires_ms > 0 else None,
        "source": source,
    }
    purchases = u.get("purchases") or []
    if not isinstance(purchases, list):
        purchases = []
    purchases.append(rec)
    if len(purchases) > PURCHASE_HISTORY_LIMIT:
        purchases = purchases[-PURCHASE_HISTORY_LIMIT:]
    u["purchases"] = purchases
    u["last_purchase_at"] = ts


def _merge_guest_into_authed(db: Dict[str, Any], authed_uid: str, guest_uid: str) -> None:
    if not guest_uid or not authed_uid:
        return
    if guest_uid == authed_uid:
        return
    if not guest_uid.startswith("guest:"):
        return

    users = db.get("users") or {}
    g = users.get(guest_uid)
    if not isinstance(g, dict):
        return

    a = _ensure_user(users.get(authed_uid))
    g = _ensure_user(g)

    a["pro_until"] = max(int(a.get("pro_until") or 0), int(g.get("pro_until") or 0))
    a["week_reset_at"] = max(int(a.get("week_reset_at") or 0), int(g.get("week_reset_at") or 0))
    a["weekly_voice_used_sec"] = max(
        int(a.get("weekly_voice_used_sec") or 0), int(g.get("weekly_voice_used_sec") or 0)
    )
    a["video_credits"] = int(a.get("video_credits") or 0) + int(g.get("video_credits") or 0)

    st_a = a.get("seen_tx") or {}
    st_g = g.get("seen_tx") or {}
    if isinstance(st_a, dict) and isinstance(st_g, dict):
        for tx, info in st_g.items():
            if tx not in st_a:
                st_a[tx] = info
        a["seen_tx"] = st_a

    p_a = a.get("purchases") or []
    p_g = g.get("purchases") or []
    if isinstance(p_a, list) and isinstance(p_g, list):
        merged = p_a + p_g
        if len(merged) > PURCHASE_HISTORY_LIMIT:
            merged = merged[-PURCHASE_HISTORY_LIMIT:]
        a["purchases"] = merged
        a["last_purchase_at"] = max(int(a.get("last_purchase_at") or 0), int(g.get("last_purchase_at") or 0))

    users[authed_uid] = a
    users.pop(guest_uid, None)
    db["users"] = users


def _snapshot(uid: str, authed: bool, u: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_week_reset(u)
    _prune_state(u)

    is_pro = _is_pro(u)
    credits = int(u.get("video_credits") or 0)

    remaining_rt, in_call, active_call_id, active_call_rem, active_call_hard_stop_ms = _weekly_remaining_realtime(u)
    remaining_base = _weekly_remaining_base(u) if is_pro else 0

    # server authoritative gates
    can_voice = bool(is_pro and remaining_rt >= VOICE_START_MIN_SEC)
    can_video = bool((is_pro if VIDEO_REQUIRE_PRO else True) and credits > 0)

    purchases = u.get("purchases") or []
    if not isinstance(purchases, list):
        purchases = []
    recent = purchases[-10:]

    return {
        "tier": _tier(authed, u),
        "authed": bool(authed),
        "user_id": uid,
        "is_pro": bool(is_pro),
        "pro_until": int(u.get("pro_until") or 0),
        "week_reset_at": int(u.get("week_reset_at") or 0),
        "voice_weekly_cap_sec": WEEKLY_VOICE_ALLOW_SEC,
        "voice_weekly_used_sec": int(u.get("weekly_voice_used_sec") or 0),

        # ✅ realtime remaining (recommended for UI)
        "voice_remaining_sec": int(remaining_rt if is_pro else 0),
        # ✅ base remaining (compat/debug)
        "voice_remaining_sec_base": int(remaining_base if is_pro else 0),

        # ✅ in-call realtime details (helps front-end debug/hud)
        "voice_in_call": bool(in_call),
        "voice_active_call_id": active_call_id,
        "voice_active_call_remaining_sec": int(active_call_rem) if active_call_rem is not None else None,
        "voice_active_call_hard_stop_at_ms": int(active_call_hard_stop_ms) if active_call_hard_stop_ms is not None else None,

        "video_credits": int(credits),
        "can_voice": bool(can_voice),
        "can_video": bool(can_video),
        "policy": {
            "video_require_pro": bool(VIDEO_REQUIRE_PRO),
            "voice_start_min_sec": int(VOICE_START_MIN_SEC),
            "video_consume_per_gen": 1,
            "voice_ping_grace_sec": int(VOICE_PING_GRACE_SEC),
        },
        "last_purchase_at": int(u.get("last_purchase_at") or 0),
        "recent_purchases": recent,
        "server_now_ms": _now_ms(),
    }


def _err(status: int, error: str, action: Optional[str], billing: Dict[str, Any]) -> JSONResponse:
    payload: Dict[str, Any] = {"ok": False, "error": error, "billing": billing}
    if action:
        payload["action"] = action
    return JSONResponse(payload, status_code=status)


# -----------------------------
# Routes
# -----------------------------
@router.get("/me")
def billing_me(req: Request, user_id: str = ""):
    rid = _rid(req)
    with _LOCK:
        db = _load_db()
        uid, authed, provided = _resolve_user_key(req, user_id)

        if authed and provided and provided != uid and provided.startswith("guest:"):
            _merge_guest_into_authed(db, uid, provided)

        users = db.get("users") or {}
        u = _ensure_user(users.get(uid))
        snap = _snapshot(uid, authed, u)
        users[uid] = u
        db["users"] = users
        _save_db(db)

    log.info(
        "[ME][%s] uid=%s authed=%s provided=%s is_pro=%s credits=%s voice_rem=%s in_call=%s active_call=%s",
        rid,
        uid,
        authed,
        provided,
        snap["is_pro"],
        snap["video_credits"],
        snap["voice_remaining_sec"],
        snap["voice_in_call"],
        snap.get("voice_active_call_id"),
    )
    return {"ok": True, "billing": snap}


@router.get("/status")
def billing_status(req: Request, user_id: str = ""):
    return billing_me(req, user_id=user_id)


@router.post("/webhook/revenuecat")
async def revenuecat_webhook(req: Request):
    """
    RevenueCat -> Your backend
    Configure in RevenueCat dashboard:
      - URL: https://<host>/billing/webhook/revenuecat
      - Authorization header value: set to REVENUECAT_WEBHOOK_AUTH
    """
    rid = _rid(req)

    if not _auth_revenuecat_webhook(req):
        log.warning("[RC][%s] unauthorized", rid)
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    try:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    def _get(*keys) -> str:
        for k in keys:
            v = body.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    event_type = _get("event", "event_type", "type")
    app_user_id = _get("app_user_id", "appUserId", "subscriber_id", "customer_app_user_id")
    product_id = _get("product_id", "productId", "product_identifier", "productIdentifier")
    environment = _get("environment", "store_environment")

    event_id = _get("id", "event_id", "eventId")
    tx_id = _get("transaction_id", "transactionId", "original_transaction_id", "originalTransactionId")

    exp_ms = _normalize_expires_ms(
        body.get("expiration_at_ms")
        or body.get("expires_at_ms")
        or body.get("expirationAtMs")
        or body.get("expiresAtMs")
    )
    purchased_ms = _normalize_expires_ms(
        body.get("purchased_at_ms") or body.get("purchasedAtMs") or body.get("purchased_at")
    )

    if not app_user_id:
        log.warning("[RC][%s] missing app_user_id keys=%s", rid, list(body.keys())[:20])
        return JSONResponse({"ok": False, "error": "missing_app_user_id"}, status_code=400)

    # idempotency key for webhook events
    if tx_id:
        key = f"rc_tx:{tx_id}"
    elif event_id:
        key = f"rc_evt:{event_id}"
    else:
        key = "rc_hash:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    log.info(
        "[RC][%s] recv type=%s uid=%s product=%s env=%s key=%s exp_ms=%s",
        rid,
        event_type,
        app_user_id,
        product_id,
        environment,
        key,
        exp_ms,
    )

    with _LOCK:
        db = _load_db()
        users = db.get("users") or {}
        u = _ensure_user(users.get(app_user_id))
        _ensure_week_reset(u)
        _prune_state(u)

        seen = u.get("seen_tx") or {}
        if not isinstance(seen, dict):
            seen = {}

        if key in seen:
            snap = _snapshot(app_user_id, True, u)
            users[app_user_id] = u
            db["users"] = users
            _save_db(db)
            return {"ok": True, "skipped": True, "billing": snap}

        now = _now()
        exp_sec = _ms_to_sec(exp_ms)

        # Apply subscription / packs
        if product_id and _match_product_id(product_id, PRO_WEEKLY_PRODUCT):
            # extend/provision
            if event_type in (
                "INITIAL_PURCHASE",
                "RENEWAL",
                "UNCANCELLATION",
                "PRODUCT_CHANGE",
                "SUBSCRIPTION_EXTENDED",
                "PURCHASE",
            ):
                if exp_sec > 0:
                    u["pro_until"] = max(int(u.get("pro_until") or 0), exp_sec)
                else:
                    u["pro_until"] = max(int(u.get("pro_until") or 0), now + 7 * 24 * 3600)

                if int(u.get("week_reset_at") or 0) == 0:
                    u["week_reset_at"] = now + 7 * 24 * 3600
                    u["weekly_voice_used_sec"] = 0

            elif event_type in ("CANCELLATION",):
                # keep access until expiration
                if exp_sec > 0:
                    u["pro_until"] = max(int(u.get("pro_until") or 0), exp_sec)

            elif event_type in ("EXPIRATION",):
                # expire
                if exp_sec > 0:
                    u["pro_until"] = min(int(u.get("pro_until") or 0) or exp_sec, exp_sec)
                else:
                    u["pro_until"] = min(int(u.get("pro_until") or 0), now)

            _add_purchase_record(u, key, product_id, exp_ms, source=f"revenuecat:{event_type or 'unknown'}")

        else:
            amt = _video_pack_amount(product_id) if product_id else None
            if amt is not None and event_type in ("NON_RENEWING_PURCHASE", "INITIAL_PURCHASE", "PURCHASE"):
                u["video_credits"] = int(u.get("video_credits") or 0) + int(amt)
                _add_purchase_record(u, key, product_id, purchased_ms, source=f"revenuecat:{event_type or 'unknown'}")

        # mark idempotency
        seen[key] = {"product_id": product_id or "", "ts": now, "event": event_type or "", "env": environment}
        u["seen_tx"] = seen

        users[app_user_id] = u
        db["users"] = users
        _save_db(db)

        snap = _snapshot(app_user_id, True, u)
        log.info(
            "[RC][%s] applied uid=%s is_pro=%s pro_until=%s credits=%s voice_rem=%s",
            rid,
            app_user_id,
            snap["is_pro"],
            snap["pro_until"],
            snap["video_credits"],
            snap["voice_remaining_sec"],
        )
        return {"ok": True, "billing": snap}


@router.post("/ingest")
async def billing_ingest(
    req: Request,
    x_idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
):
    rid = _rid(req)
    try:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    user_id = (body.get("user_id") or body.get("userId") or "").strip()
    product_id = (body.get("product_id") or body.get("productId") or "").strip()
    tx_id = str(body.get("tx_id") or body.get("txId") or "").strip()
    expires_ms = body.get("expires_ms", body.get("expiresMs"))
    jws = (body.get("jws") or "").strip()

    if x_idempotency_key and not tx_id:
        tx_id = x_idempotency_key.strip()

    if not product_id:
        if jws:
            payload = _decode_jws_payload(jws)
            product_id = (payload.get("productId") or "").strip() or (body.get("product_id") or "").strip()
            tx_id = str(payload.get("transactionId") or body.get("tx_id") or tx_id or "").strip()
            expires_ms = payload.get("expiresDate")
        else:
            return JSONResponse({"ok": False, "error": "missing_product_id_or_jws"}, status_code=400)

    if not product_id:
        return JSONResponse({"ok": False, "error": "missing_product_id"}, status_code=400)
    if not tx_id:
        return JSONResponse({"ok": False, "error": "missing_tx_id"}, status_code=400)

    expires_ms = _normalize_expires_ms(expires_ms)

    with _LOCK:
        db = _load_db()
        uid, authed, provided = _resolve_user_key(req, user_id)

        if authed and provided and provided != uid and provided.startswith("guest:"):
            _merge_guest_into_authed(db, uid, provided)

        users = db.get("users") or {}
        u = _ensure_user(users.get(uid))
        _ensure_week_reset(u)
        _prune_state(u)

        log.info(
            "[INGEST][%s] recv uid=%s authed=%s provided=%s product=%s tx=%s expires_ms=%s",
            rid,
            uid,
            authed,
            provided,
            product_id,
            tx_id,
            expires_ms,
        )

        seen_tx = u.get("seen_tx") or {}
        if isinstance(seen_tx, dict) and tx_id in seen_tx:
            snap = _snapshot(uid, authed, u)
            users[uid] = u
            db["users"] = users
            _save_db(db)
            log.info("[INGEST][%s] skipped duplicate tx=%s product=%s", rid, tx_id, product_id)
            return {"ok": True, "skipped": True, "product_id": product_id, "tx_id": tx_id, "billing": snap}

        now = _now()

        if _match_product_id(product_id, PRO_WEEKLY_PRODUCT):
            if isinstance(expires_ms, int) and expires_ms > 0:
                pro_until = int(expires_ms / 1000)
            else:
                pro_until = now + 7 * 24 * 3600
            u["pro_until"] = max(int(u.get("pro_until") or 0), pro_until)

            if int(u.get("week_reset_at") or 0) == 0:
                u["week_reset_at"] = now + 7 * 24 * 3600
                u["weekly_voice_used_sec"] = 0

            _add_purchase_record(u, tx_id, product_id, expires_ms, source="client_ingest")

        else:
            amt = _video_pack_amount(product_id)
            if amt is not None:
                u["video_credits"] = int(u.get("video_credits") or 0) + int(amt)
                _add_purchase_record(u, tx_id, product_id, None, source="client_ingest")
            else:
                snap = _snapshot(uid, authed, u)
                users[uid] = u
                db["users"] = users
                _save_db(db)
                return _err(400, f"unknown_product:{product_id}", None, snap)

        if not isinstance(seen_tx, dict):
            seen_tx = {}
        seen_tx[tx_id] = {"product_id": product_id, "ts": now}
        u["seen_tx"] = seen_tx

        users[uid] = u
        db["users"] = users
        _save_db(db)

        snap = _snapshot(uid, authed, u)
        log.info(
            "[INGEST][%s] ok uid=%s is_pro=%s pro_until=%s credits=%s voice_rem=%s",
            rid,
            uid,
            snap["is_pro"],
            snap["pro_until"],
            snap["video_credits"],
            snap["voice_remaining_sec"],
        )
        return {"ok": True, "product_id": product_id, "tx_id": tx_id, "billing": snap}


@router.post("/voice/start")
async def billing_voice_start(
    req: Request,
    x_idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
):
    rid = _rid(req)
    body = {}
    try:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    user_id = (body.get("user_id") or body.get("userId") or "").strip()

    with _LOCK:
        db = _load_db()
        uid, authed, provided = _resolve_user_key(req, user_id)

        if authed and provided and provided != uid and provided.startswith("guest:"):
            _merge_guest_into_authed(db, uid, provided)

        users = db.get("users") or {}
        u = _ensure_user(users.get(uid))

        snap = _snapshot(uid, authed, u)
        log.info(
            "[VOICE_START][%s] uid=%s authed=%s is_pro=%s credits=%s voice_rem=%s in_call=%s",
            rid,
            uid,
            authed,
            snap["is_pro"],
            snap["video_credits"],
            snap["voice_remaining_sec"],
            snap["voice_in_call"],
        )

        if not snap["is_pro"]:
            return _err(402, "not_pro", "open_paywall", snap)

        rem = int(snap.get("voice_remaining_sec") or 0)
        if rem < VOICE_START_MIN_SEC:
            return _err(402, "voice_quota_low", "open_paywall", snap)

        now = _now()

        # Idempotency: same key reuses existing call
        idem = (x_idempotency_key or "").strip()
        if idem:
            calls = u.get("voice_calls") or {}
            if isinstance(calls, dict):
                for cid, c in calls.items():
                    if isinstance(c, dict) and (c.get("idempotency_key") or "") == idem:
                        snap2 = _snapshot(uid, authed, u)
                        log.info(
                            "[VOICE_START][%s] reused uid=%s call_id=%s idem=%s rem_start=%s hard_stop_at=%s",
                            rid,
                            uid,
                            cid,
                            idem,
                            c.get("rem_at_start"),
                            c.get("hard_stop_at"),
                        )
                        return {
                            "ok": True,
                            "call_id": cid,
                            "callId": cid,
                            "hard_stop_at_ms": int((int(c.get("hard_stop_at") or (now + rem))) * 1000),
                            "billing": snap2,
                        }

        call_id = uuid.uuid4().hex

        calls = u.get("voice_calls") or {}
        if not isinstance(calls, dict):
            calls = {}
        calls[call_id] = {
            "start": now,
            "last_seen": now,
            "rem_at_start": rem,
            "hard_stop_at": now + rem,
            "idempotency_key": idem or "",
        }
        u["voice_calls"] = calls

        users[uid] = u
        db["users"] = users
        _save_db(db)

        snap2 = _snapshot(uid, authed, u)

        log.info(
            "[VOICE_START][%s] started uid=%s call_id=%s rem_start=%s hard_stop_at=%s voice_rem_now=%s",
            rid,
            uid,
            call_id,
            rem,
            now + rem,
            snap2.get("voice_remaining_sec"),
        )
        return {
            "ok": True,
            "call_id": call_id,
            "callId": call_id,
            "hard_stop_at_ms": int((now + rem) * 1000),
            "billing": snap2,
        }


@router.post("/voice/ping")
async def billing_voice_ping(req: Request):
    body = {}
    try:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    user_id = (body.get("user_id") or body.get("userId") or "").strip()
    call_id = (body.get("call_id") or body.get("callId") or "").strip()
    if not call_id:
        return JSONResponse({"ok": False, "error": "missing_call_id"}, status_code=400)

    rid = _rid(req)

    with _LOCK:
        db = _load_db()
        uid, authed, provided = _resolve_user_key(req, user_id)

        if authed and provided and provided != uid and provided.startswith("guest:"):
            _merge_guest_into_authed(db, uid, provided)

        users = db.get("users") or {}
        u = _ensure_user(users.get(uid))

        calls = u.get("voice_calls") or {}
        if not isinstance(calls, dict):
            calls = {}
        c = calls.get(call_id)
        if not isinstance(c, dict):
            snap = _snapshot(uid, authed, u)
            users[uid] = u
            db["users"] = users
            _save_db(db)
            log.info("[VOICE_PING][%s] uid=%s call_id=%s missing -> skipped", rid, uid, call_id)
            return {"ok": True, "skipped": True, "remaining_sec": 0, "billing": snap}

        now = _now()
        start = int(c.get("start") or now)
        rem_at_start = int(c.get("rem_at_start") or 0)
        hard_stop_at = int(c.get("hard_stop_at") or (start + rem_at_start))

        c["last_seen"] = now
        calls[call_id] = c
        u["voice_calls"] = calls

        elapsed = max(0, now - start)
        remaining = max(0, rem_at_start - elapsed) if rem_at_start > 0 else 0

        # Double safety: hard stop timestamp
        if now >= hard_stop_at:
            remaining = 0

        snap = _snapshot(uid, authed, u)

        will_hangup = False
        hangup_reason = ""

        if not snap["is_pro"]:
            will_hangup = True
            hangup_reason = "not_pro"
        elif remaining <= 0:
            will_hangup = True
            hangup_reason = "voice_quota_used"

        log.info(
            "[VOICE_PING][%s] uid=%s call_id=%s authed=%s is_pro=%s elapsed=%s rem_start=%s remaining_call=%s remaining_rt=%s will_hangup=%s reason=%s weekly_used=%s",
            rid,
            uid,
            call_id,
            authed,
            snap["is_pro"],
            elapsed,
            rem_at_start,
            remaining,
            snap.get("voice_remaining_sec"),
            will_hangup,
            hangup_reason,
            snap.get("voice_weekly_used_sec"),
        )

        if not snap["is_pro"]:
            dur = min(elapsed, MAX_SINGLE_VOICE_CHARGE_SEC)
            if rem_at_start > 0:
                dur = min(dur, rem_at_start)
            used_before = int(u.get("weekly_voice_used_sec") or 0)
            u["weekly_voice_used_sec"] = min(WEEKLY_VOICE_ALLOW_SEC, used_before + dur)
            calls = u.get("voice_calls") or {}
            if isinstance(calls, dict):
                calls.pop(call_id, None)
            u["voice_calls"] = calls
            snap2 = _snapshot(uid, authed, u)
            users[uid] = u
            db["users"] = users
            _save_db(db)
            log.info(
                "[VOICE_PING][%s] hangup-settle uid=%s call_id=%s charged_sec=%s weekly_used_before=%s weekly_used_after=%s",
                rid,
                uid,
                call_id,
                dur,
                used_before,
                u.get("weekly_voice_used_sec"),
            )
            return _err(402, "not_pro", "hangup", snap2)

        if remaining <= 0:
            dur = min(elapsed, MAX_SINGLE_VOICE_CHARGE_SEC)
            if rem_at_start > 0:
                dur = min(dur, rem_at_start)

            _ensure_week_reset(u)
            used_before = int(u.get("weekly_voice_used_sec") or 0)
            u["weekly_voice_used_sec"] = min(WEEKLY_VOICE_ALLOW_SEC, used_before + dur)

            calls = u.get("voice_calls") or {}
            if isinstance(calls, dict):
                calls.pop(call_id, None)
            u["voice_calls"] = calls

            snap2 = _snapshot(uid, authed, u)
            users[uid] = u
            db["users"] = users
            _save_db(db)

            log.info(
                "[VOICE_PING][%s] quota-used uid=%s call_id=%s charged_sec=%s weekly_used_before=%s weekly_used_after=%s -> 402 hangup",
                rid,
                uid,
                call_id,
                dur,
                used_before,
                u.get("weekly_voice_used_sec"),
            )
            return _err(402, "voice_quota_used", "hangup", snap2)

        users[uid] = u
        db["users"] = users
        _save_db(db)
        return {
            "ok": True,
            "call_id": call_id,
            "callId": call_id,
            "remaining_sec": int(remaining),
            "hard_stop_at_ms": int(hard_stop_at * 1000),
            "billing": _snapshot(uid, authed, u),
        }


@router.post("/voice/end")
async def billing_voice_end(req: Request):
    body = {}
    try:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    user_id = (body.get("user_id") or body.get("userId") or "").strip()
    call_id = (body.get("call_id") or body.get("callId") or "").strip()
    if not call_id:
        return JSONResponse({"ok": False, "error": "missing_call_id"}, status_code=400)

    rid = _rid(req)

    with _LOCK:
        db = _load_db()
        uid, authed, provided = _resolve_user_key(req, user_id)

        if authed and provided and provided != uid and provided.startswith("guest:"):
            _merge_guest_into_authed(db, uid, provided)

        users = db.get("users") or {}
        u = _ensure_user(users.get(uid))

        calls = u.get("voice_calls") or {}
        if not isinstance(calls, dict):
            calls = {}
        c = calls.pop(call_id, None)
        u["voice_calls"] = calls

        if not isinstance(c, dict):
            users[uid] = u
            db["users"] = users
            _save_db(db)
            snap = _snapshot(uid, authed, u)
            log.info("[VOICE_END][%s] uid=%s call_id=%s missing -> skipped", rid, uid, call_id)
            return {"ok": True, "skipped": True, "charged_sec": 0, "billing": snap}

        now = _now()
        start = int((c or {}).get("start") or now)
        last_seen = int((c or {}).get("last_seen") or now)
        rem_at_start = int((c or {}).get("rem_at_start") or 0)

        end_at = min(now, last_seen) if last_seen else now
        dur = max(0, end_at - start)
        dur = min(dur, MAX_SINGLE_VOICE_CHARGE_SEC)
        if rem_at_start > 0:
            dur = min(dur, rem_at_start)

        _ensure_week_reset(u)
        used_before = int(u.get("weekly_voice_used_sec") or 0)
        u["weekly_voice_used_sec"] = min(WEEKLY_VOICE_ALLOW_SEC, used_before + dur)

        users[uid] = u
        db["users"] = users
        _save_db(db)

        snap = _snapshot(uid, authed, u)

        log.info(
            "[VOICE_END][%s] uid=%s call_id=%s charged_sec=%s weekly_used_before=%s weekly_used_after=%s voice_rem_now=%s",
            rid,
            uid,
            call_id,
            int(dur),
            used_before,
            u.get("weekly_voice_used_sec"),
            snap.get("voice_remaining_sec"),
        )
        return {"ok": True, "charged_sec": int(dur), "billing": snap}


@router.post("/video/consume")
async def billing_video_consume(
    req: Request,
    x_idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
):
    body = {}
    try:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    user_id = (body.get("user_id") or body.get("userId") or "").strip()
    amount = body.get("amount", 1)
    reason = (body.get("reason") or "gen").strip()[:64]

    try:
        amount = int(amount)
    except Exception:
        amount = 1
    if amount <= 0:
        amount = 1
    amount = min(amount, MAX_VIDEO_CONSUME_AMOUNT)

    rid = _rid(req)

    with _LOCK:
        db = _load_db()
        uid, authed, provided = _resolve_user_key(req, user_id)

        if authed and provided and provided != uid and provided.startswith("guest:"):
            _merge_guest_into_authed(db, uid, provided)

        users = db.get("users") or {}
        u = _ensure_user(users.get(uid))

        snap = _snapshot(uid, authed, u)

        if VIDEO_REQUIRE_PRO and not snap["is_pro"]:
            log.info("[VIDEO_CONSUME][%s] uid=%s blocked not_pro", rid, uid)
            return _err(402, "not_pro", "open_paywall", snap)

        credits = int(u.get("video_credits") or 0)
        if credits < amount:
            log.info("[VIDEO_CONSUME][%s] uid=%s blocked no_credits credits=%s need=%s", rid, uid, credits, amount)
            return _err(402, "no_video_credits", "open_paywall", snap)

        consume_id = (x_idempotency_key or "").strip() or uuid.uuid4().hex

        consumes = u.get("video_consumes") or {}
        if not isinstance(consumes, dict):
            consumes = {}
        if consume_id in consumes:
            snap2 = _snapshot(uid, authed, u)
            users[uid] = u
            db["users"] = users
            _save_db(db)
            log.info("[VIDEO_CONSUME][%s] uid=%s consume_id=%s skipped", rid, uid, consume_id)
            return {"ok": True, "consume_id": consume_id, "skipped": True, "billing": snap2}

        u["video_credits"] = credits - amount
        consumes[consume_id] = {"amount": amount, "ts": _now(), "refunded": False, "reason": reason}
        u["video_consumes"] = consumes

        users[uid] = u
        db["users"] = users
        _save_db(db)

        snap2 = _snapshot(uid, authed, u)
        log.info(
            "[VIDEO_CONSUME][%s] uid=%s consume_id=%s amount=%s credits_before=%s credits_after=%s reason=%s",
            rid,
            uid,
            consume_id,
            amount,
            credits,
            u.get("video_credits"),
            reason,
        )
        return {"ok": True, "consume_id": consume_id, "billing": snap2}


@router.post("/video/refund")
async def billing_video_refund(req: Request):
    body = {}
    try:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    user_id = (body.get("user_id") or body.get("userId") or "").strip()
    consume_id = (body.get("consume_id") or body.get("consumeId") or "").strip()
    if not consume_id:
        return JSONResponse({"ok": False, "error": "missing_consume_id"}, status_code=400)

    rid = _rid(req)

    with _LOCK:
        db = _load_db()
        uid, authed, provided = _resolve_user_key(req, user_id)

        if authed and provided and provided != uid and provided.startswith("guest:"):
            _merge_guest_into_authed(db, uid, provided)

        users = db.get("users") or {}
        u = _ensure_user(users.get(uid))

        consumes = u.get("video_consumes") or {}
        if not isinstance(consumes, dict):
            consumes = {}
        c = consumes.get(consume_id)
        if not isinstance(c, dict):
            snap = _snapshot(uid, authed, u)
            users[uid] = u
            db["users"] = users
            _save_db(db)
            log.info("[VIDEO_REFUND][%s] uid=%s consume_id=%s missing -> skipped", rid, uid, consume_id)
            return {"ok": True, "skipped": True, "billing": snap}

        if bool(c.get("refunded")):
            snap = _snapshot(uid, authed, u)
            users[uid] = u
            db["users"] = users
            _save_db(db)
            log.info("[VIDEO_REFUND][%s] uid=%s consume_id=%s already_refunded -> skipped", rid, uid, consume_id)
            return {"ok": True, "skipped": True, "billing": snap}

        amount = int(c.get("amount") or 0)
        credits_before = int(u.get("video_credits") or 0)
        if amount > 0:
            u["video_credits"] = credits_before + amount

        c["refunded"] = True
        consumes[consume_id] = c
        u["video_consumes"] = consumes

        users[uid] = u
        db["users"] = users
        _save_db(db)

        snap = _snapshot(uid, authed, u)
        log.info(
            "[VIDEO_REFUND][%s] uid=%s consume_id=%s amount=%s credits_before=%s credits_after=%s",
            rid,
            uid,
            consume_id,
            amount,
            credits_before,
            u.get("video_credits"),
        )
        return {"ok": True, "billing": snap}

