# billing.py (Postgres / 商用多实例闭环版)
# -------------------------------------------------------------------
# 目标（保持兼容 + 后端强制门禁）：
# 1) 仍然提供 router=/billing，保证 server_session.py 的 `from billing import router as billing_router` 不用改
# 2) 保持旧接口可用：/billing/me /billing/credits /billing/ingest /billing/voice/start|ping|end（含别名）
# 3) 升级为 Postgres 多实例可用：plan/额度/窗口重置/原子扣减
# 4) 暴露后端护卫层函数供 server_session.py 调用（强制拦截）：
#    - billing_get_effective_plan(user_id, requested_plan_raw)
#    - billing_guard_or_403(user_id, feature, want=1, consume=False, check_quota=True)
#
# 说明：
# - 强烈建议生产环境设置 DATABASE_URL（或 BILLING_DATABASE_URL）
# - 若未设置数据库，将自动退化为 DEV 内存记账（仅为了本地跑通；重启丢失）
# -------------------------------------------------------------------

from __future__ import annotations

import os
import time
import uuid
import json
import base64
import hmac
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Request, HTTPException
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

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# -----------------------------
# Public constants (server_session imports these)
# -----------------------------
FEATURE_TEXT = "text"
FEATURE_IMAGE = "image"
FEATURE_REALTIME = "realtime"
FEATURE_VIDEO = "video"

PLAN_GUEST = "guest"
PLAN_PRO = "pro"
PLAN_ULTRA = "ultra"
PLAN_CODER = "coder"

router = APIRouter(prefix="/billing", tags=["billing"])

# -----------------------------
# ENV / Config
# -----------------------------
BILLING_BYPASS_GATES = os.getenv("BILLING_BYPASS_GATES", "0").strip().lower() in ("1", "true", "yes", "on")

# Postgres DSN
DATABASE_URL = (os.getenv("BILLING_DATABASE_URL") or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or "").strip()

# Billing timezone (用于“每天/每周/每月”窗口)
_TZ_NAME = (os.getenv("BILLING_TZ") or "America/Los_Angeles").strip()
BILLING_TZ = None
if ZoneInfo is not None:
    try:
        BILLING_TZ = ZoneInfo(_TZ_NAME)
    except Exception:
        BILLING_TZ = ZoneInfo("UTC")

# Quota defaults (可用 env 覆盖)
# - 约定：<0 表示不限；0 表示 0（通常意味着不开放该能力）
GUEST_TEXT_PER_DAY = int(os.getenv("BILLING_GUEST_TEXT_PER_DAY", os.getenv("GUEST_TEXT_PER_DAY", "30")))
GUEST_IMAGE_PER_DAY = int(os.getenv("BILLING_GUEST_IMAGE_PER_DAY", os.getenv("GUEST_IMAGE_PER_DAY", "5")))

PRO_VOICE_SECONDS_PER_WEEK = int(os.getenv("BILLING_PRO_VOICE_SECONDS_PER_WEEK", os.getenv("PRO_VOICE_SECONDS_PER_WEEK", "6000")))
ULTRA_VOICE_SECONDS_PER_WEEK = int(os.getenv("BILLING_ULTRA_VOICE_SECONDS_PER_WEEK", os.getenv("ULTRA_VOICE_SECONDS_PER_WEEK", "12000")))
CODER_VOICE_SECONDS_PER_WEEK = int(os.getenv("BILLING_CODER_VOICE_SECONDS_PER_WEEK", os.getenv("CODER_VOICE_SECONDS_PER_WEEK", "20000")))

ULTRA_VIDEO_PER_MONTH = int(os.getenv("BILLING_ULTRA_VIDEO_PER_MONTH", os.getenv("ULTRA_VIDEO_PER_MONTH", "30")))
CODER_VIDEO_PER_MONTH = int(os.getenv("BILLING_CODER_VIDEO_PER_MONTH", os.getenv("CODER_VIDEO_PER_MONTH", "60")))

# 兼容旧 /billing/credits 的“voice credits”单位：默认 60 秒 = 1 credit
VOICE_CREDIT_UNIT_SECONDS = int(os.getenv("VOICE_CREDIT_UNIT_SECONDS", "60"))
VOICE_PING_COST_SECONDS = int(os.getenv("VOICE_PING_COST_SECONDS", "20"))  # 兼容旧 ping 扣费：每 ping 默认扣 20 秒

# Purchase -> plan 映射（可配置）
# 逗号分隔 productId；不配也可以用启发式匹配
PRODUCT_IDS_PRO = [s.strip() for s in (os.getenv("BILLING_PRODUCT_IDS_PRO") or "").split(",") if s.strip()]
PRODUCT_IDS_ULTRA = [s.strip() for s in (os.getenv("BILLING_PRODUCT_IDS_ULTRA") or "").split(",") if s.strip()]
PRODUCT_IDS_CODER = [s.strip() for s in (os.getenv("BILLING_PRODUCT_IDS_CODER") or "").split(",") if s.strip()]

# Video pack (可选)
VIDEO_PACK_5_IDS = [s.strip() for s in (os.getenv("BILLING_VIDEO_PACK_5_IDS") or "solara_video_pack_5").split(",") if s.strip()]
VIDEO_PACK_10_IDS = [s.strip() for s in (os.getenv("BILLING_VIDEO_PACK_10_IDS") or "solara_video_pack_10").split(",") if s.strip()]

# -----------------------------
# JWT decode (与 server_session.py 保持一致：CHATAGI_JWT_SECRET)
# -----------------------------
JWT_SECRET = os.getenv("CHATAGI_JWT_SECRET") or ""

def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))

def _get_bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None

def _decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    # 与 server_session.py 一样：HS256(header.payload, JWT_SECRET)
    try:
        if not JWT_SECRET:
            return None
        header_b64, payload_b64, sig_b64 = token.split(".", 2)
        msg = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = base64.urlsafe_b64encode(
            hmac.new(JWT_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
        ).rstrip(b"=").decode("utf-8")
        if not hmac.compare_digest(expected_sig, sig_b64):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None

def _sanitize_user_key(key: str) -> str:
    k = (key or "").strip()
    if not k:
        return "default"
    if len(k) > 128:
        return hashlib.sha256(k.encode("utf-8")).hexdigest()
    import re
    return re.sub(r"[^a-zA-Z0-9_\-:.]", "_", k)

def _derive_user_id(req: Request, body: Dict[str, Any]) -> str:
    """尽量与 server_session.py 的 _derive_user_key 保持一致"""
    # 1) Bearer JWT -> sub
    try:
        tok = _get_bearer_token(req)
        if tok:
            payload = _decode_access_token(tok)
            if isinstance(payload, dict):
                sub = str(payload.get("sub") or "").strip()
                if sub:
                    return _sanitize_user_key(sub)
    except Exception:
        pass

    # 2) explicit user id
    uid = str(
        (body or {}).get("user_id")
        or (body or {}).get("userId")
        or (body or {}).get("uid")
        or (body or {}).get("account_id")
        or (body or {}).get("accountId")
        or ""
    ).strip()
    if not uid:
        uid = str(
            req.headers.get("x-user-id")
            or req.headers.get("x-uid")
            or req.headers.get("x-account-id")
            or ""
        ).strip()
    if uid:
        return _sanitize_user_key(uid)

    # 3) anonymous client id
    cid = str((body or {}).get("client_id") or (body or {}).get("clientId") or (body or {}).get("user_key") or (body or {}).get("userKey") or "").strip()
    if not cid:
        cid = str(req.headers.get("x-client-id") or req.headers.get("x-user-key") or "").strip()
    if cid:
        return _sanitize_user_key(cid)

    # 4) ip
    ip = req.client.host if req.client else "unknown"
    return _sanitize_user_key(ip)

# -----------------------------
# Postgres layer (psycopg2)
# -----------------------------
_PG_AVAILABLE = False
_PG_POOL = None

try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
    from psycopg2.pool import ThreadedConnectionPool  # type: ignore

    _PG_AVAILABLE = True
except Exception:
    _PG_AVAILABLE = False

def _pg_pool():
    global _PG_POOL
    if _PG_POOL is not None:
        return _PG_POOL
    if not DATABASE_URL:
        return None
    if not _PG_AVAILABLE:
        raise RuntimeError("Postgres enabled but psycopg2 is not installed. Add psycopg2-binary to requirements.")
    maxconn = int(os.getenv("BILLING_PG_MAXCONN", "10"))
    minconn = int(os.getenv("BILLING_PG_MINCONN", "1"))
    # Note: psycopg2 accepts postgres:// and postgresql://
    _PG_POOL = ThreadedConnectionPool(minconn=minconn, maxconn=maxconn, dsn=DATABASE_URL)
    return _PG_POOL

def _pg_exec(sql: str, params: Tuple[Any, ...] = ()) -> None:
    pool = _pg_pool()
    if pool is None:
        return
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
    finally:
        pool.putconn(conn)

def _pg_init():
    pool = _pg_pool()
    if pool is None:
        return
    # Create tables (idempotent)
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS billing_users (
          user_id TEXT PRIMARY KEY,
          plan TEXT NOT NULL DEFAULT 'guest',
          expire_at TIMESTAMPTZ NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

          day_anchor DATE NOT NULL,
          week_anchor DATE NOT NULL,
          month_anchor DATE NOT NULL,

          text_used_day INTEGER NOT NULL DEFAULT 0,
          image_used_day INTEGER NOT NULL DEFAULT 0,
          voice_used_week INTEGER NOT NULL DEFAULT 0,
          video_used_month INTEGER NOT NULL DEFAULT 0
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS billing_purchases (
          tx_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          product_id TEXT NOT NULL,
          expires_at TIMESTAMPTZ NULL,
          raw JSONB NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_billing_purchases_user ON billing_purchases(user_id, created_at DESC);
        """,
        """
        CREATE TABLE IF NOT EXISTS billing_voice_calls (
          call_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_ping_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          ended_at TIMESTAMPTZ NULL
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_billing_voice_calls_user ON billing_voice_calls(user_id, started_at DESC);
        """,
    ]
    pool = _pg_pool()
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                for s in ddl:
                    cur.execute(s)
    finally:
        pool.putconn(conn)

# -----------------------------
# DEV fallback store (if no DATABASE_URL)
# -----------------------------
_DEV_USERS: Dict[str, Dict[str, Any]] = {}

def _now_ts() -> int:
    return int(time.time())

def _anchors(now: Optional[datetime] = None) -> Tuple[str, str, str, int, int, int]:
    tz = BILLING_TZ
    if now is None:
        now = datetime.now(tz) if tz else datetime.utcnow()
    if tz:
        # ensure tz-aware
        if now.tzinfo is None:
            now = now.replace(tzinfo=tz)
    # day anchor
    day = now.date()
    # week anchor (Mon)
    week_start = (now - timedelta(days=now.weekday())).date()
    # month anchor
    month_start = now.replace(day=1).date()
    # next resets
    next_day = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=tz) if tz else datetime.combine(day + timedelta(days=1), datetime.min.time())
    next_week = datetime.combine(week_start + timedelta(days=7), datetime.min.time(), tzinfo=tz) if tz else datetime.combine(week_start + timedelta(days=7), datetime.min.time())
    # next month
    if month_start.month == 12:
        nm = datetime(month_start.year + 1, 1, 1, tzinfo=tz) if tz else datetime(month_start.year + 1, 1, 1)
    else:
        nm = datetime(month_start.year, month_start.month + 1, 1, tzinfo=tz) if tz else datetime(month_start.year, month_start.month + 1, 1)
    return (
        day.isoformat(),
        week_start.isoformat(),
        month_start.isoformat(),
        int(next_day.timestamp()),
        int(next_week.timestamp()),
        int(nm.timestamp()),
    )

def _normalize_plan(raw: str) -> str:
    raw0 = (raw or "").strip()
    s = raw0.lower()
    if not s:
        return ""
    import re
    s2 = re.sub(r"[\s_\-]+", "", s)
    # English keywords
    if "ultra" in s2:
        return PLAN_ULTRA
    if "pro" in s2 or "voice" in s2:
        return PLAN_PRO
    if "coder" in s2 or "advanced" in s2 or "code" in s2:
        return PLAN_CODER
    if "guest" in s2 or "basic" in s2 or "free" in s2:
        return PLAN_GUEST
    # Chinese keywords (UI display names)
    if "视频" in raw0 or "影片" in raw0:
        return PLAN_ULTRA
    if "语音" in raw0 or "通话" in raw0:
        return PLAN_PRO
    if "编程" in raw0 or "高级" in raw0:
        return PLAN_CODER
    if "基础" in raw0 or "文本" in raw0 or "免费" in raw0:
        return PLAN_GUEST
    if s in (PLAN_GUEST, PLAN_PRO, PLAN_ULTRA, PLAN_CODER):
        return s
    return s

def _plan_rank(plan: str) -> int:
    p = _normalize_plan(plan)
    if p == PLAN_GUEST:
        return 0
    if p == PLAN_PRO:
        return 1
    if p == PLAN_ULTRA:
        return 2
    if p == PLAN_CODER:
        return 3
    return 0

def _rank_to_plan(rank: int) -> str:
    return {0: PLAN_GUEST, 1: PLAN_PRO, 2: PLAN_ULTRA, 3: PLAN_CODER}.get(int(rank), PLAN_GUEST)

def _quota_for_plan(plan: str) -> Dict[str, int]:
    p = _normalize_plan(plan) or PLAN_GUEST
    if p == PLAN_GUEST:
        return {
            "text_per_day": GUEST_TEXT_PER_DAY,
            "image_per_day": GUEST_IMAGE_PER_DAY,
            "voice_seconds_per_week": 0,
            "video_per_month": 0,
        }
    if p == PLAN_PRO:
        return {
            "text_per_day": -1,
            "image_per_day": -1,
            "voice_seconds_per_week": PRO_VOICE_SECONDS_PER_WEEK,
            "video_per_month": 0,
        }
    if p == PLAN_ULTRA:
        return {
            "text_per_day": -1,
            "image_per_day": -1,
            "voice_seconds_per_week": ULTRA_VOICE_SECONDS_PER_WEEK,
            "video_per_month": ULTRA_VIDEO_PER_MONTH,
        }
    # coder
    return {
        "text_per_day": -1,
        "image_per_day": -1,
        "voice_seconds_per_week": CODER_VOICE_SECONDS_PER_WEEK,
        "video_per_month": CODER_VIDEO_PER_MONTH,
    }

def _feature_allowed(plan: str, feature: str) -> bool:
    p = _normalize_plan(plan) or PLAN_GUEST
    r = _plan_rank(p)
    if feature in (FEATURE_TEXT, FEATURE_IMAGE):
        return True
    if feature == FEATURE_REALTIME:
        return r >= 1  # pro+
    if feature == FEATURE_VIDEO:
        return r >= 2  # ultra+
    return False

# -----------------------------
# Postgres helpers: load/reset/consume (atomic)
# -----------------------------
@dataclass
class _State:
    user_id: str
    plan: str
    expire_at: Optional[int]
    day_anchor: str
    week_anchor: str
    month_anchor: str
    text_used_day: int
    image_used_day: int
    voice_used_week: int
    video_used_month: int

def _state_from_row(row: Dict[str, Any]) -> _State:
    exp = row.get("expire_at")
    if exp is not None:
        try:
            exp_ts = int(exp.timestamp())  # datetime -> ts
        except Exception:
            exp_ts = None
    else:
        exp_ts = None
    return _State(
        user_id=str(row["user_id"]),
        plan=str(row.get("plan") or PLAN_GUEST),
        expire_at=exp_ts,
        day_anchor=str(row.get("day_anchor")),
        week_anchor=str(row.get("week_anchor")),
        month_anchor=str(row.get("month_anchor")),
        text_used_day=int(row.get("text_used_day") or 0),
        image_used_day=int(row.get("image_used_day") or 0),
        voice_used_week=int(row.get("voice_used_week") or 0),
        video_used_month=int(row.get("video_used_month") or 0),
    )

def _ensure_user_row_pg(cur, user_id: str) -> None:
    d, w, m, _, _, _ = _anchors()
    cur.execute(
        """
        INSERT INTO billing_users(user_id, plan, day_anchor, week_anchor, month_anchor)
        VALUES (%s, 'guest', %s::date, %s::date, %s::date)
        ON CONFLICT (user_id) DO NOTHING
        """,
        (user_id, d, w, m),
    )

def _load_state_pg(cur, user_id: str) -> _State:
    cur.execute(
        """SELECT * FROM billing_users WHERE user_id=%s FOR UPDATE""",
        (user_id,),
    )
    row = cur.fetchone()
    if not row:
        # should not happen after ensure; retry
        cur.execute("""SELECT * FROM billing_users WHERE user_id=%s FOR UPDATE""", (user_id,))
        row = cur.fetchone()
    if not row:
        # fallback
        d, w, m, _, _, _ = _anchors()
        return _State(user_id=user_id, plan=PLAN_GUEST, expire_at=None, day_anchor=d, week_anchor=w, month_anchor=m,
                      text_used_day=0, image_used_day=0, voice_used_week=0, video_used_month=0)
    return _state_from_row(dict(row))

def _maybe_reset_windows_pg(cur, st: _State) -> _State:
    d, w, m, _, _, _ = _anchors()
    updates = {}
    if st.day_anchor != d:
        updates.update({"day_anchor": d, "text_used_day": 0, "image_used_day": 0})
        st.day_anchor = d
        st.text_used_day = 0
        st.image_used_day = 0
    if st.week_anchor != w:
        updates.update({"week_anchor": w, "voice_used_week": 0})
        st.week_anchor = w
        st.voice_used_week = 0
    if st.month_anchor != m:
        updates.update({"month_anchor": m, "video_used_month": 0})
        st.month_anchor = m
        st.video_used_month = 0
    # subscription expiry -> downgrade to guest
    if st.expire_at is not None and st.expire_at < _now_ts():
        if _normalize_plan(st.plan) != PLAN_GUEST:
            updates.update({"plan": PLAN_GUEST, "expire_at": None})
            st.plan = PLAN_GUEST
            st.expire_at = None

    if updates:
        sets = ", ".join([f"{k}=%s" for k in updates.keys()]) + ", updated_at=now()"
        cur.execute(
            f"UPDATE billing_users SET {sets} WHERE user_id=%s",
            tuple(list(updates.values()) + [st.user_id]),
        )
    return st

def _guard_and_optional_consume_pg(user_id: str, feature: str, want: int = 1, *, consume: bool = False, check_quota: bool = True) -> _State:
    if BILLING_BYPASS_GATES:
        # ensure row exists even when bypass (so /me works)
        st = _get_state(user_id)
        return st

    pool = _pg_pool()
    if pool is None:
        # DEV fallback
        return _guard_and_optional_consume_dev(user_id, feature, want=want, consume=consume, check_quota=check_quota)

    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                _ensure_user_row_pg(cur, user_id)
                st = _load_state_pg(cur, user_id)
                st = _maybe_reset_windows_pg(cur, st)

                # plan gate
                if not _feature_allowed(st.plan, feature):
                    raise HTTPException(
                        status_code=402,
                        detail={
                            "ok": False,
                            "error": "upgrade_required",
                            "feature": feature,
                            "plan": st.plan,
                            "ts": _now_ts(),
                        },
                    )

                if check_quota:
                    q = _quota_for_plan(st.plan)
                    # text/image: -1 unlimited
                    if feature == FEATURE_TEXT:
                        limit = int(q["text_per_day"])
                        if limit >= 0 and (st.text_used_day + int(want)) > limit:
                            raise HTTPException(
                                status_code=402,
                                detail={
                                    "ok": False,
                                    "error": "quota_exceeded",
                                    "feature": feature,
                                    "plan": st.plan,
                                    "limit": limit,
                                    "used": st.text_used_day,
                                    "ts": _now_ts(),
                                },
                            )
                    elif feature == FEATURE_IMAGE:
                        limit = int(q["image_per_day"])
                        if limit >= 0 and (st.image_used_day + int(want)) > limit:
                            raise HTTPException(
                                status_code=402,
                                detail={
                                    "ok": False,
                                    "error": "quota_exceeded",
                                    "feature": feature,
                                    "plan": st.plan,
                                    "limit": limit,
                                    "used": st.image_used_day,
                                    "ts": _now_ts(),
                                },
                            )
                    elif feature == FEATURE_VIDEO:
                        limit = int(q["video_per_month"])
                        if limit >= 0 and (st.video_used_month + int(want)) > limit:
                            raise HTTPException(
                                status_code=402,
                                detail={
                                    "ok": False,
                                    "error": "quota_exceeded",
                                    "feature": feature,
                                    "plan": st.plan,
                                    "limit": limit,
                                    "used": st.video_used_month,
                                    "ts": _now_ts(),
                                },
                            )
                    elif feature == FEATURE_REALTIME:
                        # want=seconds
                        limit = int(q["voice_seconds_per_week"])
                        if limit >= 0 and (st.voice_used_week + int(want)) > limit:
                            raise HTTPException(
                                status_code=402,
                                detail={
                                    "ok": False,
                                    "error": "quota_exceeded",
                                    "feature": feature,
                                    "plan": st.plan,
                                    "limit": limit,
                                    "used": st.voice_used_week,
                                    "ts": _now_ts(),
                                },
                            )

                if consume and int(want) > 0:
                    if feature == FEATURE_TEXT:
                        st.text_used_day += int(want)
                        cur.execute(
                            "UPDATE billing_users SET text_used_day=text_used_day+%s, updated_at=now() WHERE user_id=%s",
                            (int(want), user_id),
                        )
                    elif feature == FEATURE_IMAGE:
                        st.image_used_day += int(want)
                        cur.execute(
                            "UPDATE billing_users SET image_used_day=image_used_day+%s, updated_at=now() WHERE user_id=%s",
                            (int(want), user_id),
                        )
                    elif feature == FEATURE_VIDEO:
                        st.video_used_month += int(want)
                        cur.execute(
                            "UPDATE billing_users SET video_used_month=video_used_month+%s, updated_at=now() WHERE user_id=%s",
                            (int(want), user_id),
                        )
                    elif feature == FEATURE_REALTIME:
                        st.voice_used_week += int(want)
                        cur.execute(
                            "UPDATE billing_users SET voice_used_week=voice_used_week+%s, updated_at=now() WHERE user_id=%s",
                            (int(want), user_id),
                        )

                return st
    finally:
        pool.putconn(conn)

# -----------------------------
# DEV fallback implementations
# -----------------------------
def _dev_get_or_create(user_id: str) -> Dict[str, Any]:
    u = _DEV_USERS.get(user_id)
    if u:
        return u
    d, w, m, _, _, _ = _anchors()
    u = {
        "user_id": user_id,
        "plan": PLAN_GUEST,
        "expire_at": None,
        "day_anchor": d,
        "week_anchor": w,
        "month_anchor": m,
        "text_used_day": 0,
        "image_used_day": 0,
        "voice_used_week": 0,
        "video_used_month": 0,
    }
    _DEV_USERS[user_id] = u
    return u

def _dev_maybe_reset(u: Dict[str, Any]) -> None:
    d, w, m, _, _, _ = _anchors()
    if u.get("day_anchor") != d:
        u["day_anchor"] = d
        u["text_used_day"] = 0
        u["image_used_day"] = 0
    if u.get("week_anchor") != w:
        u["week_anchor"] = w
        u["voice_used_week"] = 0
    if u.get("month_anchor") != m:
        u["month_anchor"] = m
        u["video_used_month"] = 0
    exp = u.get("expire_at")
    if exp is not None and int(exp) < _now_ts():
        u["plan"] = PLAN_GUEST
        u["expire_at"] = None

def _guard_and_optional_consume_dev(user_id: str, feature: str, want: int = 1, *, consume: bool = False, check_quota: bool = True) -> _State:
    u = _dev_get_or_create(user_id)
    _dev_maybe_reset(u)
    if not _feature_allowed(str(u.get("plan")), feature):
        raise HTTPException(status_code=402, detail={"ok": False, "error": "upgrade_required", "feature": feature, "plan": u.get("plan"), "ts": _now_ts()})

    if check_quota:
        q = _quota_for_plan(str(u.get("plan")))
        if feature == FEATURE_TEXT:
            limit = int(q["text_per_day"])
            if limit >= 0 and int(u.get("text_used_day", 0)) + int(want) > limit:
                raise HTTPException(status_code=402, detail={"ok": False, "error": "quota_exceeded", "feature": feature, "limit": limit, "used": int(u.get("text_used_day", 0)), "ts": _now_ts()})
        if feature == FEATURE_IMAGE:
            limit = int(q["image_per_day"])
            if limit >= 0 and int(u.get("image_used_day", 0)) + int(want) > limit:
                raise HTTPException(status_code=402, detail={"ok": False, "error": "quota_exceeded", "feature": feature, "limit": limit, "used": int(u.get("image_used_day", 0)), "ts": _now_ts()})
        if feature == FEATURE_VIDEO:
            limit = int(q["video_per_month"])
            if limit >= 0 and int(u.get("video_used_month", 0)) + int(want) > limit:
                raise HTTPException(status_code=402, detail={"ok": False, "error": "quota_exceeded", "feature": feature, "limit": limit, "used": int(u.get("video_used_month", 0)), "ts": _now_ts()})
        if feature == FEATURE_REALTIME:
            limit = int(q["voice_seconds_per_week"])
            if limit >= 0 and int(u.get("voice_used_week", 0)) + int(want) > limit:
                raise HTTPException(status_code=402, detail={"ok": False, "error": "quota_exceeded", "feature": feature, "limit": limit, "used": int(u.get("voice_used_week", 0)), "ts": _now_ts()})

    if consume and int(want) > 0:
        if feature == FEATURE_TEXT:
            u["text_used_day"] = int(u.get("text_used_day", 0)) + int(want)
        elif feature == FEATURE_IMAGE:
            u["image_used_day"] = int(u.get("image_used_day", 0)) + int(want)
        elif feature == FEATURE_VIDEO:
            u["video_used_month"] = int(u.get("video_used_month", 0)) + int(want)
        elif feature == FEATURE_REALTIME:
            u["voice_used_week"] = int(u.get("voice_used_week", 0)) + int(want)

    return _State(
        user_id=user_id,
        plan=str(u.get("plan") or PLAN_GUEST),
        expire_at=int(u["expire_at"]) if u.get("expire_at") else None,
        day_anchor=str(u.get("day_anchor")),
        week_anchor=str(u.get("week_anchor")),
        month_anchor=str(u.get("month_anchor")),
        text_used_day=int(u.get("text_used_day") or 0),
        image_used_day=int(u.get("image_used_day") or 0),
        voice_used_week=int(u.get("voice_used_week") or 0),
        video_used_month=int(u.get("video_used_month") or 0),
    )

def _get_state(user_id: str) -> _State:
    # Non-consuming read (with window reset)
    if BILLING_BYPASS_GATES:
        # still want a coherent payload
        pass
    pool = _pg_pool()
    if pool is None:
        u = _dev_get_or_create(user_id)
        _dev_maybe_reset(u)
        return _State(
            user_id=user_id,
            plan=str(u.get("plan") or PLAN_GUEST),
            expire_at=int(u["expire_at"]) if u.get("expire_at") else None,
            day_anchor=str(u.get("day_anchor")),
            week_anchor=str(u.get("week_anchor")),
            month_anchor=str(u.get("month_anchor")),
            text_used_day=int(u.get("text_used_day") or 0),
            image_used_day=int(u.get("image_used_day") or 0),
            voice_used_week=int(u.get("voice_used_week") or 0),
            video_used_month=int(u.get("video_used_month") or 0),
        )

    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                _ensure_user_row_pg(cur, user_id)
                st = _load_state_pg(cur, user_id)
                st = _maybe_reset_windows_pg(cur, st)
                return st
    finally:
        pool.putconn(conn)

# -----------------------------
# Public API (server_session uses)
# -----------------------------
def billing_get_effective_plan(user_id: str, requested_plan_raw: str = "") -> str:
    """返回“最终生效 plan”（服务器 plan 上限 + 客户端可选降级）"""
    try:
        st = _get_state(user_id)
        server_plan = _normalize_plan(st.plan) or PLAN_GUEST
    except Exception:
        server_plan = PLAN_GUEST

    req = _normalize_plan(requested_plan_raw or "")
    if not req:
        return server_plan

    # cap: requested cannot exceed server_plan
    eff_rank = min(_plan_rank(req), _plan_rank(server_plan))
    return _rank_to_plan(eff_rank)

def billing_guard_or_403(user_id: str, feature: str, want: int = 1, *, consume: bool = False, check_quota: bool = True) -> Dict[str, Any]:
    """核心护卫：不通过就抛 HTTPException；通过则返回简要 billing 状态"""
    st = _guard_and_optional_consume_pg(user_id, feature, want=want, consume=consume, check_quota=check_quota)
    return billing_payload_from_state(st)

# -----------------------------
# Payload builders
# -----------------------------
def billing_payload_from_state(st: _State) -> Dict[str, Any]:
    q = _quota_for_plan(st.plan)
    d, w, m, next_day_ts, next_week_ts, next_month_ts = _anchors()

    def _remain(limit: int, used: int) -> Optional[int]:
        if limit < 0:
            return None
        return max(0, int(limit) - int(used))

    remaining_text = _remain(int(q["text_per_day"]), st.text_used_day)
    remaining_image = _remain(int(q["image_per_day"]), st.image_used_day)
    remaining_voice = _remain(int(q["voice_seconds_per_week"]), st.voice_used_week)
    remaining_video = _remain(int(q["video_per_month"]), st.video_used_month)

    return {
        "ok": True,
        "user_key": st.user_id,
        "plan": _normalize_plan(st.plan) or PLAN_GUEST,
        "is_subscribed": _normalize_plan(st.plan) != PLAN_GUEST and (st.expire_at is None or st.expire_at > _now_ts()),
        "quota": {
            "text_per_day": int(q["text_per_day"]),
            "image_per_day": int(q["image_per_day"]),
            "voice_seconds_per_week": int(q["voice_seconds_per_week"]),
            "video_per_month": int(q["video_per_month"]),
        },
        "usage": {
            "text_used_day": int(st.text_used_day),
            "image_used_day": int(st.image_used_day),
            "voice_used_week": int(st.voice_used_week),
            "video_used_month": int(st.video_used_month),
        },
        "remaining": {
            "text_remaining_day": remaining_text,
            "image_remaining_day": remaining_image,
            "voice_remaining_week_seconds": remaining_voice,
            "video_remaining_month": remaining_video,
        },
        "reset_at": {
            "day": int(next_day_ts),
            "week": int(next_week_ts),
            "month": int(next_month_ts),
        },
        # 兼容旧客户端：credits（text/video=次数；voice=按 unit 换算）
        "credits": {
            "text": int(remaining_text or 0) if remaining_text is not None else 999999,
            "voice": int((remaining_voice or 0) // max(1, VOICE_CREDIT_UNIT_SECONDS)) if remaining_voice is not None else 999999,
            "video": int(remaining_video or 0) if remaining_video is not None else 999999,
            "image": int(remaining_image or 0) if remaining_image is not None else 999999,
        },
        "ts": _now_ts(),
    }

def billing_payload(user_id: str) -> Dict[str, Any]:
    st = _get_state(user_id)
    return billing_payload_from_state(st)

# -----------------------------
# Purchase ingest (productId/txId) -> plan update + dedup
# -----------------------------
class IngestPurchaseReq(_BaseModel):
    productId: str
    txId: str
    expiresMs: Optional[int] = None

def _infer_plan_from_product(product_id: str) -> Optional[str]:
    pid = (product_id or "").strip()
    if not pid:
        return None
    pid_lower = pid.lower()

    if pid in PRODUCT_IDS_CODER or any(x and x.lower() == pid_lower for x in PRODUCT_IDS_CODER):
        return PLAN_CODER
    if pid in PRODUCT_IDS_ULTRA or any(x and x.lower() == pid_lower for x in PRODUCT_IDS_ULTRA):
        return PLAN_ULTRA
    if pid in PRODUCT_IDS_PRO or any(x and x.lower() == pid_lower for x in PRODUCT_IDS_PRO):
        return PLAN_PRO

    # heuristic
    if "coder" in pid_lower or "advanced" in pid_lower or "code" in pid_lower:
        return PLAN_CODER
    if "ultra" in pid_lower or "video" in pid_lower:
        return PLAN_ULTRA
    if "pro" in pid_lower or "voice" in pid_lower or "realtime" in pid_lower:
        return PLAN_PRO
    return None

def _is_video_pack(pid: str) -> int:
    pid0 = (pid or "").strip()
    pidl = pid0.lower()
    for x in VIDEO_PACK_10_IDS:
        if x and x.lower() == pidl:
            return 10
    for x in VIDEO_PACK_5_IDS:
        if x and x.lower() == pidl:
            return 5
    # heuristic fallback
    if "video_pack_10" in pidl or pidl.endswith("_10"):
        return 10
    if "video_pack_5" in pidl or pidl.endswith("_5"):
        return 5
    return 0

def _apply_purchase_pg(user_id: str, pid: str, tx: str, expires_ms: Optional[int], raw: Dict[str, Any]) -> Dict[str, Any]:
    pool = _pg_pool()
    if pool is None:
        # DEV fallback: just set plan
        u = _dev_get_or_create(user_id)
        plan = _infer_plan_from_product(pid)
        if plan:
            u["plan"] = plan
            if expires_ms:
                u["expire_at"] = int(expires_ms // 1000)
        return {"plan": u.get("plan"), "added_video": 0, "dedup": False}

    expires_at = None
    if expires_ms:
        try:
            expires_at = datetime.fromtimestamp(int(expires_ms) / 1000.0, tz=BILLING_TZ)
        except Exception:
            expires_at = None

    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # dedup by tx_id
                cur.execute("SELECT tx_id FROM billing_purchases WHERE tx_id=%s", (tx,))
                if cur.fetchone():
                    st = _get_state(user_id)
                    return {"plan": st.plan, "added_video": 0, "dedup": True}

                cur.execute(
                    """INSERT INTO billing_purchases(tx_id, user_id, product_id, expires_at, raw) VALUES (%s,%s,%s,%s,%s)""",
                    (tx, user_id, pid, expires_at, psycopg2.extras.Json(raw)),
                )

                _ensure_user_row_pg(cur, user_id)
                st = _load_state_pg(cur, user_id)
                st = _maybe_reset_windows_pg(cur, st)

                plan = _infer_plan_from_product(pid)
                if plan:
                    # upgrade plan if higher
                    old_rank = _plan_rank(st.plan)
                    new_rank = _plan_rank(plan)
                    eff_plan = plan if new_rank >= old_rank else st.plan
                    cur.execute(
                        "UPDATE billing_users SET plan=%s, expire_at=%s, updated_at=now() WHERE user_id=%s",
                        (eff_plan, expires_at, user_id),
                    )

                # video pack: 在“月用量”上给负数补偿不太优雅；这里只做兼容占位（建议以后用 add-on 表）
                added_video = _is_video_pack(pid)
                if added_video > 0:
                    # 简化：把 video_used_month 往回减，相当于多给次数（最低到 0）
                    cur.execute(
                        "UPDATE billing_users SET video_used_month=GREATEST(0, video_used_month-%s), updated_at=now() WHERE user_id=%s",
                        (int(added_video), user_id),
                    )

                st2 = _load_state_pg(cur, user_id)
                st2 = _maybe_reset_windows_pg(cur, st2)
                return {"plan": st2.plan, "added_video": int(added_video), "dedup": False}
    finally:
        pool.putconn(conn)

# -----------------------------
# Router endpoints
# -----------------------------
@router.get("/health")
def billing_health():
    if not DATABASE_URL:
        return {"ok": True, "mode": "dev_memory", "ts": _now_ts()}
    try:
        pool = _pg_pool()
        if pool is None:
            return {"ok": True, "mode": "dev_memory", "ts": _now_ts()}
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        finally:
            pool.putconn(conn)
        return {"ok": True, "mode": "postgres", "ts": _now_ts()}
    except Exception as e:
        return JSONResponse({"ok": False, "mode": "postgres", "error": str(e), "ts": _now_ts()}, status_code=500)

@router.get("/me")
def billing_me(req: Request):
    # keep compatibility: old clients don't send body; use headers/ip
    user_id = _derive_user_id(req, {})
    return billing_payload(user_id)

@router.get("/credits")
def billing_credits(req: Request):
    user_id = _derive_user_id(req, {})
    payload = billing_payload(user_id)
    # old shape: {ok,text,voice,video,ts}
    return {
        "ok": True,
        "text": int(payload.get("credits", {}).get("text", 0)),
        "voice": int(payload.get("credits", {}).get("voice", 0)),
        "video": int(payload.get("credits", {}).get("video", 0)),
        "image": int(payload.get("credits", {}).get("image", 0)),
        "plan": payload.get("plan", PLAN_GUEST),
        "ts": _now_ts(),
    }

@router.post("/ingest")
async def billing_ingest(req: Request, body: IngestPurchaseReq):
    user_id = _derive_user_id(req, {})
    pid = (body.productId or "").strip()
    tx = (body.txId or "").strip()
    if not pid or not tx:
        return JSONResponse(status_code=400, content={"ok": False, "error": "missing productId/txId", "ts": _now_ts()})

    raw = {"productId": pid, "txId": tx, "expiresMs": body.expiresMs, "headers": {"ua": req.headers.get("user-agent", "")}}
    r = _apply_purchase_pg(user_id, pid, tx, body.expiresMs, raw)
    payload = billing_payload(user_id)
    payload.update({
        "purchase": {"productId": pid, "txId": tx, "dedup": bool(r.get("dedup")), "added_video": int(r.get("added_video") or 0)},
        "ts": _now_ts(),
    })
    return payload

# -----------------------------
# Voice gating endpoints (兼容旧 VoiceBillingGuard)
# -----------------------------
class VoiceStartReq(_BaseModel):
    pass

class VoicePingReq(_BaseModel):
    callId: str
    seconds: Optional[int] = None  # 新版可传本次消耗秒数（不传则按 VOICE_PING_COST_SECONDS）

class VoiceEndReq(_BaseModel):
    callId: str
    seconds: Optional[int] = None  # 可传最终通话秒数

def _voice_start_impl(req: Request) -> Dict[str, Any]:
    user_id = _derive_user_id(req, {})
    # plan gate only（是否允许实时语音）
    billing_guard_or_403(user_id, FEATURE_REALTIME, check_quota=False)

    call_id = uuid.uuid4().hex

    # 写入 DB（多实例必须）
    pool = _pg_pool()
    if pool is not None:
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO billing_voice_calls(call_id, user_id, started_at, last_ping_at) VALUES (%s,%s,now(),now())",
                        (call_id, user_id),
                    )
        finally:
            pool.putconn(conn)
    else:
        # dev: ignore
        pass

    return {"ok": True, "callId": call_id, **billing_payload(user_id), "ts": _now_ts()}

def _voice_ping_impl(req: Request, call_id: str, seconds: Optional[int]) -> Dict[str, Any]:
    user_id = _derive_user_id(req, {})
    billing_guard_or_403(user_id, FEATURE_REALTIME, check_quota=False)

    # update last_ping
    pool = _pg_pool()
    if pool is not None:
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE billing_voice_calls SET last_ping_at=now() WHERE call_id=%s AND user_id=%s",
                        (call_id, user_id),
                    )
        finally:
            pool.putconn(conn)

    # 兼容旧逻辑：ping 扣一小段 seconds
    cost = int(seconds) if seconds is not None and int(seconds) > 0 else int(VOICE_PING_COST_SECONDS)
    # 如果你不想 ping 扣费，把 VOICE_PING_COST_SECONDS=0
    if cost > 0:
        billing_guard_or_403(user_id, FEATURE_REALTIME, want=cost, consume=True, check_quota=True)

    return {"ok": True, "callId": call_id, **billing_payload(user_id), "ts": _now_ts()}

def _voice_end_impl(req: Request, call_id: str, seconds: Optional[int]) -> Dict[str, Any]:
    user_id = _derive_user_id(req, {})
    billing_guard_or_403(user_id, FEATURE_REALTIME, check_quota=False)

    # mark ended
    pool = _pg_pool()
    if pool is not None:
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE billing_voice_calls SET ended_at=now() WHERE call_id=%s AND user_id=%s",
                        (call_id, user_id),
                    )
        finally:
            pool.putconn(conn)

    # end 时可一次性扣 seconds（若客户端愿意上报最终时长）
    if seconds is not None and int(seconds) > 0:
        billing_guard_or_403(user_id, FEATURE_REALTIME, want=int(seconds), consume=True, check_quota=True)

    return {"ok": True, "callId": call_id, **billing_payload(user_id), "ts": _now_ts()}

@router.post("/voice/start")
@router.post("/start_voice")
def billing_voice_start(req: Request):
    return _voice_start_impl(req)

@router.post("/voice/ping")
@router.post("/ping_voice")
def billing_voice_ping(req: Request, body: VoicePingReq):
    return _voice_ping_impl(req, body.callId, body.seconds)

@router.post("/voice/end")
@router.post("/end_voice")
def billing_voice_end(req: Request, body: VoiceEndReq):
    return _voice_end_impl(req, body.callId, body.seconds)

# -----------------------------
# Optional helper for other routes/modules
# -----------------------------
def require_credit(req: Request, kind: str = "text", cost: int = 1) -> Optional[Dict[str, Any]]:
    """兼容旧 helper：返回 None 表示放行；返回 dict 表示拒绝原因。"""
    user_id = _derive_user_id(req, {})
    feature = {
        "text": FEATURE_TEXT,
        "image": FEATURE_IMAGE,
        "video": FEATURE_VIDEO,
        "voice": FEATURE_REALTIME,
        "realtime": FEATURE_REALTIME,
    }.get(kind, FEATURE_TEXT)

    try:
        # 对 voice：cost 视为 seconds；对其它：cost 视为次数
        billing_guard_or_403(user_id, feature, want=int(cost), consume=True, check_quota=True)
        return None
    except HTTPException as e:
        # 统一为 dict（不抛出）
        try:
            detail = e.detail if isinstance(e.detail, dict) else {"error": str(e.detail)}
        except Exception:
            detail = {"error": "billing_blocked"}
        return detail

# Init PG tables on import (when DATABASE_URL is set)
try:
    if DATABASE_URL:
        _pg_init()
except Exception:
    # Don't block boot; /billing/health will show error
    pass

