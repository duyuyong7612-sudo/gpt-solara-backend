"""
memory_identity.py — ChatAGI Commercial Memory Identity Layer
==============================================================

Goal:
- Every commercial user gets an isolated memory namespace.
- Never use raw IP as a long-term commercial identity when a stable account/device id exists.
- Never trust unauthenticated user_id/account_id headers for account identity.

User key format:
- Authenticated: tenant:{tenant_id}:user:{user_id}
- Guest/device:   tenant:guest:device:{device_id}
- Local fallback: tenant:dev:ip:{hashed_ip}   (only outside production)

Recommended production env:
  APP_ENV=production
  CHATAGI_JWT_SECRET=<stable-secret>
  MEMORY_REQUIRE_STABLE_ID=1
  MEMORY_ALLOW_GUEST_MEMORY=1

Notes:
- This module intentionally does not import server_session.py to avoid circular imports.
- It can verify the existing HS256 access token issued by auth.py/server_session.py.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, Request

_SAFE_RE = re.compile(r"[^a-zA-Z0-9_\-:.]")


def _bool_env(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes", "on")


def _is_production() -> bool:
    return (os.getenv("APP_ENV") or os.getenv("ENV") or os.getenv("RENDER") or "").strip().lower() in (
        "prod",
        "production",
        "render",
    )


def sanitize_memory_key(value: str, *, max_len: int = 180) -> str:
    """Sanitize one memory namespace component or full key."""
    v = (value or "").strip()
    if not v:
        return "default"
    v = _SAFE_RE.sub("_", v)
    if len(v) > max_len:
        digest = hashlib.sha256(v.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
    return v


def sanitize_component(value: str, *, fallback: str = "default", max_len: int = 80) -> str:
    v = sanitize_memory_key(value or fallback, max_len=max_len)
    # Avoid accidental nested namespace injection in individual components.
    return v.replace(":", "_") or fallback


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s or "") + pad)


def _json_dumps_compact(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _verify_hs256_jwt(token: str, secret: str) -> Optional[Dict[str, Any]]:
    """Verify existing HS256 JWT and return payload."""
    if not token or not secret:
        return None
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        msg = f"{parts[0]}.{parts[1]}".encode("utf-8")
        expected = base64.urlsafe_b64encode(hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()).decode("utf-8").rstrip("=")
        if not hmac.compare_digest(expected, parts[2]):
            return None
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        exp = payload.get("exp")
        if exp is not None and float(exp) < time.time():
            return None
        return payload
    except Exception:
        return None


def _decode_unverified_payload(token: str) -> Optional[Dict[str, Any]]:
    """Development-only fallback for inspecting a token payload."""
    try:
        parts = (token or "").split(".")
        if len(parts) < 2:
            return None
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return ""


def _jwt_secrets() -> Tuple[str, ...]:
    # server_session.py uses CHATAGI_JWT_SECRET; older auth.py used JWT_SECRET.
    vals = [
        os.getenv("CHATAGI_JWT_SECRET") or "",
        os.getenv("JWT_SECRET") or "",
    ]
    out = []
    for v in vals:
        v = (v or "").strip()
        if v and v not in out:
            out.append(v)
    return tuple(out)


def _token_payload(request: Request) -> Optional[Dict[str, Any]]:
    token = _bearer_token(request)
    if not token:
        return None
    for sec in _jwt_secrets():
        payload = _verify_hs256_jwt(token, sec)
        if payload:
            return payload
    # local-only debug fallback, disabled by default
    if not _is_production() and _bool_env("MEMORY_ALLOW_UNVERIFIED_DEV_JWT", "0"):
        return _decode_unverified_payload(token)
    return None


def _body_value(body: Dict[str, Any], *keys: str) -> str:
    if not isinstance(body, dict):
        return ""
    for k in keys:
        v = body.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _header_value(request: Request, *keys: str) -> str:
    for k in keys:
        v = request.headers.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _query_value(request: Request, *keys: str) -> str:
    try:
        for k in keys:
            v = request.query_params.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
    except Exception:
        return ""
    return ""


@dataclass(frozen=True)
class MemoryIdentity:
    user_key: str
    tenant_id: str
    account_user_id: str
    identity_type: str  # authenticated | guest | local_ip
    project_id: str = "default"
    session_id: str = ""
    source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_memory_identity(request: Request, body: Optional[Dict[str, Any]] = None) -> MemoryIdentity:
    """Resolve the commercial memory namespace for the current request.

    Security model:
    - Authenticated account id comes only from verified Bearer JWT `sub`.
    - Tenant id comes from JWT (`tenant_id`, `org_id`, `workspace_id`) or defaults to `personal`.
    - Anonymous memory uses stable device/client id only; no arbitrary account_user_id is trusted.
    - In production, IP-only fallback is blocked by default.
    """
    body = body or {}
    payload = _token_payload(request)

    project_id = _body_value(body, "project_id", "projectId") or _header_value(request, "x-project-id") or _query_value(request, "project_id", "projectId") or "default"
    session_id = _body_value(body, "session_id", "sessionId", "conversation_id", "conversationId") or _header_value(request, "x-session-id") or ""

    if payload:
        raw_sub = str(payload.get("sub") or payload.get("user_id") or "").strip()
        if raw_sub:
            raw_tenant = str(payload.get("tenant_id") or payload.get("org_id") or payload.get("workspace_id") or "personal").strip() or "personal"
            tenant_id = sanitize_component(raw_tenant, fallback="personal")
            account_user_id = sanitize_component(raw_sub, fallback="user")
            user_key = f"tenant:{tenant_id}:user:{account_user_id}"
            return MemoryIdentity(
                user_key=user_key,
                tenant_id=tenant_id,
                account_user_id=account_user_id,
                identity_type="authenticated",
                project_id=sanitize_component(project_id, fallback="default"),
                session_id=sanitize_memory_key(session_id) if session_id else "",
                source="bearer_jwt",
            )

    # Stable guest/device identity. This is acceptable for Guest mode but still isolated.
    allow_guest = _bool_env("MEMORY_ALLOW_GUEST_MEMORY", "1")
    device_id = (
        _body_value(body, "device_id", "deviceId", "client_id", "clientId")
        or _header_value(request, "x-device-id", "x-client-id", "X-Device-ID", "X-Client-ID")
        or _query_value(request, "device_id", "deviceId", "client_id", "clientId")
    )
    if allow_guest and device_id:
        did = sanitize_component(device_id, fallback="device")
        user_key = f"tenant:guest:device:{did}"
        return MemoryIdentity(
            user_key=user_key,
            tenant_id="guest",
            account_user_id=did,
            identity_type="guest",
            project_id=sanitize_component(project_id, fallback="default"),
            session_id=sanitize_memory_key(session_id) if session_id else "",
            source="stable_device_id",
        )

    require_stable = _bool_env("MEMORY_REQUIRE_STABLE_ID", "1")
    if _is_production() or require_stable:
        raise HTTPException(status_code=401, detail="Memory requires an authenticated user or stable device_id/client_id")

    # Local development fallback only. Hash IP to avoid leaking raw IP in memory keys.
    ip = request.client.host if request.client else "unknown"
    hashed = hashlib.sha256(str(ip).encode("utf-8")).hexdigest()[:16]
    return MemoryIdentity(
        user_key=f"tenant:dev:ip:{hashed}",
        tenant_id="dev",
        account_user_id=f"ip_{hashed}",
        identity_type="local_ip",
        project_id=sanitize_component(project_id, fallback="default"),
        session_id=sanitize_memory_key(session_id) if session_id else "",
        source="local_ip_fallback",
    )


def memory_identity_response(request: Request, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return resolve_memory_identity(request, body or {}).as_dict()
