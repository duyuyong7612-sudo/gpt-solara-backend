# auth.py
# MVP Auth for launch (B方案):
# - POST /auth/apple with {identity_token}
# - decode Apple JWT payload to get sub/email (NO signature verify for MVP)
# - issue our own access_token (HS256) for backend auth
#
# IMPORTANT:
# - Your iOS expects response contains "access_token".
# - Make sure server includes this router: app.include_router(auth.router)

import os
import json
import time
import base64
import hmac
import hashlib
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/auth", tags=["auth"])

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_TTL_SEC = int(os.getenv("JWT_TTL_SEC", str(60 * 60 * 24 * 30)))  # 30 days default
DEBUG_AUTH = os.getenv("DEBUG_AUTH", "0") == "1"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _decode_jwt_payload(jwt_str: str) -> Dict[str, Any]:
    """
    Decode JWT payload without verifying signature (MVP).
    Apple identity_token is a JWT: header.payload.signature
    """
    try:
        parts = (jwt_str or "").split(".")
        if len(parts) < 2:
            return {}
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _issue_access_token(user_id: str) -> str:
    """
    Issue our own HS256 JWT:
    sub = user_id (e.g., apple:<sub>)
    """
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"sub": user_id, "iat": now, "exp": now + JWT_TTL_SEC}

    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    msg = f"{h}.{p}".encode("utf-8")

    sig = hmac.new(JWT_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
    s = _b64url_encode(sig)
    return f"{h}.{p}.{s}"


@router.post("/apple")
async def auth_apple(req: Request):
    """
    Body: { "identity_token": "<apple_identity_token>" }
    Returns:
      { ok, user_id, access_token, email }
    """
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid_json"}, status_code=400)

    identity_token = (body.get("identity_token") or "").strip()
    if not identity_token:
        return JSONResponse({"ok": False, "error": "missing_identity_token"}, status_code=400)

    payload = _decode_jwt_payload(identity_token)
    sub = (payload.get("sub") or "").strip()
    email = (payload.get("email") or "").strip()

    # Dev fallback (optional)
    if not sub:
        sub = (body.get("sub") or "").strip()

    if not sub:
        # keep error key stable for your current debug
        resp = {"ok": False, "error": "missing_sub"}
        if DEBUG_AUTH:
            resp["debug"] = {"payload_keys": list(payload.keys())[:20]}
        return JSONResponse(resp, status_code=400)

    user_id = f"apple:{sub}"
    access_token = _issue_access_token(user_id)

    resp = {
        "ok": True,
        "user_id": user_id,
        "access_token": access_token,
        "email": email or None,
    }

    if DEBUG_AUTH:
        # help you confirm server is running new code
        resp["debug"] = {
            "jwt_ttl_sec": JWT_TTL_SEC,
            "has_email": bool(email),
            "server_time": int(time.time()),
        }

    return resp
