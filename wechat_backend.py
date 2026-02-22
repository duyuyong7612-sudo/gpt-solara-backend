"""
Solara WeChat Module Backend (FastAPI)
- 人与人聊天：会话/消息（SQLite 持久化）
- 实时推送：WebSocket /ws
- 视频通话信令：/ws 透传 call.* / webrtc.*（媒体层可自行接 WebRTC / SFU / TURN）

运行：
  pip install -r requirements.txt
  uvicorn wechat_backend:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


DB_PATH = os.environ.get("WECHAT_DB", "wechat.db")


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        device_id TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        created_at REAL NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tokens (
        token TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        created_at REAL NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY,
        type TEXT NOT NULL,            -- direct / group
        title TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversation_members (
        conversation_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        PRIMARY KEY(conversation_id, user_id),
        FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id),
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        message_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        sender_id TEXT NOT NULL,
        kind TEXT NOT NULL,            -- text (扩展: image / file / call etc)
        text TEXT NOT NULL,
        created_at REAL NOT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id),
        FOREIGN KEY(sender_id) REFERENCES users(user_id)
    )
    """)

    conn.commit()
    conn.close()


class LoginReq(BaseModel):
    device_id: str = Field(..., min_length=2)
    display_name: str = Field(..., min_length=1, max_length=64)


class LoginResp(BaseModel):
    ok: bool = True
    user_id: str
    token: str
    display_name: str


class DirectConvReq(BaseModel):
    peer_id: str = Field(..., min_length=1)


class ConversationSummary(BaseModel):
    id: str
    type: str
    title: str
    peer_id: Optional[str] = None
    last_text: Optional[str] = None
    updated_at: float


class ListConversationsResp(BaseModel):
    ok: bool = True
    conversations: List[ConversationSummary]


class SendMessageReq(BaseModel):
    conversation_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=8000)


class MessageDTO(BaseModel):
    id: str
    conversation_id: str
    sender_id: str
    kind: str
    text: str
    created_at: float


class SendMessageResp(BaseModel):
    ok: bool = True
    message: MessageDTO


class ListMessagesResp(BaseModel):
    ok: bool = True
    messages: List[MessageDTO]


class CreateDirectResp(BaseModel):
    ok: bool = True
    conversation: ConversationSummary


def create_user_if_needed(conn: sqlite3.Connection, device_id: str, display_name: str) -> Tuple[str, str]:
    cur = conn.cursor()
    cur.execute("SELECT user_id, display_name FROM users WHERE device_id = ?", (device_id,))
    row = cur.fetchone()
    if row:
        user_id = row["user_id"]
        cur.execute("UPDATE users SET display_name = ? WHERE user_id = ?", (display_name, user_id))
        conn.commit()
        return user_id, display_name

    user_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO users(user_id, device_id, display_name, created_at) VALUES (?,?,?,?)",
        (user_id, device_id, display_name, time.time()),
    )
    conn.commit()
    return user_id, display_name


def issue_token(conn: sqlite3.Connection, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tokens(token, user_id, created_at) VALUES (?,?,?)",
        (token, user_id, time.time()),
    )
    conn.commit()
    return token


def auth_user_from_header(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="missing Authorization")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="invalid Authorization")
    token = parts[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="invalid token")

    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM tokens WHERE token = ?", (token,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="token not found")
    return row["user_id"]


def current_user_id(authorization: Optional[str] = Header(default=None)) -> str:
    return auth_user_from_header(authorization)


def get_display_name(conn: sqlite3.Connection, user_id: str) -> str:
    cur = conn.cursor()
    cur.execute("SELECT display_name FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    return row["display_name"] if row else user_id


def ensure_user_exists(conn: sqlite3.Connection, user_id: str) -> None:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail=f"user not found: {user_id}")


def find_direct_conversation(conn: sqlite3.Connection, a: str, b: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("""
    SELECT c.conversation_id
    FROM conversations c
    JOIN conversation_members m1 ON m1.conversation_id = c.conversation_id AND m1.user_id = ?
    JOIN conversation_members m2 ON m2.conversation_id = c.conversation_id AND m2.user_id = ?
    WHERE c.type = 'direct'
    """, (a, b))
    row = cur.fetchone()
    return row["conversation_id"] if row else None


def create_direct_conversation(conn: sqlite3.Connection, a: str, b: str) -> str:
    conv_id = str(uuid.uuid4())
    now = time.time()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversations(conversation_id, type, title, created_at, updated_at) VALUES (?,?,?,?,?)",
        (conv_id, "direct", "direct", now, now),
    )
    cur.execute("INSERT INTO conversation_members(conversation_id, user_id) VALUES (?,?)", (conv_id, a))
    cur.execute("INSERT INTO conversation_members(conversation_id, user_id) VALUES (?,?)", (conv_id, b))
    conn.commit()
    return conv_id


def list_conversation_members(conn: sqlite3.Connection, conv_id: str) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM conversation_members WHERE conversation_id = ?", (conv_id,))
    return [r["user_id"] for r in cur.fetchall()]


def last_message_text(conn: sqlite3.Connection, conv_id: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 1",
        (conv_id,),
    )
    row = cur.fetchone()
    return row["text"] if row else None


def conv_updated_at(conn: sqlite3.Connection, conv_id: str) -> float:
    cur = conn.cursor()
    cur.execute("SELECT updated_at FROM conversations WHERE conversation_id = ?", (conv_id,))
    row = cur.fetchone()
    return float(row["updated_at"]) if row else 0.0


def update_conv_updated_at(conn: sqlite3.Connection, conv_id: str, ts: float) -> None:
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET updated_at = ? WHERE conversation_id = ?", (ts, conv_id))
    conn.commit()


def conv_summary_for_user(conn: sqlite3.Connection, user_id: str, conv_id: str) -> ConversationSummary:
    cur = conn.cursor()
    cur.execute("SELECT type FROM conversations WHERE conversation_id = ?", (conv_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="conversation not found")

    ctype = row["type"]
    members = list_conversation_members(conn, conv_id)
    peer_id: Optional[str] = None
    title = "群聊"
    if ctype == "direct":
        peer_id = next((m for m in members if m != user_id), None)
        title = get_display_name(conn, peer_id) if peer_id else "direct"

    return ConversationSummary(
        id=conv_id,
        type=ctype,
        title=title,
        peer_id=peer_id,
        last_text=last_message_text(conn, conv_id),
        updated_at=conv_updated_at(conn, conv_id),
    )


def insert_message(conn: sqlite3.Connection, conv_id: str, sender_id: str, kind: str, text: str) -> MessageDTO:
    now = time.time()
    mid = str(uuid.uuid4())
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages(message_id, conversation_id, sender_id, kind, text, created_at) VALUES (?,?,?,?,?,?)",
        (mid, conv_id, sender_id, kind, text, now),
    )
    update_conv_updated_at(conn, conv_id, now)
    conn.commit()
    return MessageDTO(
        id=mid,
        conversation_id=conv_id,
        sender_id=sender_id,
        kind=kind,
        text=text,
        created_at=now,
    )


def list_messages(conn: sqlite3.Connection, conv_id: str, limit: int = 80) -> List[MessageDTO]:
    cur = conn.cursor()
    cur.execute(
        "SELECT message_id, conversation_id, sender_id, kind, text, created_at "
        "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
        (conv_id, limit),
    )
    out: List[MessageDTO] = []
    for r in cur.fetchall():
        out.append(
            MessageDTO(
                id=r["message_id"],
                conversation_id=r["conversation_id"],
                sender_id=r["sender_id"],
                kind=r["kind"],
                text=r["text"],
                created_at=float(r["created_at"]),
            )
        )
    return out


@dataclass
class WSConn:
    user_id: str
    ws: WebSocket


class Hub:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._conns: Dict[str, Set[WebSocket]] = {}

    async def add(self, user_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._conns.setdefault(user_id, set()).add(ws)

    async def remove(self, user_id: str, ws: WebSocket) -> None:
        async with self._lock:
            if user_id in self._conns:
                self._conns[user_id].discard(ws)
                if not self._conns[user_id]:
                    del self._conns[user_id]

    async def send_to(self, user_id: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._conns.get(user_id, set()))
        if not targets:
            return
        msg = json.dumps(payload, ensure_ascii=False)
        for t in targets:
            try:
                await t.send_text(msg)
            except Exception:
                pass

    async def broadcast(self, user_ids: List[str], payload: Dict[str, Any]) -> None:
        await asyncio.gather(*(self.send_to(uid, payload) for uid in set(user_ids)), return_exceptions=True)


hub = Hub()

app = FastAPI(title="Solara WeChat Module Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产请改白名单
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    db_init()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/auth/login", response_model=LoginResp)
def login(req: LoginReq) -> LoginResp:
    conn = db_connect()
    user_id, display_name = create_user_if_needed(conn, req.device_id, req.display_name)
    token = issue_token(conn, user_id)
    conn.close()
    return LoginResp(user_id=user_id, token=token, display_name=display_name)


@app.get("/conversations", response_model=ListConversationsResp)
def list_conversations(user_id: str = Depends(current_user_id)) -> ListConversationsResp:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
    SELECT c.conversation_id
    FROM conversations c
    JOIN conversation_members m ON m.conversation_id = c.conversation_id
    WHERE m.user_id = ?
    ORDER BY c.updated_at DESC
    """, (user_id,))
    ids = [r["conversation_id"] for r in cur.fetchall()]
    convs = [conv_summary_for_user(conn, user_id, cid) for cid in ids]
    conn.close()
    return ListConversationsResp(conversations=convs)


@app.post("/conversations/direct", response_model=CreateDirectResp)
def create_or_get_direct(req: DirectConvReq, user_id: str = Depends(current_user_id)) -> CreateDirectResp:
    conn = db_connect()
    ensure_user_exists(conn, req.peer_id)

    cid = find_direct_conversation(conn, user_id, req.peer_id)
    if not cid:
        cid = create_direct_conversation(conn, user_id, req.peer_id)

    conv = conv_summary_for_user(conn, user_id, cid)
    conn.close()
    return CreateDirectResp(conversation=conv)


@app.get("/messages", response_model=ListMessagesResp)
def get_messages(
    conversation_id: str = Query(..., min_length=1),
    user_id: str = Depends(current_user_id),
    limit: int = Query(80, ge=1, le=200),
) -> ListMessagesResp:
    conn = db_connect()
    members = list_conversation_members(conn, conversation_id)
    if user_id not in members:
        conn.close()
        raise HTTPException(status_code=403, detail="not a member")

    msgs = list_messages(conn, conversation_id, limit=limit)
    conn.close()
    return ListMessagesResp(messages=msgs)


@app.post("/messages", response_model=SendMessageResp)
async def send_message(req: SendMessageReq, user_id: str = Depends(current_user_id)) -> SendMessageResp:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    conn = db_connect()
    members = list_conversation_members(conn, req.conversation_id)
    if user_id not in members:
        conn.close()
        raise HTTPException(status_code=403, detail="not a member")

    msg = insert_message(conn, req.conversation_id, user_id, "text", text)
    conn.close()

    await hub.broadcast(members, {
        "type": "msg.new",
        "conversation_id": req.conversation_id,
        "message": msg.dict(),
        "server_ts": time.time(),
    })

    return SendMessageResp(message=msg)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: str = Query(..., min_length=8)) -> None:
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM tokens WHERE token = ?", (token,))
    row = cur.fetchone()
    conn.close()

    if not row:
        await ws.close(code=4401)
        return

    user_id = row["user_id"]
    await ws.accept()
    await hub.add(user_id, ws)

    await hub.send_to(user_id, {"type": "ws.ready", "user_id": user_id, "server_ts": time.time()})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            mtype = str(obj.get("type", ""))

            # 信令透传：call.* / webrtc.* / custom.*
            if mtype.startswith("call.") or mtype.startswith("webrtc.") or mtype.startswith("custom."):
                to = obj.get("to")
                if isinstance(to, str) and to:
                    obj["from"] = user_id
                    obj["server_ts"] = time.time()
                    await hub.send_to(to, obj)
                continue

            if mtype == "ping":
                await ws.send_text(json.dumps({"type": "pong", "server_ts": time.time()}))
                continue

    except WebSocketDisconnect:
        pass
    finally:
        await hub.remove(user_id, ws)
