# -*- coding: utf-8 -*-
"""
memory_module.py  (Plan A)

A dedicated, backend-only "memory module" for ChatAGI / Solara backend.

Design goals
- ✅ Keep existing DB schema compatibility (memory_items + memory_facts) so you can drop-in upgrade.
- ✅ Make memory writes NON-BLOCKING via an async write queue (critical for Render free-tier + iOS timeouts).
- ✅ Provide both:
    1) Vector memory (semantic recall)
    2) Facts memory (query-independent "important facts / preferences")
- ✅ Optional "episode memory" table for future hierarchical memory compression.

This module is intentionally dependency-light:
- standard library + requests
"""

from __future__ import annotations

import os
import re
import json
import time
import uuid
import math
import queue
import hashlib
import threading
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from array import array

import requests


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class MemoryConfig:
    db_path: str

    # Embedding
    openai_api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    embed_provider: str = "auto"  # auto|openai|local
    openai_timeout_sec: float = 25.0

    # Vector memory limits
    enabled: bool = True
    max_items_per_user: int = 2000
    candidate_limit: int = 1200  # how many recent items to score
    top_k_default: int = 6
    min_score_default: float = 0.25

    # Prompt formatting
    context_max_chars: int = 1600
    item_max_chars: int = 240

    # Async writer
    write_async: bool = True
    write_queue_maxsize: int = 512

    # Facts memory
    facts_enabled: bool = True
    facts_max_items_per_user: int = 400
    facts_prompt_limit: int = 12
    facts_context_max_chars: int = 1200
    facts_item_max_chars: int = 220

    # Facts extraction (optional)
    facts_extract_enabled: bool = True
    facts_extract_model_openai: str = "gpt-4o-mini"
    facts_importance_min: int = 2  # only save >= this importance
    # Episode memory (hierarchical compression)
    episodes_enabled: bool = True
    episodes_top_k_default: int = 4
    episodes_min_score_default: float = 0.23

    # Episode prompt formatting
    episodes_prompt_limit: int = 6
    episodes_context_max_chars: int = 1800
    episodes_item_max_chars: int = 340

    # Whether to store embeddings for episodes (recommended)
    episodes_store_embedding: bool = True

    # Episode summarization (optional; used by server-side compaction)
    episodes_summarize_enabled: bool = True
    episodes_summarize_model_openai: str = "gpt-4o-mini"
    episodes_summary_max_chars: int = 360
    episodes_title_max_chars: int = 40
    episodes_keywords_max: int = 8




    # Heuristics
    min_chars: int = 8
    max_store_chars: int = 1200


# -----------------------------
# Helpers
# -----------------------------

_USERKEY_SAFE_RE = re.compile(r"[^a-zA-Z0-9_\-:.]")


def sanitize_user_key(key: str) -> str:
    k = (key or "").strip()
    if not k:
        return "default"
    if len(k) > 128:
        return hashlib.sha256(k.encode("utf-8")).hexdigest()
    return _USERKEY_SAFE_RE.sub("_", k)


def _pack_f32(vec: List[float]) -> bytes:
    return array("f", [float(x) for x in vec]).tobytes()


def _unpack_f32(blob: bytes) -> array:
    a = array("f")
    a.frombytes(blob)
    return a


def _cosine(a: array, b: array) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb) + 1e-9
    return dot / denom


def _local_embed(text: str, dim: int = 384) -> List[float]:
    """
    Local, no-network embedding (feature hashing + L2 norm).
    Good enough for recall without paid embeddings.
    """
    t = (text or "").strip().lower()
    if not t:
        return [0.0] * dim

    toks = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", t)
    cjk = re.findall(r"[\u4e00-\u9fff]", t)
    if len(cjk) >= 2:
        toks.extend([cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)])

    vec = [0.0] * dim
    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 31) & 1) == 0 else -1.0
        vec[idx] += sign

    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def _parse_json_array_best_effort(s: str) -> List[Dict[str, Any]]:
    if not s:
        return []
    t = s.strip()
    t = t.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    if "[" in t and "]" in t:
        t = t[t.find("[") : t.rfind("]") + 1]
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        return []
    return []



def _parse_json_object_best_effort(s: str) -> Dict[str, Any]:
    """Parse a JSON object from an LLM output. Best-effort, safe fallback to {}."""
    if not s:
        return {}
    t = s.strip()
    t = t.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    if "{" in t and "}" in t:
        t = t[t.find("{") : t.rfind("}") + 1]
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _ensure_sqlite_column(con: sqlite3.Connection, table: str, col: str, col_def_sql: str) -> None:
    """Idempotently add a column to a SQLite table if missing."""
    try:
        cur = con.execute(f"PRAGMA table_info({table});")
        cols = {str(r[1]) for r in (cur.fetchall() or [])}
        if col in cols:
            return
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col_def_sql};")
    except Exception:
        return


# -----------------------------
# Memory Engine
# -----------------------------

class MemoryEngine:
    """
    Thread-safe memory engine.

    - Uses SQLite (WAL mode) for durability.
    - Embeddings: OpenAI (optional) or local hashing.
    - Async writer thread (optional).
    """

    def __init__(self, cfg: MemoryConfig) -> None:
        self.cfg = cfg
        self._q: "queue.Queue[Tuple[str, str]]" = queue.Queue(maxsize=max(1, int(cfg.write_queue_maxsize)))
        self._writer: Optional[threading.Thread] = None
        self._writer_started = False
        self._start_lock = threading.Lock()

    # --- DB ---
    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.cfg.db_path, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def init_db(self) -> None:
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(self.cfg.db_path), exist_ok=True)
        except Exception:
            pass

        with self._conn() as con:
            # Vector memory
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                  id TEXT PRIMARY KEY,
                  user_key TEXT NOT NULL,
                  text TEXT NOT NULL,
                  text_sha1 TEXT NOT NULL,
                  embedding BLOB NOT NULL,
                  dim INTEGER NOT NULL,
                  created_at REAL NOT NULL,
                  last_used_at REAL NOT NULL
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_used ON memory_items(user_key, last_used_at DESC);")
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_mem_user_sha1 ON memory_items(user_key, text_sha1);")

            # Facts memory
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_facts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_key TEXT NOT NULL,
                  content TEXT NOT NULL,
                  tags TEXT DEFAULT '',
                  importance INTEGER DEFAULT 1,
                  hash TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  UNIQUE(user_key, hash)
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_memfacts_user_imp_time ON memory_facts(user_key, importance DESC, created_at DESC);")

            # Episode memory (for future hierarchical memory compression)
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_episodes (
                  id TEXT PRIMARY KEY,
                  user_key TEXT NOT NULL,
                  conversation_id TEXT DEFAULT '',
                  summary TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_memepisodes_user_time ON memory_episodes(user_key, updated_at DESC);")
            # Optional columns for ranged/embedded episodes (idempotent upgrades)
            _ensure_sqlite_column(con, "memory_episodes", "title", "title TEXT DEFAULT ''")
            _ensure_sqlite_column(con, "memory_episodes", "keywords", "keywords TEXT DEFAULT ''")
            _ensure_sqlite_column(con, "memory_episodes", "start_at", "start_at REAL DEFAULT 0")
            _ensure_sqlite_column(con, "memory_episodes", "end_at", "end_at REAL DEFAULT 0")
            _ensure_sqlite_column(con, "memory_episodes", "embedding", "embedding BLOB")
            _ensure_sqlite_column(con, "memory_episodes", "dim", "dim INTEGER DEFAULT 0")
            _ensure_sqlite_column(con, "memory_episodes", "source_meta", "source_meta TEXT DEFAULT ''")
            con.execute("CREATE INDEX IF NOT EXISTS idx_memepisodes_user_conv_time ON memory_episodes(user_key, conversation_id, end_at DESC);")

        # Start writer if enabled
        if self.cfg.write_async:
            self._ensure_writer()

    # --- Embeddings ---
    def _embed(self, text: str) -> Optional[List[float]]:
        t = (text or "").strip()
        if not t:
            return None

        provider = (self.cfg.embed_provider or "auto").strip().lower()
        if provider not in ("auto", "openai", "local"):
            provider = "auto"

        # 1) OpenAI
        if provider in ("auto", "openai") and self.cfg.openai_api_key:
            try:
                r = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.cfg.openai_api_key}", "Content-Type": "application/json"},
                    json={"model": self.cfg.embed_model, "input": t},
                    timeout=float(self.cfg.openai_timeout_sec),
                )
                if r.status_code < 400:
                    obj = r.json()
                    emb = (obj.get("data") or [{}])[0].get("embedding")
                    if isinstance(emb, list) and emb:
                        return emb
            except Exception:
                # fallthrough to local
                pass

        # 2) Local fallback
        try:
            return _local_embed(t)
        except Exception:
            return None

    # --- Async writer ---
    def _ensure_writer(self) -> None:
        if self._writer_started:
            return
        with self._start_lock:
            if self._writer_started:
                return
            self._writer_started = True
            self._writer = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer.start()

    def _writer_loop(self) -> None:
        while True:
            try:
                user_key, text = self._q.get()
            except Exception:
                continue
            try:
                self.add_vector_sync(user_key, text)
            except Exception:
                # Never crash the writer thread
                pass

    # --- Vector memory ---
    def add_vector(self, user_key: str, text: str) -> None:
        """
        Public API. Non-blocking if write_async is enabled.
        """
        if not self.cfg.enabled:
            return
        u = sanitize_user_key(user_key)
        t = (text or "").strip()
        if not t:
            return
        if len(t) > int(self.cfg.max_store_chars):
            t = t[: int(self.cfg.max_store_chars)].rstrip() + "…"

        if self.cfg.write_async:
            self._ensure_writer()
            try:
                self._q.put_nowait((u, t))
                return
            except queue.Full:
                # fall back to sync insert if queue is full
                pass

        self.add_vector_sync(u, t)

    def add_vector_sync(self, user_key: str, text: str) -> None:
        if not self.cfg.enabled:
            return
        u = sanitize_user_key(user_key)
        t = (text or "").strip()
        if not t:
            return

        emb = self._embed(t)
        if not emb:
            return

        sha = hashlib.sha1(t.encode("utf-8")).hexdigest()
        now = time.time()

        with self._conn() as con:
            con.execute(
                """
                INSERT INTO memory_items(id,user_key,text,text_sha1,embedding,dim,created_at,last_used_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(user_key,text_sha1)
                DO UPDATE SET text=excluded.text, embedding=excluded.embedding, dim=excluded.dim, last_used_at=excluded.last_used_at;
                """,
                (uuid.uuid4().hex, u, t, sha, _pack_f32(emb), len(emb), now, now),
            )
            # prune
            con.execute(
                """
                DELETE FROM memory_items
                WHERE id IN (
                  SELECT id FROM memory_items WHERE user_key=? ORDER BY last_used_at DESC LIMIT -1 OFFSET ?
                );
                """,
                (u, int(self.cfg.max_items_per_user)),
            )

    def search_vectors(self, user_key: str, query: str, k: Optional[int] = None, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        if not self.cfg.enabled:
            return []
        u = sanitize_user_key(user_key)
        q = (query or "").strip()
        if not q:
            return []
        k = max(1, int(k or self.cfg.top_k_default))
        min_score = float(min_score if min_score is not None else self.cfg.min_score_default)

        qv_list = self._embed(q)
        if not qv_list:
            return []

        qv = array("f", [float(x) for x in qv_list])

        with self._conn() as con:
            cur = con.execute(
                "SELECT id,text,embedding,dim FROM memory_items WHERE user_key=? ORDER BY last_used_at DESC LIMIT ?",
                (u, int(self.cfg.candidate_limit)),
            )
            rows = cur.fetchall()

        scored: List[Tuple[float, str, str]] = []
        for rid, txt, blob, dim in rows:
            try:
                v = _unpack_f32(blob)
                if len(v) != int(dim):
                    continue
                s = _cosine(qv, v)
                if s >= min_score:
                    scored.append((s, str(rid), str(txt or "")))
            except Exception:
                continue

        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:k]

        # touch recency
        if top:
            now = time.time()
            with self._conn() as con:
                for s, rid, _ in top:
                    con.execute("UPDATE memory_items SET last_used_at=? WHERE id=? AND user_key=?", (now, rid, u))

        return [{"id": rid, "text": txt, "score": float(s)} for (s, rid, txt) in top]

    def build_vector_context(self, user_key: str, query: str, k: Optional[int] = None, min_score: Optional[float] = None) -> str:
        hits = self.search_vectors(user_key, query, k=k, min_score=min_score)
        if not hits:
            return ""
        lines: List[str] = ["以下是【长期记忆】（仅供参考；若有冲突/不确定请向用户确认）："]
        total = 0
        for h in hits:
            t = (h.get("text") or "").strip()
            if not t:
                continue
            if len(t) > int(self.cfg.item_max_chars):
                t = t[: int(self.cfg.item_max_chars)].rstrip() + "…"
            add = f"- {t}"
            if total + len(add) > int(self.cfg.context_max_chars):
                break
            lines.append(add)
            total += len(add)
        return "\n".join(lines).strip()

    # --- Facts memory ---
    def _fact_hash(self, user_key: str, content: str) -> str:
        raw = f"{sanitize_user_key(user_key)}:{(content or '').strip().lower()}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def facts_save(self, user_key: str, content: str, tags: str = "", importance: int = 1) -> None:
        if not self.cfg.facts_enabled:
            return
        u = sanitize_user_key(user_key)
        c = re.sub(r"\s+", " ", (content or "").strip())
        if not c:
            return
        if len(c) > 420:
            c = c[:420].rstrip() + "…"

        try:
            imp = int(importance or 1)
        except Exception:
            imp = 1
        imp = max(1, min(imp, 5))

        h = self._fact_hash(u, c)
        now = time.time()

        with self._conn() as con:
            con.execute(
                """
                INSERT OR IGNORE INTO memory_facts (user_key, content, tags, importance, hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (u, c, (tags or "").strip(), imp, h, now),
            )
            con.execute(
                """
                DELETE FROM memory_facts
                WHERE id IN (
                  SELECT id FROM memory_facts
                  WHERE user_key=?
                  ORDER BY importance DESC, created_at DESC
                  LIMIT -1 OFFSET ?
                )
                """,
                (u, int(self.cfg.facts_max_items_per_user)),
            )

    def facts_list(self, user_key: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.cfg.facts_enabled:
            return []
        u = sanitize_user_key(user_key)
        lim = max(1, min(int(limit or self.cfg.facts_prompt_limit), 200))
        with self._conn() as con:
            cur = con.execute(
                "SELECT id, content, tags, importance, created_at FROM memory_facts WHERE user_key=? ORDER BY importance DESC, created_at DESC LIMIT ?",
                (u, lim),
            )
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for rid, content, tags, imp, created_at in rows:
            out.append(
                {
                    "id": int(rid),
                    "content": str(content or ""),
                    "tags": str(tags or ""),
                    "importance": int(imp or 1),
                    "created_at": float(created_at or 0.0),
                }
            )
        return out

    def facts_build_prompt(self, user_key: str, limit: Optional[int] = None) -> str:
        if not self.cfg.facts_enabled:
            return ""
        mems = self.facts_list(user_key, limit=limit or self.cfg.facts_prompt_limit)
        if not mems:
            return ""
        lines = ["以下是【长期记忆】（重要事实/偏好；如不确定请向用户确认）："]
        total = 0
        for m in mems:
            c = (m.get("content") or "").strip()
            if not c:
                continue
            if len(c) > int(self.cfg.facts_item_max_chars):
                c = c[: int(self.cfg.facts_item_max_chars)].rstrip() + "…"
            add = f"- {c}"
            if total + len(add) > int(self.cfg.facts_context_max_chars):
                break
            lines.append(add)
            total += len(add)
        return "\n".join(lines).strip()

    # --- Facts extraction ---
    def _extract_facts_openai(self, user_msg: str, ai_reply: str) -> List[Dict[str, Any]]:
        if not self.cfg.openai_api_key:
            return []
        prompt = (
            "你是记忆提取助手。请从下面对话中提取值得长期记住的用户信息。\n"
            "只提取对未来有帮助的事实：姓名/称呼、身份/职业、偏好、目标、重要约束、长期项目等。\n"
            "不要提取泛泛聊天、情绪化表达、临时问题细节。\n"
            "如果没有值得记忆的内容，返回空数组 []。\n\n"
            f"用户说：{user_msg}\n"
            f"助手说：{ai_reply}\n\n"
            "以 JSON 数组返回，格式：\n"
            '[{"content":"记忆内容","tags":"标签1,标签2","importance":1到5的整数}]\n'
            "只返回 JSON，不要解释。"
        )
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.cfg.openai_api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.cfg.facts_extract_model_openai,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 400,
                },
                timeout=float(self.cfg.openai_timeout_sec),
            )
            if r.status_code >= 400:
                return []
            j = r.json()
            content = (((j.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
            return _parse_json_array_best_effort(str(content))
        except Exception:
            return []

    def extract_and_save_facts(self, user_key: str, user_msg: str, ai_reply: str) -> None:
        """
        Best-effort: extract & save structured memories (facts).
        Never raises.
        """
        if not (self.cfg.facts_enabled and self.cfg.facts_extract_enabled):
            return
        um = (user_msg or "").strip()
        ar = (ai_reply or "").strip()
        if not um and not ar:
            return

        # cheap gate
        gate_text = (um + "\n" + ar).strip()
        if len(gate_text) < 24:
            return

        items: List[Dict[str, Any]] = self._extract_facts_openai(um, ar)
        if not items:
            return

        for it in items:
            try:
                c = str(it.get("content") or "").strip()
                if not c:
                    continue
                tags = str(it.get("tags") or "").strip()
                try:
                    imp = int(it.get("importance") or 1)
                except Exception:
                    imp = 1
                if imp < int(self.cfg.facts_importance_min):
                    continue
                self.facts_save(user_key, c, tags=tags, importance=imp)
            except Exception:
                continue

    def extract_and_save_facts_async(self, user_key: str, user_msg: str, ai_reply: str) -> None:
        if not (self.cfg.facts_enabled and self.cfg.facts_extract_enabled):
            return
        threading.Thread(
            target=self.extract_and_save_facts,
            args=(user_key, user_msg, ai_reply),
            daemon=True,
        ).start()

# --- Episode memory (hierarchical / episodic) ---
def episode_save(
    self,
    user_key: str,
    conversation_id: str,
    summary: str,
    *,
    start_at: float = 0.0,
    end_at: float = 0.0,
    title: str = "",
    keywords: str = "",
    source_meta: str = "",
) -> str:
    """Save one episodic summary.

    Returns episode_id (hex).
    """
    if not bool(self.cfg.episodes_enabled):
        return ""
    u = sanitize_user_key(user_key)
    cid = (conversation_id or "").strip()
    s = (summary or "").strip()
    if not s:
        return ""

    # Normalize
    t = (title or "").strip()
    if t and len(t) > int(self.cfg.episodes_title_max_chars):
        t = t[: int(self.cfg.episodes_title_max_chars)].rstrip() + "…"

    kw = re.sub(r"\s+", " ", (keywords or "").strip())
    if kw and len(kw) > 240:
        kw = kw[:240].rstrip() + "…"

    if len(s) > int(self.cfg.episodes_summary_max_chars):
        s = s[: int(self.cfg.episodes_summary_max_chars)].rstrip() + "…"

    # Embed (optional)
    emb_blob = None
    dim = 0
    if bool(self.cfg.episodes_store_embedding):
        try:
            emb = self._embed((t + "\n" + s).strip() if t else s)
            if emb:
                emb_blob = _pack_f32(emb)
                dim = int(len(emb))
        except Exception:
            emb_blob = None
            dim = 0

    now = time.time()
    eid = uuid.uuid4().hex

    with self._conn() as con:
        try:
            con.execute(
                """
                INSERT INTO memory_episodes(
                  id, user_key, conversation_id, title, summary, keywords,
                  start_at, end_at, embedding, dim,
                  source_meta, created_at, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    eid, u, cid, t, s, kw,
                    float(start_at or 0.0), float(end_at or 0.0),
                    emb_blob, int(dim or 0),
                    (source_meta or "").strip(),
                    now, now,
                ),
            )
        except Exception:
            # Fallback to legacy schema (older DB without added columns)
            con.execute(
                """
                INSERT INTO memory_episodes(id, user_key, conversation_id, summary, created_at, updated_at)
                VALUES(?,?,?,?,?,?)
                """,
                (eid, u, cid, s, now, now),
            )
    return eid

def episode_list(self, user_key: str, limit: int = 20, conversation_id: str = "") -> List[Dict[str, Any]]:
    u = sanitize_user_key(user_key)
    lim = max(1, min(int(limit or 20), 200))
    cid = (conversation_id or "").strip()

    q = (
        "SELECT id, conversation_id, title, summary, keywords, start_at, end_at, created_at, updated_at "
        "FROM memory_episodes WHERE user_key=? "
    )
    args: List[Any] = [u]
    if cid:
        q += "AND conversation_id=? "
        args.append(cid)
    q += "ORDER BY updated_at DESC LIMIT ?"
    args.append(lim)

    with self._conn() as con:
        try:
            cur = con.execute(q, tuple(args))
            rows = cur.fetchall()
        except Exception:
            # Legacy schema
            cur = con.execute(
                "SELECT id, conversation_id, '' as title, summary, '' as keywords, 0 as start_at, 0 as end_at, created_at, updated_at "
                "FROM memory_episodes WHERE user_key=? ORDER BY updated_at DESC LIMIT ?",
                (u, lim),
            )
            rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for rid, cid2, title, summ, kw, sa, ea, ca, ua in rows:
        out.append(
            {
                "id": str(rid),
                "conversation_id": str(cid2 or ""),
                "title": str(title or ""),
                "summary": str(summ or ""),
                "keywords": str(kw or ""),
                "start_at": float(sa or 0.0),
                "end_at": float(ea or 0.0),
                "created_at": float(ca or 0.0),
                "updated_at": float(ua or 0.0),
            }
        )
    return out

def episode_last_end_at(self, user_key: str, conversation_id: str) -> float:
    """Return the last covered end_at for a conversation (0.0 if none)."""
    u = sanitize_user_key(user_key)
    cid = (conversation_id or "").strip()
    if not cid:
        return 0.0
    with self._conn() as con:
        # Prefer end_at column when present
        try:
            cur = con.execute(
                "SELECT MAX(end_at) FROM memory_episodes WHERE user_key=? AND conversation_id=? AND end_at>0",
                (u, cid),
            )
            v = (cur.fetchone() or [0])[0]
            if v:
                return float(v)
        except Exception:
            pass
        # Fallback to updated_at (legacy)
        try:
            cur = con.execute(
                "SELECT MAX(updated_at) FROM memory_episodes WHERE user_key=? AND conversation_id=?",
                (u, cid),
            )
            v2 = (cur.fetchone() or [0])[0]
            return float(v2 or 0.0)
        except Exception:
            return 0.0

def episode_search(
    self,
    user_key: str,
    query: str,
    *,
    conversation_id: str = "",
    k: Optional[int] = None,
    min_score: Optional[float] = None,
    candidate_limit: int = 400,
) -> List[Dict[str, Any]]:
    if not bool(self.cfg.episodes_enabled):
        return []
    u = sanitize_user_key(user_key)
    q = (query or "").strip()
    if not q:
        return []
    cid = (conversation_id or "").strip()
    k = max(1, int(k or self.cfg.episodes_top_k_default))
    min_score = float(min_score if min_score is not None else self.cfg.episodes_min_score_default)

    qv_list = self._embed(q)
    if not qv_list:
        return []
    qv = array("f", [float(x) for x in qv_list])

    # Load recent episodes (per conversation if provided)
    with self._conn() as con:
        try:
            if cid:
                cur = con.execute(
                    "SELECT id, title, summary, keywords, start_at, end_at, embedding, dim, updated_at "
                    "FROM memory_episodes WHERE user_key=? AND conversation_id=? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (u, cid, int(candidate_limit)),
                )
            else:
                cur = con.execute(
                    "SELECT id, title, summary, keywords, start_at, end_at, embedding, dim, updated_at "
                    "FROM memory_episodes WHERE user_key=? ORDER BY updated_at DESC LIMIT ?",
                    (u, int(candidate_limit)),
                )
            rows = cur.fetchall()
        except Exception:
            # Legacy schema: no embedding, no title/keywords/start/end
            if cid:
                cur = con.execute(
                    "SELECT id, '' as title, summary, '' as keywords, 0 as start_at, 0 as end_at, NULL as embedding, 0 as dim, updated_at "
                    "FROM memory_episodes WHERE user_key=? AND conversation_id=? ORDER BY updated_at DESC LIMIT ?",
                    (u, cid, int(candidate_limit)),
                )
            else:
                cur = con.execute(
                    "SELECT id, '' as title, summary, '' as keywords, 0 as start_at, 0 as end_at, NULL as embedding, 0 as dim, updated_at "
                    "FROM memory_episodes WHERE user_key=? ORDER BY updated_at DESC LIMIT ?",
                    (u, int(candidate_limit)),
                )
            rows = cur.fetchall()

    scored: List[Tuple[float, Dict[str, Any]]] = []
    to_backfill: List[Tuple[str, bytes, int]] = []

    for rid, title, summ, kw, sa, ea, blob, dim, ua in rows:
        title_s = str(title or "")
        summ_s = str(summ or "")
        if not summ_s:
            continue

        try:
            # If embedding missing, compute once and (optionally) backfill.
            v = None
            if isinstance(blob, (bytes, bytearray)) and int(dim or 0) > 0:
                v = _unpack_f32(blob)
                if len(v) != int(dim or 0):
                    v = None
            if v is None and bool(self.cfg.episodes_store_embedding):
                emb = self._embed((title_s + "\n" + summ_s).strip() if title_s else summ_s)
                if emb:
                    blob2 = _pack_f32(emb)
                    dim2 = int(len(emb))
                    v = _unpack_f32(blob2)
                    to_backfill.append((str(rid), blob2, dim2))
            if v is None:
                continue

            s = _cosine(qv, v)
            if s >= min_score:
                scored.append(
                    (
                        float(s),
                        {
                            "id": str(rid),
                            "title": title_s,
                            "summary": summ_s,
                            "keywords": str(kw or ""),
                            "start_at": float(sa or 0.0),
                            "end_at": float(ea or 0.0),
                            "updated_at": float(ua or 0.0),
                        },
                    )
                )
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    # Best-effort embedding backfill
    if to_backfill:
        try:
            with self._conn() as con:
                for rid, blob2, dim2 in to_backfill[:50]:
                    con.execute(
                        "UPDATE memory_episodes SET embedding=?, dim=? WHERE id=? AND user_key=?",
                        (blob2, int(dim2), rid, u),
                    )
        except Exception:
            pass

    return [{"score": s, **d} for (s, d) in top]

def episodes_build_prompt(
    self,
    user_key: str,
    conversation_id: str,
    query: str,
    *,
    limit: Optional[int] = None,
    k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> str:
    if not bool(self.cfg.episodes_enabled):
        return ""
    lim = max(1, min(int(limit or self.cfg.episodes_prompt_limit), 50))
    hits = self.episode_search(
        user_key,
        query,
        conversation_id=(conversation_id or "").strip(),
        k=k or lim,
        min_score=min_score,
    )
    if not hits:
        return ""

    lines: List[str] = ["以下是【历史摘要】（对话分段压缩；需要更细节可触发“回放”）："]
    total = 0
    for h in hits:
        summ = (h.get("summary") or "").strip()
        if not summ:
            continue
        title = (h.get("title") or "").strip()
        sa = float(h.get("start_at") or 0.0)
        ea = float(h.get("end_at") or 0.0)
        when = ""
        try:
            if sa > 0:
                when = time.strftime("%Y-%m-%d", time.localtime(sa))
        except Exception:
            when = ""

        head = ""
        if when and title:
            head = f"{when}｜{title}"
        elif when:
            head = when
        elif title:
            head = title

        text_line = f"- {head + '：' if head else ''}{summ}"
        if len(text_line) > int(self.cfg.episodes_item_max_chars):
            text_line = text_line[: int(self.cfg.episodes_item_max_chars)].rstrip() + "…"
        if total + len(text_line) > int(self.cfg.episodes_context_max_chars):
            break
        lines.append(text_line)
        total += len(text_line)

    return "\n".join(lines).strip()

# --- Episode summarization helpers (used by server-side compaction) ---
def episode_summarize_openai(self, transcript: str, *, model: Optional[str] = None) -> Dict[str, Any]:
    if not self.cfg.openai_api_key:
        return {}
    txt = (transcript or "").strip()
    if not txt:
        return {}
    model = (model or self.cfg.episodes_summarize_model_openai or "gpt-4o-mini").strip()

    prompt = (
        "你是一个“对话压缩器”。请把下面的对话片段压缩为一个可长期存储的“情节记忆”。\n"
        "要求：\n"
        "1) title：<= 40字，概括主题\n"
        "2) summary：<= 360字，中文，包含关键信息（目标/决定/约束/结论/待办）\n"
        "3) keywords：3-8个关键词（用逗号分隔）\n"
        "只返回 JSON 对象，不要解释。\n\n"
        f"对话片段：\n{txt}\n"
    )

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.openai_api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500,
            },
            timeout=float(self.cfg.openai_timeout_sec),
        )
        if r.status_code >= 400:
            return {}
        j = r.json()
        content = (((j.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        obj = _parse_json_object_best_effort(str(content))
        if not obj:
            return {}

        title = str(obj.get("title") or "").strip()
        summary = str(obj.get("summary") or "").strip()
        keywords = obj.get("keywords")
        if isinstance(keywords, list):
            keywords = ",".join([str(x).strip() for x in keywords if str(x).strip()])
        keywords_s = str(keywords or "").strip()

        # Apply caps
        if title and len(title) > int(self.cfg.episodes_title_max_chars):
            title = title[: int(self.cfg.episodes_title_max_chars)].rstrip() + "…"
        if summary and len(summary) > int(self.cfg.episodes_summary_max_chars):
            summary = summary[: int(self.cfg.episodes_summary_max_chars)].rstrip() + "…"

        if keywords_s:
            # normalize separators
            keywords_s = re.sub(r"[，;；\s]+", ",", keywords_s).strip(", ")
            parts = [p.strip() for p in keywords_s.split(",") if p.strip()]
            parts = parts[: int(self.cfg.episodes_keywords_max)]
            keywords_s = ",".join(parts)

        out = {"title": title, "summary": summary, "keywords": keywords_s}
        if not out["summary"]:
            return {}
        return out
    except Exception:
        return {}

def episode_summarize(self, transcript: str, *, model: Optional[str] = None) -> Dict[str, Any]:
    """Summarize transcript -> {title, summary, keywords}. Best-effort."""
    if bool(self.cfg.episodes_summarize_enabled):
        obj = self.episode_summarize_openai(transcript, model=model)
        if obj:
            return obj

    # Fallback: no network
    t = re.sub(r"\s+", " ", (transcript or "").strip())
    if not t:
        return {}
    return {
        "title": "",
        "summary": (t[: int(self.cfg.episodes_summary_max_chars)].rstrip() + "…") if len(t) > int(self.cfg.episodes_summary_max_chars) else t,
        "keywords": "",
    }

# -----------------------------
# Heuristics (can be shared)
# -----------------------------

_MEMORY_WORTHY_PATTERNS = [
    r"\b我叫\b",
    r"\b我的名字\b",
    r"\b叫我\b",
    r"\b我是\b",
    r"\b记住\b",
    r"\b以后\b",
    r"\b偏好\b",
    r"\b喜欢\b",
    r"\b不喜欢\b",
    r"\b生日\b",
    r"\b住在\b",
    r"\b来自\b",
    r"\b公司\b",
    r"\b工作\b",
    r"\bmy name is\b",
    r"\bcall me\b",
    r"\bremember\b",
    r"\bi like\b",
    r"\bi don't like\b",
]
_memory_worthy_re = re.compile("|".join(_MEMORY_WORTHY_PATTERNS), re.IGNORECASE)


def should_store_memory(text: str, min_chars: int = 8) -> bool:
    """
    Conservative heuristic: only store durable, user-relevant stuff.
    """
    t = (text or "").strip()
    if not t:
        return False
    if "```" in t:
        return False
    if len(t) < int(min_chars):
        return False
    if t.lower() in ("hi", "hello", "ok", "okay") or t in ("你好", "在吗", "嗯", "好的", "收到"):
        return False
    if _memory_worthy_re.search(t):
        return True
    if len(t) <= 220 and ("我" in t or "my " in t.lower() or t.lower().startswith("i ")):
        return True
    return False

