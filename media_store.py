# media_store.py
import os, time, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "./_media")).resolve()
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

MEDIA_TTL_S = int(os.getenv("MEDIA_TTL_S", "7200"))

@dataclass
class MediaMeta:
    id: str
    path: Path
    mime: str
    created_at: float

class MediaStore:
    def __init__(self):
        self._db: Dict[str, MediaMeta] = {}

    def put_bytes(self, data: bytes, mime: str, ext: str) -> MediaMeta:
        mid = f"media_{uuid.uuid4().hex[:18]}"
        p = MEDIA_DIR / f"{mid}.{ext}"
        p.write_bytes(data)
        meta = MediaMeta(id=mid, path=p, mime=mime, created_at=time.time())
        self._db[mid] = meta
        return meta

    def get(self, mid: str) -> Optional[MediaMeta]:
        meta = self._db.get(mid)
        if not meta:
            return None
        if time.time() - meta.created_at > MEDIA_TTL_S:
            try:
                meta.path.unlink(missing_ok=True)
            except Exception:
                pass
            self._db.pop(mid, None)
            return None
        return meta

store = MediaStore()
