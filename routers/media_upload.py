# routers/media_upload.py
# ✅ Media upload router (image/audio) backed by in-memory MediaStore.
#    - POST /upload/image  (multipart form-data: file)
#    - POST /upload/audio  (multipart form-data: file)
#    - GET  /media/{media_id}
#
# Notes:
# - This router MUST be included in FastAPI app: app.include_router(media_upload.router)
# - media_store.py should live at backend root (same level as server_session.py) and expose `store`.

import mimetypes
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from media_store import store as media_store

router = APIRouter()


class UploadedMedia(BaseModel):
    id: str
    mime: str
    url: Optional[str] = None


def _guess_ext(filename: Optional[str], mime: Optional[str], default_ext: str) -> str:
    """Return extension WITH a leading dot, e.g. ".jpg"."""
    # Prefer original filename ext
    if filename and "." in filename:
        ext = "." + filename.split(".")[-1].strip().lower()
        if 1 < len(ext) <= 8:
            return ext

    # Use mimetypes mapping
    if mime:
        ext = mimetypes.guess_extension(mime.split(";")[0].strip())
        if ext:
            return ext

    return default_ext


def _ext_without_dot(ext: str) -> str:
    return ext[1:] if ext.startswith(".") else ext


@router.post("/upload/image", response_model=UploadedMedia)
async def upload_image(request: Request, file: UploadFile = File(...)):
    ctype = (file.content_type or "").strip().lower()
    if not ctype.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"expected image/*, got {ctype or 'unknown'}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    ext = _guess_ext(file.filename, ctype, default_ext=".jpg")
    meta = media_store.put_bytes(data=data, mime=ctype, ext=_ext_without_dot(ext))

    base = str(request.base_url).rstrip("/")
    return UploadedMedia(id=meta.id, mime=meta.mime, url=f"{base}/media/{meta.id}")


@router.post("/upload/audio", response_model=UploadedMedia)
async def upload_audio(request: Request, file: UploadFile = File(...)):
    ctype = (file.content_type or "").strip().lower()

    # iOS m4a often comes as audio/mp4; some clients send application/octet-stream.
    ok = (
        ctype.startswith("audio/")
        or ctype in ("video/mp4", "application/octet-stream")
        or "m4a" in (file.filename or "").lower()
    )
    if not ok:
        raise HTTPException(status_code=400, detail=f"expected audio/*, got {ctype or 'unknown'}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    ext = _guess_ext(file.filename, ctype, default_ext=".m4a")
    # If content-type is unknown, normalize to audio/mp4 for m4a
    mime = ctype if ctype and ctype != "application/octet-stream" else "audio/mp4"
    meta = media_store.put_bytes(data=data, mime=mime, ext=_ext_without_dot(ext))

    base = str(request.base_url).rstrip("/")
    return UploadedMedia(id=meta.id, mime=meta.mime, url=f"{base}/media/{meta.id}")


@router.get("/media/{media_id}")
async def get_media(media_id: str):
    meta = media_store.get(media_id)
    if not meta:
        raise HTTPException(status_code=404, detail="media not found or expired")
    return FileResponse(path=str(meta.path), media_type=meta.mime, filename=meta.path.name)

