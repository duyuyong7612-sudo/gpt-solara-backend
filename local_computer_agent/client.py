from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class LocalComputerAgentClient:
    """
    调用本地电脑总代理的轻量客户端。

    默认假设你的后端和电脑总代理都挂在同一个 FastAPI 服务下：
      http://127.0.0.1:8000/computer-agent/...

    用法：
        client = LocalComputerAgentClient()
        data = await client.health()
        data = await client.probe()
        data = await client.execute("列出目录", "backend", async_mode=False)
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def health(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(self._url("/computer-agent/health"))
            r.raise_for_status()
            return r.json()

    async def probe(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(self._url("/computer-agent/probe"))
            r.raise_for_status()
            return r.json()

    async def registry(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(self._url("/computer-agent/registry"))
            r.raise_for_status()
            return r.json()

    async def execute(self, goal: str, target: str = "", async_mode: bool = True) -> Dict[str, Any]:
        payload = {
            "goal": goal,
            "target": target,
            "async_mode": async_mode,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(self._url("/computer-agent/execute"), json=payload)
            r.raise_for_status()
            return r.json()

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(self._url(f"/computer-agent/task/{task_id}"))
            r.raise_for_status()
            return r.json()

    async def list_tasks(self, limit: int = 20) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(self._url("/computer-agent/tasks"), params={"limit": limit})
            r.raise_for_status()
            return r.json()

    async def abort(self, task_id: str, reason: str = "aborted_by_user") -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                self._url(f"/computer-agent/abort/{task_id}"),
                json={"reason": reason},
            )
            r.raise_for_status()
            return r.json()

    async def get_file_meta(self, path: str) -> Dict[str, Any]:
        """
        这里只返回可下载 URL，不直接下载文件。
        真要下载就自己再 GET /computer-agent/file?path=...
        """
        return {
            "ok": True,
            "url": self._url("/computer-agent/file"),
            "path": path,
        }c