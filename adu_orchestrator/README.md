# adu_orchestrator — minimal multi-tool dispatcher

A thin "orchestrator" layer that takes one task string and routes it to one
of three executors. V1 wires only what we already have on this Mac;
Codex CLI, Claude Code, and browser-use are adapter stubs.

## Layout

```
~/Desktop/backend/adu_orchestrator/
├── __init__.py
├── dispatcher.py         run_task(task, mode, project_dir, safety_level)
├── executors/
│   ├── __init__.py
│   ├── code.py           wraps adu_code_agent (search/list/read); Codex/Claude = stubs
│   ├── browser.py        stub
│   └── desktop.py        calls local-agent :4317
├── routes.py             FastAPI APIRouter at /adu/orchestrator
└── README.md
```

## HTTP endpoint

```
POST /api/adu/orchestrator/run
GET  /api/adu/orchestrator/info
```

Request:
```json
{
  "task": "search gpt-realtime-2 in /Users/.../little-beijing-edge-box",
  "mode": "code",
  "project_dir": "/Users/.../little-beijing-edge-box",
  "safety_level": "normal"
}
```

Response:
```json
{
  "plan": ["adu_code_agent.search_text(...)"],
  "actions": [
    {"tool": "adu_code_agent.search_text", "ok": true, "match_count": 2}
  ],
  "result": "2 match(es) for 'gpt-realtime-2' in ...",
  "needs_user_confirmation": false,
  "mode": "code",
  "engine": "adu_code_agent",
  "task": "...",
  "safety_level": "normal",
  "raw": { ... }
}
```

## Wire into the running ChatAGI 8000 backend

In `~/Desktop/backend/server_session.py`:

```python
from adu_orchestrator.routes import router as orch_router
app.include_router(orch_router, prefix="/api")
```

Then restart 8000. Endpoints are then available at
`http://127.0.0.1:8000/api/adu/orchestrator/{info,run}`.

## What V1 actually does

| Mode | Engine wired in V1 | Handles today | Next step |
|------|--------------------|---------------|-----------|
| code | adu_code_agent | `search X [in PATH]`, `list PATH`, `read PATH[:start-end]` | wire Codex CLI / Claude Code for free-form tasks |
| browser | (stub) | returns plan-only `needs_user_confirmation=true` | wire browser-use or agent-browser |
| desktop | local-agent :4317 | screenshot / mouse_position / active_window / click / double_click / type / paste / press | richer multi-step via `/api/brain/adu/act` |
| auto | (router) | classifies by keywords → code by default | replace with LLM classifier if needed |

Every response is shaped as:

```ts
{
  plan: string[];                  // declared steps
  actions: {tool, ok, ...}[];      // what actually ran
  result: string;                  // human-readable summary
  needs_user_confirmation: bool;   // true ⇒ pause and ask
  mode, task, safety_level;        // echo-back for the caller
  engine?: string;                 // which executor took it
  raw?: any;                       // underlying tool result
}
```

## Continuing — wiring Codex CLI

[Codex CLI](https://github.com/openai/codex) is OpenAI's command-line
agent. To plug it in:

1. Install on the box: `npm i -g @openai/codex` (or whatever the current
   package name is on the day you do this).
2. In `executors/code.py`, add a helper:
   ```python
   import subprocess
   def _codex_cli_run(prompt: str, cwd: str | None) -> dict:
       p = subprocess.run(
           ["codex", "--print", prompt],          # check current flag name
           cwd=cwd, capture_output=True, text=True, timeout=180,
       )
       return {"ok": p.returncode == 0,
               "stdout": p.stdout[-4000:],
               "stderr": p.stderr[-1000:]}
   ```
3. Have `run(task, ...)` call it AFTER `_local_handle(...)` returns
   None — replace the current "plan-only fallback" branch.
4. Keep `OPENAI_API_KEY` in env only; never echo it.

## Continuing — wiring Claude Code

[Claude Code](https://docs.claude.com/en/docs/claude-code) ships a CLI
named `claude`. Pattern:

```python
subprocess.run(["claude", "-p", task, "--permission-mode", "plan",
                "--allowed-tools", "Read,Grep,Glob",
                "--max-turns", "8"],
               cwd=project_dir, capture_output=True, text=True, timeout=180)
```

The existing `~/Desktop/backend/local_agent_4317.py` already exposes a
`/dev_agent/ask` endpoint that does exactly this in read-only `--permission-mode plan`. The simplest wiring is to POST to that instead
of shelling out from this process:

```python
import httpx
r = httpx.post("http://127.0.0.1:4317/dev_agent/ask",
               headers={"Authorization": "Bearer local-dev-token"},
               json={"prompt": task, "project_root": project_dir,
                     "timeout_ms": 90_000, "read_only": True},
               timeout=120)
```

## Continuing — wiring browser-use

[browser-use](https://github.com/browser-use/browser-use) drives a real
Playwright Chromium with an LLM:

1. `pip install browser-use playwright langchain-openai`
2. `playwright install chromium`
3. Replace `executors/browser.py:run` with:
   ```python
   import asyncio
   from browser_use import Agent
   from langchain_openai import ChatOpenAI
   def run(task, project_dir=None, safety_level="normal"):
       async def _go():
           agent = Agent(task=task, llm=ChatOpenAI(model="gpt-4o-mini"))
           return await agent.run()
       result = asyncio.run(_go())
       return {"plan": [task], "actions": [...], "result": str(result),
               "needs_user_confirmation": False, "engine": "browser_use"}
   ```
4. Hard-gate on `safety_level`: when "strict", refuse to submit
   forms / payments / messages — return `needs_user_confirmation=true`.

## Safety notes for V1

- `code` executor is **read-only** (search / list / read). It never writes.
- `browser` executor is a stub — no real network requests.
- `desktop` executor goes to local-agent, which has its own auth (Bearer
  `local-dev-token`) and is bound to `127.0.0.1`.
- `safety_level` is currently recorded but not enforced; "strict" is
  where confirmation gates will plug in once browser/codex executors are wired.
- `needs_user_confirmation=true` whenever the executor can't finish the
  task on its own — the UI should pause and ask.

## Smoke test (no server needed)

```bash
cd ~/Desktop/backend
python3 -c "
from adu_orchestrator import dispatcher
import json
print(json.dumps(dispatcher.run_task(
    'search gpt-realtime-2 in /Users/a12345/Desktop/little-beijing-edge-box',
    mode='code'), indent=2, ensure_ascii=False)[:600])
"
```
