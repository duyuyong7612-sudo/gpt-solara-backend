from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .executor import LocalComputerAgent
from .registry import get_registry

router = APIRouter(prefix='/computer-agent', tags=['computer-agent'])
_agent = LocalComputerAgent()


class ExecuteRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    target: str = Field('', min_length=0)


@router.get('/health')
async def health():
    reg = get_registry()
    return {
        'ok': True,
        'source': 'local',
        'platform': reg.data.get('system', {}).get('platform'),
        'apps_count': len(reg.data.get('apps') or {}),
        'workspace_aliases_count': len(reg.data.get('workspace_aliases') or {}),
    }


@router.post('/probe')
async def probe():
    reg = get_registry()
    data = reg.probe_and_persist()
    return {'ok': True, 'source': 'local', 'registry': data}

@router.get('/registry')
async def registry():
    reg = get_registry()
    return {'ok': True, 'source': 'local', 'registry': reg.data}


@router.post('/execute')
async def execute(body: ExecuteRequest):
    try:
        result = await _agent.execute(body.goal, body.target)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
