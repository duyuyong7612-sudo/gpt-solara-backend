"""adu_orchestrator — minimal multi-tool dispatcher (code / browser / desktop).

V1 wires the local tools we already have (adu_code_agent, local-agent :4317).
Codex CLI, Claude Code, browser-use are adapter stubs — see README.md.
"""
__version__ = "0.1.0"

from . import dispatcher  # noqa: F401

__all__ = ["dispatcher", "__version__"]
