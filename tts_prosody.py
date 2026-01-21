"""TTS prosody helpers for Solara backend.

Drop-in module to provide:
- Default speed = 1.2 (env: TTS_SPEED)
- Adaptive speed per sentence length (env: TTS_DYNAMIC_SPEED=1)
- Emotion + prosody instructions generation per segment

Designed for OpenAI /v1/audio/speech payload:
  { model, voice, input, instructions?, speed?, response_format? }
"""

from __future__ import annotations

import os
import re
from typing import Optional

# -----------------------------
# Defaults (override with env)
# -----------------------------

TTS_SPEED_DEFAULT: float = float(os.getenv("TTS_SPEED") or "1.2")

TTS_DYNAMIC_SPEED: bool = (os.getenv("TTS_DYNAMIC_SPEED") or "1").strip().lower() in (
    "1", "true", "yes", "on"
)

# Keep this short-ish; model-side instructions are not meant to be a full prompt.
TTS_STYLE_BASE: str = (
    os.getenv("TTS_STYLE_BASE")
    or "Speak in a clear, friendly tone with natural rhythm."
).strip()

# -----------------------------
# Helpers
# -----------------------------

def clamp_speed(v: float) -> float:
    try:
        f = float(v)
    except Exception:
        f = TTS_SPEED_DEFAULT
    if f < 0.25:
        return 0.25
    if f > 4.0:
        return 4.0
    return f


def pick_speed(text: str, base: float = TTS_SPEED_DEFAULT) -> float:
    """Adaptive speed:
    - Base defaults to 1.2
    - Short segments: slightly slower (more expressive)
    - Long segments: slightly faster (keep pace)
    """
    t = (text or "").strip()
    n = len(t)

    # Length-based
    if n <= 10:
        delta = -0.12
    elif n <= 22:
        delta = -0.05
    elif n <= 48:
        delta = 0.0
    elif n <= 90:
        delta = 0.06
    else:
        delta = 0.12

    # Many comma-like pauses -> slightly slower
    pauses = sum(t.count(ch) for ch in [",", "，", ";", "；", ":", "："])
    if pauses >= 4 and n >= 40:
        delta -= 0.05

    # Emotion hints
    if re.search(r"[！？!]", t):
        delta += 0.03
    if re.search(r"[？?]", t):
        delta -= 0.02

    return clamp_speed(base + delta)


def build_instructions(text: str, base: Optional[str] = None) -> str:
    """Generate compact emotional / prosody instructions."""
    b = (base or TTS_STYLE_BASE or "").strip()
    t = (text or "").strip()

    if re.search(r"[！？!]", t):
        style = "Sound lively and enthusiastic with brighter intonation."
    elif re.search(r"[？?]", t):
        style = "Sound curious and helpful with a gentle rising intonation."
    elif re.search(r"(抱歉|对不起|遗憾|难过|我理解|理解你|辛苦了|谢谢|感谢)", t):
        style = "Sound gentle, empathetic, and reassuring."
    else:
        style = "Sound warm, expressive, and natural; avoid monotone."

    prosody = (
        "Use natural pauses at commas and sentence ends. "
        "Lightly emphasize key words. Keep pronunciation crisp."
    )

    out = f"{b} {style} {prosody}"
    out = re.sub(r"\s+", " ", out).strip()

    # Keep instructions reasonably short
    return out[:360]
