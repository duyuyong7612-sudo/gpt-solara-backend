# remix_profile.py
# ✅ One Source of Truth for Remix Assistant (Realtime instructions + markers + prompt normalization)
# ✅ 修复点：
#   1) 语言跟随使用者：REMIX: 后面的 prompt 也跟随用户语言；送入 Remix API 的 prompt 也跟随用户语言
#   2) “客户已确定”不是用户口令，而是助手在语义确认后自动附加的内部门禁标记（##CONFIRM=客户已确定）
#   3) 仍保留强防误触发：未语义确认“现在执行”前禁止输出 REMIX 触发行
import re
from typing import Optional, Dict


REMIX_BEGIN = "[REMIX_REQUEST]"
REMIX_END = "[/REMIX_REQUEST]"

# =========================
# Protocol constants (统一前后端协议)
# =========================
TRIGGER_PREFIX = "REMIX:"
CONFIRM_PHRASE = "客户已确定"  # ✅ INTERNAL protocol mark appended by assistant on trigger line; user never needs to say it
CONFIRM_MARK = f"##CONFIRM={CONFIRM_PHRASE}"
CODE_MARK_PREFIX = "##CODE="

# 可选：更严格确认（你后续加 4 位确认码时用）
_CONFIRM_CODE_RE = re.compile(r"([0-9]{4})")
_PROTOCOL_TAG_RE = re.compile(r"(##CONFIRM=.*$|##CODE=.*$)", flags=re.I)

# =========================
# Language detection + stability anchors
# =========================

def detect_user_language(text: str) -> str:
    """
    粗略检测用户语言（够用、稳定、不引入依赖）：
    - 包含中文 => zh
    - 包含日文假名 => ja
    - 包含韩文 => ko
    - 否则 => en
    """
    s = (text or "").strip()
    if not s:
        return "en"

    if re.search(r"[\u4e00-\u9fff]", s):
        return "zh"
    if re.search(r"[\u3040-\u30ff]", s):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", s):
        return "ko"
    return "en"


# 多语言“稳定锚点”模板：保证不漂移成新视频
# 你要求“送入模型文本也是使用者的语言”，所以这里锚点也用对应语言
STABLE_ANCHORS = {
    "en": {
        "must_keep": "Keep the same scene, subject, background, composition, and camera motion as the original video.",
        "only_change": "Only change:",
        "do_not_change": "Do not change:",
        "dnc_default": "subject identity, background/layout, composition/framing, or camera path/motion.",
        "punct": ".",
        "fallback_change": "Make subtle cinematic improvements with minimal change.",
    },
    "zh": {
        "must_keep": "保持原视频的场景、主体、背景、构图与镜头运动完全一致。",
        "only_change": "只改动：",
        "do_not_change": "不要改动：",
        "dnc_default": "主体身份、背景/布局、构图/取景、镜头路径/运动。",
        "punct": "。",
        "fallback_change": "做轻微的电影质感提升，尽量不改变原有内容。",
    },
    "ja": {
        "must_keep": "元の動画と同じシーン、被写体、背景、構図、カメラの動きを維持する。",
        "only_change": "変更するのは：",
        "do_not_change": "変更しないで：",
        "dnc_default": "被写体の同一性、背景/レイアウト、構図/フレーミング、カメラ経路/動き。",
        "punct": "。",
        "fallback_change": "最小限の変更で、映画的な質感を少しだけ向上させる。",
    },
    "ko": {
        "must_keep": "원본 영상과 동일한 장면, 피사체, 배경, 구도, 카메라 움직임을 그대로 유지하세요.",
        "only_change": "바꿀 것:",
        "do_not_change": "바꾸지 말 것:",
        "dnc_default": "피사체 정체성, 배경/레이아웃, 구도/프레이밍, 카메라 경로/움직임.",
        "punct": ".",
        "fallback_change": "최소한의 변경으로 영화적 질감을 약간 향상하세요.",
    },
}


def _get_anchors(lang: str) -> Dict[str, str]:
    lang = (lang or "").strip().lower()
    return STABLE_ANCHORS.get(lang, STABLE_ANCHORS["en"])


# =========================
# ✅ Realtime: Remix 专用高维定义（唯一真源）
# =========================
REMIX_REALTIME_INSTRUCTIONS = r"""
You are Solara Remix Assistant (single responsibility: video remix/edit).

IDENTITY
- You ONLY do remix, always based on a base_video_id (the original OpenAI video).
- You are NOT generating from scratch. You must preserve the original video's:
  subject identity, background/location, composition, camera path/motion, main actions and timing.
- Only change what the user explicitly requests.

CORE TASK
1) Briefly clarify what to change (ask at most 1-2 short questions only if unclear).
2) Propose 2-4 clear remix options (each: what changes + expected result + risk/uncertainty).
3) Produce a stable remix prompt IN THE USER'S LANGUAGE with:
   - KEEP: what must stay the same
   - CHANGE: what to change
   - DO NOT: what must not change
4) Avoid drifting into a totally different new video. If user wants to change subject/location/background,
   warn that it becomes closer to re-creating a new video.

LANGUAGE (must follow the user)
- ALWAYS chat in the user's language (match the user's latest message language).
- The final remix prompt after REMIX: must also be in the user's language.
- Do NOT translate unless the user asks.

SEMANTIC FINAL CONFIRM (anti-early-trigger, strict)
- "客户已确定" is an INTERNAL protocol mark that YOU (assistant) append to the trigger line.
  The user does NOT need to say it.
- You MUST NOT output the trigger line until you are semantically sure the user has issued a FINAL "generate/execute now" command.

Treat as CONFIRMED only if the user intent is unmistakably "execute now", for example:
- Direct go-ahead: "start generating", "generate now", "just do it", "OK do it",
  "就按这个生成", "现在生成", "开始二次生成", "直接做吧", "就这样生成"
- Or the user clearly accepts your last draft without requesting any further change (e.g., "OK", "就这样", "可以了")
  AND context shows they are done.

NOT CONFIRMED if the user is still:
- asking questions, requesting revisions, comparing options, hesitating ("maybe", "wait", "hold on", "not yet"), or adding new constraints.

If unclear, ask a short yes/no question like:
- "要我现在就开始二次生成吗？"
(Do NOT ask the user to say any special phrase.)

OUTPUT PROTOCOL (machine-stable, strict)
A) First: output a short confirmation draft in user language (KEEP / CHANGE / DO NOT, 3-6 lines).
B) Only after semantic confirmation, output ONE single standalone trigger line:
   REMIX: <ONE single-line prompt in the user's language> ##CONFIRM=客户已确定
- The trigger line MUST be standalone and start with "REMIX:".
- Output the trigger line only once per task.
- Do NOT include any extra text in the same message as the trigger line.
- Optional: if the user provides a 4-digit confirmation code in their final go-ahead (e.g., "ok 4821"),
  append: "##CODE=4821".

STABILITY RULES (must include stable anchors in user's language)
- The final prompt must include an explicit KEEP anchor, plus "Only change / Do not change" equivalents in the user's language.
- Do NOT mention API/models/backend.

QUESTION STYLE
- Ask only if missing: style/color/lighting/speed/effects, and strength (subtle/strong).
- Do NOT ask about duration or technical details. Do NOT mention API/models/backend.
""".strip()


# =========================
# Helpers
# =========================
def normalize_video_id(vid: str) -> str:
    s = (vid or "").strip()
    m = re.search(r"(video_[A-Za-z0-9]+)", s)
    return m.group(1) if m else s


def _extract_block(raw: str) -> str:
    s = raw or ""
    if REMIX_BEGIN not in s:
        return ""
    if REMIX_END in s:
        try:
            return s.split(REMIX_BEGIN, 1)[1].split(REMIX_END, 1)[0]
        except Exception:
            return ""
    try:
        return s.split(REMIX_BEGIN, 1)[1]
    except Exception:
        return ""


def _strip_protocol_tags(text: str) -> str:
    """
    把触发行里的协议字段剥掉，避免污染送入 Remix 的 prompt：
      - 去掉 'REMIX:' 前缀（如果存在）
      - 去掉 '##CONFIRM=...' / '##CODE=...' 及其后内容
      - 单行化
    """
    s = (text or "").strip()
    if not s:
        return ""

    if TRIGGER_PREFIX in s:
        s = s.split(TRIGGER_PREFIX, 1)[1].strip()

    if "##CONFIRM=" in s:
        s = s.split("##CONFIRM=", 1)[0].strip()
    if "##CODE=" in s:
        s = s.split("##CODE=", 1)[0].strip()

    # 兜底：如果 tag 写法异常，用正则截断
    s = _PROTOCOL_TAG_RE.sub("", s).strip()

    # 单行化
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ensure_stable_remix_prompt(core: str, lang: str) -> str:
    """
    强制补齐三句稳定锚点（按用户语言），避免漂移成新视频。
    - 必须包含 must_keep
    - 必须包含 only_change / do_not_change（或其用户语言等价表述）
    """
    a = _get_anchors(lang)
    must_keep = a["must_keep"]
    only_change_phrase = a["only_change"]
    do_not_change_phrase = a["do_not_change"]
    dnc_default = a["dnc_default"]
    punct = a["punct"]
    fallback_change = a["fallback_change"]

    c = (core or "").strip()
    c = re.sub(r"\s+", " ", c).strip()

    if not c:
        c = fallback_change

    # 检测是否已包含锚点（按当前语言；英文不区分大小写）
    if lang == "en":
        low = c.lower()
        has_keep = must_keep.lower() in low
        has_only = only_change_phrase.lower() in low
        has_dnc = do_not_change_phrase.lower() in low
    else:
        has_keep = must_keep in c
        has_only = only_change_phrase in c
        has_dnc = do_not_change_phrase in c

    if not (has_keep and has_only and has_dnc):
        change_payload = c
        c = (
            f"{must_keep} "
            f"{only_change_phrase} {change_payload}{punct} "
            f"{do_not_change_phrase} {dnc_default}"
        )

    c = re.sub(r"\s+", " ", c).strip()
    return c


def parse_remix_request(raw_prompt: str) -> Optional[Dict[str, str]]:
    """
    支持 iOS 发送的结构：
    [REMIX_REQUEST]
    base_video_id: video_xxx
    user_instruction: ...
    assistant_rewrite: REMIX: ... ##CONFIRM=客户已确定   (可选，但强烈推荐)
    user_language: zh                               (可选：建议显式传，避免误判)
    [/REMIX_REQUEST]
    """
    s = (raw_prompt or "").strip()
    if not s or REMIX_BEGIN not in s:
        return None

    block = _extract_block(s) or s

    # base_video_id
    base_id = ""
    m = re.search(r"base_video_id\s*:\s*(video_[A-Za-z0-9]+)", block, flags=re.I)
    if m:
        base_id = m.group(1)
    else:
        m2 = re.search(r"(video_[A-Za-z0-9]+)", block)
        if m2:
            base_id = m2.group(1)

    base_id = normalize_video_id(base_id)
    if not base_id:
        return None

    # user_instruction
    user_instr = ""
    m3 = re.search(r"user_instruction\s*:\s*([\s\S]*?)(?:\n[A-Za-z_]+\s*:|\Z)", block, flags=re.I)
    if m3:
        user_instr = (m3.group(1) or "").strip()

    # assistant_rewrite (preferred)
    assistant_rewrite = ""
    m4 = re.search(r"assistant_rewrite\s*:\s*([\s\S]*?)(?:\n[A-Za-z_]+\s*:|\Z)", block, flags=re.I)
    if m4:
        assistant_rewrite = (m4.group(1) or "").strip()

    # user_language (optional)
    user_lang = ""
    ml = re.search(r"user_language\s*:\s*([A-Za-z_-]+)", block, flags=re.I)
    if ml:
        user_lang = (ml.group(1) or "").strip().lower()

    # 若 assistant_rewrite 里包含触发行，提取其 prompt 并剥离协议 tag
    if assistant_rewrite:
        assistant_rewrite = _strip_protocol_tags(assistant_rewrite)

    # 兜底：从用户指令/改写检测
    if not user_lang:
        user_lang = detect_user_language((user_instr or "") + "\n" + (assistant_rewrite or ""))

    return {
        "base_video_id": base_id,
        "user_instruction": user_instr,
        "assistant_rewrite": assistant_rewrite,
        "user_language": user_lang,
    }


def build_remix_api_prompt(
    base_video_id: str,
    user_instruction: str,
    assistant_rewrite: str,
    user_language: str = "",
) -> str:
    """
    ✅ 关键：Remix API 只需要“稳定的改造指令”。
    ✅ 修复：按你要求，送入模型文本=使用者语言（不再强制英文）。

    优先级：
    1) assistant_rewrite（来自 Remix Assistant 最终触发前的稳定 prompt，用户语言）
    2) user_instruction（用户原话兜底）

    并且：
    - 自动剥离 ##CONFIRM / ##CODE 等协议字段
    - 强制补齐 must_keep / only change / do not change 的用户语言锚点
    - 单行化
    """
    _ = normalize_video_id(base_video_id)  # 保留接口一致性（未来可用于日志或更强约束）

    lang = (user_language or "").strip().lower()
    if not lang:
        lang = detect_user_language((assistant_rewrite or "") + "\n" + (user_instruction or ""))

    rewrite = _strip_protocol_tags(assistant_rewrite or "")
    instr = (user_instruction or "").strip()

    core = rewrite if rewrite else instr
    core = _strip_protocol_tags(core)
    core = _ensure_stable_remix_prompt(core, lang)

    return core
