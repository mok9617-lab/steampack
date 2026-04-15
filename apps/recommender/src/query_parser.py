from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


GENRE_KEYWORDS = {
    "RPG": ["rpg", "알피지", "역할수행", "역할"],
    "Action": ["action", "액션"],
    "Adventure": ["adventure", "어드벤처", "모험"],
    "Strategy": ["strategy", "전략", "전술", "턴제"],
    "Simulation": ["simulation", "시뮬레이션", "시뮬"],
    "Survival": ["survival", "생존", "서바이벌"],
    "FPS": ["fps", "first person shooter", "슈팅"],
    "Horror": ["horror", "공포", "호러", "무서운"],
    "Indie": ["indie", "인디"],
    "Free To Play": ["free to play", "f2p", "무료", "무료플레이"],
    "Casual": ["casual", "캐주얼"],
    "Puzzle": ["puzzle", "퍼즐"],
    "Platformer": ["platformer", "플랫포머", "플랫폼"],
    "Rhythm": ["rhythm", "리듬"],
    "Racing": ["racing", "레이싱", "레이스"],
    "Sports": ["sports", "스포츠"],
    "Visual Novel": ["visual novel", "비주얼노벨"],
    "Card Game": ["card game", "card", "덱빌딩", "카드"],
}

SOFT_PREFERENCE_KEYWORDS = {
    "StoryRich": ["story", "narrative", "스토리", "서사", "이야기", "엔딩", "여운", "감동"],
    "Healing": ["chill", "relax", "cozy", "힐링", "편안", "느긋", "잔잔"],
    "Challenge": ["hard", "challenging", "도전", "어려운"],
    "FastPaced": ["fast", "fast-paced", "빠른", "속도감", "템포"],
    "HiddenGem": ["숨겨진", "꿀겜", "덜 알려진", "마이너", "hidden gem", "underrated"],
}

PLAY_STYLE_KEYWORDS = {
    "Relaxed": ["여유", "편하게", "캐주얼", "chill", "relax", "cozy"],
    "Competitive": ["경쟁", "랭크", "pvp", "competitive", "빡센"],
    "Exploration": ["탐험", "오픈월드", "explore", "exploration"],
    "BuildCraft": ["건설", "제작", "크래프팅", "craft", "build"],
    "Narrative": ["스토리", "서사", "narrative", "story"],
}

SESSION_LENGTH_KEYWORDS = {
    "Short": ["짧게", "잠깐", "30분", "한판", "라이트", "quick"],
    "Long": ["오래", "장기", "오랜", "시간 투자", "long play"],
}

DIFFICULTY_KEYWORDS = {
    "Easy": ["쉬운", "입문", "초보", "편한 난이도", "easy", "beginner"],
    "Hard": ["어려운", "하드코어", "고난도", "도전", "hard", "challenging"],
}

FOCUS_KEYWORDS = {
    "Combat": ["전투", "액션", "combat", "싸움", "슈팅"],
    "Story": ["스토리", "서사", "엔딩", "감동", "story"],
    "Growth": ["성장", "빌드", "육성", "파밍", "레벨"],
    "Puzzle": ["퍼즐", "추리", "미스터리", "puzzle"],
    "Management": ["경영", "운영", "자원관리", "management"],
}

MUST_HAVE_KEYWORDS = {
    "Singleplayer": ["single", "singleplayer", "싱글", "솔플", "혼자"],
    "Multiplayer": ["multi", "multiplayer", "co-op", "online", "멀티", "협동", "친구랑", "mmo", "mmorpg"],
}

EXCLUSION_CLUSTERS = {
    "Horror": ["horror", "scary", "공포", "호러", "무서운", "점프스케어"],
    "Free To Play": ["free to play", "f2p", "무료", "무료플레이"],
    "Multiplayer": ["multiplayer", "co-op", "coop", "online", "mmo", "멀티", "협동"],
    "RPG": ["rpg", "알피지", "역할수행"],
    "Action": ["action", "액션"],
    "Adventure": ["adventure", "어드벤처", "모험"],
    "Strategy": ["strategy", "전략", "전술", "턴제"],
    "Simulation": ["simulation", "시뮬레이션", "시뮬"],
    "Survival": ["survival", "생존", "서바이벌"],
    "FPS": ["fps", "first person shooter", "슈팅"],
    "Indie": ["indie", "인디"],
    "Casual": ["casual", "캐주얼"],
    "Racing": ["racing", "레이싱", "레이스"],
    "Sports": ["sports", "스포츠"],
    "Puzzle": ["puzzle", "퍼즐"],
    "Platformer": ["platformer", "플랫포머", "플랫폼"],
    "Rhythm": ["rhythm", "리듬"],
    "Visual Novel": ["visual novel", "비주얼노벨"],
    "Open World": ["open world", "오픈월드"],
    "Crafting": ["crafting", "크래프팅", "제작"],
    "Anime": ["anime", "애니", "서브컬처"],
    "Card Game": ["card", "카드", "덱빌딩"],
    "Turn-Based": ["turn-based", "turn based", "턴제"],
    "Violence/Gore": ["gore", "violent", "유혈", "잔인", "폭력", "고어"],
    "Sexual Content": ["sexual", "nudity", "성인", "선정", "노출"],
}

EXCLUDE_INTENT_HINTS = [
    "제외",
    "빼고",
    "빼줘",
    "말고",
    "싫어",
    "싫고",
    "싫은",
    "원치",
    "안 원",
    "금지",
    "빼줘",
    "하지마",
    "말아",
    "없이",
    "없는",
    "없고",
    "아닌",
    "아니고",
    "아님",
    "원치 않아",
    "원치않아",
    "빼는",
    "제외한",
    "without",
    "exclude",
    "not ",
    "no ",
]

PUNCT_RE = re.compile(r"[\.,!?;:/\\\(\)\[\]\{\}\|\"']")
SPACE_RE = re.compile(r"\s+")
HANGUL_RE = re.compile(r"[가-힣]")


@dataclass
class ParsedQuery:
    raw_query: str
    normalized_query: str
    preferred_genres: list[str] = field(default_factory=list)
    excluded_genres_or_moods: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)
    must_have: list[str] = field(default_factory=list)
    soft_preferences: list[str] = field(default_factory=list)
    play_style: list[str] = field(default_factory=list)
    session_length: list[str] = field(default_factory=list)
    difficulty: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "preferred_genres": self.preferred_genres,
            "excluded_genres_or_moods": self.excluded_genres_or_moods,
            "excluded_terms": self.excluded_terms,
            "must_have": self.must_have,
            "soft_preferences": self.soft_preferences,
            "play_style": self.play_style,
            "session_length": self.session_length,
            "difficulty": self.difficulty,
            "focus": self.focus,
        }


def _normalize_text(query: str) -> str:
    raw = sanitize_user_query(query)
    if not raw:
        return ""

    q = unicodedata.normalize("NFKC", raw).lower()
    q = PUNCT_RE.sub(" ", q)
    q = SPACE_RE.sub(" ", q).strip()
    if q:
        return q

    # Guard: if punctuation stripping wipes everything out, preserve readable tokens.
    fallback = re.sub(r"[^0-9a-z가-힣\s]", " ", unicodedata.normalize("NFKC", raw).lower())
    fallback = SPACE_RE.sub(" ", fallback).strip()
    return fallback


def _hangul_count(text: str) -> int:
    return len(HANGUL_RE.findall(text or ""))


def _text_quality_score(text: str) -> tuple[int, int, int]:
    if not text:
        return (-1, -1, -1)
    hangul = _hangul_count(text)
    valid = len(re.findall(r"[0-9a-zA-Z가-힣\s]", text))
    weird = len(re.findall(r"[^\x20-\x7E가-힣]", text))
    return (hangul, valid, -weird)


def _try_redecode(text: str, src_enc: str, dst_enc: str) -> str:
    try:
        return text.encode(src_enc, errors="ignore").decode(dst_enc, errors="ignore").strip()
    except Exception:
        return ""


def _recover_mojibake(text: str) -> str:
    base = unicodedata.normalize("NFKC", text or "").strip()
    if not base:
        return ""

    candidates = {base}
    # Common mojibake routes for Korean text.
    pairs = [
        ("latin-1", "utf-8"),
        ("cp1252", "utf-8"),
        ("utf-8", "cp949"),
        ("utf-8", "euc-kr"),
        ("cp949", "utf-8"),
        ("euc-kr", "utf-8"),
    ]
    for src, dst in pairs:
        fixed = _try_redecode(base, src, dst)
        if fixed:
            candidates.add(unicodedata.normalize("NFKC", fixed))

    best = base
    best_score = _text_quality_score(best)
    for cand in candidates:
        score = _text_quality_score(cand)
        if score > best_score:
            best = cand
            best_score = score
    return best


def sanitize_user_query(query: str) -> str:
    raw = (query or "").strip()
    if not raw:
        return ""

    best = _recover_mojibake(raw)

    best = best.replace("\uFFFD", " ").strip()
    best = SPACE_RE.sub(" ", best)
    return best


def _contains_any(query: str, terms: list[str]) -> bool:
    return any(t.lower() in query for t in terms)


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _has_exclude_intent(query: str) -> bool:
    return any(h in query for h in EXCLUDE_INTENT_HINTS)


def _extract_excluded_terms(query: str) -> list[str]:
    exclusion_aliases = {
        kw.lower()
        for kws in EXCLUSION_CLUSTERS.values()
        for kw in kws
        if isinstance(kw, str) and kw.strip()
    }

    def _clean_phrase(raw: str) -> str:
        s = raw.strip()
        s = re.sub(
            r"\b(추천해줘|추천|게임|좀|해줘|말해줘|찾아줘|비슷한|같은|유사한|류는|스타일)\b",
            " ",
            s,
        )
        s = SPACE_RE.sub(" ", s).strip()
        if not s:
            return ""
        tokens = [t for t in s.split(" ") if t]
        if not tokens:
            return ""
        cleaned_tokens = [re.sub(r"(은|는|을|를|이|가|랑|와|과)$", "", t) for t in tokens]
        cleaned_tokens = [t for t in cleaned_tokens if t]
        if not cleaned_tokens:
            return ""
        phrase = " ".join(cleaned_tokens).lower()
        multiword_aliases = [a for a in exclusion_aliases if " " in a]
        multiword_aliases.sort(key=len, reverse=True)
        for alias in multiword_aliases:
            if alias in phrase:
                return alias
        # Prefer the token that actually looks like an exclusion target.
        # This avoids "rpg 공포" being interpreted as excluding both RPG and Horror.
        matched = [
            t
            for t in cleaned_tokens
            if t.lower() in exclusion_aliases
        ]
        if matched:
            return matched[-1]
        return cleaned_tokens[-1]

    terms: list[str] = []
    patterns = [
        r"([가-힣a-z0-9\-]+(?:\s+[가-힣a-z0-9\-]+){0,2})(?:은|는|을|를|이|가)?\s*(빼고|빼줘|제외하고|제외|말고|없이)",
        r"([가-힣a-z0-9\-]+(?:\s+[가-힣a-z0-9\-]+){0,2})(?:은|는|을|를|이|가)?\s*(싫어|싫고|원치 않아|원치않아|안좋아해|원치 않음)",
        r"([가-힣a-z0-9\-]+(?:\s+[가-힣a-z0-9\-]+){0,2})(?:은|는|을|를|이|가)?\s*(없는|없고|아닌|아니고|아님)",
        r"(?:without|exclude|no|not)\s+([a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,3})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, query):
            cand = _clean_phrase(m.group(1) or "")
            if len(cand) >= 2:
                terms.append(cand)
    return _dedup_keep_order(terms)


def _extract_exclusion_contexts(normalized_query: str) -> list[str]:
    def _strip_particle(token: str) -> str:
        return re.sub(r"(은|는|을|를|이|가|랑|와|과)$", "", token)

    stop_tokens = {
        "추천",
        "추천해줘",
        "추천좀",
        "게임",
        "해줘",
        "좀",
        "하는",
        "중에서",
    }
    contexts: list[str] = []
    patterns = [
        r"([가-힣a-z0-9\-]+(?:\s+[가-힣a-z0-9\-]+){0,3})\s*(?:빼고|빼줘|제외하고|제외|말고|없이|싫어|싫고|없는|없고|아닌|아니고|아님)",
        r"(?:without|exclude|no|not)\s+([a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,3})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, normalized_query):
            chunk = SPACE_RE.sub(" ", (m.group(1) or "").strip())
            if chunk:
                contexts.append(chunk)

    # Heuristic fallback: use local token window before explicit exclusion intent markers.
    tokens = [t for t in normalized_query.split(" ") if t]
    marker_tokens = {
        "빼고",
        "빼줘",
        "제외",
        "제외하고",
        "말고",
        "없이",
        "싫어",
        "싫고",
        "없는",
        "없고",
        "아닌",
        "아니고",
        "아님",
        "without",
        "exclude",
        "no",
        "not",
    }
    for i, tok in enumerate(tokens):
        if tok not in marker_tokens:
            continue
        if i == 0:
            continue
        start = max(0, i - 4)
        local = []
        for t in tokens[start:i]:
            v = _strip_particle(t)
            if not v or v in stop_tokens:
                continue
            local.append(v)
        if local:
            contexts.append(" ".join(local))
    return _dedup_keep_order(contexts)


def _map_excluded_terms_to_clusters(excluded_terms: list[str], exclusion_contexts: list[str]) -> list[str]:
    out: list[str] = []
    joined_terms = " ".join(excluded_terms)
    joined_context = " ".join(exclusion_contexts)
    for cluster, kws in EXCLUSION_CLUSTERS.items():
        if any(k.lower() in joined_terms for k in kws):
            out.append(cluster)
            continue
        # Also map clusters from exclusion-local context phrase.
        if joined_context and any(k.lower() in joined_context for k in kws):
            out.append(cluster)
    return _dedup_keep_order(out)


def _has_explicit_exclusion_support(cluster: str, excluded_terms: list[str]) -> bool:
    aliases = [a.lower() for a in EXCLUSION_CLUSTERS.get(cluster, [])]
    if not aliases:
        return False
    joined_terms = " ".join(t.lower() for t in excluded_terms)
    return any(a in joined_terms for a in aliases)


def parse_query(query: str) -> ParsedQuery:
    normalized = _normalize_text(query)
    parsed = ParsedQuery(raw_query=query, normalized_query=normalized)

    for genre, kws in GENRE_KEYWORDS.items():
        if _contains_any(normalized, kws):
            parsed.preferred_genres.append(genre)

    for cond, kws in MUST_HAVE_KEYWORDS.items():
        if _contains_any(normalized, kws):
            parsed.must_have.append(cond)

    if _has_exclude_intent(normalized):
        parsed.excluded_terms.extend(_extract_excluded_terms(normalized))
        exclusion_contexts = _extract_exclusion_contexts(normalized)
        parsed.excluded_genres_or_moods.extend(
            _map_excluded_terms_to_clusters(parsed.excluded_terms, exclusion_contexts)
        )

    # Guard against accidental collision: when a genre appears as preferred and excluded
    # without explicit exclusion-term support, keep it as preferred.
    always_exclusion_priority = {"Horror", "Free To Play", "Multiplayer"}
    filtered_excluded: list[str] = []
    for ex in parsed.excluded_genres_or_moods:
        if (
            ex in parsed.preferred_genres
            and ex not in always_exclusion_priority
            and not _has_explicit_exclusion_support(ex, parsed.excluded_terms)
        ):
            continue
        filtered_excluded.append(ex)
    parsed.excluded_genres_or_moods = filtered_excluded

    for pref, kws in SOFT_PREFERENCE_KEYWORDS.items():
        if _contains_any(normalized, kws):
            parsed.soft_preferences.append(pref)
    for style, kws in PLAY_STYLE_KEYWORDS.items():
        if _contains_any(normalized, kws):
            parsed.play_style.append(style)
    for slen, kws in SESSION_LENGTH_KEYWORDS.items():
        if _contains_any(normalized, kws):
            parsed.session_length.append(slen)
    for diff, kws in DIFFICULTY_KEYWORDS.items():
        if _contains_any(normalized, kws):
            parsed.difficulty.append(diff)
    for f, kws in FOCUS_KEYWORDS.items():
        if _contains_any(normalized, kws):
            parsed.focus.append(f)

    if parsed.excluded_genres_or_moods:
        ex_set = set(parsed.excluded_genres_or_moods)
        parsed.preferred_genres = [g for g in parsed.preferred_genres if g not in ex_set]

    if "Singleplayer" in parsed.must_have and "Multiplayer" in parsed.must_have:
        parsed.must_have = [x for x in parsed.must_have if x != "Multiplayer"]
    if "Multiplayer" in parsed.excluded_genres_or_moods:
        parsed.must_have.append("Singleplayer")
        parsed.must_have = [x for x in parsed.must_have if x != "Multiplayer"]

    parsed.preferred_genres = _dedup_keep_order(parsed.preferred_genres)
    parsed.excluded_genres_or_moods = _dedup_keep_order(parsed.excluded_genres_or_moods)
    parsed.excluded_terms = _dedup_keep_order(parsed.excluded_terms)
    parsed.must_have = _dedup_keep_order(parsed.must_have)
    parsed.soft_preferences = _dedup_keep_order(parsed.soft_preferences)
    parsed.play_style = _dedup_keep_order(parsed.play_style)
    parsed.session_length = _dedup_keep_order(parsed.session_length)
    parsed.difficulty = _dedup_keep_order(parsed.difficulty)
    parsed.focus = _dedup_keep_order(parsed.focus)
    return parsed
