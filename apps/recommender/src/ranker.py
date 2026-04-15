from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer

from .db import get_connection
from .llm_openai import OpenAILLM
from .query_parser import ParsedQuery, parse_query, sanitize_user_query


_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_NAME_INDEX_CACHE: dict[str, tuple[list[int], list[str], np.ndarray]] = {}
_PROFILE_INDEX_CACHE: dict[str, dict] = {}

GENRE_ALIAS = {
    "RPG": {"rpg"},
    "Action": {"action"},
    "Adventure": {"adventure"},
    "Strategy": {"strategy"},
    "Simulation": {"simulation"},
    "Survival": {"survival"},
    "FPS": {"fps", "first person shooter"},
    "Horror": {"horror"},
    "Indie": {"indie"},
    "Free To Play": {"free to play", "f2p"},
    "Casual": {"casual"},
}

NEGATIVE_REVIEW_HINTS = [
    "bad",
    "boring",
    "not recommended",
    "terrible",
    "awful",
    "bug",
    "boring",
    "not fun",
    "bad game",
    "do not recommend",
]

EXCLUDED_SIGNAL_TERMS = {
    "Horror": {"horror", "scary", "fear"},
    "Free To Play": {"free to play", "f2p"},
    "Multiplayer": {"multiplayer", "co-op", "coop", "online", "mmo"},
    "RPG": {"rpg"},
    "Action": {"action"},
    "Adventure": {"adventure"},
    "Strategy": {"strategy"},
    "Simulation": {"simulation"},
    "Survival": {"survival"},
    "FPS": {"fps", "first person shooter"},
    "Indie": {"indie"},
    "Casual": {"casual"},
    "Racing": {"racing"},
    "Sports": {"sports"},
    "Puzzle": {"puzzle"},
    "Platformer": {"platformer"},
    "Rhythm": {"rhythm"},
    "Visual Novel": {"visual novel"},
    "Open World": {"open world"},
    "Crafting": {"crafting"},
    "Anime": {"anime"},
    "Card Game": {"card"},
    "Turn-Based": {"turn-based", "turn based"},
    "Violence/Gore": {"gore", "violent"},
    "Sexual Content": {"sexual", "nudity"},
}


def _get_model(model_name: str) -> SentenceTransformer:
    model = _MODEL_CACHE.get(model_name)
    if model is None:
        offline = os.getenv("HF_HUB_OFFLINE", "").strip() == "1" or os.getenv(
            "TRANSFORMERS_OFFLINE", ""
        ).strip() == "1"
        try:
            if offline:
                model = SentenceTransformer(model_name)
            else:
                model = SentenceTransformer(model_name)
        except Exception as exc:
            # Retry once in strict offline mode to survive blocked outbound networks.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                model = SentenceTransformer(model_name)
            except Exception:
                try:
                    model = SentenceTransformer(model_name, local_files_only=True)
                except Exception:
                    raise exc
        _MODEL_CACHE[model_name] = model
    return model


def _deserialize_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


@dataclass
class Candidate:
    app_id: int
    name: str
    genres: list[str]
    tags: list[str]
    positive_ratio_1y: float
    recent_review_count: int
    median_playtime_1y: float
    sim: float


def _extract_reference_game_hint(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    patterns = [
        r"(.+?)\s*(?:같은|비슷한|유사한|닮은)\s*(?:게임)?",
        r"(.+?)\s*(?:like|similar to)\s*",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if not m:
            continue
        hint = (m.group(1) or "").strip()
        if not hint:
            continue
        parts = [p for p in re.split(r"\s+", hint) if p]
        candidate = " ".join(parts[-4:]).strip()
        generic_terms = {"게임", "추천", "비슷한", "유사한", "같은"}
        if not candidate or candidate in generic_terms:
            continue
        return candidate
    return ""


def _rewrite_query_rule_based(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip())
    if not q:
        return ""
    enrich: list[str] = []
    if any(k in q.lower() for k in ["스토리", "서사", "엔딩", "감동", "story", "narrative"]):
        enrich.append("스토리 중심")
    if any(k in q.lower() for k in ["싱글", "혼자", "single"]):
        enrich.append("싱글플레이")
    if any(k in q.lower() for k in ["멀티", "협동", "친구", "multiplayer", "co-op"]):
        enrich.append("멀티플레이")
    if any(k in q.lower() for k in ["힐링", "편안", "cozy", "relax"]):
        enrich.append("힐링")
    if any(k in q.lower() for k in ["짧게", "잠깐", "30분", "quick"]):
        enrich.append("짧은 세션")
    if enrich:
        return f"{q}. {' / '.join(dict.fromkeys(enrich))}"
    return q


def _extract_query_terms(query: str) -> list[str]:
    text = (query or "").lower()
    tokens = re.findall(r"[a-z0-9가-힣]{2,}", text)
    stop = {
        "추천",
        "게임",
        "스팀",
        "뭐",
        "있음",
        "있냐",
        "해주세요",
        "해줘",
        "좀",
    }
    out: list[str] = []
    for t in tokens:
        if t in stop:
            continue
        if t not in out:
            out.append(t)
    return out[:12]


def _build_intent_bias_text(parsed: ParsedQuery, raw_query: str) -> str:
    parts: list[str] = []
    q = (raw_query or "").lower()
    if "Singleplayer" in parsed.must_have:
        parts.append("singleplayer solo")
    if "Multiplayer" in parsed.must_have:
        parts.append("multiplayer coop online")
    if "StoryRich" in parsed.soft_preferences or any(k in q for k in ["story", "narrative", "스토리", "서사"]):
        parts.append("story narrative ending emotional")
    if "Healing" in parsed.soft_preferences:
        parts.append("healing cozy relax chill")
    if "Challenge" in parsed.soft_preferences:
        parts.append("challenging difficult hardcore")
    if "FastPaced" in parsed.soft_preferences:
        parts.append("fast paced action")
    if "HiddenGem" in parsed.soft_preferences:
        parts.append("hidden gem underrated")
    if "Relaxed" in parsed.play_style:
        parts.append("relaxed cozy")
    if "Competitive" in parsed.play_style:
        parts.append("competitive rank pvp")
    if "Exploration" in parsed.play_style:
        parts.append("exploration open world")
    if "BuildCraft" in parsed.play_style:
        parts.append("building crafting")
    if "Narrative" in parsed.play_style:
        parts.append("narrative story")
    if "Short" in parsed.session_length:
        parts.append("short session quick")
    if "Long" in parsed.session_length:
        parts.append("long playtime")
    if "Easy" in parsed.difficulty:
        parts.append("easy beginner")
    if "Hard" in parsed.difficulty:
        parts.append("hardcore difficult")
    if "Combat" in parsed.focus:
        parts.append("combat action")
    if "Story" in parsed.focus:
        parts.append("story ending")
    if "Growth" in parsed.focus:
        parts.append("growth progression build")
    if "Puzzle" in parsed.focus:
        parts.append("puzzle mystery")
    if "Management" in parsed.focus:
        parts.append("management tycoon")
    for g in parsed.preferred_genres:
        parts.append(g.lower())
    for ex in parsed.excluded_genres_or_moods:
        parts.append(f"{ex.lower()} exclude")
    return " ".join(parts).strip()

def _build_name_index(conn, model: SentenceTransformer) -> tuple[list[int], list[str], np.ndarray]:
    key = str(id(model))
    cached = _NAME_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    rows = conn.execute(
        """
        SELECT g.app_id, g.name
        FROM games g
        JOIN game_profiles p ON p.app_id = g.app_id
        """
    ).fetchall()
    app_ids = [int(r["app_id"]) for r in rows]
    names = [str(r["name"] or "") for r in rows]
    if names:
        vecs = model.encode(names, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    else:
        vecs = np.zeros((0, 384), dtype=np.float32)
    cached = (app_ids, names, vecs)
    _NAME_INDEX_CACHE[key] = cached
    return cached


def _load_profile_index(conn, db_key: str) -> dict:
    cached = _PROFILE_INDEX_CACHE.get(db_key)
    if cached is not None:
        return cached

    rows = []
    for attempt in range(6):
        try:
            rows = conn.execute(
                """
                SELECT g.app_id, g.name, g.genres, g.tags, p.profile_embedding,
                       p.positive_ratio_1y, p.recent_review_count, p.median_playtime_1y
                FROM game_profiles p
                JOIN games g ON g.app_id = p.app_id
                WHERE p.profile_embedding IS NOT NULL
                """
            ).fetchall()
            break
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt == 5:
                raise
            time.sleep(0.25 * (attempt + 1))

    app_ids: list[int] = []
    names: list[str] = []
    genres_list: list[list[str]] = []
    tags_list: list[list[str]] = []
    vecs: list[np.ndarray] = []
    pos_ratios: list[float] = []
    recent_counts: list[int] = []
    med_playtimes: list[float] = []

    for row in rows:
        app_ids.append(int(row["app_id"]))
        names.append(str(row["name"] or ""))
        try:
            genres = json.loads(row["genres"] or "[]")
        except Exception:
            genres = []
        try:
            tags = json.loads(row["tags"] or "[]")
        except Exception:
            tags = []
        genres_list.append(genres)
        tags_list.append(tags)
        vecs.append(_deserialize_vector(row["profile_embedding"]))
        pos_ratios.append(float(row["positive_ratio_1y"] or 0.0))
        recent_counts.append(int(row["recent_review_count"] or 0))
        med_playtimes.append(float(row["median_playtime_1y"] or 0.0))

    mat = np.vstack(vecs) if vecs else np.zeros((0, 384), dtype=np.float32)
    cached = {
        "app_ids": app_ids,
        "names": names,
        "genres_list": genres_list,
        "tags_list": tags_list,
        "vectors": mat,
        "positive_ratio_1y": pos_ratios,
        "recent_review_count": recent_counts,
        "median_playtime_1y": med_playtimes,
    }
    _PROFILE_INDEX_CACHE[db_key] = cached
    return cached


def _resolve_reference_game(
    conn, hint: str, model: SentenceTransformer, extra_hints: list[str] | None = None
) -> tuple[int, str] | None:
    candidates = [hint] + [h for h in (extra_hints or []) if h]
    seen: set[str] = set()

    # 1) Lexical matching across primary hint + LLM-generated aliases.
    for h in candidates:
        h = (h or "").strip()
        if not h:
            continue
        key = h.lower()
        if key in seen:
            continue
        seen.add(key)

        rows = conn.execute(
            """
            SELECT g.app_id, g.name
            FROM games g
            JOIN game_profiles p ON p.app_id = g.app_id
            WHERE LOWER(g.name) LIKE ?
            LIMIT 120
            """,
            (f"%{key}%",),
        ).fetchall()
        if not rows:
            continue

        best = None
        best_score = 0.0
        for r in rows:
            name = str(r["name"] or "")
            score = SequenceMatcher(None, key, name.lower()).ratio()
            if key in name.lower():
                score += 0.35
            if score > best_score:
                best_score = score
                best = (int(r["app_id"]), name)
        if best and best_score >= 0.52:
            return best

    # 2) If hint is still Hangul-only and no lexical match was found, avoid noisy fallback.
    base_hint = (hint or "").strip()
    if not base_hint:
        return None
    if _contains_hangul(base_hint):
        return None

    # 3) High-confidence embedding fallback (avoid noisy switches).
    app_ids, names, vecs = _build_name_index(conn, model)
    if len(names) > 0:
        qv = model.encode([base_hint], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        sims = vecs @ qv
        idx = int(np.argmax(sims))
        if float(sims[idx]) >= 0.70:
            return (app_ids[idx], names[idx])
    return None


def _hint_matches_title(candidate_name: str, hint: str) -> bool:
    if not hint:
        return False
    c = _normalize_title_for_variant(candidate_name)
    h = _normalize_title_for_variant(hint)
    if not c or not h:
        return False
    if h in c or c in h:
        return True
    return SequenceMatcher(None, c, h).ratio() >= 0.72


def _contains_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


def _load_profile_vector(conn, app_id: int) -> np.ndarray | None:
    row = conn.execute(
        "SELECT profile_embedding FROM game_profiles WHERE app_id = ?",
        (app_id,),
    ).fetchone()
    if not row or row["profile_embedding"] is None:
        return None
    vec = _deserialize_vector(row["profile_embedding"])
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _normalize_title_for_variant(name: str) -> str:
    t = (name or "").lower()
    for token in [
        " enhanced",
        " definitive edition",
        " definitive",
        " remastered",
        " game of the year",
        " goty",
        " edition",
        " dlc",
        " expansion",
        " pack",
        " soundtrack",
        " ost",
    ]:
        t = t.replace(token, "")
    t = re.sub(r"[^a-z0-9가-힣]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_reference_variant(candidate_name: str, reference_name: str) -> bool:
    c = _normalize_title_for_variant(candidate_name)
    r = _normalize_title_for_variant(reference_name)
    if not c or not r:
        return False
    if c == r:
        return True
    if r in c or c in r:
        return True
    return False


def _is_hard_filtered(parsed: ParsedQuery, genres: list[str], tags: list[str], game_name: str) -> bool:
    genre_set = {g.lower() for g in (genres or [])}
    tag_set = {t.lower() for t in (tags or [])}
    topic_set = genre_set.union(tag_set)
    name_l = game_name.lower()

    if parsed.preferred_genres:
        enforceable = []
        for pref in parsed.preferred_genres:
            alias = GENRE_ALIAS.get(pref)
            if alias:
                enforceable.append(alias)
        if enforceable and not any(topic_set.intersection(a) for a in enforceable):
            return True

    # Explicit exclusions by known categories
    for ex in parsed.excluded_genres_or_moods:
        aliases = GENRE_ALIAS.get(ex) or EXCLUDED_SIGNAL_TERMS.get(ex) or {ex.lower()}
        if any(a in topic_set for a in aliases):
            return True
        if any(a in name_l for a in aliases):
            return True

    # Generic exclusion terms extracted from user utterance.
    for term in parsed.excluded_terms:
        t = term.lower().strip()
        if len(t) < 2:
            continue
        if any(t in g for g in topic_set):
            return True
        if t in name_l:
            return True

    if "Singleplayer" in parsed.must_have and "massively multiplayer" in topic_set:
        return True

    return False


def _has_excluded_signal(
    parsed: ParsedQuery, genres: list[str], tags: list[str], evidence_texts: list[str]
) -> bool:
    genre_set = {g.lower() for g in (genres or [])}
    tag_set = {t.lower() for t in (tags or [])}
    topic_set = genre_set.union(tag_set)
    joined = " ".join(evidence_texts).lower()
    for ex in parsed.excluded_genres_or_moods:
        aliases = EXCLUDED_SIGNAL_TERMS.get(ex) or GENRE_ALIAS.get(ex) or {ex.lower()}
        if any(a in topic_set for a in aliases):
            return True
        if any(a in joined for a in aliases):
            return True
    for term in parsed.excluded_terms:
        t = term.lower().strip()
        if len(t) < 2:
            continue
        if any(t in g for g in topic_set):
            return True
        if t in joined:
            return True
    return False


def _fetch_query_relevant_evidence(
    conn,
    app_id: int,
    query_vec: np.ndarray | None,
    query_terms: list[str] | None = None,
    limit: int = 5,
    pool_limit: int = 160,
):
    rows = conn.execute(
        """
        SELECT cleaned_text, trust_label, votes_up, voted_up, playtime_forever, review_date, embedding
        FROM reviews
        WHERE app_id = ?
          AND cleaned_text IS NOT NULL
          AND cleaned_text != ''
          AND trust_label IN ('high', 'medium')
          AND voted_up = 1
          AND embedding IS NOT NULL
        ORDER BY review_date DESC
        LIMIT ?
        """,
        (app_id, pool_limit),
    ).fetchall()

    scored = []
    for row in rows:
        text = row["cleaned_text"] or ""
        if _is_negative_evidence_text(text):
            continue
        try:
            if query_vec is not None:
                r_vec = _deserialize_vector(row["embedding"])
                sim = float(np.dot(query_vec, r_vec))
            else:
                sim = 0.0
        except Exception:
            continue
        text_l = text.lower()
        lexical_hit = 0
        if query_terms:
            lexical_hit = sum(1 for t in query_terms if t in text_l)
        score = sim + (0.035 * lexical_hit)
        trust_rank = 1 if row["trust_label"] == "high" else 0
        scored.append(
            (
                score,
                lexical_hit,
                trust_rank,
                int(row["votes_up"] or 0),
                int(row["playtime_forever"] or 0),
                text,
            )
        )

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
    return [x[5] for x in scored[:limit]]


def _fetch_negative_evidence(
    conn,
    app_id: int,
    query_terms: list[str] | None = None,
    limit: int = 2,
    pool_limit: int = 100,
):
    rows = conn.execute(
        """
        SELECT cleaned_text, trust_label, votes_up, review_date
        FROM reviews
        WHERE app_id = ?
          AND cleaned_text IS NOT NULL
          AND cleaned_text != ''
          AND voted_up = 0
        ORDER BY review_date DESC
        LIMIT ?
        """,
        (app_id, pool_limit),
    ).fetchall()

    scored = []
    for row in rows:
        text = row["cleaned_text"] or ""
        text_l = text.lower()
        lexical_hit = 0
        if query_terms:
            lexical_hit = sum(1 for t in query_terms if t in text_l)
        trust_rank = 1 if str(row["trust_label"] or "").lower() == "high" else 0
        scored.append(
            (
                lexical_hit,
                trust_rank,
                int(row["votes_up"] or 0),
                str(row["review_date"] or ""),
                text,
            )
        )

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    out: list[str] = []
    for _, _, _, _, text in scored:
        if not text:
            continue
        out.append(_brief_note(text, max_chars=90))
        if len(out) >= limit:
            break
    return out


def _brief_note(text: str, max_chars: int = 90) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return ""
    # Keep only first sentence-like chunk for compact caution notes.
    t = re.split(r"[.!?。！？]\s*", t, maxsplit=1)[0].strip()
    if len(t) <= max_chars:
        return t
    return t[: max(20, max_chars - 1)].rstrip() + "…"


def _candidate_lexical_score(
    query_terms: list[str], name: str, genres: list[str], tags: list[str]
) -> float:
    if not query_terms:
        return 0.0
    name_l = (name or "").lower()
    genre_l = " ".join(g.lower() for g in (genres or []))
    tags_l = " ".join(t.lower() for t in (tags or []))
    hits = 0
    for t in query_terms:
        if t in name_l or t in genre_l or t in tags_l:
            hits += 1
    return hits / max(len(query_terms), 1)


def _has_multiplayer_signal(genres: list[str], evidence_texts: list[str]) -> bool:
    genre_set = {g.lower() for g in genres}
    if "multiplayer" in genre_set or "massively multiplayer" in genre_set:
        return True
    joined = " ".join(evidence_texts).lower()
    signals = ["multiplayer", "co-op", "coop", "online", "party", "friends"]
    return any(s in joined for s in signals)


def _soft_match_count(texts: list[str], soft_preferences: list[str]) -> int:
    if not soft_preferences:
        return 0
    joined = " ".join(texts).lower()
    pref_to_terms = {
        "StoryRich": ["story", "narrative", "ending", "plot"],
        "Healing": ["chill", "relax", "cozy", "comfort"],
        "Challenge": ["hard", "challeng", "difficult", "skill"],
        "FastPaced": ["fast", "paced", "speed", "tempo"],
        "HiddenGem": ["underrated", "hidden gem", "unknown", "niche"],
    }
    count = 0
    for pref in soft_preferences:
        terms = pref_to_terms.get(pref, [])
        if any(t in joined for t in terms):
            count += 1
    return count


def _hidden_gem_score(recent_count: int, positive_ratio: float, median_playtime: float) -> float:
    # Prefer low-to-mid popularity while keeping quality/engagement signals.
    popularity_penalty = min(max(recent_count, 0) / 260.0, 1.0)
    long_tail = 1.0 - popularity_penalty
    quality = min(max(positive_ratio, 0.0), 1.0)
    engagement = min(max(median_playtime, 0.0) / 1200.0, 1.0)
    return (0.6 * long_tail) + (0.3 * quality) + (0.1 * engagement)


def _preferred_genre_signal_count(
    preferred_genres: list[str], genres: list[str], tags: list[str], evidence_texts: list[str]
) -> int:
    if not preferred_genres:
        return 0
    genre_set = {g.lower() for g in (genres or [])}
    tag_set = {t.lower() for t in (tags or [])}
    topic_set = genre_set.union(tag_set)
    joined = " ".join(evidence_texts).lower()
    hit = 0
    for pref in preferred_genres:
        aliases = GENRE_ALIAS.get(pref, {pref.lower()})
        if any(a in topic_set for a in aliases) or any(a in joined for a in aliases):
            hit += 1
    return hit


def _query_alignment_score(
    query_terms: list[str], genres: list[str], tags: list[str], evidence_texts: list[str]
) -> float:
    if not query_terms:
        return 0.0
    joined = " ".join(evidence_texts).lower()
    genre_text = " ".join(g.lower() for g in (genres or []))
    tags_text = " ".join(t.lower() for t in (tags or []))
    hits = 0
    for t in query_terms:
        if t in joined or t in genre_text or t in tags_text:
            hits += 1
    return min(hits / max(len(query_terms), 1), 1.0)


def _confidence_label(recent_count: int, median_playtime: float, evidence_count: int) -> str:
    if evidence_count >= 3 and recent_count >= 80 and median_playtime >= 120:
        return "high"
    if evidence_count >= 2 and recent_count >= 30:
        return "medium"
    return "low"


def _is_negative_evidence_text(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in NEGATIVE_REVIEW_HINTS)


def _merge_parsed(rule_parsed: ParsedQuery, llm_parsed: ParsedQuery) -> ParsedQuery:
    preferred = list(dict.fromkeys(llm_parsed.preferred_genres + rule_parsed.preferred_genres))
    excluded = list(
        dict.fromkeys(rule_parsed.excluded_genres_or_moods + llm_parsed.excluded_genres_or_moods)
    )
    excluded_terms = list(dict.fromkeys(rule_parsed.excluded_terms + llm_parsed.excluded_terms))
    must_have = list(dict.fromkeys(rule_parsed.must_have + llm_parsed.must_have))
    soft = list(dict.fromkeys(llm_parsed.soft_preferences + rule_parsed.soft_preferences))
    play_style = list(dict.fromkeys(llm_parsed.play_style + rule_parsed.play_style))
    session_length = list(dict.fromkeys(llm_parsed.session_length + rule_parsed.session_length))
    difficulty = list(dict.fromkeys(llm_parsed.difficulty + rule_parsed.difficulty))
    focus = list(dict.fromkeys(llm_parsed.focus + rule_parsed.focus))

    if "Horror" in excluded:
        preferred = [g for g in preferred if g != "Horror"]
    if "Free To Play" in excluded:
        preferred = [g for g in preferred if g != "Free To Play"]
    if "Singleplayer" in must_have and "Multiplayer" in must_have:
        must_have = [x for x in must_have if x != "Multiplayer"]

    ex_text = " ".join(excluded_terms).lower()
    if ex_text:
        for genre, aliases in GENRE_ALIAS.items():
            if genre in preferred and any(a in ex_text for a in aliases):
                preferred = [g for g in preferred if g != genre]
                if genre not in excluded:
                    excluded.append(genre)

    return ParsedQuery(
        raw_query=rule_parsed.raw_query,
        normalized_query=rule_parsed.normalized_query,
        preferred_genres=preferred,
        excluded_genres_or_moods=excluded,
        excluded_terms=excluded_terms,
        must_have=must_have,
        soft_preferences=soft,
        play_style=play_style,
        session_length=session_length,
        difficulty=difficulty,
        focus=focus,
    )


def _merge_parsed_llm_primary(rule_parsed: ParsedQuery, llm_parsed: ParsedQuery) -> ParsedQuery:
    # LLM-first: keep LLM intent as primary and only fill missing slots with rule-based fallback.
    preferred = list(llm_parsed.preferred_genres) or list(rule_parsed.preferred_genres)
    excluded = list(llm_parsed.excluded_genres_or_moods) or list(rule_parsed.excluded_genres_or_moods)
    excluded_terms = list(llm_parsed.excluded_terms) or list(rule_parsed.excluded_terms)
    must_have = list(llm_parsed.must_have) or list(rule_parsed.must_have)
    soft = list(llm_parsed.soft_preferences) or list(rule_parsed.soft_preferences)
    play_style = list(llm_parsed.play_style) or list(rule_parsed.play_style)
    session_length = list(llm_parsed.session_length) or list(rule_parsed.session_length)
    difficulty = list(llm_parsed.difficulty) or list(rule_parsed.difficulty)
    focus = list(llm_parsed.focus) or list(rule_parsed.focus)

    merged = ParsedQuery(
        raw_query=rule_parsed.raw_query,
        normalized_query=rule_parsed.normalized_query,
        preferred_genres=preferred,
        excluded_genres_or_moods=excluded,
        excluded_terms=excluded_terms,
        must_have=must_have,
        soft_preferences=soft,
        play_style=play_style,
        session_length=session_length,
        difficulty=difficulty,
        focus=focus,
    )
    return _merge_parsed(merged, merged)

def _select_diverse_results(items: list[dict], top_k: int, diversity_weight: float = 0.2) -> list[dict]:
    if len(items) <= 1:
        return items[:top_k]

    remaining = list(items)
    selected: list[dict] = []

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")
        for i, cand in enumerate(remaining):
            relevance = float(cand.get("final_score", cand.get("similarity", 0.0)))
            vec = cand.get("_vector")
            if selected and vec is not None:
                max_sim = max(float(np.dot(vec, s.get("_vector"))) for s in selected if s.get("_vector") is not None)
            else:
                max_sim = 0.0
            mmr = (1.0 - diversity_weight) * relevance - diversity_weight * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


def _reason_has_contradiction(reason_ko: str, parsed: ParsedQuery) -> bool:
    text = (reason_ko or "").lower()
    if not text:
        return True
    if any(x in text for x in ["not recommend", "bad", "worst", "boring"]):
        return True
    # Korean contradiction/negative recommendation patterns.
    if any(
        x in text
        for x in [
            "추천하기 어렵",
            "추천하기 힘들",
            "추천하기 힘듭",
            "비추천",
            "적합하지 않",
            "맞지 않",
            "장르가 다르",
            "유사하지 않",
            "비슷하지 않",
        ]
    ):
        return True
    if "Horror" in parsed.excluded_genres_or_moods:
        if any(x in text for x in ["horror", "scary", "fear", "terror"]):
            return True
    for term in parsed.excluded_terms:
        t = term.lower().strip()
        if len(t) >= 2 and t in text:
            return True
    return False


def recommend_games(
    db_path: Path,
    query: str,
    top_k: int = 5,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4.1-mini",
) -> dict:
    original_query = query
    query = sanitize_user_query(query)
    runtime_errors: list[str] = []
    no_fallback = os.getenv("STRICT_NO_FALLBACK", "").strip() == "1"

    def _fail_result(message: str, mode: str = "strict_no_fallback") -> dict:
        errs = [message]
        if llm is not None:
            errs = list(llm.errors) + errs
        errs = errs + runtime_errors
        return {
            "query": original_query,
            "normalized_input_query": query,
            "rewritten_query": rewritten_query,
            "effective_query": effective_query,
            "mode": mode,
            "reference_game": None,
            "parsed_query": parse_query(effective_query).to_dict(),
            "results": [],
            "llm_errors": errs,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    llm: OpenAILLM | None = OpenAILLM(api_key=openai_api_key, model=openai_model) if openai_api_key else None
    rewritten_query = ""
    effective_query = query
    llm_parsed_payload: dict | None = None

    if no_fallback and llm is None:
        return _fail_result("fallback_disabled: OPENAI_API_KEY is required")

    if llm is not None:
        llm_payload = llm.rewrite_and_parse_query(query)
        llm_rewritten = str(llm_payload.get("rewritten_query", "")).strip()
        llm_has_parse_signal = any(
            llm_payload.get(k)
            for k in [
                "preferred_genres",
                "excluded_genres_or_moods",
                "excluded_terms",
                "must_have",
                "soft_preferences",
                "play_style",
                "session_length",
                "difficulty",
                "focus",
            ]
        )
        if llm_rewritten:
            rewritten_query = llm_rewritten
            effective_query = llm_rewritten
            llm_parsed_payload = llm_payload
        elif llm_has_parse_signal:
            llm_parsed_payload = llm_payload
            rewritten_query = _rewrite_query_rule_based(query)
            effective_query = rewritten_query or query
        else:
            if no_fallback:
                return _fail_result("fallback_disabled: llm_rewrite_and_parse_failed")
            rewritten_query = _rewrite_query_rule_based(query)
            effective_query = rewritten_query or query
    else:
        rewritten_query = _rewrite_query_rule_based(query)
        effective_query = rewritten_query or query

    rule_parsed = parse_query(effective_query)
    if no_fallback and not rule_parsed.normalized_query:
        return _fail_result("input_normalization_failed: normalized_query is empty")

    query_terms = _extract_query_terms(effective_query)

    if llm is not None:
        if llm_parsed_payload is None:
            err_before_parse = len(llm.errors)
            llm_parsed_payload = llm.parse_query(effective_query)
            llm_parse_failed = len(llm.errors) > err_before_parse
        else:
            llm_parse_failed = False
        p = llm_parsed_payload or {}
        llm_parsed = ParsedQuery(
            raw_query=effective_query,
            normalized_query=effective_query,
            preferred_genres=list(p.get("preferred_genres", [])),
            excluded_genres_or_moods=list(p.get("excluded_genres_or_moods", [])),
            excluded_terms=list(p.get("excluded_terms", [])),
            must_have=list(p.get("must_have", [])),
            soft_preferences=list(p.get("soft_preferences", [])),
            play_style=list(p.get("play_style", [])),
            session_length=list(p.get("session_length", [])),
            difficulty=list(p.get("difficulty", [])),
            focus=list(p.get("focus", [])),
        )
        if no_fallback and llm_parse_failed:
            return _fail_result("fallback_disabled: llm_parse_failed")
        parsed = _merge_parsed_llm_primary(rule_parsed, llm_parsed) if not llm_parse_failed else rule_parsed
    else:
        parsed = rule_parsed

    model: SentenceTransformer | None = None
    model_load_error = ""
    try:
        model = _get_model(model_name)
    except Exception as exc:
        model = None
        model_load_error = f"embedding_model_load_failed: {exc}"
        runtime_errors.append(model_load_error)
        if llm is not None:
            llm.errors.append(model_load_error)
        if no_fallback:
            return _fail_result("fallback_disabled: embedding_model_unavailable")
    mode = "query"
    reference_game: dict | None = None

    candidates: list[Candidate] = []
    with get_connection(db_path) as conn:
        q_vec: np.ndarray | None = None
        # Important: similar_to mode must be triggered by the original user query, not rewritten query.
        reference_hint = _extract_reference_game_hint(query)
        if model is not None and reference_hint:
            alias_hints = llm.guess_game_titles(reference_hint) if llm is not None else []
            resolved = _resolve_reference_game(conn, reference_hint, model, extra_hints=alias_hints)
            if resolved is not None:
                ref_app_id, ref_name = resolved
                ref_vec = _load_profile_vector(conn, ref_app_id)
                if ref_vec is not None:
                    q_vec = ref_vec
                    mode = "similar_to"
                    reference_game = {"app_id": ref_app_id, "name": ref_name, "hint": reference_hint}
                else:
                    q_vec = model.encode([effective_query], convert_to_numpy=True, normalize_embeddings=True)[0]
            else:
                q_vec = model.encode([effective_query], convert_to_numpy=True, normalize_embeddings=True)[0]
        else:
            if model is not None:
                q_vec = model.encode([effective_query], convert_to_numpy=True, normalize_embeddings=True)[0]
            else:
                if no_fallback:
                    return _fail_result("fallback_disabled: query_vector_unavailable")
                mode = "query_lexical_fallback"

        # Intent-bias blending to separate vague queries with different user intents.
        if mode == "query" and model is not None and q_vec is not None:
            bias_text = _build_intent_bias_text(parsed, effective_query)
            if bias_text:
                bias_vec = model.encode([bias_text], convert_to_numpy=True, normalize_embeddings=True)[0]
                q_vec = (0.8 * q_vec) + (0.2 * bias_vec)
                norm = np.linalg.norm(q_vec)
                if norm > 0:
                    q_vec = q_vec / norm

        profile_index = _load_profile_index(conn, str(Path(db_path).resolve()))
        app_ids = profile_index["app_ids"]
        names = profile_index["names"]
        genres_list = profile_index["genres_list"]
        tags_list = profile_index["tags_list"]
        vectors = profile_index["vectors"]
        vector_by_app_id = {app_ids[i]: vectors[i] for i in range(len(app_ids))}
        pos_ratios = profile_index["positive_ratio_1y"]
        recent_counts = profile_index["recent_review_count"]
        med_playtimes = profile_index["median_playtime_1y"]
        sims = vectors @ q_vec if (len(app_ids) > 0 and q_vec is not None) else np.array([], dtype=np.float32)

        for i in range(len(app_ids)):
            app_id = app_ids[i]
            name = names[i]
            genres = genres_list[i]
            tags = tags_list[i]

            if reference_game is not None:
                if app_id == int(reference_game["app_id"]):
                    continue
                if _is_reference_variant(name, str(reference_game["name"])):
                    continue
                if _hint_matches_title(name, str(reference_game.get("hint", ""))):
                    continue

            if _is_hard_filtered(parsed, genres, tags, name):
                continue

            sim = (
                float(sims[i])
                if q_vec is not None
                else _candidate_lexical_score(query_terms, name, genres, tags)
            )
            if no_fallback and not np.isfinite(sim):
                continue
            candidates.append(
                Candidate(
                    app_id=app_id,
                    name=name,
                    genres=genres,
                    tags=tags,
                    positive_ratio_1y=pos_ratios[i],
                    recent_review_count=recent_counts[i],
                    median_playtime_1y=med_playtimes[i],
                    sim=sim,
                )
            )

        candidates.sort(key=lambda x: x.sim, reverse=True)
        stage1 = candidates[: max(top_k * 8, 50)]

        reranked: list[dict] = []
        hidden_gem_mode = "HiddenGem" in parsed.soft_preferences
        for c in stage1:
            evidence_texts = _fetch_query_relevant_evidence(
                conn, c.app_id, q_vec, query_terms=query_terms, limit=5
            )
            if not evidence_texts:
                continue
            if _has_excluded_signal(parsed, c.genres, c.tags, evidence_texts):
                continue

            multiplayer_signal = _has_multiplayer_signal(c.genres, evidence_texts)
            if "Multiplayer" in parsed.must_have and not multiplayer_signal:
                continue
            if "Singleplayer" in parsed.must_have and multiplayer_signal:
                continue

            pref_hits = _preferred_genre_signal_count(
                parsed.preferred_genres, c.genres, c.tags, evidence_texts
            )
            alignment = _query_alignment_score(query_terms, c.genres, c.tags, evidence_texts)
            reranked.append(
                {
                    "app_id": c.app_id,
                    "name": c.name,
                    "steam_url": f"https://store.steampowered.com/app/{c.app_id}/",
                    "image_url": f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{c.app_id}/header.jpg",
                    "genres": c.genres,
                    "tags": c.tags,
                    "similarity": round(c.sim, 4),
                    "recent_review_count": c.recent_review_count,
                    "positive_ratio_1y": round(c.positive_ratio_1y, 4),
                    "median_playtime_1y": round(c.median_playtime_1y, 1),
                    "preferred_genre_hits": pref_hits,
                    "query_alignment_score": round(alignment, 4),
                    "soft_match_count": _soft_match_count(evidence_texts, parsed.soft_preferences),
                    "multiplayer_signal": multiplayer_signal,
                    "hidden_gem_score": round(
                        _hidden_gem_score(
                            c.recent_review_count,
                            c.positive_ratio_1y,
                            c.median_playtime_1y,
                        ),
                        4,
                    ),
                    "confidence": _confidence_label(
                        c.recent_review_count, c.median_playtime_1y, len(evidence_texts)
                    ),
                    "evidence_reviews": evidence_texts[:5],
                    "_vector": vector_by_app_id.get(c.app_id),
                }
            )

        if hidden_gem_mode:
            reranked.sort(
                key=lambda x: (
                    x["query_alignment_score"],
                    x["hidden_gem_score"],
                    x["similarity"],
                    x["preferred_genre_hits"],
                    x["soft_match_count"],
                    x["positive_ratio_1y"],
                ),
                reverse=True,
            )
            for x in reranked:
                x["final_score"] = (
                    (0.32 * float(x["query_alignment_score"]))
                    + (0.30 * float(x["hidden_gem_score"]))
                    + (0.24 * float(x["similarity"]))
                    + (0.08 * float(x["preferred_genre_hits"]))
                    + (0.06 * float(x["soft_match_count"]))
                )
        else:
            reranked.sort(
                key=lambda x: (
                    x["query_alignment_score"],
                    x["similarity"],
                    x["preferred_genre_hits"],
                    x["soft_match_count"],
                    x["recent_review_count"],
                    x["median_playtime_1y"],
                ),
                reverse=True,
            )
            for x in reranked:
                x["final_score"] = (
                    (0.34 * float(x["query_alignment_score"]))
                    + (0.40 * float(x["similarity"]))
                    + (0.12 * float(x["preferred_genre_hits"]))
                    + (0.08 * float(x["soft_match_count"]))
                    + (0.06 * min(float(x["recent_review_count"]) / 300.0, 1.0))
                )

    diverse = _select_diverse_results(reranked, top_k=max(top_k * 3, top_k), diversity_weight=0.22)
    final_results: list[dict] = []
    if llm is not None:
        for item in diverse:
            combined = llm.summarize_and_reason_ko(
                query=effective_query,
                game_name=item.get("name", ""),
                genres=item.get("genres", []),
                reviews=item.get("evidence_reviews", []),
            )
            summaries = list(combined.get("summaries", []))
            reason = str(combined.get("reason", "")).strip()
            # If LLM emits a contradictory reason, drop this candidate entirely.
            if reason and _reason_has_contradiction(reason, parsed):
                continue
            item["evidence_summaries_ko"] = summaries
            item["reason_ko"] = reason
            one_liner = llm.generate_one_liner_ko(
                query=effective_query,
                game_name=item.get("name", ""),
                reason_ko=reason,
                caution_notes=list(item.get("caution_notes", []) or []),
            )
            item["one_liner_ko"] = one_liner
            final_results.append(item)
            if len(final_results) >= top_k:
                break
        if len(final_results) < top_k:
            used = {int(x.get("app_id", -1)) for x in final_results}
            for item in diverse:
                app_id = int(item.get("app_id", -1))
                if app_id in used:
                    continue
                final_results.append(item)
                used.add(app_id)
                if len(final_results) >= top_k:
                    break
    else:
        final_results = diverse[:top_k]

    for item in final_results:
        item.pop("_vector", None)
        item.pop("final_score", None)

    return {
        "query": original_query,
        "normalized_input_query": query,
        "rewritten_query": rewritten_query,
        "effective_query": effective_query,
        "mode": mode,
        "reference_game": reference_game,
        "parsed_query": parsed.to_dict(),
        "results": final_results,
        "llm_errors": ((llm.errors if llm is not None else []) + runtime_errors),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

