from __future__ import annotations

import re
from functools import lru_cache


CONFIDENCE_KO = {
    "high": "높음",
    "medium": "보통",
    "low": "낮음",
}

GENRE_KO = {
    "Action": "액션",
    "Adventure": "어드벤처",
    "Massively Multiplayer": "대규모 멀티플레이",
    "RPG": "RPG",
    "Strategy": "전략",
    "Free To Play": "무료 플레이",
    "Casual": "캐주얼",
    "Indie": "인디",
    "Simulation": "시뮬레이션",
    "Sports": "스포츠",
    "Racing": "레이싱",
    "Survival": "생존",
    "FPS": "FPS",
    "Horror": "호러",
}

_EN_RE = re.compile(r"[A-Za-z]")


def confidence_to_ko(label: str) -> str:
    return CONFIDENCE_KO.get((label or "").lower(), label or "정보 없음")


def genre_to_ko(genre: str) -> str:
    return GENRE_KO.get(genre, genre)


def _looks_english(text: str) -> bool:
    return bool(_EN_RE.search(text or ""))


@lru_cache(maxsize=1)
def _load_translator():
    # Lazy-load translator only when needed.
    from transformers import pipeline

    return pipeline("translation_en_to_ko", model="Helsinki-NLP/opus-mt-en-ko")


def translate_en_to_ko(text: str) -> str:
    if not text:
        return text
    if not _looks_english(text):
        return text
    try:
        translator = _load_translator()
        out = translator(text, max_length=512)
        if isinstance(out, list) and out:
            return out[0].get("translation_text", text)
    except Exception:
        return text
    return text
