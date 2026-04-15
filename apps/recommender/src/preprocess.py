from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .db import get_connection


URL_RE = re.compile(r"https?://\S+")
MULTISPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def _token_count(text: str) -> int:
    return len([t for t in text.split(" ") if t])


def _is_repetitive(text: str) -> bool:
    tokens = [t for t in text.lower().split(" ") if t]
    if len(tokens) < 6:
        return False
    top_count = Counter(tokens).most_common(1)[0][1]
    return (top_count / len(tokens)) > 0.5


def _trust_label(
    playtime_forever: int,
    steam_purchase: int,
    received_for_free: int,
    votes_up: int,
    token_count: int,
    repetitive: bool,
) -> tuple[str, str]:
    reasons: list[str] = []
    if playtime_forever < 30:
        reasons.append("very_low_playtime")
    elif playtime_forever >= 180:
        reasons.append("enough_playtime")

    if steam_purchase:
        reasons.append("steam_purchase")
    if received_for_free:
        reasons.append("received_for_free")
    if votes_up >= 5:
        reasons.append("community_helpful")
    if repetitive:
        reasons.append("repetitive_text")
    if token_count >= 20:
        reasons.append("rich_text")

    if repetitive or token_count < 5 or playtime_forever < 15:
        return "low", ",".join(reasons)

    if playtime_forever >= 120 and steam_purchase and not received_for_free:
        return "high", ",".join(reasons)

    return "medium", ",".join(reasons)


def preprocess_reviews(db_path: Path, min_tokens: int = 5) -> int:
    updated = 0
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT review_id, review_text, playtime_forever, steam_purchase,
                   received_for_free, votes_up
            FROM reviews
            """
        ).fetchall()
        updates: list[tuple[str, str, str]] = []

        for row in rows:
            cleaned = _clean_text(row["review_text"])
            token_count = _token_count(cleaned)
            repetitive = _is_repetitive(cleaned)

            if token_count < min_tokens:
                cleaned = ""
                label = "low"
                reasons = "too_short"
            else:
                label, reasons = _trust_label(
                    playtime_forever=int(row["playtime_forever"] or 0),
                    steam_purchase=int(row["steam_purchase"] or 0),
                    received_for_free=int(row["received_for_free"] or 0),
                    votes_up=int(row["votes_up"] or 0),
                    token_count=token_count,
                    repetitive=repetitive,
                )

            updates.append((cleaned, label, reasons, row["review_id"]))
            updated += 1

        if updates:
            conn.executemany(
                """
                UPDATE reviews
                SET cleaned_text = ?, trust_label = ?, trust_reasons = ?
                WHERE review_id = ?
                """,
                updates,
            )
        conn.commit()

    return updated
