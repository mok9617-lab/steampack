from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from .db import get_connection, init_db


APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
APP_REVIEWS_URL = "https://store.steampowered.com/appreviews/{app_id}"
STEAMSPY_URL = "https://steamspy.com/api.php"
STEAM_SEARCH_RESULTS_URL = "https://store.steampowered.com/search/results/"
APP_ID_RE = re.compile(r'data-ds-appid="(\d+)"')

_CANONICAL_GENRES = {
    "rpg": "RPG",
    "action": "Action",
    "adventure": "Adventure",
    "strategy": "Strategy",
    "simulation": "Simulation",
    "survival": "Survival",
    "fps": "FPS",
    "first-person shooter": "FPS",
    "first person shooter": "FPS",
    "horror": "Horror",
    "indie": "Indie",
    "free to play": "Free To Play",
    "f2p": "Free To Play",
    "casual": "Casual",
}


def _http_get_json(url: str, params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    full_url = f"{url}?{urlencode(params)}"
    backoff = 1.5
    for attempt in range(retries):
        try:
            with urlopen(full_url, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Steam may throttle with HTTP 429.
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 1.8
                continue
            raise
        except URLError:
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 1.6
                continue
            raise
    return {}


def _fetch_steamspy_tags(app_id: int) -> list[str]:
    try:
        payload = _http_get_json(STEAMSPY_URL, {"request": "appdetails", "appid": app_id})
    except Exception:
        return []
    tags = payload.get("tags")
    if isinstance(tags, dict):
        return [str(k).strip() for k in tags.keys() if str(k).strip()]
    if isinstance(tags, list):
        return [str(x).strip() for x in tags if str(x).strip()]
    return []


def _normalize_genres(genres: list[str], tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        v = x.strip()
        if not v:
            return
        key = v.lower()
        canonical = _CANONICAL_GENRES.get(key, v)
        if canonical.lower() in seen:
            return
        seen.add(canonical.lower())
        out.append(canonical)

    for g in genres:
        add(g)
    for t in tags:
        add(t)
    return out


def _fetch_app_metadata(app_id: int) -> dict[str, Any] | None:
    data = _http_get_json(APP_DETAILS_URL, {"appids": app_id, "l": "english"})
    item = data.get(str(app_id), {})
    if not item.get("success"):
        return None
    payload = item.get("data") or {}
    genres = [g.get("description") for g in payload.get("genres", []) if g.get("description")]
    tags = _fetch_steamspy_tags(app_id)
    normalized_genres = _normalize_genres(genres, tags)
    return {
        "app_id": app_id,
        "name": payload.get("name") or f"app_{app_id}",
        "release_date": (payload.get("release_date") or {}).get("date"),
        "genres": json.dumps(normalized_genres, ensure_ascii=False),
        "tags": json.dumps(tags, ensure_ascii=False),
    }


def _extract_app_ids(payload: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for key in payload.keys():
        if str(key).isdigit():
            ids.append(int(key))
    return ids


def _fetch_store_search_app_ids(limit: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    # Mix different sort views to reduce one-list bias.
    sort_modes = ["Released_DESC", "Reviews_DESC", "Name_ASC"]

    for sort_by in sort_modes:
        start = 0
        while len(out) < limit and start < 5000:
            try:
                payload = _http_get_json(
                    STEAM_SEARCH_RESULTS_URL,
                    {
                        "query": "",
                        "start": start,
                        "count": 50,
                        "dynamic_data": "",
                        "sort_by": sort_by,
                        "supportedlang": "english",
                        "infinite": 1,
                    },
                )
            except Exception:
                # Skip current sort stream when rate-limited or temporarily blocked.
                time.sleep(3.0)
                break
            html = str(payload.get("results_html") or "")
            if not html:
                break
            ids = [int(x) for x in APP_ID_RE.findall(html)]
            if not ids:
                break
            for app_id in ids:
                if app_id in seen:
                    continue
                seen.add(app_id)
                out.append(app_id)
                if len(out) >= limit:
                    return out[:limit]
            start += 50
            time.sleep(0.25)

    return out[:limit]


def fetch_top_games_from_steamspy(limit: int = 100) -> list[int]:
    # For <=100, recent top list is enough.
    if limit <= 100:
        payload = _http_get_json(STEAMSPY_URL, {"request": "top100in2weeks"})
        return _extract_app_ids(payload)[:limit]

    # For larger limits, aggregate multiple SteamSpy sources.
    seen: set[int] = set()
    merged: list[int] = []

    for request_name in ("top100in2weeks", "top100forever", "top100owned"):
        payload = _http_get_json(STEAMSPY_URL, {"request": request_name})
        for app_id in _extract_app_ids(payload):
            if app_id in seen:
                continue
            seen.add(app_id)
            merged.append(app_id)
            if len(merged) >= limit:
                return merged[:limit]

    # If still short, fill from Steam store search endpoint
    # (avoids SteamSpy paginated `all` instability / 500 errors).
    if len(merged) < limit:
        for app_id in _fetch_store_search_app_ids(limit=max(limit * 2, 2000)):
            if app_id in seen:
                continue
            seen.add(app_id)
            merged.append(app_id)
            if len(merged) >= limit:
                return merged[:limit]

    return merged[:limit]


def _fetch_reviews(
    app_id: int,
    min_timestamp: int,
    max_reviews: int,
    language: str = "all",
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    cursor = "*"

    while len(reviews) < max_reviews:
        payload = _http_get_json(
            APP_REVIEWS_URL.format(app_id=app_id),
            {
                "json": 1,
                "language": language,
                "filter": "recent",
                "num_per_page": 100,
                "cursor": cursor,
                "purchase_type": "all",
            },
        )
        batch = payload.get("reviews", [])
        if not batch:
            break

        stop = False
        for review in batch:
            created = int(review.get("timestamp_created", 0))
            if created < min_timestamp:
                stop = True
                break
            reviews.append(review)
            if len(reviews) >= max_reviews:
                break

        if stop:
            break

        new_cursor = payload.get("cursor")
        if not new_cursor or new_cursor == cursor:
            break
        cursor = new_cursor
        time.sleep(0.2)

    return reviews


def _upsert_game(db_path: Path, game: dict[str, Any]) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO games (app_id, name, release_date, genres, tags, positive_ratio, review_count, updated_at)
            VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
            ON CONFLICT(app_id) DO UPDATE SET
              name=excluded.name,
              release_date=excluded.release_date,
              genres=excluded.genres,
              tags=excluded.tags,
              updated_at=excluded.updated_at
            """,
            (
                game["app_id"],
                game["name"],
                game["release_date"],
                game["genres"],
                game["tags"],
                now_iso,
            ),
        )
        conn.commit()


def _upsert_reviews(db_path: Path, app_id: int, reviews: list[dict[str, Any]]) -> int:
    inserted = 0
    with get_connection(db_path) as conn:
        for review in reviews:
            language = (review.get("language", "") or "").lower()
            if language in {"korean", "koreana"}:
                normalized_language = "ko"
            elif language == "english":
                normalized_language = "en"
            else:
                continue

            review_id = str(review.get("recommendationid"))
            review_text = (review.get("review") or "").strip()
            if not review_id or not review_text:
                continue

            conn.execute(
                """
                INSERT INTO reviews (
                  review_id, app_id, language, review_text, cleaned_text, voted_up, votes_up,
                  weighted_vote_score, steam_purchase, received_for_free, playtime_forever,
                  playtime_at_review, review_date, sentiment_score, trust_label, trust_reasons, embedding
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
                ON CONFLICT(review_id) DO UPDATE SET
                  language=excluded.language,
                  review_text=excluded.review_text,
                  voted_up=excluded.voted_up,
                  votes_up=excluded.votes_up,
                  weighted_vote_score=excluded.weighted_vote_score,
                  steam_purchase=excluded.steam_purchase,
                  received_for_free=excluded.received_for_free,
                  playtime_forever=excluded.playtime_forever,
                  playtime_at_review=excluded.playtime_at_review,
                  review_date=excluded.review_date
                """,
                (
                    review_id,
                    app_id,
                    normalized_language,
                    review_text,
                    int(bool(review.get("voted_up"))),
                    int(review.get("votes_up", 0)),
                    float(review.get("weighted_vote_score", 0.0)),
                    int(bool(review.get("steam_purchase"))),
                    int(bool(review.get("received_for_free"))),
                    int((review.get("author") or {}).get("playtime_forever", 0)),
                    int((review.get("author") or {}).get("playtime_at_review", 0)),
                    datetime.fromtimestamp(
                        int(review.get("timestamp_created", 0)), tz=timezone.utc
                    ).isoformat(),
                ),
            )
            inserted += 1

        conn.commit()
    return inserted


def collect_reviews_for_app_ids(
    db_path: Path,
    app_ids: list[int],
    days: int = 365,
    max_reviews_per_game: int = 500,
    language: str = "all",
    min_raw_reviews_per_game: int = 0,
) -> int:
    init_db(db_path)
    min_timestamp = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    total = 0

    for app_id in app_ids:
        try:
            metadata = _fetch_app_metadata(app_id)
            if metadata is None:
                continue
            _upsert_game(db_path, metadata)
            reviews = _fetch_reviews(
                app_id,
                min_timestamp=min_timestamp,
                max_reviews=max_reviews_per_game,
                language=language,
            )
            if len(reviews) < min_raw_reviews_per_game:
                continue
            total += _upsert_reviews(db_path, app_id, reviews)
            time.sleep(0.3)
        except Exception:
            # Skip problematic apps and continue large-batch collection.
            continue

    return total
