from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .db import get_connection

TRUST_WEIGHTS = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.3,
}


def _serialize_vector(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    offline = os.getenv("HF_HUB_OFFLINE", "").strip() == "1" or os.getenv(
        "TRANSFORMERS_OFFLINE", ""
    ).strip() == "1"
    first_exc: Exception | None = None

    try:
        if offline:
            return SentenceTransformer(model_name, local_files_only=True)
        return SentenceTransformer(model_name)
    except Exception as exc:
        first_exc = exc

    # Retry once in offline mode for restricted-network environments.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as second_exc:
        raise RuntimeError(
            "Failed to load embedding model. "
            "Network may be blocked and local model cache is missing. "
            "Try once on a network-enabled environment to populate cache, "
            "or set HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1 with cached files."
        ) from (first_exc or second_exc)


def _batched(items: list[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_review_and_game_embeddings(
    db_path: Path,
    model_name: str,
    batch_size: int = 64,
) -> dict[str, int]:
    model = _load_sentence_transformer(model_name)

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT review_id, app_id, cleaned_text, trust_label, review_date, voted_up, playtime_forever
            FROM reviews
            WHERE cleaned_text IS NOT NULL AND cleaned_text != ''
            """
        ).fetchall()

        valid_rows = rows
        texts = [r["cleaned_text"] for r in valid_rows]

        review_count = 0
        if texts:
            for batch_rows in _batched(valid_rows, batch_size):
                batch_texts = [r["cleaned_text"] for r in batch_rows]
                emb = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                for row, vec in zip(batch_rows, emb):
                    conn.execute(
                        "UPDATE reviews SET embedding = ? WHERE review_id = ?",
                        (_serialize_vector(vec), row["review_id"]),
                    )
                    review_count += 1
            conn.commit()

        app_to_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
        app_to_dates: dict[int, list[datetime]] = defaultdict(list)
        app_to_votes: dict[int, list[int]] = defaultdict(list)
        app_to_playtime: dict[int, list[int]] = defaultdict(list)
        app_to_weights: dict[int, list[float]] = defaultdict(list)

        vec_rows = conn.execute(
            """
            SELECT app_id, embedding, review_date, voted_up, playtime_forever, trust_label
            FROM reviews
            WHERE embedding IS NOT NULL
            """
        ).fetchall()

        for row in vec_rows:
            app_id = int(row["app_id"])
            app_to_vectors[app_id].append(_deserialize_vector(row["embedding"]))
            app_to_votes[app_id].append(int(row["voted_up"] or 0))
            app_to_playtime[app_id].append(int(row["playtime_forever"] or 0))
            trust = str(row["trust_label"] or "").lower()
            app_to_weights[app_id].append(float(TRUST_WEIGHTS.get(trust, 0.3)))
            try:
                app_to_dates[app_id].append(datetime.fromisoformat(row["review_date"]))
            except Exception:
                pass

        profile_count = 0
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=365)
        for app_id, vectors in app_to_vectors.items():
            if not vectors:
                continue
            mat = np.vstack(vectors)
            weights = np.asarray(app_to_weights.get(app_id, []), dtype=np.float32)
            if weights.shape[0] != mat.shape[0] or float(weights.sum()) <= 0.0:
                profile = mat.mean(axis=0)
            else:
                w = weights / float(weights.sum())
                profile = (mat * w[:, None]).sum(axis=0)
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm

            dates = app_to_dates.get(app_id, [])
            recent_review_count = sum(1 for d in dates if d >= cutoff)
            votes = app_to_votes.get(app_id, [])
            positive_ratio = float(sum(votes) / len(votes)) if votes else 0.0
            playtimes = app_to_playtime.get(app_id, [])
            median_playtime = float(np.median(playtimes)) if playtimes else 0.0

            conn.execute(
                """
                INSERT INTO game_profiles (
                  app_id, profile_embedding, top_keywords, mood_tags,
                  recent_review_count, positive_ratio_1y, median_playtime_1y, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(app_id) DO UPDATE SET
                  profile_embedding=excluded.profile_embedding,
                  recent_review_count=excluded.recent_review_count,
                  positive_ratio_1y=excluded.positive_ratio_1y,
                  median_playtime_1y=excluded.median_playtime_1y,
                  updated_at=excluded.updated_at
                """,
                (
                    app_id,
                    _serialize_vector(profile),
                    json.dumps([], ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                    int(recent_review_count),
                    float(positive_ratio),
                    float(median_playtime),
                    now.isoformat(),
                ),
            )
            profile_count += 1
        conn.commit()

    return {"review_count": review_count, "profile_count": profile_count}
