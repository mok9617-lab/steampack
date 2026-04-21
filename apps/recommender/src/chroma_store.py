from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .db import get_connection


def _deserialize_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _resolve_chroma_path(db_path: Path, chroma_path: str | None = None) -> Path:
    if chroma_path:
        return Path(chroma_path)
    env_path = (os.getenv("CHROMA_PATH") or "").strip()
    if env_path:
        return Path(env_path)
    return db_path.parent / "chroma"


def _resolve_collection_name(collection_name: str | None = None) -> str:
    if collection_name:
        return collection_name
    env_name = (os.getenv("CHROMA_COLLECTION") or "").strip()
    if env_name:
        return env_name
    return "steam_game_profiles"


def sync_game_profiles_to_chroma(
    db_path: Path,
    collection_name: str | None = None,
    chroma_path: str | None = None,
) -> dict[str, Any]:
    try:
        import chromadb
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "chromadb is required. Install with: pip install chromadb"
        ) from exc

    db_path = Path(db_path)
    persist_path = _resolve_chroma_path(db_path, chroma_path)
    persist_path.mkdir(parents=True, exist_ok=True)
    resolved_collection = _resolve_collection_name(collection_name)

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT g.app_id, g.name, g.genres, g.tags, p.profile_embedding,
                   p.positive_ratio_1y, p.recent_review_count, p.median_playtime_1y
            FROM game_profiles p
            JOIN games g ON g.app_id = p.app_id
            WHERE p.profile_embedding IS NOT NULL
            """
        ).fetchall()

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name=resolved_collection,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []
    documents: list[str] = []

    for row in rows:
        vec = _deserialize_vector(row["profile_embedding"]).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        ids.append(str(int(row["app_id"])))
        embeddings.append(vec.tolist())
        metadatas.append(
            {
                "app_id": int(row["app_id"]),
                "name": str(row["name"] or ""),
                "genres_json": str(row["genres"] or "[]"),
                "tags_json": str(row["tags"] or "[]"),
                "positive_ratio_1y": float(row["positive_ratio_1y"] or 0.0),
                "recent_review_count": int(row["recent_review_count"] or 0),
                "median_playtime_1y": float(row["median_playtime_1y"] or 0.0),
            }
        )
        documents.append(str(row["name"] or ""))

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    return {
        "count": len(ids),
        "collection": resolved_collection,
        "chroma_path": str(persist_path),
    }


def query_profiles_from_chroma(
    db_path: Path,
    query_vector: np.ndarray,
    n_results: int,
    collection_name: str | None = None,
    chroma_path: str | None = None,
) -> list[dict[str, Any]]:
    try:
        import chromadb
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "chromadb is required. Install with: pip install chromadb"
        ) from exc

    db_path = Path(db_path)
    persist_path = _resolve_chroma_path(db_path, chroma_path)
    resolved_collection = _resolve_collection_name(collection_name)

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name=resolved_collection,
        metadata={"hnsw:space": "cosine"},
    )

    q = np.asarray(query_vector, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm > 0:
        q = q / norm

    result = collection.query(
        query_embeddings=[q.tolist()],
        n_results=max(1, int(n_results)),
        include=["metadatas", "embeddings", "distances"],
    )

    metadatas = (result.get("metadatas") or [[]])[0]
    embeddings = (result.get("embeddings") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    out: list[dict[str, Any]] = []
    for idx, metadata in enumerate(metadatas):
        if not metadata:
            continue
        genres_raw = str(metadata.get("genres_json") or "[]")
        tags_raw = str(metadata.get("tags_json") or "[]")
        try:
            genres = json.loads(genres_raw)
        except Exception:
            genres = []
        try:
            tags = json.loads(tags_raw)
        except Exception:
            tags = []

        distance = float(distances[idx]) if idx < len(distances) else 1.0
        sim = max(-1.0, min(1.0, 1.0 - distance))
        vec = np.asarray(embeddings[idx], dtype=np.float32) if idx < len(embeddings) else None

        out.append(
            {
                "app_id": int(metadata.get("app_id") or 0),
                "name": str(metadata.get("name") or ""),
                "genres": genres if isinstance(genres, list) else [],
                "tags": tags if isinstance(tags, list) else [],
                "positive_ratio_1y": float(metadata.get("positive_ratio_1y") or 0.0),
                "recent_review_count": int(metadata.get("recent_review_count") or 0),
                "median_playtime_1y": float(metadata.get("median_playtime_1y") or 0.0),
                "similarity": float(sim),
                "vector": vec,
            }
        )

    return out
