from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_APP_IDS = [
    730,  # Counter-Strike 2
    570,  # Dota 2
    271590,  # GTA V
    292030,  # Witcher 3
    1091500,  # Cyberpunk 2077
]


@dataclass(frozen=True)
class Settings:
    db_path: Path
    app_ids: list[int]
    openai_api_key: str | None
    openai_model: str


def _parse_app_ids(raw: str | None) -> list[int]:
    if not raw:
        return DEFAULT_APP_IDS

    app_ids: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid app id: {token}")
        app_ids.append(int(token))
    return app_ids or DEFAULT_APP_IDS


def load_settings() -> Settings:
    _load_env_file(Path(".env"))
    db_raw = os.getenv("STEAM_DB_PATH", "data/steam_mvp.db")
    app_ids_raw = os.getenv("STEAM_APP_IDS")
    openai_api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPEN_API_KEY")
        or os.getenv("OPEN_API")
        or os.getenv("OEPN_API")
        or None
    )
    # Default model pinned to gpt-4.1-mini unless user explicitly overrides.
    openai_model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    db_path = Path(db_raw)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        db_path=db_path,
        app_ids=_parse_app_ids(app_ids_raw),
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        # If an env var exists but is empty, allow .env value to fill it.
        if key and (key not in os.environ or not (os.environ.get(key) or "").strip()):
            os.environ[key] = value
