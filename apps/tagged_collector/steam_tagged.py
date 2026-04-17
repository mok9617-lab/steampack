from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
STEAMSPY_URL = "https://steamspy.com/api.php"
STEAM_SEARCH_RESULTS_URL = "https://store.steampowered.com/search/results/"
APP_ID_RE = re.compile(r'data-ds-appid="(\d+)"')


def _http_get_json(url: str, params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    full_url = f"{url}?{urlencode(params)}"
    backoff = 1.0
    for attempt in range(retries):
        try:
            with urlopen(full_url, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
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


def _extract_app_ids(payload: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for key in payload.keys():
        if str(key).isdigit():
            out.append(int(key))
    return out


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


def _fetch_app_metadata(app_id: int, tags: list[str]) -> dict[str, Any] | None:
    try:
        data = _http_get_json(APP_DETAILS_URL, {"appids": app_id, "l": "english"})
    except Exception:
        return None
    item = data.get(str(app_id), {})
    if not item.get("success"):
        return None
    payload = item.get("data") or {}
    if payload.get("type") != "game":
        return None
    genres = [
        g.get("description")
        for g in (payload.get("genres") or [])
        if isinstance(g, dict) and g.get("description")
    ]
    return {
        "app_id": app_id,
        "name": payload.get("name") or f"app_{app_id}",
        "release_date": (payload.get("release_date") or {}).get("date"),
        "genres": genres,
        "tags": tags,
    }


def _fetch_candidate_app_ids(max_candidates: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []

    for request_name in ("top100in2weeks", "top100forever", "top100owned"):
        try:
            payload = _http_get_json(STEAMSPY_URL, {"request": request_name})
        except Exception:
            continue
        for app_id in _extract_app_ids(payload):
            if app_id in seen:
                continue
            seen.add(app_id)
            out.append(app_id)
            if len(out) >= max_candidates:
                return out

    sort_modes = ["Released_DESC", "Reviews_DESC", "Name_ASC"]
    for sort_by in sort_modes:
        start = 0
        while len(out) < max_candidates and start < 50000:
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
                if len(out) >= max_candidates:
                    return out

            start += 50
            time.sleep(0.2)

    return out


def _has_any_tag(tags: list[str], required_tags: list[str]) -> bool:
    tag_set = {x.lower() for x in tags}
    return any(target.lower() in tag_set for target in required_tags)


def collect_games_by_tags(
    required_tags: list[str],
    limit: int,
    max_candidates: int,
    sleep_seconds: float = 0.15,
) -> list[dict[str, Any]]:
    candidates = _fetch_candidate_app_ids(max_candidates=max_candidates)
    matched: list[dict[str, Any]] = []

    for idx, app_id in enumerate(candidates, start=1):
        if len(matched) >= limit:
            break

        tags = _fetch_steamspy_tags(app_id)
        if not tags or not _has_any_tag(tags, required_tags):
            continue

        game = _fetch_app_metadata(app_id, tags=tags)
        if game is None:
            continue

        matched.append(game)
        print(
            f"[{len(matched)}/{limit}] matched: {game['name']} (app_id={app_id}) "
            f"- scanned {idx}/{len(candidates)}"
        )
        time.sleep(sleep_seconds)

    return matched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Steam games by specific SteamSpy tag(s)."
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=None,
        help='Required tag (repeatable). Default: "Sexual Content"',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of games to collect (default: 100)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=30000,
        help="Maximum number of candidate app IDs to scan (default: 30000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tagged/sexual_content_games.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--appids-output",
        type=str,
        default="data/tagged/sexual_content_app_ids.txt",
        help="Output TXT path for app IDs only",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raw_tags = args.tags or ["Sexual Content"]
    required_tags = [t.strip() for t in raw_tags if t and t.strip()]
    if not required_tags:
        parser.error("At least one non-empty --tag is required.")
    if args.limit <= 0:
        parser.error("--limit must be > 0")
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be > 0")

    games = collect_games_by_tags(
        required_tags=required_tags,
        limit=args.limit,
        max_candidates=args.max_candidates,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "required_tags": required_tags,
                "requested_limit": args.limit,
                "collected_count": len(games),
                "games": games,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    appids_path = Path(args.appids_output)
    appids_path.parent.mkdir(parents=True, exist_ok=True)
    appids_path.write_text(
        "\n".join(str(game["app_id"]) for game in games),
        encoding="utf-8",
    )

    print(f"Collected games: {len(games)} / requested {args.limit}")
    print(f"Saved JSON: {output_path}")
    print(f"Saved app IDs: {appids_path}")


if __name__ == "__main__":
    main()
