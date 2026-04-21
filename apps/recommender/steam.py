from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from apps.recommender.src.collector import collect_reviews_for_app_ids
    from apps.recommender.src.chroma_store import sync_game_profiles_to_chroma
    from apps.recommender.src.config import load_settings
    from apps.recommender.src.db import init_db
    from apps.recommender.src.evaluator import run_evaluation, save_report
    from apps.recommender.src.features import build_review_and_game_embeddings
    from apps.recommender.src.network_diag import print_network_diagnosis
    from apps.recommender.src.preprocess import preprocess_reviews
    from apps.recommender.src.ranker import recommend_games
    from apps.recommender.src.web_ui import run_server
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from apps.recommender.src.collector import collect_reviews_for_app_ids
    from apps.recommender.src.chroma_store import sync_game_profiles_to_chroma
    from apps.recommender.src.config import load_settings
    from apps.recommender.src.db import init_db
    from apps.recommender.src.evaluator import run_evaluation, save_report
    from apps.recommender.src.features import build_review_and_game_embeddings
    from apps.recommender.src.network_diag import print_network_diagnosis
    from apps.recommender.src.preprocess import preprocess_reviews
    from apps.recommender.src.ranker import recommend_games
    from apps.recommender.src.web_ui import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Steam review-based recommendation MVP pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Create SQLite tables")

    collect = sub.add_parser("collect", help="Collect recent Steam reviews")
    collect.add_argument(
        "--days",
        type=int,
        default=365,
        help="How many recent days to collect (default: 365)",
    )
    collect.add_argument(
        "--max-reviews-per-game",
        type=int,
        default=500,
        help="Maximum reviews to fetch per game (default: 500)",
    )
    collect.add_argument(
        "--auto-top-games",
        type=int,
        default=0,
        help="If >0, auto-select top N games from SteamSpy",
    )
    collect.add_argument(
        "--language",
        type=str,
        default="all",
        choices=["all", "koreana", "english"],
        help="Steam review language filter",
    )
    collect.add_argument(
        "--min-raw-reviews-per-game",
        type=int,
        default=0,
        help="Skip game if fetched raw review count is below this threshold",
    )

    preprocess = sub.add_parser("preprocess", help="Clean and label collected reviews")
    preprocess.add_argument(
        "--min-tokens",
        type=int,
        default=5,
        help="Minimum token length threshold after cleaning",
    )

    embed = sub.add_parser("embed", help="Build review/game embeddings")
    embed.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    embed.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size",
    )

    recommend = sub.add_parser("recommend", help="Recommend games from user query")
    recommend.add_argument("--query", type=str, required=True, help="Korean user query")
    recommend.add_argument("--top-k", type=int, default=5, help="Top K recommendations")
    recommend.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    webui = sub.add_parser("webui", help="Run local web UI for recommendation testing")
    webui.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    webui.add_argument("--port", type=int, default=8000, help="Bind port")
    webui.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    reactflow_ui = sub.add_parser(
        "reactflow-ui",
        help="Run ReactFlow dashboard UI (FastAPI + React)",
    )
    reactflow_ui.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    reactflow_ui.add_argument("--port", type=int, default=8010, help="Bind port")
    evaluate = sub.add_parser("evaluate", help="Run automatic evaluation loop")
    evaluate.add_argument("--top-k", type=int, default=5, help="Top K for recommendation eval")
    evaluate.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    evaluate.add_argument(
        "--output",
        type=str,
        default="data/eval_report.json",
        help="Path to save evaluation report JSON",
    )
    sub.add_parser("diagnose-network", help="Check OpenAI/HuggingFace network connectivity")
    chroma_sync = sub.add_parser("chroma-sync", help="Sync game profile vectors into Chroma DB")
    chroma_sync.add_argument(
        "--collection-name",
        type=str,
        default="steam_game_profiles",
        help="Chroma collection name",
    )
    chroma_sync.add_argument(
        "--chroma-path",
        type=str,
        default="",
        help="Chroma persistence directory (default: <db_dir>/chroma or CHROMA_PATH)",
    )
    sub.add_parser("reset-db", help="Delete all data in DB tables")

    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings()

    if args.command == "init-db":
        init_db(settings.db_path)
        print(f"DB initialized: {settings.db_path}")
        return

    if args.command == "collect":
        app_ids = settings.app_ids
        if args.auto_top_games and args.auto_top_games > 0:
            from apps.recommender.src.collector import fetch_top_games_from_steamspy

            app_ids = fetch_top_games_from_steamspy(limit=args.auto_top_games)
            print(f"Auto-selected games from SteamSpy: {len(app_ids)}")

        count = collect_reviews_for_app_ids(
            db_path=settings.db_path,
            app_ids=app_ids,
            days=args.days,
            max_reviews_per_game=args.max_reviews_per_game,
            language=args.language,
            min_raw_reviews_per_game=args.min_raw_reviews_per_game,
        )
        print(f"Collected/updated reviews: {count}")
        return

    if args.command == "preprocess":
        updated = preprocess_reviews(
            db_path=settings.db_path,
            min_tokens=args.min_tokens,
        )
        print(f"Preprocessed reviews: {updated}")
        return

    if args.command == "embed":
        result = build_review_and_game_embeddings(
            db_path=settings.db_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
        print(
            "Embedded reviews: {review_count}, game profiles: {profile_count}".format(
                review_count=result["review_count"],
                profile_count=result["profile_count"],
            )
        )
        return

    if args.command == "recommend":
        result = recommend_games(
            db_path=settings.db_path,
            query=args.query,
            top_k=args.top_k,
            model_name=args.model_name,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "webui":
        run_server(
            db_path=settings.db_path,
            host=args.host,
            port=args.port,
            model_name=args.model_name,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )
        return

    if args.command == "reactflow-ui":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "uvicorn is required for reactflow-ui. Install with: pip install uvicorn fastapi"
            ) from exc

        uvicorn.run(
            "apps.recommender.reactflow_server:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
        return

    if args.command == "evaluate":
        report = run_evaluation(
            db_path=settings.db_path,
            model_name=args.model_name,
            top_k=args.top_k,
        )
        save_report(report, Path(args.output))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Saved evaluation report: {args.output}")
        return

    if args.command == "reset-db":
        from apps.recommender.src.db import reset_db_data

        reset_db_data(settings.db_path)
        print(f"DB data reset: {settings.db_path}")
        return

    if args.command == "diagnose-network":
        print_network_diagnosis()
        return

    if args.command == "chroma-sync":
        result = sync_game_profiles_to_chroma(
            db_path=settings.db_path,
            collection_name=args.collection_name or None,
            chroma_path=(args.chroma_path or None),
        )
        print(
            "Chroma synced profiles: {count} | collection: {collection} | path: {chroma_path}".format(
                count=result["count"],
                collection=result["collection"],
                chroma_path=result["chroma_path"],
            )
        )
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
