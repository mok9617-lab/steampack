from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from apps.recommender.src.config import load_settings
    from apps.recommender.src.dashboard_formatter import (
        build_parsed_value_meaning_rows,
        build_pipeline_stages,
        build_result_rows,
        build_score_breakdown_rows,
    )
    from apps.recommender.src.ranker import recommend_games
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from apps.recommender.src.config import load_settings
    from apps.recommender.src.dashboard_formatter import (
        build_parsed_value_meaning_rows,
        build_pipeline_stages,
        build_result_rows,
        build_score_breakdown_rows,
    )
    from apps.recommender.src.ranker import recommend_games


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

METRIC_RULES = {
    "pass_rate": {"label": "Parser Pass Rate", "higher_is_better": True},
    "nonempty_rate": {"label": "Non-empty Rate", "higher_is_better": True},
    "preferred_genre_hit_rate": {"label": "Genre Hit Rate", "higher_is_better": True},
    "hard_constraint_violation_rate": {"label": "Hard Violation Rate", "higher_is_better": False},
}


def _db_stats(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {"games": 0, "reviews": 0, "profiles": 0}
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        games = int(cur.execute("SELECT COUNT(*) FROM games").fetchone()[0])
        reviews = int(cur.execute("SELECT COUNT(*) FROM reviews").fetchone()[0])
        profiles = int(cur.execute("SELECT COUNT(*) FROM game_profiles").fetchone()[0])
    return {"games": games, "reviews": reviews, "profiles": profiles}


def _load_eval_history(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("eval_report_v*.json")):
        try:
            version = int(path.stem.split("_v")[-1])
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pm = payload.get("parser_metrics") or {}
        rm = payload.get("recommendation_metrics") or {}
        rows.append(
            {
                "version": version,
                "generated_at": str(payload.get("generated_at", "")),
                "pass_rate": float(pm.get("pass_rate", 0.0)),
                "nonempty_rate": float(rm.get("nonempty_rate", 0.0)),
                "preferred_genre_hit_rate": float(rm.get("preferred_genre_hit_rate", 0.0)),
                "hard_constraint_violation_rate": float(rm.get("hard_constraint_violation_rate", 0.0)),
                "parse_failures_count": len(payload.get("parse_failures") or []),
                "recommendation_failures_count": len(payload.get("recommendation_failures") or []),
            }
        )
    rows.sort(key=lambda x: x["version"])
    return rows


def _change_label(delta: float, higher_is_better: bool) -> str:
    if abs(delta) < 1e-9:
        return "unchanged"
    if higher_is_better:
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def _render_eval_panel(data_dir: Path) -> None:
    history = _load_eval_history(data_dir)
    st.subheader("Model Improvement History")
    if not history:
        st.info("No eval_report_v*.json found.")
        return

    chart_rows = [
        {
            "version": row["version"],
            "pass_rate": row["pass_rate"],
            "nonempty_rate": row["nonempty_rate"],
            "genre_hit_rate": row["preferred_genre_hit_rate"],
            "hard_violation_rate": row["hard_constraint_violation_rate"],
        }
        for row in history
    ]
    st.line_chart(chart_rows, x="version")

    blocks: list[dict[str, Any]] = []
    for i in range(1, len(history)):
        prev = history[i - 1]
        cur = history[i]
        improved: list[str] = []
        cautions: list[str] = []

        for key, rule in METRIC_RULES.items():
            delta = float(cur[key]) - float(prev[key])
            if abs(delta) < 1e-9:
                continue
            msg = f"{rule['label']}: {prev[key]:.4f} -> {cur[key]:.4f} ({delta:+.4f})"
            if _change_label(delta, bool(rule["higher_is_better"])) == "improved":
                improved.append(msg)
            else:
                cautions.append(msg)

        pf_delta = int(cur["parse_failures_count"]) - int(prev["parse_failures_count"])
        if pf_delta != 0:
            msg = (
                f"Parse failures: {prev['parse_failures_count']} -> "
                f"{cur['parse_failures_count']} ({pf_delta:+d})"
            )
            (improved if pf_delta < 0 else cautions).append(msg)

        rf_delta = int(cur["recommendation_failures_count"]) - int(
            prev["recommendation_failures_count"]
        )
        if rf_delta != 0:
            msg = (
                f"Recommendation failures: {prev['recommendation_failures_count']} -> "
                f"{cur['recommendation_failures_count']} ({rf_delta:+d})"
            )
            (improved if rf_delta < 0 else cautions).append(msg)

        if improved or cautions:
            blocks.append(
                {
                    "range": f"v{prev['version']} -> v{cur['version']}",
                    "improved": improved,
                    "cautions": cautions,
                }
            )

    if not blocks:
        st.info("No significant metric changes between versions.")
        return

    for block in blocks:
        with st.container(border=True):
            st.markdown(f"**{block['range']}**")
            left, right = st.columns(2)
            with left:
                st.markdown("Improved")
                if block["improved"]:
                    for x in block["improved"]:
                        st.write(f"- {x}")
                else:
                    st.write("- None")
            with right:
                st.markdown("Caution")
                if block["cautions"]:
                    for x in block["cautions"]:
                        st.write(f"- {x}")
                else:
                    st.write("- None")


def _render_pipeline_overview() -> None:
    st.subheader("Pipeline Overview")
    st.markdown(
        """
1. Input normalize and query rewrite  
2. Query parsing (rule + optional LLM parse)  
3. Mode decision (`query`, `similar_to`, `similar_to_llm_profile_fallback`)  
4. Candidate retrieval (embedding similarity)  
5. Re-rank and filtering  
6. Evidence summary and reason generation (LLM optional)  
7. Final top-k output
        """
    )
    with st.expander("Where LLM is used", expanded=False):
        st.write("- rewrite_and_parse_query")
        st.write("- parse_query (fallback)")
        st.write("- guess_game_titles (similar_to alias hints)")
        st.write("- infer_game_profile_for_similarity (new fallback when ref game is not in DB)")
        st.write("- summarize_and_reason_ko")
        st.write("- generate_one_liner_ko")


def _render_live_result(result: dict[str, Any], elapsed: float, query: str, top_k: int, model: str, openai_model: str, llm_on: bool) -> None:
    mode = str(result.get("mode") or "query")
    rows = result.get("results") or []
    llm_errors = result.get("llm_errors") or []
    ref = result.get("reference_game")
    fallback = result.get("similar_to_fallback")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mode", mode)
    m2.metric("Latency (sec)", f"{elapsed:.2f}")
    m3.metric("Returned", len(rows))
    m4.metric("LLM Errors", len(llm_errors))

    if ref:
        st.success(
            f"Reference game resolved: {ref.get('name')} (app_id={ref.get('app_id')}, hint={ref.get('hint')})"
        )
    elif fallback:
        st.warning(
            "Reference game was not resolved in DB, "
            "so LLM profile fallback was used to build an alternate query."
        )
        st.json(fallback)
    else:
        st.info("No reference-game path was used. Standard query mode executed.")

    stages = build_pipeline_stages(
        result,
        input_query=query,
        top_k=top_k,
        model_name=model,
        openai_model=openai_model,
        has_openai_key=llm_on,
    )
    with st.expander("Stage Cards (Presentation Detail)", expanded=True):
        for stage in stages:
            with st.container(border=True):
                st.markdown(f"**{stage.name}**")
                st.caption(stage.goal)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("INPUT")
                    st.json(stage.input_data)
                with c2:
                    st.markdown("PROCESS")
                    st.json(stage.process_data)
                with c3:
                    st.markdown("OUTPUT")
                    st.json(stage.output_data)

    st.subheader("Parsed Intent")
    parsed_rows = build_parsed_value_meaning_rows(result)
    if parsed_rows:
        st.dataframe(parsed_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No parsed intent values.")

    st.subheader("Top Results")
    result_rows = build_result_rows(result)
    if result_rows:
        st.dataframe(result_rows, use_container_width=True, hide_index=True)
        for i, item in enumerate(rows, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {item.get('name', 'Unknown')}**")
                st.caption(
                    f"confidence={item.get('confidence')} | "
                    f"similarity={item.get('similarity')} | "
                    f"query_alignment_score={item.get('query_alignment_score')}"
                )
                st.markdown("Score Breakdown")
                st.dataframe(
                    build_score_breakdown_rows(item),
                    use_container_width=True,
                    hide_index=True,
                )
                reason_ko = (item.get("reason_ko") or "").strip()
                one_liner = (item.get("one_liner_ko") or "").strip()
                st.markdown("Reason")
                st.write(reason_ko if reason_ko else "N/A")
                st.markdown("One-liner")
                st.write(one_liner if one_liner else "N/A")
    else:
        st.warning("No recommendation result.")

    with st.expander("LLM/Runtime Logs", expanded=False):
        if llm_errors:
            for err in llm_errors:
                st.write(f"- {err}")
        else:
            st.write("None")

    with st.expander("Raw Output JSON", expanded=False):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def main() -> None:
    st.set_page_config(
        page_title="Steam Recommender Presentation",
        page_icon="SR",
        layout="wide",
    )
    settings = load_settings()

    st.title("Steam Recommender Presentation Dashboard")
    st.caption("Presentation-first view. Recommendation logic remains unchanged.")

    with st.sidebar:
        st.header("Run Settings")
        db_path_text = st.text_input("DB Path", value=str(settings.db_path))
        model_name = st.text_input("Embedding Model", value=DEFAULT_MODEL)
        top_k = st.slider("Top-K", min_value=3, max_value=10, value=5, step=1)
        llm_on = st.toggle("Use LLM", value=bool(settings.openai_api_key))
        openai_model = st.text_input("OpenAI Model", value=settings.openai_model)
        st.caption("TODO(connect): Keep this UI thin and route all logic through recommend_games.")

    db_path = Path(db_path_text)
    stats = _db_stats(db_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Games", f"{stats['games']:,}")
    c2.metric("Reviews", f"{stats['reviews']:,}")
    c3.metric("Game Profiles", f"{stats['profiles']:,}")

    tab1, tab2, tab3 = st.tabs(["Overview", "Improvement", "Live Demo"])

    with tab1:
        _render_pipeline_overview()
        st.subheader("Mode Rule")
        st.info(
            "If reference game is resolved in DB -> similar_to mode. "
            "If not resolved and '~same as' hint exists -> similar_to_llm_profile_fallback may run. "
            "Otherwise -> query mode."
        )

    with tab2:
        _render_eval_panel(Path("data"))

    with tab3:
        query = st.text_input(
            "Input Query",
            value="Recommend games similar to Hades, but avoid horror.",
        )
        run = st.button("Run Recommendation", type="primary", use_container_width=True)

        if not run:
            st.info("Run Recommendation to visualize one live execution.")
            return
        if not query.strip():
            st.error("Please enter a query.")
            return

        api_key = settings.openai_api_key if llm_on else None
        start = time.perf_counter()
        result = recommend_games(
            db_path=db_path,
            query=query,
            top_k=top_k,
            model_name=model_name.strip() or DEFAULT_MODEL,
            openai_api_key=api_key,
            openai_model=openai_model.strip() or settings.openai_model,
        )
        elapsed = time.perf_counter() - start

        _render_live_result(
            result=result,
            elapsed=elapsed,
            query=query,
            top_k=top_k,
            model=model_name.strip() or DEFAULT_MODEL,
            openai_model=openai_model.strip() or settings.openai_model,
            llm_on=llm_on,
        )


if __name__ == "__main__":
    main()
