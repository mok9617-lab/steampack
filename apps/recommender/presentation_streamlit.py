from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import streamlit as st

try:
    from apps.recommender.src.config import load_settings
    from apps.recommender.src.ranker import recommend_games
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from apps.recommender.src.config import load_settings
    from apps.recommender.src.ranker import recommend_games


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

METRIC_RULES = {
    "pass_rate": {"label": "Parser Pass Rate", "higher_is_better": True},
    "nonempty_rate": {"label": "Non-empty Rate", "higher_is_better": True},
    "preferred_genre_hit_rate": {"label": "Genre Hit Rate", "higher_is_better": True},
    "hard_constraint_violation_rate": {
        "label": "Hard Violation Rate",
        "higher_is_better": False,
    },
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


def _load_eval_history(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
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
                "file": path.name,
                "generated_at": str(payload.get("generated_at", "")),
                "pass_rate": float(pm.get("pass_rate", 0.0)),
                "nonempty_rate": float(rm.get("nonempty_rate", 0.0)),
                "preferred_genre_hit_rate": float(rm.get("preferred_genre_hit_rate", 0.0)),
                "hard_constraint_violation_rate": float(
                    rm.get("hard_constraint_violation_rate", 0.0)
                ),
                "parse_failures_count": len(payload.get("parse_failures") or []),
                "recommendation_failures_count": len(payload.get("recommendation_failures") or []),
                "recommendation_failure_queries": [
                    str(x.get("query", "")) for x in (payload.get("recommendation_failures") or [])
                ],
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


def _build_korean_change_lines(prev: dict, cur: dict) -> tuple[list[str], list[str]]:
    improved: list[str] = []
    caution: list[str] = []
    metric_keys = [
        "pass_rate",
        "nonempty_rate",
        "preferred_genre_hit_rate",
        "hard_constraint_violation_rate",
    ]
    for k in metric_keys:
        delta = float(cur[k]) - float(prev[k])
        if abs(delta) < 1e-9:
            continue
        label = str(METRIC_RULES[k]["label"])
        line = f"{label}가 {prev[k]:.4f}에서 {cur[k]:.4f}로 변했습니다."
        status = _change_label(delta, bool(METRIC_RULES[k]["higher_is_better"]))
        if status == "improved":
            improved.append(line)
        else:
            caution.append(line)

    parse_delta = cur["parse_failures_count"] - prev["parse_failures_count"]
    if parse_delta < 0:
        improved.append(
            f"파싱 실패 건수가 {prev['parse_failures_count']}건에서 {cur['parse_failures_count']}건으로 감소했습니다."
        )
    elif parse_delta > 0:
        caution.append(
            f"파싱 실패 건수가 {prev['parse_failures_count']}건에서 {cur['parse_failures_count']}건으로 증가했습니다."
        )

    rec_delta = cur["recommendation_failures_count"] - prev["recommendation_failures_count"]
    if rec_delta < 0:
        improved.append(
            f"추천 실패 건수가 {prev['recommendation_failures_count']}건에서 {cur['recommendation_failures_count']}건으로 감소했습니다."
        )
    elif rec_delta > 0:
        caution.append(
            f"추천 실패 건수가 {prev['recommendation_failures_count']}건에서 {cur['recommendation_failures_count']}건으로 증가했습니다."
        )
    return improved, caution


def _render_korean_version_story(history: list[dict]) -> None:
    st.subheader("버전별 변경 요약 (한국어)")
    shown = 0
    for i in range(1, len(history)):
        prev = history[i - 1]
        cur = history[i]
        improved, caution = _build_korean_change_lines(prev, cur)
        if not improved and not caution:
            continue
        shown += 1
        with st.container(border=True):
            st.markdown(f"**v{prev['version']} -> v{cur['version']}**")
            if improved:
                st.markdown("개선된 점")
                for line in improved:
                    st.write(f"- {line}")
            if caution:
                st.markdown("주의할 점")
                for line in caution:
                    st.write(f"- {line}")
    if shown == 0:
        st.info("버전 간 변경점이 없어 요약할 내용이 없습니다.")


def _infer_change_causes(prev: dict, cur: dict) -> list[str]:
    causes: list[str] = []
    pass_delta = float(cur["pass_rate"]) - float(prev["pass_rate"])
    nonempty_delta = float(cur["nonempty_rate"]) - float(prev["nonempty_rate"])
    genre_delta = float(cur["preferred_genre_hit_rate"]) - float(prev["preferred_genre_hit_rate"])
    hard_delta = float(cur["hard_constraint_violation_rate"]) - float(prev["hard_constraint_violation_rate"])

    if pass_delta > 0:
        causes.append("입력 정규화/의도 파싱 규칙(또는 LLM 파싱 폴백)을 보강해 파서 안정성이 개선된 것으로 보입니다.")
    if nonempty_delta > 0 and hard_delta >= 0:
        causes.append("후보 생성 범위 또는 폴백 경로를 넓혀 빈 결과를 줄인 것으로 해석됩니다.")
    if hard_delta < 0:
        causes.append("하드 필터(제외 조건, must-have) 적용 순서/조건을 강화해 제약 위반이 감소한 것으로 보입니다.")
    if hard_delta > 0:
        causes.append("결과 반환률을 높이는 과정에서 하드 필터 강도가 상대적으로 완화되어 제약 위반이 늘어난 것으로 보입니다.")
    if genre_delta > 0:
        causes.append("질의-장르 정합 재랭킹(정렬 가중치 또는 쿼리 정렬 점수)이 개선된 것으로 보입니다.")
    if nonempty_delta < 0 and hard_delta <= 0:
        causes.append("필터 강화를 적용하면서 일부 질의에서 후보가 과도하게 제거되어 결과 수가 줄어든 것으로 보입니다.")

    prev_fail = set(prev.get("recommendation_failure_queries") or [])
    cur_fail = set(cur.get("recommendation_failure_queries") or [])
    fixed = sorted([q for q in prev_fail - cur_fail if q])
    new = sorted([q for q in cur_fail - prev_fail if q])
    if fixed:
        causes.append(f"이전 실패 질의 일부가 해결되었습니다: {', '.join(fixed)}")
    if new:
        causes.append(f"신규 실패 질의가 발생했습니다: {', '.join(new)}")

    if not causes:
        causes.append("지표 변화가 거의 없어 내부 튜닝 영향이 제한적이거나 안정화 단계로 해석됩니다.")
    return causes


def _render_version_cause_story(history: list[dict]) -> None:
    st.subheader("버전별 변경 이유 (발표용 설명)")
    st.caption("주의: 코드 커밋 로그가 없어, 아래는 지표/실패 케이스 변화에 기반한 근거 추론입니다.")
    shown = 0
    for i in range(1, len(history)):
        prev = history[i - 1]
        cur = history[i]
        improved, caution = _build_korean_change_lines(prev, cur)
        if not improved and not caution:
            continue
        shown += 1
        causes = _infer_change_causes(prev, cur)
        with st.container(border=True):
            st.markdown(f"**v{prev['version']} -> v{cur['version']}**")
            for line in causes:
                st.write(f"- {line}")
    if shown == 0:
        st.info("변경점이 없어 원인 설명 섹션도 생략됩니다.")


def _render_change_highlights(history: list[dict]) -> None:
    st.subheader("Version Changes (Only Changed Steps)")
    st.caption("vN -> vN+1 단위로 변경점만 표시하고, 변경 없는 버전 구간은 생략합니다.")

    metric_keys = [
        "pass_rate",
        "nonempty_rate",
        "preferred_genre_hit_rate",
        "hard_constraint_violation_rate",
    ]
    shown_blocks = 0
    for i in range(1, len(history)):
        prev = history[i - 1]
        cur = history[i]
        improved: list[str] = []
        regressed: list[str] = []

        for k in metric_keys:
            delta = float(cur[k]) - float(prev[k])
            if abs(delta) < 1e-9:
                continue
            label = str(METRIC_RULES[k]["label"])
            msg = f"{label}: {prev[k]:.4f} -> {cur[k]:.4f} ({delta:+.4f})"
            if _change_label(delta, bool(METRIC_RULES[k]["higher_is_better"])) == "improved":
                improved.append(msg)
            else:
                regressed.append(msg)

        parse_delta = cur["parse_failures_count"] - prev["parse_failures_count"]
        if parse_delta != 0:
            msg = (
                f"Parse failures: {prev['parse_failures_count']} -> "
                f"{cur['parse_failures_count']} ({parse_delta:+d})"
            )
            if parse_delta < 0:
                improved.append(msg)
            else:
                regressed.append(msg)

        rec_delta = cur["recommendation_failures_count"] - prev["recommendation_failures_count"]
        if rec_delta != 0:
            msg = (
                f"Recommendation failures: {prev['recommendation_failures_count']} -> "
                f"{cur['recommendation_failures_count']} ({rec_delta:+d})"
            )
            if rec_delta < 0:
                improved.append(msg)
            else:
                regressed.append(msg)

        if not improved and not regressed:
            continue

        shown_blocks += 1
        with st.container(border=True):
            st.markdown(f"**v{prev['version']} -> v{cur['version']}**")
            left, right = st.columns(2)
            with left:
                st.markdown("Improved")
                for x in improved:
                    st.write(f"- {x}")
            with right:
                st.markdown("Regressed / Caution")
                if regressed:
                    for x in regressed:
                        st.write(f"- {x}")
                else:
                    st.write("- None")

    if shown_blocks == 0:
        st.info("버전 간 실질적인 변화가 없습니다.")


def _render_improvement_section(data_dir: Path) -> None:
    history = _load_eval_history(data_dir)
    if not history:
        st.subheader("Model Improvement")
        st.warning("eval_report_v*.json files were not found.")
        return

    chart_rows = [
        {
            "v": r["version"],
            "pass_rate": r["pass_rate"],
            "nonempty_rate": r["nonempty_rate"],
            "genre_hit_rate": r["preferred_genre_hit_rate"],
            "hard_violation_rate": r["hard_constraint_violation_rate"],
        }
        for r in history
    ]
    st.line_chart(chart_rows, x="v")
    _render_change_highlights(history)
    _render_korean_version_story(history)
    _render_version_cause_story(history)


def _pill_row(title: str, values: list[str]) -> None:
    st.markdown(f"**{title}**")
    if not values:
        st.caption("None")
        return
    html = "".join(
        f"<span style='display:inline-block;padding:4px 10px;margin:2px 6px 2px 0;"
        f"border-radius:999px;background:#EAF2FF;color:#1E3A8A;font-size:12px;'>{v}</span>"
        for v in values
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_pipeline(mode: str, has_reference: bool, llm_enabled: bool) -> None:
    steps = [
        ("1. Query Input", "#E2E8F0"),
        ("2. Normalize", "#DBEAFE"),
        ("3. LLM Parse/Rewrite" if llm_enabled else "3. Rule Parse", "#FDE68A"),
        ("4. Similarity Search", "#BFDBFE"),
        ("5. Re-rank + Filter", "#C7D2FE"),
        ("6. Evidence & Reason", "#FBCFE8"),
        ("7. Final Top-K", "#BBF7D0"),
    ]
    if mode == "similar_to" and has_reference:
        steps[3] = ("4. Similar-To Vector Search", "#93C5FD")

    cols = st.columns(len(steps))
    for col, (label, color) in zip(cols, steps):
        col.markdown(
            (
                "<div style='padding:10px 8px;border-radius:10px;text-align:center;"
                "font-size:13px;font-weight:600;background:"
                f"{color};'>{label}</div>"
            ),
            unsafe_allow_html=True,
        )


def _render_mode_overview() -> None:
    st.subheader("Mode Overview")
    left, right = st.columns(2)

    with left:
        st.markdown("### Query Mode")
        st.caption("Default mode: interpret query text and retrieve candidates.")
        st.info("Reference vector: user query embedding")

    with right:
        st.markdown("### Similar-To Mode")
        st.caption("Conditional mode: enabled when reference game is resolved.")
        st.info("Reference vector: matched game's profile embedding")

    st.success("Rule: reference-game match success -> similar_to, otherwise -> query")


def _render_result_card(item: dict, rank: int) -> None:
    title = item.get("name", f"Game {rank}")
    genres = ", ".join(item.get("genres") or [])
    reason = item.get("reason_ko") or "(No generated reason)"
    one_liner = item.get("one_liner_ko") or ""
    confidence = item.get("confidence", "n/a")
    sim = item.get("similarity", 0.0)
    align = item.get("query_alignment_score", 0.0)
    st.markdown(f"### {rank}. {title}")
    st.caption(f"Confidence: {confidence} | Similarity: {sim} | Alignment: {align}")
    if genres:
        st.caption(f"Genres: {genres}")
    st.write(reason)
    if one_liner:
        st.info(f"One-liner: {one_liner}")
    ev = item.get("evidence_summaries_ko") or item.get("evidence_reviews") or []
    if ev:
        st.markdown("Evidence")
        for line in ev[:3]:
            st.write(f"- {line}")
    steam_url = item.get("steam_url")
    if steam_url:
        st.markdown(f"[Steam Page]({steam_url})")
    st.divider()


def main() -> None:
    st.set_page_config(
        page_title="Steam Recommender - Presentation",
        page_icon="🎮",
        layout="wide",
    )

    settings = load_settings()
    st.title("Steam Recommender Presentation Dashboard")
    st.caption("Visualization only: underlying recommendation logic is unchanged.")
    _render_mode_overview()
    _render_improvement_section(Path("data"))
    st.divider()
    st.subheader("Live Demo (Optional)")

    with st.sidebar:
        st.header("Run Settings")
        db_path_text = st.text_input("DB Path", value=str(settings.db_path))
        model_name = st.text_input("Embedding Model", value=DEFAULT_MODEL)
        top_k = st.slider("Top-K", min_value=3, max_value=10, value=5, step=1)
        llm_on = st.toggle("Use LLM", value=bool(settings.openai_api_key))
        st.caption("LLM OFF reduces natural-language reason generation only.")

    query = st.text_input(
        "Input Query",
        value="엘든링 같은 게임 추천해줘. 너무 공포스러운 건 빼고",
    )
    run = st.button("Run Recommendation", type="primary", use_container_width=True)

    db_path = Path(db_path_text)
    stats = _db_stats(db_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Games", f"{stats['games']:,}")
    c2.metric("Reviews", f"{stats['reviews']:,}")
    c3.metric("Game Profiles", f"{stats['profiles']:,}")

    if not run:
        st.info("Run Recommendation to visualize a live pipeline execution.")
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
        openai_model=settings.openai_model,
    )
    elapsed = time.perf_counter() - start

    mode = str(result.get("mode") or "query")
    ref = result.get("reference_game")
    llm_errors = result.get("llm_errors") or []
    rows = result.get("results") or []

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mode", mode)
    m2.metric("Latency (sec)", f"{elapsed:.2f}")
    m3.metric("Returned", len(rows))
    m4.metric("LLM Errors", len(llm_errors))

    st.subheader("Pipeline Flow")
    _render_pipeline(mode=mode, has_reference=bool(ref), llm_enabled=llm_on)

    if ref:
        st.success(
            f"Reference game matched: {ref.get('name')} "
            f"(app_id={ref.get('app_id')}, hint='{ref.get('hint')}')"
        )
    else:
        st.warning("Reference game not resolved, so query mode was used.")

    parsed = (result.get("parsed_query") or {}) if isinstance(result, dict) else {}
    st.subheader("Parsed Intent")
    _pill_row("Preferred Genres", list(parsed.get("preferred_genres") or []))
    _pill_row("Excluded", list(parsed.get("excluded_genres_or_moods") or []))
    _pill_row("Must Have", list(parsed.get("must_have") or []))
    _pill_row("Soft Prefs", list(parsed.get("soft_preferences") or []))

    st.subheader("Top Results")
    if not rows:
        st.error("No recommendation result.")
    else:
        for i, item in enumerate(rows, start=1):
            _render_result_card(item, i)

    with st.expander("Raw Output JSON"):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
