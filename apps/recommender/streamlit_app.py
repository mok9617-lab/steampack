from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from apps.recommender.src.config import load_settings
    from apps.recommender.src.dashboard_formatter import (
        PipelineStage,
        build_evidence_rows,
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
        PipelineStage,
        build_evidence_rows,
        build_parsed_value_meaning_rows,
        build_pipeline_stages,
        build_result_rows,
        build_score_breakdown_rows,
    )
    from apps.recommender.src.ranker import recommend_games


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _inject_dark_theme() -> None:
    st.markdown(
        """
        <style>
          :root {
            --text-main: #f3f4f6;
            --text-sub: #d1d5db;
            --text-muted: #cbd5e1;
            --card-bg: #0f172a;
            --card-line: #334155;
            --accent: #93c5fd;
          }
          .stApp {
            background: radial-gradient(circle at 15% 0%, #1f2937 0%, transparent 35%),
                        radial-gradient(circle at 90% 100%, #111827 0%, transparent 40%),
                        #0b1220;
            color: var(--text-main);
            font-family: "Segoe UI", "Noto Sans KR", "Pretendard", sans-serif;
            font-size: 17px;
            line-height: 1.7;
          }
          .block-container {
            max-width: 1320px;
            padding-top: 1.2rem;
          }
          h1, h2, h3 {
            color: #ffffff !important;
            letter-spacing: 0.1px;
            line-height: 1.35;
          }
          p, li, label, .stMarkdown, .stCaption, .stAlert, .stInfo, .stWarning {
            color: var(--text-main) !important;
            font-size: 1rem !important;
            line-height: 1.7 !important;
          }
          .stCaption {
            color: var(--text-sub) !important;
            font-size: 0.94rem !important;
          }
          .stTextInput input, .stTextArea textarea {
            background: #0b1324 !important;
            color: var(--text-main) !important;
            border: 1px solid #334155 !important;
            font-size: 1rem !important;
          }
          .stButton button {
            font-size: 1rem !important;
            font-weight: 700 !important;
            padding: 0.6rem 1rem !important;
            border-radius: 10px !important;
          }
          [data-testid="stMetric"] {
            background: #101827;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 12px;
          }
          [data-testid="stMetricLabel"] {
            color: var(--text-sub) !important;
            font-size: 0.92rem !important;
          }
          [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.55rem !important;
            font-weight: 800 !important;
          }
          [data-testid="stDataFrame"] table {
            font-size: 0.98rem !important;
          }
          [data-testid="stDataFrame"] th {
            font-size: 0.98rem !important;
            color: #ffffff !important;
          }
          [data-testid="stDataFrame"] td {
            color: #f3f4f6 !important;
            line-height: 1.55 !important;
          }
          .stage-card {
            background: var(--card-bg);
            border: 1px solid var(--card-line);
            border-radius: 16px;
            padding: 16px 18px;
            margin: 10px 0 16px 0;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
          }
          .stage-title {
            font-weight: 800;
            font-size: 1.08rem;
            margin-bottom: 6px;
            color: #ffffff;
          }
          .stage-goal {
            color: var(--accent);
            font-size: 1rem;
            margin-bottom: 7px;
            font-weight: 600;
          }
          .stage-note {
            color: var(--text-muted);
            font-size: 0.97rem;
            margin-bottom: 12px;
          }
          .block-title {
            font-size: 0.93rem;
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: 0.3px;
          }
          .stExpander {
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _as_inline(value: Any) -> str:
    if value is None:
        return "없음"
    if isinstance(value, bool):
        return "예" if value else "아니오"
    text = str(value).strip()
    return text if text else "없음"


def _render_value(value: Any) -> None:
    # TODO(raw-object): dict/list를 사람이 읽는 형태로 펼쳐서 출력합니다.
    if isinstance(value, dict):
        if not value:
            st.write("해당 없음")
            return
        for k, v in value.items():
            st.write(f"- {k}: {_as_inline(v)}")
        return
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if not values:
            st.write("해당 없음")
            return
        for v in values:
            st.write(f"- {_as_inline(v)}")
        return
    st.write(_as_inline(value))


def _render_stage(stage: PipelineStage) -> None:
    st.markdown("<div class='stage-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='stage-title'>{stage.name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='stage-goal'>목적: {stage.goal}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='stage-note'>초심자 가이드: {stage.beginner_note}</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='block-title'>INPUT</div>", unsafe_allow_html=True)
        for k, v in stage.input_data.items():
            st.markdown(f"**{k}**")
            _render_value(v)
    with c2:
        st.markdown("<div class='block-title'>PROCESS</div>", unsafe_allow_html=True)
        for k, v in stage.process_data.items():
            st.markdown(f"**{k}**")
            _render_value(v)
    with c3:
        st.markdown("<div class='block-title'>OUTPUT</div>", unsafe_allow_html=True)
        for k, v in stage.output_data.items():
            st.markdown(f"**{k}**")
            _render_value(v)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_beginner_guide() -> None:
    with st.expander("초심자용 읽는 법", expanded=True):
        st.write("1) 단계는 위에서 아래로 순서대로 진행됩니다.")
        st.write("2) INPUT은 들어온 값, PROCESS는 처리 규칙, OUTPUT은 다음 단계로 넘기는 결과입니다.")
        st.write("3) 값이 없으면 억지로 만들지 않고 `없음/해당 없음/현재 파이프라인에 없음`으로 표시합니다.")
        st.write("4) 최종 추천 표에서 점수 컬럼을 함께 보면서 왜 이 게임이 올라왔는지 확인할 수 있습니다.")


def _render_score_glossary() -> None:
    with st.expander("점수 용어 설명", expanded=False):
        st.write("- similarity: 질의와 게임 프로필 임베딩 유사도")
        st.write("- query_alignment_score: 질의 키워드가 장르/태그/근거 리뷰와 맞는 정도")
        st.write("- preferred_genre_hits: 선호 장르가 실제 후보에 맞아떨어진 횟수")
        st.write("- soft_match_count: soft preference와 근거 리뷰가 맞는 개수")
        st.write("- hidden_gem_score: 롱테일/품질/플레이타임 신호를 합친 값")
        st.write("- confidence: 리뷰 수/플레이타임/근거 수 기반 신뢰도 라벨")


def _render_confidence_glossary() -> None:
    with st.expander("confidence 라벨 의미", expanded=False):
        st.write("- high: 근거 리뷰 수/리뷰량/플레이타임이 모두 충분한 편")
        st.write("- medium: 근거와 데이터가 보통 수준")
        st.write("- low: 근거 리뷰나 데이터량이 상대적으로 적음")


def _render_result_detail(rank: int, item: dict[str, Any]) -> None:
    title = f"{rank}. {item.get('name', '게임명 없음')}"
    with st.container(border=True):
        st.subheader(title)

        summary_cols = st.columns(3)
        summary_cols[0].metric("Similarity", _as_inline(item.get("similarity")))
        summary_cols[1].metric("Alignment", _as_inline(item.get("query_alignment_score")))
        summary_cols[2].metric("Confidence", _as_inline(item.get("confidence")))

        st.markdown("**기본 정보**")
        st.write(f"- app_id: {_as_inline(item.get('app_id'))}")
        st.write(f"- 장르: {_as_inline(', '.join(item.get('genres', []) or []))}")
        st.write(f"- 태그: {_as_inline(', '.join(item.get('tags', []) or []))}")
        st.write(f"- 스팀 링크: {_as_inline(item.get('steam_url'))}")

        st.markdown("**점수 분해**")
        st.dataframe(build_score_breakdown_rows(item), use_container_width=True, hide_index=True)

        reason_ko = (item.get("reason_ko") or "").strip()
        one_liner = (item.get("one_liner_ko") or "").strip()
        summaries = item.get("evidence_summaries_ko", []) or []
        evidence_rows = build_evidence_rows(item)

        st.markdown("**추천 이유 (모델 문장)**")
        st.write(reason_ko if reason_ko else "이유 생성 실패 / 미지원")
        st.markdown("**한 줄 요약**")
        st.write(one_liner if one_liner else "해당 없음")

        st.markdown("**근거 요약**")
        if summaries:
            for s in summaries:
                st.write(f"- {s}")
        else:
            st.write("해당 없음")

        st.markdown("**근거 리뷰 일부**")
        if evidence_rows:
            for row in evidence_rows[:2]:
                st.write(f"- [{row['번호']}] {row['리뷰 발췌']}")
        else:
            st.write("검색 결과 없음 / 근거 없음")

        with st.expander("근거 리뷰 전체 보기"):
            if evidence_rows:
                st.dataframe(evidence_rows, use_container_width=True, hide_index=True)
            else:
                st.info("해당 없음")


def main() -> None:
    st.set_page_config(
        page_title="Steam Recommender Pipeline Dashboard",
        page_icon="SR",
        layout="wide",
    )
    _inject_dark_theme()

    settings = load_settings()
    st.title("게임 추천 모델 대시보드")
    st.caption("기존 추천 로직은 유지하고, 파이프라인 설명력을 높인 Streamlit 화면입니다.")

    _render_beginner_guide()
    _render_score_glossary()
    _render_confidence_glossary()

    with st.sidebar:
        st.header("실행 옵션")
        model_name = st.text_input("임베딩 모델", value=DEFAULT_MODEL)
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=5)
        openai_model = st.text_input("OpenAI 모델", value=settings.openai_model or "gpt-4.1-mini")
        st.caption(f"DB: `{settings.db_path}`")
        st.caption("TODO(connect): 추가 옵션이 필요하면 이 영역에서 입력받아 recommend_games 인자로 연결하세요.")

    query = st.text_area(
        "사용자 질의",
        value="스토리 좋고 싱글 위주, 공포는 제외한 게임 추천해줘",
        height=90,
    )
    run_clicked = st.button("추천 실행", type="primary")

    if not run_clicked:
        st.info("질의를 입력하고 `추천 실행`을 누르면 단계별 처리와 결과가 표시됩니다.")
        return

    if not (query or "").strip():
        st.warning("질의를 입력해 주세요.")
        return

    # TODO(connect): 기존 핵심 로직 연결 지점 (recommend_games). 로직 변경 없이 화면만 구조화합니다.
    with st.spinner("추천 파이프라인 실행 중..."):
        result = recommend_games(
            db_path=settings.db_path,
            query=query,
            top_k=top_k,
            model_name=model_name,
            openai_api_key=settings.openai_api_key,
            openai_model=openai_model,
        )

    stages = build_pipeline_stages(
        result,
        input_query=query,
        top_k=top_k,
        model_name=model_name,
        openai_model=openai_model,
        has_openai_key=bool(settings.openai_api_key),
    )

    st.markdown("## 단계별 파이프라인")
    for stage in stages:
        _render_stage(stage)

    parsed_meaning_rows = build_parsed_value_meaning_rows(result)
    with st.expander("파싱 값(영어 코드) 의미 설명", expanded=False):
        if parsed_meaning_rows:
            st.dataframe(parsed_meaning_rows, use_container_width=True, hide_index=True)
        else:
            st.info("해당 없음")

    rows = build_result_rows(result)
    st.markdown("## 최종 추천 결과")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("반환 결과 수", str(len(rows)))
    m2.metric(
        "reason_ko 생성 수",
        str(sum(1 for x in (result.get("results", []) or []) if (x.get("reason_ko") or "").strip())),
    )
    m3.metric(
        "evidence 보유 수",
        str(sum(1 for x in (result.get("results", []) or []) if (x.get("evidence_reviews") or []))),
    )
    m4.metric("오류 로그 수", str(len(result.get("llm_errors", []) or [])))

    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        for idx, item in enumerate(result.get("results", []) or [], start=1):
            _render_result_detail(idx, item)
    else:
        st.warning("검색 결과 없음")

    with st.expander("LLM/런타임 로그"):
        errors = result.get("llm_errors", []) or []
        if errors:
            for err in errors:
                st.write(f"- {err}")
        else:
            st.write("해당 없음")

    with st.expander("Raw JSON (디버깅용)"):
        st.json(result)


if __name__ == "__main__":
    main()
