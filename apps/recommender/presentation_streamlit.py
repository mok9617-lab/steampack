from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st


PIPELINE_STAGES = [
    {
        "id": "input",
        "label": "1) 질의 입력",
        "input": "사용자 자연어 질의",
        "process": "질의 원문 수집, 옵션(top-k 등) 결합",
        "output": "정규화 대상 텍스트",
        "variables": [
            {"name": "query", "desc": "사용자 원문 질의"},
            {"name": "top_k", "desc": "최종 추천 개수"},
        ],
        "weights": [],
    },
    {
        "id": "parse",
        "label": "2) 질의 정규화/파싱",
        "input": "원문 질의",
        "process": "장르/제외조건/필수조건/소프트 선호를 추출",
        "output": "parsed_query 객체",
        "variables": [
            {"name": "preferred_genres", "desc": "선호 장르"},
            {"name": "excluded_genres_or_moods", "desc": "제외 장르/분위기"},
            {"name": "must_have", "desc": "반드시 포함할 조건(예: Singleplayer)"},
            {"name": "soft_preferences", "desc": "가중치 반영용 취향"},
            {"name": "play_style / difficulty / focus", "desc": "플레이 성향"},
        ],
        "weights": [],
    },
    {
        "id": "retrieve",
        "label": "3) 후보 검색 + 하드 필터",
        "input": "parsed_query + query vector",
        "process": "임베딩 유사 후보 탐색 후 하드 제약(제외/필수) 필터",
        "output": "1차 후보군(stage1)",
        "variables": [
            {"name": "similarity", "desc": "질의-게임 프로필 임베딩 유사도"},
            {"name": "_is_hard_filtered", "desc": "하드 제약 위반 후보 제거"},
            {"name": "stage1", "desc": "상위 후보 풀(top_k*8 또는 50)"},
        ],
        "weights": [
            {
                "name": "Intent blend",
                "value": "0.8 * q_vec + 0.2 * bias_vec",
                "why": "원문 의도를 유지(80%)하면서 취향 신호(20%)를 보강해 모호한 질의 분리",
            }
        ],
    },
    {
        "id": "score",
        "label": "4) 점수 계산/재정렬",
        "input": "후보군 + 근거 리뷰",
        "process": "정합도/유사도/선호 적중/소프트 매치 기반 final_score 계산",
        "output": "재정렬 후보 + 다양성 반영 결과",
        "variables": [
            {"name": "query_alignment_score", "desc": "질의 키워드-장르/태그/근거 정합도"},
            {"name": "preferred_genre_hits", "desc": "선호 장르 적중 개수"},
            {"name": "soft_match_count", "desc": "소프트 선호 매치 개수"},
            {"name": "hidden_gem_score", "desc": "저평가 보석 성향 점수"},
            {"name": "recent_review_count", "desc": "신뢰도 보정용 최근 리뷰 수"},
        ],
        "weights": [
            {
                "name": "일반 모드 final_score",
                "value": "0.34*alignment + 0.40*similarity + 0.12*genre_hits + 0.08*soft + 0.06*review_norm",
                "why": "기본 검색 품질은 similarity(0.40)+alignment(0.34)에 가장 크게 두고, 취향 미세조정은 보조 가중치로 설계",
            },
            {
                "name": "HiddenGem 모드 final_score",
                "value": "0.32*alignment + 0.30*hidden_gem + 0.24*similarity + 0.08*genre_hits + 0.06*soft",
                "why": "숨은 보석 탐색에서는 hidden_gem_score 비중을 크게 올려(0.30) 인기작 편향을 줄이기 위해",
            },
            {
                "name": "MMR 다양성",
                "value": "(1-0.22)*relevance - 0.22*max_sim",
                "why": "상위 결과가 서로 너무 비슷해지는 중복을 줄이되(22%), 관련성은 주로 유지(78%)",
            },
        ],
    },
    {
        "id": "reason",
        "label": "5) 추천 이유 생성",
        "input": "최종 후보 + 근거 리뷰",
        "process": "근거 요약/추천 사유/한줄 요약 생성",
        "output": "사용자에게 보여줄 추천 카드",
        "variables": [
            {"name": "evidence_reviews", "desc": "근거 리뷰 텍스트"},
            {"name": "reason_ko", "desc": "추천 사유"},
            {"name": "one_liner_ko", "desc": "한 줄 요약"},
        ],
        "weights": [],
    },
]


TRANSITION_NOTES = {
    "v1->v2": {
        "what": "파서 성공률/결과 비어있음 개선(0.9167→1.0, 0.75→1.0), 하드 제약 위반 증가(0.0→0.1667)",
        "why": "파싱/리콜을 올리는 과정에서 제약 필터가 느슨해진 구간으로 해석",
    },
    "v2->v3": {"what": "지표 거의 동일", "why": "기능 추가보다 안정화/리팩터링 단계"},
    "v3->v4": {"what": "지표 거의 동일", "why": "대형 로직 변경 없이 유지"},
    "v4->v5": {
        "what": "하드 제약 위반 제거(0.1667→0.0), 추천 실패 0",
        "why": "제외/필수 조건 필터 순서 또는 강도 보정 효과",
    },
    "v5->v6": {
        "what": "결과 비어있음 악화(1.0→0.8333)",
        "why": "필터가 너무 강해져 일부 질의에서 결과가 사라진 과필터링",
    },
    "v6->v7": {
        "what": "non-empty 회복(0.8333→1.0), 장르 적중 개선(0.8333→0.9167)",
        "why": "v6 과필터를 완화하면서 정밀도 균형 복구",
    },
    "v7->v8": {
        "what": "파서 성공률 급락(1.0→0.7375, parse_fail 0→20), 추천 지표는 상대적으로 유지",
        "why": "평가셋 확장(12→60)과 난이도 상승으로 파서 취약점 노출",
    },
    "v8->v9": {
        "what": "파서 일부 회복(0.7375→0.7833), 장르 적중 1.0",
        "why": "파싱 규칙/정규화 개선 반영",
    },
    "v9->v10": {
        "what": "파서 추가 회복(0.7833→0.8167), 하드 위반 0.0, 추천 실패 0",
        "why": "제약 필터 안정화 + 추천 파이프라인 완성도 상승",
    },
}


VERSION_NOTES = {
    1: "초기 기준선. 파싱/추천 실패가 존재.",
    2: "pass 1.0, nonempty 1.0로 개선. 대신 hard_violation 0.1667 증가.",
    3: "v2와 지표 거의 동일. 중간 안정화.",
    4: "v3와 지표 거의 동일. 대형 로직 변경 없음.",
    5: "hard_violation 0.0, reco_fail 0으로 개선.",
    6: "nonempty 0.8333 하락. 과필터링 징후.",
    7: "nonempty 1.0 회복, genre_hit 0.9167 상승.",
    8: "평가셋 확장(12→60)으로 pass 급락, parse_fail 20.",
    9: "pass 일부 회복, genre_hit 1.0.",
    10: "추천 품질 안정권, 파서도 추가 회복(완전 복구 전).",
}


def _inject_style() -> None:
    st.markdown(
        """
        <style>
          :root {
            --bg0: #071018;
            --bg1: #0e1f2b;
            --card: #132a3a;
            --line: #2f4f63;
            --txt: #eaf4fb;
            --muted: #b4c7d6;
            --accent: #6ec5ff;
            --good: #6fd08c;
            --warn: #ffcc66;
          }
          .stApp {
            background:
              radial-gradient(circle at 8% 0%, #173348 0%, transparent 32%),
              radial-gradient(circle at 92% 100%, #1a2b38 0%, transparent 32%),
              linear-gradient(180deg, var(--bg1), var(--bg0));
            color: var(--txt);
          }
          .block-container { max-width: 1320px; padding-top: 1rem; }
          h1, h2, h3, h4 { color: var(--txt) !important; letter-spacing: 0.2px; }
          p, li, label, .stMarkdown, .stCaption { color: var(--txt) !important; line-height: 1.55; }
          .metric-box {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(19, 42, 58, 0.72);
            padding: 10px 12px;
          }
          .hint {
            border-left: 4px solid var(--accent);
            padding: 10px 12px;
            background: rgba(19, 42, 58, 0.65);
            border-radius: 8px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_eval_history(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("eval_report_v*.json")):
        match = re.search(r"_v(\d+)", path.stem)
        if not match:
            continue
        version = int(match.group(1))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        parser_metrics = payload.get("parser_metrics") or {}
        reco_metrics = payload.get("recommendation_metrics") or {}
        parse_failures = payload.get("parse_failures") or []
        reco_failures = payload.get("recommendation_failures") or []

        rows.append(
            {
                "version": version,
                "case_count": int(payload.get("case_count", 0) or 0),
                "pass_rate": float(parser_metrics.get("pass_rate", 0.0) or 0.0),
                "parse_fail_count": len(parse_failures),
                "nonempty_rate": float(reco_metrics.get("nonempty_rate", 0.0) or 0.0),
                "genre_hit_rate": float(reco_metrics.get("preferred_genre_hit_rate", 0.0) or 0.0),
                "precision_at_k": float(reco_metrics.get("precision_at_k", 0.0) or 0.0),
                "ndcg_at_k": float(reco_metrics.get("ndcg_at_k", 0.0) or 0.0),
                "hard_violation_rate": float(
                    reco_metrics.get("hard_constraint_violation_rate", 0.0) or 0.0
                ),
                "reco_fail_count": len(reco_failures),
            }
        )

    rows.sort(key=lambda x: x["version"])
    return rows


def _render_pipeline_graph() -> None:
    graph = """
    digraph Pipeline {
      rankdir=LR;
      splines=ortho;
      nodesep=0.55;
      ranksep=0.7;
      bgcolor="transparent";

      node [shape=box, style="rounded,filled", fillcolor="#132a3a", color="#2f4f63", fontcolor="#eaf4fb", fontsize=12, penwidth=1.4];
      edge [color="#6ec5ff", penwidth=1.5];

      input [label="1) 질의 입력"];
      parse [label="2) 질의 정규화/파싱"];
      retrieve [label="3) 후보 검색 + 하드 필터"];
      score [label="4) 점수 계산/재정렬"];
      reason [label="5) 추천 이유 생성"];

      input -> parse -> retrieve -> score -> reason;
    }
    """
    st.graphviz_chart(graph, use_container_width=True)


def _render_stage_selector() -> str:
    if "selected_stage" not in st.session_state:
        st.session_state["selected_stage"] = PIPELINE_STAGES[0]["id"]

    cols = st.columns(len(PIPELINE_STAGES))
    for idx, stage in enumerate(PIPELINE_STAGES):
        with cols[idx]:
            if st.button(stage["label"], key=f"stage_btn_{stage['id']}", use_container_width=True):
                st.session_state["selected_stage"] = stage["id"]

    return str(st.session_state["selected_stage"])


def _render_stage_detail(stage_id: str) -> None:
    stage = next((x for x in PIPELINE_STAGES if x["id"] == stage_id), None)
    if stage is None:
        st.info("선택한 단계 정보를 찾을 수 없습니다.")
        return

    st.markdown(f"### {stage['label']}")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**INPUT**")
    c1.write(stage["input"])
    c2.markdown("**PROCESS**")
    c2.write(stage["process"])
    c3.markdown("**OUTPUT**")
    c3.write(stage["output"])

    st.markdown("**핵심 변수 설명**")
    st.dataframe(
        [{"변수": x["name"], "설명": x["desc"]} for x in stage["variables"]],
        use_container_width=True,
        hide_index=True,
    )

    if stage["weights"]:
        st.markdown("**가중치/공식 + 배경 설명**")
        st.dataframe(
            [
                {
                    "항목": x["name"],
                    "공식/가중치": x["value"],
                    "왜 이 비중인가": x["why"],
                }
                for x in stage["weights"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def _fmt_delta(prev: float, curr: float) -> str:
    return f"{curr - prev:+.4f}"


def _render_version_table(history: list[dict[str, Any]]) -> None:
    rows = []
    for row in history:
        rows.append(
            {
                "버전": f"v{row['version']}",
                "평가 케이스": row["case_count"],
                "pass_rate": row["pass_rate"],
                "parse_fail": row["parse_fail_count"],
                "nonempty_rate": row["nonempty_rate"],
                "genre_hit_rate": row["genre_hit_rate"],
                "precision@k": row["precision_at_k"],
                "ndcg@k": row["ndcg_at_k"],
                "hard_violation_rate": row["hard_violation_rate"],
                "reco_fail": row["reco_fail_count"],
                "해석": VERSION_NOTES.get(row["version"], "-"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_transition_cards(history: list[dict[str, Any]]) -> None:
    st.markdown("### 버전 전환 해석(무엇이/왜)")
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        key = f"v{prev['version']}->v{curr['version']}"
        note = TRANSITION_NOTES.get(key)

        with st.container(border=True):
            st.markdown(f"**{key}**")
            if note:
                st.write(f"- 무엇이 바뀌었나: {note['what']}")
                st.write(f"- 왜 바뀌었나: {note['why']}")

            st.write(
                "- 수치 변화: "
                f"pass_rate {_fmt_delta(prev['pass_rate'], curr['pass_rate'])}, "
                f"nonempty {_fmt_delta(prev['nonempty_rate'], curr['nonempty_rate'])}, "
                f"genre_hit {_fmt_delta(prev['genre_hit_rate'], curr['genre_hit_rate'])}, "
                f"hard_violation {_fmt_delta(prev['hard_violation_rate'], curr['hard_violation_rate'])}, "
                f"parse_fail {curr['parse_fail_count'] - prev['parse_fail_count']:+d}, "
                f"reco_fail {curr['reco_fail_count'] - prev['reco_fail_count']:+d}"
            )

            if prev["case_count"] != curr["case_count"]:
                st.caption(
                    f"평가셋 크기 변경: {prev['case_count']} -> {curr['case_count']} (난이도/분포 변화 영향 가능)"
                )


def main() -> None:
    st.set_page_config(page_title="추천 파이프라인 대시보드", page_icon="SR", layout="wide")
    _inject_style()

    st.title("추천 시스템 분석 대시보드")
    st.caption("탭 1: 파이프라인 프로세스 / 탭 2: 버전별 업그레이드")

    tab1, tab2 = st.tabs(["파이프라인 프로세스", "버전별 업그레이드"])

    with tab1:
        st.subheader("프로세스 흐름 시각화")
        _render_pipeline_graph()
        st.markdown(
            "<div class='hint'>박스를 클릭하면 해당 단계의 입력/처리/출력, 변수, 가중치, 가중치 배경을 확인할 수 있습니다.</div>",
            unsafe_allow_html=True,
        )

        selected = _render_stage_selector()
        _render_stage_detail(selected)

        with st.expander("한눈에 보는 가중치 설계 원칙", expanded=True):
            st.write("- 기본 모드: 관련성(similarity, alignment)을 가장 크게 두고, 취향 변수는 보조 가중치로 반영")
            st.write("- HiddenGem 모드: hidden_gem_score 비중을 확대해 대중 인기 편향을 완화")
            st.write("- 다양성(MMR): 같은 타입 결과가 줄줄이 나오는 문제를 제어")

    with tab2:
        history = _load_eval_history(Path("data"))
        if not history:
            st.info("`data/eval_report_v*.json` 파일이 없어 버전 분석을 표시할 수 없습니다.")
            return

        st.subheader("버전별 핵심 변화 (수치 기반)")
        _render_version_table(history)

        st.markdown("### 전체 해석")
        st.write("- v1~v7: 기본 안정화 + 제약 준수/결과 반환 밸런스 조정")
        st.write("- v8: 평가 확장(12→60)으로 파서 취약점 노출")
        st.write("- v9~v10: 복합 제외/인코딩/충돌 처리 보강으로 hard 케이스 개선, 제약 위반 0 달성")

        _render_transition_cards(history)

        with st.expander("한 줄 결론", expanded=True):
            st.write("v1→v7: 파싱/추천 모두 개선 흐름")
            st.write("v8: 파싱 회귀")
            st.write("v9→v10: 추천 품질은 거의 완성, 파싱은 부분 회복 상태")


if __name__ == "__main__":
    main()
