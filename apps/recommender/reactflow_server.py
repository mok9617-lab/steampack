from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


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
        "id": "route",
        "label": "3) 질의 유형 분기",
        "input": "parsed_query + 원문 패턴",
        "process": "일반 질의 vs '~같은 게임' 질의 라우팅",
        "output": "일반 경로 또는 similar 경로 선택",
        "variables": [
            {"name": "mode", "desc": "query / similar_to / similar_to_llm_profile_fallback"},
            {"name": "reference_hint", "desc": "'~같은 게임'에서 참조 게임 후보 텍스트"},
        ],
        "weights": [],
    },
    {
        "id": "normal_path",
        "label": "4-A) 일반 질의 경로",
        "input": "일반 질의(query mode)",
        "process": "참조 게임 없이 바로 검색 질의로 진행",
        "output": "공통 합류 지점으로 전달",
        "variables": [
            {"name": "effective_query", "desc": "정규화/리라이트 후 최종 검색 질의"},
        ],
        "weights": [],
    },
    {
        "id": "similar_path",
        "label": "4-B) '~같은 게임' 경로",
        "input": "similar_to 패턴 질의",
        "process": "참조 게임명을 추출하고 DB 매칭 시도",
        "output": "DB 존재/미존재 분기",
        "variables": [
            {"name": "reference_game_hint", "desc": "추출된 참조 게임 힌트"},
            {"name": "reference_game", "desc": "DB 매칭된 게임 객체(있으면)"},
        ],
        "weights": [],
    },
    {
        "id": "similar_db_found",
        "label": "5-B1) similar + DB 존재",
        "input": "DB에 존재하는 참조 게임",
        "process": "참조 게임 벡터/특성으로 유사 후보 탐색",
        "output": "공통 합류 지점으로 전달",
        "variables": [
            {"name": "mode=similar_to", "desc": "DB 참조 기반 similar 모드"},
        ],
        "weights": [],
    },
    {
        "id": "similar_db_missing",
        "label": "5-B2) similar + DB 미존재(LLM 프로파일)",
        "input": "DB에 없는 참조 게임",
        "process": "LLM이 참조 게임 프로파일을 추정해 대체 질의 생성",
        "output": "공통 합류 지점으로 전달",
        "variables": [
            {"name": "similar_to_fallback", "desc": "LLM 추정 프로파일/대체 질의 정보"},
            {"name": "mode=similar_to_llm_profile_fallback", "desc": "LLM 프로파일 fallback 모드"},
        ],
        "weights": [
            {
                "name": "Fallback 트리거",
                "value": "DB miss -> LLM profile inference",
                "why": "DB에 참조 게임이 없어도 '~같은 게임' 질의를 버리지 않고 설명 가능한 대체 경로를 제공",
            }
        ],
    },
    {
        "id": "merge",
        "label": "6) 공통 합류",
        "input": "일반 경로 / similar(DB hit) / similar(DB miss)",
        "process": "후속 검색 파이프라인 입력 형태를 단일화",
        "output": "표준화된 검색 입력",
        "variables": [
            {"name": "effective_query", "desc": "공통 검색 질의"},
            {"name": "parsed_query", "desc": "공통 필터/선호 조건"},
        ],
        "weights": [],
    },
    {
        "id": "retrieve",
        "label": "7) 후보 검색 + 하드 필터",
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
                "why": "원문 의도를 유지(80%)하면서 취향 신호(20%)를 보강해 모호한 질의를 분리",
            }
        ],
    },
    {
        "id": "score",
        "label": "8) 점수 계산/재정렬",
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
                "why": "기본 검색 품질은 similarity(0.40)+alignment(0.34)에 가장 크게 두고 취향 변수는 보조 반영",
            },
            {
                "name": "HiddenGem 모드 final_score",
                "value": "0.32*alignment + 0.30*hidden_gem + 0.24*similarity + 0.08*genre_hits + 0.06*soft",
                "why": "숨은 보석 탐색에서는 hidden_gem_score 비중을 높여(0.30) 인기작 편향을 완화",
            },
            {
                "name": "MMR 다양성",
                "value": "(1-0.22)*relevance - 0.22*max_sim",
                "why": "결과 중복을 줄이되(22%) 관련성은 주로 유지(78%)",
            },
        ],
    },
    {
        "id": "reason",
        "label": "9) 추천 이유 생성",
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
    {
        "id": "final_output",
        "label": "10) 최종 결과 출력",
        "input": "최종 후보 + reason_ko + one_liner_ko",
        "process": "UI 카드/표로 렌더링",
        "output": "사용자가 보는 최종 추천 결과",
        "variables": [
            {"name": "results", "desc": "최종 top-k 추천 목록"},
            {"name": "reason_ko / one_liner_ko", "desc": "사용자 친화 설명 텍스트"},
        ],
        "weights": [],
    },
]

PIPELINE_FLOW = {
    "nodes": [
        {"id": "input", "label": "1) 질의 입력", "x": 30, "y": 30},
        {"id": "parse", "label": "2) 질의 정규화/파싱", "x": 250, "y": 30},
        {"id": "route", "label": "3) 질의 유형 분기", "x": 500, "y": 30},
        {"id": "normal_path", "label": "4-A) 일반 질의", "x": 740, "y": 10},
        {"id": "similar_path", "label": "4-B) ~같은 게임", "x": 740, "y": 90},
        {"id": "similar_db_found", "label": "5-B1) DB 존재", "x": 980, "y": 60},
        {"id": "similar_db_missing", "label": "5-B2) DB 미존재(LLM)", "x": 980, "y": 140},
        {"id": "merge", "label": "6) 공통 합류", "x": 1230, "y": 90},
        {"id": "retrieve", "label": "7) 후보 검색+하드필터", "x": 1470, "y": 90},
        {"id": "score", "label": "8) 점수 계산/재정렬", "x": 1730, "y": 90},
        {"id": "reason", "label": "9) 추천 이유 생성", "x": 1980, "y": 90},
        {"id": "final_output", "label": "10) 최종 결과 출력", "x": 2230, "y": 90},
    ],
    "edges": [
        {"source": "input", "target": "parse"},
        {"source": "parse", "target": "route"},
        {"source": "route", "target": "normal_path"},
        {"source": "route", "target": "similar_path"},
        {"source": "normal_path", "target": "merge"},
        {"source": "similar_path", "target": "similar_db_found"},
        {"source": "similar_path", "target": "similar_db_missing"},
        {"source": "similar_db_found", "target": "merge"},
        {"source": "similar_db_missing", "target": "merge"},
        {"source": "merge", "target": "retrieve"},
        {"source": "retrieve", "target": "score"},
        {"source": "score", "target": "reason"},
        {"source": "reason", "target": "final_output"},
    ],
}


TRANSITION_NOTES = {
    "v1->v2": {
        "what": "파서 성공률/결과 비어있음 개선(0.9167→1.0, 0.75→1.0), 하드 제약 위반 증가(0.0→0.1667)",
        "why": "파싱/리콜 개선 과정에서 제약 필터가 느슨해진 구간으로 해석",
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


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
UI_DIR = Path(__file__).resolve().parent / "reactflow_ui"

app = FastAPI(title="Recommender ReactFlow UI", version="1.0.0")
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


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
                "note": VERSION_NOTES.get(version, "-"),
            }
        )

    rows.sort(key=lambda x: x["version"])
    return rows


def _build_transitions(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        key = f"v{prev['version']}->v{curr['version']}"
        note = TRANSITION_NOTES.get(key, {"what": "해석 없음", "why": "해석 없음"})
        out.append(
            {
                "key": key,
                "from_version": prev["version"],
                "to_version": curr["version"],
                "what": note["what"],
                "why": note["why"],
                "deltas": {
                    "pass_rate": round(curr["pass_rate"] - prev["pass_rate"], 4),
                    "nonempty_rate": round(curr["nonempty_rate"] - prev["nonempty_rate"], 4),
                    "genre_hit_rate": round(curr["genre_hit_rate"] - prev["genre_hit_rate"], 4),
                    "hard_violation_rate": round(
                        curr["hard_violation_rate"] - prev["hard_violation_rate"], 4
                    ),
                    "parse_fail_count": curr["parse_fail_count"] - prev["parse_fail_count"],
                    "reco_fail_count": curr["reco_fail_count"] - prev["reco_fail_count"],
                },
                "case_count_changed": prev["case_count"] != curr["case_count"],
                "case_count_from": prev["case_count"],
                "case_count_to": curr["case_count"],
            }
        )
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/favicon.ico")
def favicon() -> Response:
    # Browser auto-requests favicon; return empty 204 to avoid noisy 404 logs.
    return Response(status_code=204)


@app.get("/api/dashboard-data")
def dashboard_data() -> JSONResponse:
    history = _load_eval_history(DATA_DIR)
    transitions = _build_transitions(history)
    return JSONResponse(
        {
            "pipeline_stages": PIPELINE_STAGES,
            "pipeline_flow": PIPELINE_FLOW,
            "version_history": history,
            "transitions": transitions,
            "summary": {
                "v1_to_v7": "기본 안정화 + 제약 준수/결과 반환 밸런스 조정",
                "v8": "평가셋 확장(12→60)으로 파서 취약점 노출",
                "v9_to_v10": "복합 제외/인코딩/충돌 처리 보강으로 hard 케이스 개선 및 제약 위반 0 달성",
                "one_liner": "v1→v7 개선, v8 파싱 회귀, v9→v10 추천 품질 안정 + 파싱 부분 회복",
            },
        }
    )
