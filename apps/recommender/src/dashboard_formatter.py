from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


NA_TEXT = "현재 파이프라인에 없음"

SCORE_KEY_MEANINGS = {
    "similarity": "질의 벡터와 게임 프로필 벡터의 유사도",
    "query_alignment_score": "질의 키워드가 장르/태그/근거 리뷰와 얼마나 맞는지",
    "preferred_genre_hits": "선호 장르와 후보 게임 신호가 맞은 횟수",
    "soft_match_count": "soft preference(예: StoryRich)와 근거 리뷰의 매칭 수",
    "hidden_gem_score": "롱테일 성향 + 품질(긍정비율) + 플레이타임을 합친 점수",
    "positive_ratio_1y": "최근 1년 긍정 리뷰 비율",
    "recent_review_count": "최근 1년 리뷰 수",
    "median_playtime_1y": "최근 1년 리뷰어 기준 중앙 플레이타임(분)",
}

PARSED_VALUE_MEANINGS = {
    "StoryRich": "스토리/서사 중심 선호",
    "Healing": "힐링/편안한 분위기 선호",
    "Challenge": "어려운 난이도/도전 선호",
    "FastPaced": "빠른 템포 선호",
    "HiddenGem": "덜 알려진 게임 선호",
    "Singleplayer": "싱글 플레이 필수",
    "Multiplayer": "멀티 플레이 필수",
    "Relaxed": "느긋한 플레이 스타일 선호",
    "Competitive": "경쟁적 플레이 스타일 선호",
    "Exploration": "탐험 중심 선호",
    "BuildCraft": "건설/제작 중심 선호",
    "Narrative": "이야기 중심 플레이 선호",
    "Short": "짧은 세션 선호",
    "Long": "긴 세션 선호",
    "Easy": "쉬운 난이도 선호",
    "Hard": "어려운 난이도 선호",
    "Combat": "전투 중심 선호",
    "Story": "스토리 중심 선호",
    "Growth": "성장/육성 중심 선호",
    "Puzzle": "퍼즐/추리 중심 선호",
    "Management": "경영/운영 중심 선호",
}


@dataclass
class PipelineStage:
    name: str
    goal: str
    beginner_note: str
    input_data: dict[str, Any]
    process_data: dict[str, Any]
    output_data: dict[str, Any]


def _na_if_empty(value: Any, na_text: str = "없음") -> Any:
    if value is None:
        return na_text
    if isinstance(value, str) and not value.strip():
        return na_text
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return na_text
    return value


def _iso_to_local(iso_text: str) -> str:
    text = (iso_text or "").strip()
    if not text:
        return "없음"
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return text


def build_pipeline_stages(
    result: dict[str, Any],
    *,
    input_query: str,
    top_k: int,
    model_name: str,
    openai_model: str,
    has_openai_key: bool,
) -> list[PipelineStage]:
    parsed = result.get("parsed_query", {}) or {}
    results = result.get("results", []) or []
    llm_errors = result.get("llm_errors", []) or []
    reference_game = result.get("reference_game")
    similar_to_fallback = result.get("similar_to_fallback")

    # TODO(connect): recommend_games 반환 스키마가 바뀌면 stage 매핑도 함께 수정하세요.
    return [
        PipelineStage(
            name="1) 입력 준비",
            goal="사용자 문장을 모델이 처리 가능한 입력 형태로 정리합니다.",
            beginner_note="여기서 문장이 다듬어져야 이후 검색/점수 계산이 흔들리지 않습니다.",
            input_data={
                "원문 질의": _na_if_empty(input_query),
                "요청 Top-K": top_k,
                "임베딩 모델": _na_if_empty(model_name),
                "OpenAI 모델": _na_if_empty(openai_model),
            },
            process_data={
                "OpenAI Key 상태": "연결됨" if has_openai_key else "미연결 (LLM 단계 일부 미지원)",
                "실행 시각": _iso_to_local(str(result.get("generated_at", ""))),
            },
            output_data={
                "normalized_input_query": _na_if_empty(result.get("normalized_input_query")),
                "rewritten_query": _na_if_empty(result.get("rewritten_query")),
                "effective_query": _na_if_empty(result.get("effective_query")),
            },
        ),
        PipelineStage(
            name="2) 질의 해석",
            goal="문장에서 조건을 추출해 구조화합니다. (선호/제외/플레이스타일 등)",
            beginner_note="추천 정확도는 이 단계의 해석 품질에 크게 좌우됩니다.",
            input_data={
                "파싱 대상": _na_if_empty(result.get("effective_query")),
            },
            process_data={
                "파싱 방식": "Rule + LLM 결합" if has_openai_key else "Rule 기반",
                "중간 토큰 로그": NA_TEXT,
            },
            output_data={
                "preferred_genres": _na_if_empty(parsed.get("preferred_genres")),
                "excluded_genres_or_moods": _na_if_empty(parsed.get("excluded_genres_or_moods")),
                "excluded_terms": _na_if_empty(parsed.get("excluded_terms")),
                "must_have": _na_if_empty(parsed.get("must_have")),
                "soft_preferences": _na_if_empty(parsed.get("soft_preferences")),
                "play_style": _na_if_empty(parsed.get("play_style")),
                "session_length": _na_if_empty(parsed.get("session_length")),
                "difficulty": _na_if_empty(parsed.get("difficulty")),
                "focus": _na_if_empty(parsed.get("focus")),
            },
        ),
        PipelineStage(
            name="3) 검색 후보 생성",
            goal="해석된 질의를 바탕으로 게임 후보를 수집하고 기본 필터를 적용합니다.",
            beginner_note="이 단계는 '많이 뽑기' 단계이고, 다음 단계에서 정밀하게 순위를 다시 매깁니다.",
            input_data={
                "검색 모드": _na_if_empty(result.get("mode")),
                "reference_game 사용": "예" if reference_game else "아니오",
            },
            process_data={
                "reference_game": reference_game or "해당 없음",
                "similar_to 실패시 LLM fallback": similar_to_fallback or "미사용",
                "후보 수(stage1)": NA_TEXT,
                "하드 필터 통과 수": NA_TEXT,
            },
            output_data={
                "최종 반환 결과 수": len(results),
                "후보 생성 상세 로그": NA_TEXT,
            },
        ),
        PipelineStage(
            name="4) 점수 계산/재정렬",
            goal="여러 신호를 합쳐 점수를 만들고 결과 순서를 재배치합니다.",
            beginner_note="similarity 하나만 보지 않고 alignment, 장르 hit, hidden_gem 성향 등을 함께 봅니다.",
            input_data={
                "재정렬 대상": f"최종 반환 결과 {len(results)}개 기준",
            },
            process_data={
                "실제 반환 점수 요소": [
                    "similarity",
                    "query_alignment_score",
                    "preferred_genre_hits",
                    "soft_match_count",
                    "hidden_gem_score",
                    "recent_review_count",
                ],
                "MMR 다양성 선택": "적용됨 (내부 동작 상세 점수는 미반환)",
            },
            output_data={
                "평균 similarity": _safe_avg(results, "similarity"),
                "평균 query_alignment_score": _safe_avg(results, "query_alignment_score"),
                "평균 hidden_gem_score": _safe_avg(results, "hidden_gem_score"),
            },
        ),
        PipelineStage(
            name="5) 근거 요약/추천 문장 생성",
            goal="근거 리뷰를 바탕으로 추천 이유(reason)를 사람이 읽기 좋은 문장으로 만듭니다.",
            beginner_note="LLM이 없거나 실패하면 빈 값일 수 있으며, 이 경우 화면에서 명확히 표시합니다.",
            input_data={
                "근거 리뷰 사용": "예 (evidence_reviews)",
                "LLM reason 생성": "예" if has_openai_key else "미지원 (OpenAI Key 없음)",
            },
            process_data={
                "LLM 오류 로그 수": len(llm_errors),
                "실패 시 처리": "빈 값 허용 (강제 생성 없음)",
            },
            output_data={
                "reason_ko 생성 결과 수": sum(1 for r in results if (r.get("reason_ko") or "").strip()),
                "one_liner_ko 생성 결과 수": sum(
                    1 for r in results if (r.get("one_liner_ko") or "").strip()
                ),
                "evidence_summaries_ko 생성 결과 수": sum(
                    1
                    for r in results
                    if isinstance(r.get("evidence_summaries_ko"), list) and r.get("evidence_summaries_ko")
                ),
            },
        ),
    ]


def build_result_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(result.get("results", []) or [], start=1):
        rows.append(
            {
                "순위": rank,
                "게임명": _na_if_empty(item.get("name")),
                "앱ID(app_id)": _na_if_empty(item.get("app_id")),
                "유사도(similarity)": _na_if_empty(item.get("similarity")),
                "질의정합도(query_alignment_score)": _na_if_empty(item.get("query_alignment_score")),
                "선호장르일치수(preferred_genre_hits)": _na_if_empty(item.get("preferred_genre_hits")),
                "소프트선호일치수(soft_match_count)": _na_if_empty(item.get("soft_match_count")),
                "숨은보석점수(hidden_gem_score)": _na_if_empty(item.get("hidden_gem_score")),
                "최근1년긍정비율(positive_ratio_1y)": _na_if_empty(item.get("positive_ratio_1y")),
                "최근1년리뷰수(recent_review_count)": _na_if_empty(item.get("recent_review_count")),
                "중앙플레이타임분(median_playtime_1y)": _na_if_empty(item.get("median_playtime_1y")),
                "신뢰도(confidence)": _na_if_empty(item.get("confidence")),
            }
        )
    return rows


def build_score_breakdown_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    # TODO(raw-object): 점수 dict를 한 줄로 보여주지 않고, 항목별 행으로 분해해 출력합니다.
    return [
        {
            "항목(영문키)": "similarity",
            "의미": SCORE_KEY_MEANINGS["similarity"],
            "값": _na_if_empty(item.get("similarity")),
        },
        {
            "항목(영문키)": "query_alignment_score",
            "의미": SCORE_KEY_MEANINGS["query_alignment_score"],
            "값": _na_if_empty(item.get("query_alignment_score")),
        },
        {
            "항목(영문키)": "preferred_genre_hits",
            "의미": SCORE_KEY_MEANINGS["preferred_genre_hits"],
            "값": _na_if_empty(item.get("preferred_genre_hits")),
        },
        {
            "항목(영문키)": "soft_match_count",
            "의미": SCORE_KEY_MEANINGS["soft_match_count"],
            "값": _na_if_empty(item.get("soft_match_count")),
        },
        {
            "항목(영문키)": "hidden_gem_score",
            "의미": SCORE_KEY_MEANINGS["hidden_gem_score"],
            "값": _na_if_empty(item.get("hidden_gem_score")),
        },
        {
            "항목(영문키)": "positive_ratio_1y",
            "의미": SCORE_KEY_MEANINGS["positive_ratio_1y"],
            "값": _na_if_empty(item.get("positive_ratio_1y")),
        },
        {
            "항목(영문키)": "recent_review_count",
            "의미": SCORE_KEY_MEANINGS["recent_review_count"],
            "값": _na_if_empty(item.get("recent_review_count")),
        },
        {
            "항목(영문키)": "median_playtime_1y",
            "의미": SCORE_KEY_MEANINGS["median_playtime_1y"],
            "값": _na_if_empty(item.get("median_playtime_1y")),
        },
    ]


def build_evidence_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, review in enumerate(item.get("evidence_reviews", []) or [], start=1):
        rows.append({"번호": i, "리뷰 발췌": _na_if_empty(review)})
    return rows


def _safe_avg(rows: list[dict[str, Any]], key: str) -> str:
    vals: list[float] = []
    for row in rows:
        v = row.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return "해당 없음"
    return f"{sum(vals) / len(vals):.4f}"


def build_parsed_value_meaning_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parsed = result.get("parsed_query", {}) or {}
    for field_name, raw_values in parsed.items():
        if not isinstance(raw_values, list):
            continue
        for value in raw_values:
            v = str(value).strip()
            if not v:
                continue
            rows.append(
                {
                    "파싱 필드": field_name,
                    "값(영문코드)": v,
                    "의미": PARSED_VALUE_MEANINGS.get(v, "코드상 사전 의미 정의 없음"),
                }
            )
    return rows
