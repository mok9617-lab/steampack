from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dash import Dash, Input, Output, State, dash_table, dcc, html

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


def _as_text(value: Any) -> str:
    if value is None:
        return "없음"
    if isinstance(value, bool):
        return "예" if value else "아니오"
    text = str(value).strip()
    return text if text else "없음"


def _value_component(value: Any) -> Any:
    # TODO(raw-object): dict/list를 사람이 읽을 수 있게 항목으로 풀어서 렌더링합니다.
    if isinstance(value, dict):
        if not value:
            return html.Div("해당 없음")
        return html.Ul([html.Li(f"{k}: {_as_text(v)}") for k, v in value.items()])
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if not values:
            return html.Div("해당 없음")
        return html.Ul([html.Li(_as_text(v)) for v in values])
    return html.Div(_as_text(value))


def _kv_block(title: str, payload: dict[str, Any]) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontWeight": "700", "color": "#93c5fd", "marginBottom": "8px"}),
            *[
                html.Div(
                    [
                        html.Div(k, style={"fontWeight": "600", "marginTop": "8px"}),
                        _value_component(v),
                    ]
                )
                for k, v in payload.items()
            ],
        ],
        style={
            "flex": "1",
            "minWidth": "280px",
            "background": "#0b1324",
            "padding": "10px",
            "borderRadius": "10px",
            "border": "1px solid #26324a",
        },
    )


def _stage_card(stage: PipelineStage) -> html.Div:
    return html.Div(
        [
            html.Div(stage.name, style={"fontWeight": "700", "fontSize": "18px", "marginBottom": "4px"}),
            html.Div(f"목적: {stage.goal}", style={"color": "#93c5fd", "marginBottom": "4px"}),
            html.Div(
                f"초심자 가이드: {stage.beginner_note}",
                style={"color": "#cbd5e1", "fontSize": "14px", "marginBottom": "10px"},
            ),
            html.Div(
                [
                    _kv_block("INPUT", stage.input_data),
                    _kv_block("PROCESS", stage.process_data),
                    _kv_block("OUTPUT", stage.output_data),
                ],
                style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
            ),
        ],
        style={
            "background": "#0f172a",
            "border": "1px solid #273449",
            "borderRadius": "14px",
            "padding": "14px",
            "marginBottom": "12px",
        },
    )


def _table(data: list[dict[str, Any]], table_id: str) -> dash_table.DataTable:
    columns = [{"name": c, "id": c} for c in (data[0].keys() if data else [])]
    return dash_table.DataTable(
        id=table_id,
        data=data,
        columns=columns,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#111827",
            "color": "#e5e7eb",
            "fontWeight": "700",
            "border": "1px solid #374151",
        },
        style_cell={
            "backgroundColor": "#0b1324",
            "color": "#e5e7eb",
            "border": "1px solid #1f2937",
            "padding": "8px",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
        },
        page_size=20,
    )


def _result_detail_cards(results: list[dict[str, Any]]) -> list[html.Div]:
    cards: list[html.Div] = []
    for idx, item in enumerate(results, start=1):
        score_rows = build_score_breakdown_rows(item)
        evidence_rows = build_evidence_rows(item)
        summaries = item.get("evidence_summaries_ko", []) or []
        reason_ko = (item.get("reason_ko") or "").strip() or "이유 생성 실패 / 미지원"
        one_liner = (item.get("one_liner_ko") or "").strip() or "해당 없음"
        cards.append(
            html.Div(
                [
                    html.H3(f"{idx}. {_as_text(item.get('name'))}"),
                    html.Div(f"app_id: {_as_text(item.get('app_id'))}"),
                    html.Div(f"장르: {_as_text(', '.join(item.get('genres', []) or []))}"),
                    html.Div(f"태그: {_as_text(', '.join(item.get('tags', []) or []))}"),
                    html.Div(f"스팀 링크: {_as_text(item.get('steam_url'))}"),
                    html.H4("점수 분해", style={"marginTop": "10px"}),
                    _table(score_rows, f"score-breakdown-{idx}"),
                    html.H4("추천 이유", style={"marginTop": "10px"}),
                    html.Div(reason_ko),
                    html.H4("한 줄 요약", style={"marginTop": "10px"}),
                    html.Div(one_liner),
                    html.H4("근거 요약", style={"marginTop": "10px"}),
                    html.Ul([html.Li(_as_text(x)) for x in summaries]) if summaries else html.Div("해당 없음"),
                    html.H4("근거 리뷰 일부", style={"marginTop": "10px"}),
                    (
                        html.Ul(
                            [
                                html.Li(f"[{row.get('번호')}] {row.get('리뷰 발췌')}")
                                for row in evidence_rows[:2]
                            ]
                        )
                        if evidence_rows
                        else html.Div("검색 결과 없음 / 근거 없음")
                    ),
                    html.Details(
                        [html.Summary("근거 리뷰 전체 보기"), _table(evidence_rows, f"evidence-{idx}")]
                        if evidence_rows
                        else [html.Summary("근거 리뷰 전체 보기"), html.Div("해당 없음")]
                    ),
                ],
                style={
                    "background": "#0f172a",
                    "border": "1px solid #273449",
                    "borderRadius": "14px",
                    "padding": "14px",
                    "marginBottom": "12px",
                },
            )
        )
    return cards


settings = load_settings()
app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("게임 추천 모델 대시보드 (Dash)"),
        html.P("기존 추천 로직을 유지하고 Dash UI로 단계 설명력을 높인 버전"),
        html.Details(
            [
                html.Summary("초심자용 읽는 법"),
                html.Ul(
                    [
                        html.Li("단계는 위에서 아래로 순서대로 진행됩니다."),
                        html.Li("INPUT=들어온 값, PROCESS=처리 규칙, OUTPUT=결과입니다."),
                        html.Li("값이 없으면 없음/해당 없음/현재 파이프라인에 없음으로 표시됩니다."),
                    ]
                ),
            ],
            open=True,
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("사용자 질의"),
                        dcc.Textarea(
                            id="query",
                            value="스토리 좋고 싱글 위주, 공포는 제외한 게임 추천해줘",
                            style={"width": "100%", "height": "90px", "background": "#0b1324", "color": "#e5e7eb"},
                        ),
                    ],
                    style={"flex": "2"},
                ),
                html.Div(
                    [
                        html.Label("Top-K"),
                        dcc.Input(id="top-k", type="number", value=5, min=1, max=10, step=1),
                        html.Label("임베딩 모델", style={"marginTop": "8px"}),
                        dcc.Input(id="model-name", type="text", value=DEFAULT_MODEL, style={"width": "100%"}),
                        html.Label("OpenAI 모델", style={"marginTop": "8px"}),
                        dcc.Input(
                            id="openai-model",
                            type="text",
                            value=settings.openai_model or "gpt-4.1-mini",
                            style={"width": "100%"},
                        ),
                        html.Button("추천 실행", id="run-btn", n_clicks=0, style={"marginTop": "10px"}),
                    ],
                    style={"flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "14px"},
        ),
        html.Div(id="run-output"),
    ],
    style={
        "backgroundColor": "#0b1220",
        "color": "#e5e7eb",
        "fontFamily": "Segoe UI, Noto Sans KR, sans-serif",
        "padding": "14px",
    },
)


@app.callback(
    Output("run-output", "children"),
    Input("run-btn", "n_clicks"),
    State("query", "value"),
    State("top-k", "value"),
    State("model-name", "value"),
    State("openai-model", "value"),
)
def run_pipeline(n_clicks: int, query: str, top_k: int, model_name: str, openai_model: str) -> Any:
    if not n_clicks:
        return html.Div("질의를 입력하고 '추천 실행'을 누르면 결과가 표시됩니다.")
    if not (query or "").strip():
        return html.Div("질의를 입력해 주세요.")

    # TODO(connect): 기존 핵심 로직 recommend_games 반환값을 Dash에 맞게 구조화해서 표시합니다.
    result = recommend_games(
        db_path=settings.db_path,
        query=query,
        top_k=int(top_k or 5),
        model_name=(model_name or DEFAULT_MODEL).strip(),
        openai_api_key=settings.openai_api_key,
        openai_model=(openai_model or settings.openai_model or "gpt-4.1-mini").strip(),
    )

    stages = build_pipeline_stages(
        result,
        input_query=query,
        top_k=int(top_k or 5),
        model_name=(model_name or DEFAULT_MODEL).strip(),
        openai_model=(openai_model or settings.openai_model or "gpt-4.1-mini").strip(),
        has_openai_key=bool(settings.openai_api_key),
    )
    rows = build_result_rows(result)
    parsed_rows = build_parsed_value_meaning_rows(result)
    errors = result.get("llm_errors", []) or []
    results = result.get("results", []) or []

    summary = html.Div(
        [
            html.H2("요약"),
            html.Ul(
                [
                    html.Li(f"반환 결과 수: {len(rows)}"),
                    html.Li(
                        "reason_ko 생성 수: "
                        + str(sum(1 for x in results if (x.get("reason_ko") or "").strip()))
                    ),
                    html.Li("evidence 보유 수: " + str(sum(1 for x in results if (x.get("evidence_reviews") or [])))),
                    html.Li(f"오류 로그 수: {len(errors)}"),
                ]
            ),
        ]
    )

    return html.Div(
        [
            html.H2("단계별 파이프라인"),
            *[_stage_card(stage) for stage in stages],
            html.Details(
                [html.Summary("파싱 값(영문코드) 의미 설명"), _table(parsed_rows, "parsed-meaning")]
                if parsed_rows
                else [html.Summary("파싱 값(영문코드) 의미 설명"), html.Div("해당 없음")]
            ),
            html.H2("최종 추천 결과"),
            summary,
            _table(rows, "result-table") if rows else html.Div("검색 결과 없음"),
            html.H2("추천 상세"),
            *(_result_detail_cards(results) if results else [html.Div("상세 결과 없음")]),
            html.Details(
                [html.Summary("LLM/런타임 로그"), html.Ul([html.Li(_as_text(e)) for e in errors])]
                if errors
                else [html.Summary("LLM/런타임 로그"), html.Div("해당 없음")]
            ),
            html.Details(
                [
                    html.Summary("Raw JSON (디버깅용)"),
                    html.Pre(
                        json.dumps(result, ensure_ascii=False, indent=2),
                        style={"whiteSpace": "pre-wrap", "wordBreak": "break-word"},
                    ),
                ]
            ),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
