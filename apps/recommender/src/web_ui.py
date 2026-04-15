from __future__ import annotations

import html
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .localize import confidence_to_ko, genre_to_ko, translate_en_to_ko
from .ranker import recommend_games


CATEGORY_TABS = [
    ("action", "액션"),
    ("rpg", "RPG"),
    ("adventure", "어드벤처"),
    ("strategy", "전략"),
    ("simulation", "시뮬레이션"),
    ("survival", "생존"),
    ("fps", "FPS"),
    ("horror", "호러"),
    ("indie", "인디"),
    ("f2p", "무료 플레이"),
]

_CATEGORY_ALIAS = {
    "action": {"action"},
    "rpg": {"rpg"},
    "adventure": {"adventure"},
    "strategy": {"strategy"},
    "simulation": {"simulation"},
    "survival": {"survival"},
    "fps": {"fps", "first person shooter"},
    "horror": {"horror"},
    "indie": {"indie"},
    "f2p": {"free to play"},
}


def _page(title: str, body: str) -> bytes:
    doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f4ed;
      --card: #ffffff;
      --text: #17212b;
      --muted: #667085;
      --line: #e5e7eb;
      --brand: #0f766e;
      --brand-weak: #ccfbf1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans KR", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 10% 0%, #def7ec 0, transparent 35%),
        radial-gradient(circle at 90% 100%, #ffe8d6 0, transparent 30%),
        var(--bg);
      min-height: 100vh;
    }}
    .wrap {{ max-width: 980px; margin: 0 auto; padding: 24px 16px 56px; }}
    h1 {{ margin: 0 0 8px; }}
    .sub {{ color: var(--muted); margin-bottom: 16px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 14px;
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.04);
    }}
    .row {{ display: grid; grid-template-columns: 1fr 110px; gap: 10px; }}
    input[type=text], input[type=number] {{
      width: 100%;
      border: 1px solid #d0d5dd;
      border-radius: 10px;
      padding: 11px 12px;
      font-size: 15px;
      background: #fff;
    }}
    button {{
      border: none;
      border-radius: 10px;
      padding: 11px 14px;
      font-size: 15px;
      background: var(--brand);
      color: #fff;
      cursor: pointer;
    }}
    button:disabled {{
      opacity: 0.75;
      cursor: wait;
    }}
    .search-status {{
      margin-top: 10px;
      color: #115e59;
      font-size: 13px;
      min-height: 18px;
    }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .pill {{
      display: inline-block;
      font-size: 12px;
      border: 1px solid #99f6e4;
      color: #115e59;
      background: #ecfeff;
      padding: 3px 8px;
      border-radius: 999px;
      margin-right: 6px;
      margin-top: 6px;
    }}
    .reason {{
      margin-top: 10px;
      padding: 10px;
      border-radius: 10px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      font-size: 14px;
      line-height: 1.5;
    }}
    .evidence {{
      margin-top: 10px;
      padding-left: 18px;
      color: #374151;
    }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px; }}
    .tab-btn {{
      border: 1px solid #cbd5e1;
      background: #fff;
      color: #0f172a;
      border-radius: 999px;
      padding: 6px 11px;
      cursor: pointer;
      font-size: 13px;
      text-decoration: none;
    }}
    .tab-btn.active {{
      background: var(--brand-weak);
      border-color: #5eead4;
      color: #115e59;
      font-weight: 600;
    }}
    .thumb {{
      width: 100%;
      max-width: 460px;
      border-radius: 10px;
      border: 1px solid #e5e7eb;
      margin: 8px 0 10px;
      display: block;
      object-fit: cover;
    }}
  </style>
</head>
<body>
  <main class="wrap">{body}</main>
  <script>
    (function () {{
      const form = document.querySelector("form[action='/recommend']");
      if (!form) return;
      const button = form.querySelector("button[type='submit']");
      const status = document.getElementById("search-status");
      const messages = [
        "질문을 해석하고 있어요...",
        "리뷰 근거를 찾는 중이에요...",
        "장르와 분위기를 맞춰보는 중이에요...",
        "추천 결과를 정리하고 있어요..."
      ];
      let timerId = null;

      form.addEventListener("submit", function () {{
        if (button) {{
          button.disabled = true;
          button.textContent = "검색 중...";
        }}
        if (!status) return;
        let idx = 0;
        status.style.display = "";
        status.textContent = messages[idx];
        timerId = window.setInterval(function () {{
          idx = (idx + 1) % messages.length;
          status.textContent = messages[idx];
        }}, 1400);
      }});

      window.addEventListener("pageshow", function () {{
        if (timerId !== null) {{
          window.clearInterval(timerId);
          timerId = null;
        }}
      }});
    }})();
  </script>
</body>
</html>
"""
    return doc.encode("utf-8")


def _normalize_categories(genres: list[str], evidence_texts: list[str]) -> list[str]:
    genre_tokens = {g.lower() for g in genres}
    ev = " ".join(evidence_texts).lower()
    out: list[str] = []
    for cat_id, _ in CATEGORY_TABS:
        aliases = _CATEGORY_ALIAS.get(cat_id, set())
        if any(a in genre_tokens for a in aliases) or any(a in ev for a in aliases):
            out.append(cat_id)
    return out


def _genre_tabs_html(selected_category: str) -> str:
    buttons = []
    all_cls = "tab-btn active" if selected_category == "all" else "tab-btn"
    buttons.append(f"<button type='button' class='{all_cls}' data-category='all'>전체</button>")
    for cat_id, label in CATEGORY_TABS:
        cls = "tab-btn active" if selected_category == cat_id else "tab-btn"
        buttons.append(
            f"<button type='button' class='{cls}' data-category='{html.escape(cat_id)}'>{html.escape(label)}</button>"
        )
    return "<div class='tabs'>" + "".join(buttons) + "</div>"


def _reason_from_item(item: dict) -> str:
    if item.get("reason_ko"):
        return str(item.get("reason_ko", "")).strip()
    evidence = item.get("evidence_summaries_ko") or item.get("evidence_reviews") or []
    if evidence:
        line = translate_en_to_ko(str(evidence[0]))
        line = _clip_evidence(line, max_chars=140)
        return f"리뷰 근거: {line}"
    conf = confidence_to_ko(item.get("confidence", "unknown"))
    pos = float(item.get("positive_ratio_1y", 0.0))
    pos_pct = int(round(pos * 100))
    recent = int(item.get("recent_review_count", 0))
    pt = int(float(item.get("median_playtime_1y", 0.0)))
    return (
        f"최근 리뷰 {recent}개 기준 신뢰도는 '{conf}'이며, 긍정 비율은 약 {pos_pct}%입니다. "
        f"중앙 플레이타임은 약 {pt}분으로 실제 플레이 패턴이 반영된 추천입니다."
    )


def _clip_evidence(text: str, max_chars: int = 220) -> str:
    t = " ".join((text or "").split()).strip()
    if len(t) <= max_chars:
        return t
    return t[: max(40, max_chars - 1)].rstrip() + "…"


def _result_cards(result: dict, query: str, top_k: int, selected_category: str) -> str:
    rows: list[str] = []
    llm_errors = result.get("llm_errors", []) or []
    if llm_errors:
        rows.append("<section class='card'>")
        rows.append("<div><strong>LLM 로그</strong></div>")
        rows.append("<ol class='evidence'>")
        for err in llm_errors[:8]:
            rows.append(f"<li>{html.escape(str(err))}</li>")
        rows.append("</ol>")
        rows.append("</section>")

    rows.append("<section class='card'>")
    rows.append("<div><strong>장르 필터</strong></div>")
    rows.append(_genre_tabs_html(selected_category=selected_category))
    rows.append("</section>")

    label_map = {k: v for k, v in CATEGORY_TABS}
    result_count = 0

    for i, item in enumerate(result.get("results", []), start=1):
        game_name = translate_en_to_ko(item.get("name", "Unknown"))
        genres_ko = ", ".join(genre_to_ko(g) for g in item.get("genres", [])) or "장르 정보 없음"
        evidence = item.get("evidence_reviews", [])
        mapped = _normalize_categories(item.get("genres", []), evidence)

        result_count += 1
        mapped_str = ",".join(mapped)

        rows.append(f"<article class='card' data-categories='{html.escape(mapped_str)}'>")
        rows.append(f"<h3>{i}. {html.escape(game_name)}</h3>")

        app_id = int(item.get("app_id", 0) or 0)
        steam_url = str(item.get("steam_url") or f"https://store.steampowered.com/app/{app_id}/")
        image_url = str(
            item.get("image_url")
            or f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg"
        )
        rows.append(
            f"<a href='{html.escape(steam_url)}' target='_blank' rel='noopener noreferrer'>"
            f"<img class='thumb' src='{html.escape(image_url)}' alt='{html.escape(game_name)} 이미지' loading='lazy' />"
            "</a>"
        )
        rows.append(
            f"<div style='margin:6px 0 2px'><a href='{html.escape(steam_url)}' target='_blank' rel='noopener noreferrer'>스팀 상점에서 보기</a></div>"
        )
        rows.append(
            "<div class='meta'>유사도 {sim} | 최근 리뷰 {cnt}개 | 긍정비율(1년) {pos} | 중앙 플레이타임 {pt}분</div>".format(
                sim=item.get("similarity"),
                cnt=item.get("recent_review_count"),
                pos=item.get("positive_ratio_1y"),
                pt=item.get("median_playtime_1y"),
            )
        )
        rows.append(
            "<div><span class='pill'>신뢰도 {}</span><span class='pill'>{}</span></div>".format(
                html.escape(confidence_to_ko(item.get("confidence", "unknown"))),
                html.escape(genres_ko),
            )
        )

        if mapped:
            mapped_labels = [label_map.get(x, x) for x in mapped]
            rows.append("<div class='meta'>분류: {}</div>".format(", ".join(mapped_labels)))

        rows.append(f"<div class='reason'><strong>추천 이유</strong><br/>{html.escape(_reason_from_item(item))}</div>")
        one_liner = str(item.get("one_liner_ko") or "").strip()
        if one_liner:
            rows.append(f"<div class='meta' style='margin-top:8px'><strong>한줄 평:</strong> {html.escape(one_liner)}</div>")

        evidence_for_view = evidence
        if evidence_for_view:
            rows.append("<div class='meta' style='margin-top:8px'>리뷰 근거</div>")
            rows.append("<ol class='evidence'>")
            for ev in evidence_for_view[:4]:
                rows.append(f"<li>{html.escape(_clip_evidence(translate_en_to_ko(ev), max_chars=220))}</li>")
            rows.append("</ol>")

        rows.append("</article>")

    rows.append(
        "<section id='filter-empty' class='card' style='display:none'>선택한 장르에 맞는 결과가 없습니다.</section>"
    )

    if not result.get("results") or result_count == 0:
        rows.append("<section class='card'>조건에 맞는 결과가 없습니다. 질문을 조금 바꿔서 다시 시도해 주세요.</section>")

    rows.append(
        """
<script>
(function() {
  const tabs = Array.from(document.querySelectorAll('.tabs .tab-btn'));
  const cards = Array.from(document.querySelectorAll('article.card[data-categories]'));
  const emptyBox = document.getElementById('filter-empty');
  if (!tabs.length || !cards.length) return;

  function setActive(target) {
    tabs.forEach((btn) => btn.classList.remove('active'));
    if (target) target.classList.add('active');
  }

  function applyFilter(category) {
    let visible = 0;
    cards.forEach((card) => {
      const cats = (card.getAttribute('data-categories') || '').split(',').filter(Boolean);
      const show = category === 'all' || cats.includes(category);
      card.style.display = show ? '' : 'none';
      if (show) visible += 1;
    });
    if (emptyBox) emptyBox.style.display = visible === 0 ? '' : 'none';
  }

  tabs.forEach((btn) => {
    btn.addEventListener('click', () => {
      const category = btn.getAttribute('data-category') || 'all';
      setActive(btn);
      applyFilter(category);
    });
  });
})();
</script>
        """
    )

    return "\n".join(rows)


def run_server(
    db_path: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4.1-mini",
) -> None:
    db_path = Path(db_path)

    class Handler(BaseHTTPRequestHandler):
        def _render(self, content: str, status: int = 200) -> None:
            payload = _page("Steam 게임 추천", content)
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _form(self, query: str = "", top_k: int = 5, extra: str = "") -> str:
            return f"""
            <h1>Steam 게임 추천 테스트</h1>
            <div class="sub">원하는 분위기나 장르를 자유롭게 입력하면 추천 결과와 근거를 보여줍니다.</div>
            <section class="card">
              <form method="post" action="/recommend">
                <label for="query"><strong>질문</strong></label>
                <div class="row" style="margin-top:8px">
                  <input id="query" name="query" type="text" required value="{html.escape(query)}" placeholder="예: 힐링되는 싱글 RPG 추천해줘. 공포는 제외" />
                  <input name="top_k" type="number" min="1" max="10" value="{int(top_k)}" />
                </div>
                <div style="margin-top:10px"><button type="submit">추천 받기</button></div>
                <div id="search-status" class="search-status" style="display:none" aria-live="polite"></div>
              </form>
            </section>
            {extra}
            """

        def _run_recommend(self, query: str, top_k: int) -> None:
            result = recommend_games(
                db_path=db_path,
                query=query,
                top_k=top_k,
                model_name=model_name,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
            )
            self._render(
                self._form(
                    query,
                    top_k,
                    _result_cards(result, query=query, top_k=top_k, selected_category="all"),
                )
            )

        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            if parsed_url.path != "/":
                self._render("<h1>404</h1><p>Not Found</p>", status=404)
                return

            qs = parse_qs(parsed_url.query)
            query = (qs.get("query", [""])[0] or "").strip()
            top_k_raw = (qs.get("top_k", ["5"])[0] or "5").strip()
            try:
                top_k = max(1, min(10, int(top_k_raw)))
            except ValueError:
                top_k = 5

            if not query:
                self._render(self._form())
                return

            try:
                self._run_recommend(query, top_k)
            except Exception as exc:
                traceback.print_exc()
                err = (
                    "<section class='card'><strong>오류</strong>"
                    f"<div class='meta'>{html.escape(str(exc))}</div></section>"
                )
                self._render(self._form(query, top_k, err), status=500)

        def do_POST(self) -> None:
            if self.path != "/recommend":
                self._render("<h1>404</h1><p>Not Found</p>", status=404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            form = parse_qs(body)
            query = (form.get("query", [""])[0] or "").strip()
            top_k_raw = (form.get("top_k", ["5"])[0] or "5").strip()
            try:
                top_k = max(1, min(10, int(top_k_raw)))
            except ValueError:
                top_k = 5

            if not query:
                self._render(self._form("", top_k, "<section class='card'>질문을 입력해 주세요.</section>"))
                return

            try:
                self._run_recommend(query, top_k)
            except Exception as exc:
                traceback.print_exc()
                err = (
                    "<section class='card'><strong>오류</strong>"
                    f"<div class='meta'>{html.escape(str(exc))}</div></section>"
                )
                self._render(self._form(query, top_k, err), status=500)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Web UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
