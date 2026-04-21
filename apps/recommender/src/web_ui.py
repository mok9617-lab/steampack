from __future__ import annotations

import html
import json
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


def _clip_evidence(text: str, max_chars: int = 220) -> str:
    t = " ".join((text or "").split()).strip()
    if len(t) <= max_chars:
        return t
    return t[: max(40, max_chars - 1)].rstrip() + "…"


def _normalize_categories(genres: list[str], evidence_texts: list[str]) -> list[str]:
    genre_tokens = {str(g).lower() for g in genres}
    ev = " ".join(str(x) for x in evidence_texts).lower()
    out: list[str] = []
    for cat_id, _ in CATEGORY_TABS:
        aliases = _CATEGORY_ALIAS.get(cat_id, set())
        if any(a in genre_tokens for a in aliases) or any(a in ev for a in aliases):
            out.append(cat_id)
    return out


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
        f"최근 리뷰 {recent}개 기준 추천 확신도는 '{conf}'이며, 최근 만족도는 약 {pos_pct}%입니다. "
        f"평균 플레이 시간은 약 {pt}분으로 실제 플레이 패턴이 반영된 추천입니다."
    )


def _prepare_result_payload(result: dict, query: str, top_k: int) -> dict:
    rows: list[dict] = []
    for item in result.get("results", []) or []:
        app_id = int(item.get("app_id", 0) or 0)
        evidence = [str(x) for x in (item.get("evidence_reviews") or [])]
        evidence_ko = [
            _clip_evidence(translate_en_to_ko(ev), max_chars=220)
            for ev in evidence[:4]
        ]
        rows.append(
            {
                "app_id": app_id,
                "name": str(item.get("name") or "Unknown"),
                "display_name": translate_en_to_ko(str(item.get("name") or "Unknown")),
                "genres": [str(g) for g in (item.get("genres") or [])],
                "genres_ko": [genre_to_ko(str(g)) for g in (item.get("genres") or [])],
                "categories": _normalize_categories(item.get("genres", []) or [], evidence),
                "similarity": item.get("similarity"),
                "recent_review_count": item.get("recent_review_count"),
                "positive_ratio_1y": item.get("positive_ratio_1y"),
                "median_playtime_1y": item.get("median_playtime_1y"),
                "confidence": str(item.get("confidence") or "unknown"),
                "confidence_ko": confidence_to_ko(item.get("confidence", "unknown")),
                "reason_ko": _reason_from_item(item),
                "one_liner_ko": str(item.get("one_liner_ko") or "").strip(),
                "evidence_ko": evidence_ko,
                "steam_url": str(item.get("steam_url") or f"https://store.steampowered.com/app/{app_id}/"),
                "image_url": str(
                    item.get("image_url")
                    or f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg"
                ),
            }
        )

    return {
        "query": query,
        "top_k": top_k,
        "llm_errors": [str(x) for x in (result.get("llm_errors", []) or [])][:8],
        "results": rows,
        "category_tabs": [{"id": k, "label": v} for k, v in CATEGORY_TABS],
    }


def _page(initial_query: str, initial_top_k: int) -> bytes:
    initial_json = json.dumps(
        {"query": initial_query, "top_k": initial_top_k},
        ensure_ascii=False,
    )
    category_json = json.dumps(
        [{"id": k, "label": v} for k, v in CATEGORY_TABS],
        ensure_ascii=False,
    )

    doc = (
        "<!doctype html>\n"
        "<html lang='ko'>\n"
        "<head>\n"
        "  <meta charset='utf-8' />\n"
        "  <meta name='viewport' content='width=device-width, initial-scale=1' />\n"
        "  <title>Steam 게임 추천</title>\n"
        "  <style>\n"
        "    :root {\n"
        "      --bg: #f6f4ed;\n"
        "      --card: #ffffff;\n"
        "      --text: #17212b;\n"
        "      --muted: #667085;\n"
        "      --line: #e5e7eb;\n"
        "      --brand: #0f766e;\n"
        "      --brand-weak: #ccfbf1;\n"
        "    }\n"
        "    * { box-sizing: border-box; }\n"
        "    body {\n"
        "      margin: 0;\n"
        "      font-family: 'Segoe UI', 'Noto Sans KR', sans-serif;\n"
        "      color: var(--text);\n"
        "      background:\n"
        "        radial-gradient(circle at 10% 0%, #def7ec 0, transparent 35%),\n"
        "        radial-gradient(circle at 90% 100%, #ffe8d6 0, transparent 30%),\n"
        "        var(--bg);\n"
        "      min-height: 100vh;\n"
        "    }\n"
        "    .wrap { max-width: 980px; margin: 0 auto; padding: 24px 16px 56px; }\n"
        "    h1 { margin: 0 0 8px; }\n"
        "    .sub { color: var(--muted); margin-bottom: 16px; }\n"
        "    .card {\n"
        "      background: var(--card);\n"
        "      border: 1px solid var(--line);\n"
        "      border-radius: 14px;\n"
        "      padding: 16px;\n"
        "      margin-bottom: 14px;\n"
        "      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.04);\n"
        "    }\n"
        "    .row { display: grid; grid-template-columns: 1fr 110px; gap: 10px; }\n"
        "    input[type=text], input[type=number] {\n"
        "      width: 100%;\n"
        "      border: 1px solid #d0d5dd;\n"
        "      border-radius: 10px;\n"
        "      padding: 11px 12px;\n"
        "      font-size: 15px;\n"
        "      background: #fff;\n"
        "    }\n"
        "    button {\n"
        "      border: none;\n"
        "      border-radius: 10px;\n"
        "      padding: 11px 14px;\n"
        "      font-size: 15px;\n"
        "      background: var(--brand);\n"
        "      color: #fff;\n"
        "      cursor: pointer;\n"
        "    }\n"
        "    button:disabled { opacity: 0.75; cursor: wait; }\n"
        "    .search-status { margin-top: 10px; color: #115e59; font-size: 13px; min-height: 18px; }\n"
        "    .meta { color: var(--muted); font-size: 13px; }\n"
        "    .pill {\n"
        "      display: inline-block;\n"
        "      font-size: 12px;\n"
        "      border: 1px solid #99f6e4;\n"
        "      color: #115e59;\n"
        "      background: #ecfeff;\n"
        "      padding: 3px 8px;\n"
        "      border-radius: 999px;\n"
        "      margin-right: 6px;\n"
        "      margin-top: 6px;\n"
        "    }\n"
        "    .reason {\n"
        "      margin-top: 10px;\n"
        "      padding: 10px;\n"
        "      border-radius: 10px;\n"
        "      background: #f8fafc;\n"
        "      border: 1px solid #e2e8f0;\n"
        "      font-size: 14px;\n"
        "      line-height: 1.5;\n"
        "    }\n"
        "    .evidence { margin-top: 10px; padding-left: 18px; color: #374151; }\n"
        "    .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px; }\n"
        "    .tab-btn {\n"
        "      border: 1px solid #cbd5e1;\n"
        "      background: #fff;\n"
        "      color: #0f172a;\n"
        "      border-radius: 999px;\n"
        "      padding: 6px 11px;\n"
        "      cursor: pointer;\n"
        "      font-size: 13px;\n"
        "    }\n"
        "    .tab-btn.active {\n"
        "      background: var(--brand-weak);\n"
        "      border-color: #5eead4;\n"
        "      color: #115e59;\n"
        "      font-weight: 600;\n"
        "    }\n"
        "    .thumb {\n"
        "      width: 100%;\n"
        "      max-width: 460px;\n"
        "      border-radius: 10px;\n"
        "      border: 1px solid #e5e7eb;\n"
        "      margin: 8px 0 10px;\n"
        "      display: block;\n"
        "      object-fit: cover;\n"
        "    }\n"
        "    .error { color: #b42318; }\n"
        "    @media (max-width: 640px) {\n"
        "      .row { grid-template-columns: 1fr; }\n"
        "    }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <main id='app' class='wrap'></main>\n"
        "  <script>\n"
        f"    window.__INITIAL_STATE__ = {initial_json};\n"
        f"    window.__CATEGORY_TABS__ = {category_json};\n"
        "  </script>\n"
        "  <script crossorigin src='https://unpkg.com/react@18/umd/react.production.min.js'></script>\n"
        "  <script crossorigin src='https://unpkg.com/react-dom@18/umd/react-dom.production.min.js'></script>\n"
        "  <script>\n"
        "    const h = React.createElement;\n"
        "\n"
        "    function App() {\n"
        "      const init = window.__INITIAL_STATE__ || { query: '', top_k: 5 };\n"
        "      const tabs = window.__CATEGORY_TABS__ || [];\n"
        "      const [query, setQuery] = React.useState(init.query || '');\n"
        "      const [topK, setTopK] = React.useState(init.top_k || 5);\n"
        "      const [loading, setLoading] = React.useState(false);\n"
        "      const [statusMsg, setStatusMsg] = React.useState('');\n"
        "      const [errorMsg, setErrorMsg] = React.useState('');\n"
        "      const [result, setResult] = React.useState(null);\n"
        "      const [activeCategory, setActiveCategory] = React.useState('all');\n"
        "\n"
        "      React.useEffect(() => {\n"
        "        if (!init.query) return;\n"
        "        submitSearch(init.query, init.top_k || 5);\n"
        "      }, []);\n"
        "\n"
        "      React.useEffect(() => {\n"
        "        if (!loading) return;\n"
        "        const messages = [\n"
        "          '질문을 해석하고 있어요...',\n"
        "          '리뷰 근거를 찾는 중이에요...',\n"
        "          '장르와 분위기를 맞춰보는 중이에요...',\n"
        "          '추천 결과를 정리하고 있어요...'\n"
        "        ];\n"
        "        let idx = 0;\n"
        "        setStatusMsg(messages[0]);\n"
        "        const timer = setInterval(() => {\n"
        "          idx = (idx + 1) % messages.length;\n"
        "          setStatusMsg(messages[idx]);\n"
        "        }, 1400);\n"
        "        return () => clearInterval(timer);\n"
        "      }, [loading]);\n"
        "\n"
        "      async function submitSearch(nextQuery, nextTopK) {\n"
        "        const q = (nextQuery || '').trim();\n"
        "        const k = Math.max(1, Math.min(10, parseInt(nextTopK, 10) || 5));\n"
        "        if (!q) {\n"
        "          setErrorMsg('질문을 입력해 주세요.');\n"
        "          return;\n"
        "        }\n"
        "\n"
        "        setLoading(true);\n"
        "        setErrorMsg('');\n"
        "        setStatusMsg('검색 준비 중...');\n"
        "        setActiveCategory('all');\n"
        "\n"
        "        try {\n"
        "          const res = await fetch('/api/recommend', {\n"
        "            method: 'POST',\n"
        "            headers: { 'Content-Type': 'application/json' },\n"
        "            body: JSON.stringify({ query: q, top_k: k })\n"
        "          });\n"
        "          const payload = await res.json();\n"
        "          if (!res.ok) throw new Error(payload.error || ('HTTP ' + res.status));\n"
        "          setResult(payload);\n"
        "          const url = new URL(window.location.href);\n"
        "          url.searchParams.set('query', q);\n"
        "          url.searchParams.set('top_k', String(k));\n"
        "          window.history.replaceState({}, '', url.toString());\n"
        "        } catch (err) {\n"
        "          setErrorMsg(String(err));\n"
        "        } finally {\n"
        "          setLoading(false);\n"
        "          setStatusMsg('');\n"
        "        }\n"
        "      }\n"
        "\n"
        "      const rows = (result && result.results) || [];\n"
        "      const filtered = rows.filter((row) => activeCategory === 'all' || (row.categories || []).includes(activeCategory));\n"
        "\n"
        "      return h(React.Fragment, null,\n"
        "        h('h1', null, 'Steam 게임 추천 테스트'),\n"
        "        h('div', { className: 'sub' }, '원하는 분위기나 장르를 자유롭게 입력하면 추천 결과와 근거를 보여줍니다.'),\n"
        "\n"
        "        h('section', { className: 'card' },\n"
        "          h('label', { htmlFor: 'query' }, h('strong', null, '질문')),\n"
        "          h('div', { className: 'row', style: { marginTop: '8px' } },\n"
        "            h('input', {\n"
        "              id: 'query',\n"
        "              type: 'text',\n"
        "              value: query,\n"
        "              onChange: (e) => setQuery(e.target.value),\n"
        "              placeholder: '예: 힐링되는 싱글 RPG 추천해줘. 공포는 제외'\n"
        "            }),\n"
        "            h('input', {\n"
        "              type: 'number',\n"
        "              min: 1,\n"
        "              max: 10,\n"
        "              value: topK,\n"
        "              onChange: (e) => setTopK(e.target.value)\n"
        "            })\n"
        "          ),\n"
        "          h('div', { style: { marginTop: '10px' } },\n"
        "            h('button', { disabled: loading, onClick: () => submitSearch(query, topK) }, loading ? '검색 중...' : '추천 받기')\n"
        "          ),\n"
        "          h('div', { className: 'search-status', 'aria-live': 'polite' }, statusMsg),\n"
        "          errorMsg ? h('div', { className: 'meta error', style: { marginTop: '8px' } }, errorMsg) : null\n"
        "        ),\n"
        "\n"
        "        result && result.llm_errors && result.llm_errors.length\n"
        "          ? h('section', { className: 'card' },\n"
        "              h('div', null, h('strong', null, 'LLM 로그')),\n"
        "              h('ol', { className: 'evidence' },\n"
        "                ...result.llm_errors.map((err, idx) => h('li', { key: 'llm-' + idx }, err))\n"
        "              )\n"
        "            )\n"
        "          : null,\n"
        "\n"
        "        result\n"
        "          ? h('section', { className: 'card' },\n"
        "              h('div', null, h('strong', null, '장르 필터')),\n"
        "              h('div', { className: 'tabs' },\n"
        "                h('button', {\n"
        "                  className: 'tab-btn ' + (activeCategory === 'all' ? 'active' : ''),\n"
        "                  onClick: () => setActiveCategory('all')\n"
        "                }, '전체'),\n"
        "                ...tabs.map((t) => h('button', {\n"
        "                  key: t.id,\n"
        "                  className: 'tab-btn ' + (activeCategory === t.id ? 'active' : ''),\n"
        "                  onClick: () => setActiveCategory(t.id)\n"
        "                }, t.label))\n"
        "              )\n"
        "            )\n"
        "          : null,\n"
        "\n"
        "        result && filtered.length === 0\n"
        "          ? h('section', { className: 'card' }, '선택한 장르에 맞는 결과가 없습니다.')\n"
        "          : null,\n"
        "\n"
        "        ...filtered.map((item, idx) => h('article', { className: 'card', key: String(item.app_id) + '-' + idx },\n"
        "          h('h3', null, (idx + 1) + '. ' + item.display_name),\n"
        "          h('a', { href: item.steam_url, target: '_blank', rel: 'noopener noreferrer' },\n"
        "            h('img', { className: 'thumb', src: item.image_url, alt: item.display_name + ' 이미지', loading: 'lazy' })\n"
        "          ),\n"
        "          h('div', { style: { margin: '6px 0 2px' } },\n"
        "            h('a', { href: item.steam_url, target: '_blank', rel: 'noopener noreferrer' }, '스팀 상점에서 보기')\n"
        "          ),\n"
        "          h('div', { className: 'meta' }, '취향 일치도 ' + item.similarity + ' | 최근 리뷰 ' + item.recent_review_count + '개 | 최근 만족도(1년) ' + item.positive_ratio_1y + ' | 평균 플레이 시간 ' + item.median_playtime_1y + '분'),\n"
        "          h('div', null,\n"
        "            h('span', { className: 'pill' }, '추천 확신도 ' + item.confidence_ko),\n"
        "            h('span', { className: 'pill' }, (item.genres_ko || []).join(', ') || '장르 정보 없음')\n"
        "          ),\n"
        "          item.categories && item.categories.length\n"
        "            ? h('div', { className: 'meta' }, '분류: ' + item.categories.join(', '))\n"
        "            : null,\n"
        "          h('div', { className: 'reason' },\n"
        "            h('strong', null, '추천 이유'), h('br'), item.reason_ko\n"
        "          ),\n"
        "          item.one_liner_ko\n"
        "            ? h('div', { className: 'meta', style: { marginTop: '8px' } }, h('strong', null, '한줄 평:'), ' ' + item.one_liner_ko)\n"
        "            : null,\n"
        "          item.evidence_ko && item.evidence_ko.length\n"
        "            ? h(React.Fragment, null,\n"
        "                h('div', { className: 'meta', style: { marginTop: '8px' } }, '리뷰 근거'),\n"
        "                h('ol', { className: 'evidence' },\n"
        "                  ...item.evidence_ko.map((ev, i) => h('li', { key: 'ev-' + i }, ev))\n"
        "                )\n"
        "              )\n"
        "            : null\n"
        "        )),\n"
        "\n"
        "        result && rows.length === 0\n"
        "          ? h('section', { className: 'card' }, '조건에 맞는 결과가 없습니다. 질문을 조금 바꿔서 다시 시도해 주세요.')\n"
        "          : null\n"
        "      );\n"
        "    }\n"
        "\n"
        "    ReactDOM.createRoot(document.getElementById('app')).render(h(App));\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )

    return doc.encode("utf-8")


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
        def _send_html(self, payload: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def _parse_query_topk(self) -> tuple[str, int]:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", "0"))
            body_bytes = self.rfile.read(length) if length > 0 else b""

            query = ""
            top_k_raw = "5"

            if "application/json" in content_type:
                try:
                    body = json.loads(body_bytes.decode("utf-8", errors="replace"))
                    query = str(body.get("query") or "").strip()
                    top_k_raw = str(body.get("top_k") or "5").strip()
                except Exception:
                    query = ""
                    top_k_raw = "5"
            else:
                form = parse_qs(body_bytes.decode("utf-8", errors="replace"))
                query = (form.get("query", [""])[0] or "").strip()
                top_k_raw = (form.get("top_k", ["5"])[0] or "5").strip()

            try:
                top_k = max(1, min(10, int(top_k_raw)))
            except ValueError:
                top_k = 5

            return query, top_k

        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            if parsed_url.path == "/":
                qs = parse_qs(parsed_url.query)
                query = (qs.get("query", [""])[0] or "").strip()
                top_k_raw = (qs.get("top_k", ["5"])[0] or "5").strip()
                try:
                    top_k = max(1, min(10, int(top_k_raw)))
                except ValueError:
                    top_k = 5

                page = _page(initial_query=query, initial_top_k=top_k)
                self._send_html(page)
                return

            if parsed_url.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return

            self._send_html(b"<h1>404</h1><p>Not Found</p>", status=404)

        def do_POST(self) -> None:
            if self.path not in {"/api/recommend", "/recommend"}:
                self._send_json({"error": "Not Found"}, status=404)
                return

            query, top_k = self._parse_query_topk()
            if not query:
                self._send_json({"error": "질문을 입력해 주세요."}, status=400)
                return

            try:
                result = recommend_games(
                    db_path=db_path,
                    query=query,
                    top_k=top_k,
                    model_name=model_name,
                    openai_api_key=openai_api_key,
                    openai_model=openai_model,
                )
                self._send_json(_prepare_result_payload(result, query=query, top_k=top_k))
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"error": html.escape(str(exc))}, status=500)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"React Web UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
