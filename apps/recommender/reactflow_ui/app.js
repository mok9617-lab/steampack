function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "className") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, String(v));
  }
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child == null) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function fmtDelta(v) {
  if (typeof v !== "number") return "-";
  return `${v >= 0 ? "+" : ""}${v.toFixed(4)}`;
}

function renderTable(columns, rows) {
  const table = el("table");
  const thead = el("thead");
  const trh = el("tr");
  columns.forEach((c) => trh.appendChild(el("th", { text: c })));
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = el("tbody");
  rows.forEach((row) => {
    const tr = el("tr");
    columns.forEach((c) => tr.appendChild(el("td", { text: row[c] ?? "-" })));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return el("div", { className: "table-wrap" }, table);
}

function renderPipelineFlow(flow, selectedId, onSelect) {
  const stages = flow?.nodes || [];
  if (stages.length === 0) {
    return el("div", { className: "flow-simple", text: "Flow data is empty." });
  }
  const labelById = Object.fromEntries(stages.map((s) => [s.id, s.label]));
  const label = (id, fallback) => labelById[id] || fallback;

  const baseWidth = 1880;
  const baseHeight = 420;
  const targetWidth = Math.max(900, Math.min(1220, window.innerWidth - 120));
  const scale = Math.min(1, targetWidth / baseWidth);

  const W = Math.round(baseWidth * scale);
  const H = Math.round(baseHeight * scale);
  const nodeW = Math.round(130 * scale);
  const nodeH = Math.round(102 * scale);

  const P = {
    n1: {
      x: Math.round(20 * scale),
      y: Math.round(140 * scale),
      id: "input",
      text: label("input", "1)"),
    },
    n2: {
      x: Math.round(210 * scale),
      y: Math.round(140 * scale),
      id: "parse",
      text: label("parse", "2)"),
    },
    n3: {
      x: Math.round(390 * scale),
      y: Math.round(140 * scale),
      id: "route",
      text: label("route", "3)"),
    },
    n4a: {
      x: Math.round(600 * scale),
      y: Math.round(40 * scale),
      id: "normal_path",
      text: label("normal_path", "4-a)"),
    },
    n4b: {
      x: Math.round(600 * scale),
      y: Math.round(230 * scale),
      id: "similar_path",
      text: label("similar_path", "4-b)"),
    },
    n4b1: {
      x: Math.round(840 * scale),
      y: Math.round(140 * scale),
      id: "similar_db_found",
      text: label("similar_db_found", "4-b1)"),
    },
    n4b2: {
      x: Math.round(840 * scale),
      y: Math.round(310 * scale),
      id: "similar_db_missing",
      text: label("similar_db_missing", "4-b2)"),
    },
    n5: {
      x: Math.round(1040 * scale),
      y: Math.round(140 * scale),
      id: "merge",
      text: label("merge", "5)"),
    },
    n6: {
      x: Math.round(1200 * scale),
      y: Math.round(140 * scale),
      id: "retrieve",
      text: label("retrieve", "6)"),
    },
    n7: {
      x: Math.round(1380 * scale),
      y: Math.round(140 * scale),
      id: "score",
      text: label("score", "7)"),
    },
    n8: {
      x: Math.round(1560 * scale),
      y: Math.round(140 * scale),
      id: "reason",
      text: label("reason", "8)"),
    },
    n9: {
      x: Math.round(1740 * scale),
      y: Math.round(140 * scale),
      id: "final_output",
      text: label("final_output", "9)"),
    },
  };

  const board = el("div", { className: "flow-board" });
  const canvas = el("div", { className: "flow-canvas-diagram", style: `width:${W}px;height:${H}px;` });
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "flow-svg-diagram");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  marker.setAttribute("id", "arrowhead");
  marker.setAttribute("markerWidth", "10");
  marker.setAttribute("markerHeight", "8");
  marker.setAttribute("refX", "8");
  marker.setAttribute("refY", "4");
  marker.setAttribute("orient", "auto");
  const arrowPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  arrowPath.setAttribute("d", "M0,0 L0,8 L8,4 z");
  arrowPath.setAttribute("fill", "#66c3ff");
  marker.appendChild(arrowPath);
  defs.appendChild(marker);
  svg.appendChild(defs);

  const draw = (pts) => {
    const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    path.setAttribute("points", pts.map((p) => `${p.x},${p.y}`).join(" "));
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#66c3ff");
    path.setAttribute("stroke-width", String(Math.max(1.4, 2 * scale)));
    path.setAttribute("marker-end", "url(#arrowhead)");
    svg.appendChild(path);
  };

  const midL = (n) => ({ x: n.x, y: n.y + Math.round(nodeH / 2) });
  const midR = (n) => ({ x: n.x + nodeW, y: n.y + Math.round(nodeH / 2) });

  draw([midR(P.n1), midL(P.n2)]);
  draw([midR(P.n2), midL(P.n3)]);

  const j1x = Math.round(560 * scale);
  draw([midR(P.n3), { x: j1x, y: midR(P.n3).y }, { x: j1x, y: midL(P.n4a).y }, midL(P.n4a)]);
  draw([midR(P.n3), { x: j1x, y: midR(P.n3).y }, { x: j1x, y: midL(P.n4b).y }, midL(P.n4b)]);

  const j2x = Math.round(780 * scale);
  draw([midR(P.n4b), { x: j2x, y: midR(P.n4b).y }, { x: j2x, y: midL(P.n4b1).y }, midL(P.n4b1)]);
  draw([midR(P.n4b), { x: j2x, y: midR(P.n4b).y }, { x: j2x, y: midL(P.n4b2).y }, midL(P.n4b2)]);

  const j3x = Math.round(1010 * scale);
  draw([midR(P.n4a), { x: j3x, y: midR(P.n4a).y }, { x: j3x, y: midL(P.n5).y }, midL(P.n5)]);
  draw([midR(P.n4b1), { x: j3x, y: midR(P.n4b1).y }, { x: j3x, y: midL(P.n5).y }, midL(P.n5)]);
  draw([midR(P.n4b2), { x: j3x, y: midR(P.n4b2).y }, { x: j3x, y: midL(P.n5).y }, midL(P.n5)]);

  draw([midR(P.n5), midL(P.n6)]);
  draw([midR(P.n6), midL(P.n7)]);
  draw([midR(P.n7), midL(P.n8)]);
  draw([midR(P.n8), midL(P.n9)]);

  canvas.appendChild(svg);

  const makeNode = (n) =>
    el(
      "button",
      {
        className: `flow-node-diagram ${selectedId === n.id ? "active" : ""}`,
        style: `left:${n.x}px; top:${n.y}px; width:${nodeW}px; height:${nodeH}px; font-size:${Math.max(9, Math.round(14 * scale))}px;`,
        onclick: () => onSelect(n.id),
      },
      n.text
    );

  [P.n1, P.n2, P.n3, P.n4a, P.n4b, P.n4b1, P.n4b2, P.n5, P.n6, P.n7, P.n8, P.n9].forEach((n) =>
    canvas.appendChild(makeNode(n))
  );

  board.appendChild(canvas);
  return board;
}

function renderApp(data) {
  const app = document.getElementById("app");
  app.innerHTML = "";

  const state = {
    activeTab: "pipeline",
    selectedStageId: data.pipeline_stages?.[0]?.id || "",
  };

  const title = el("h1", { className: "header-title", text: "추천 시스템 대시보드" });
  const sub = el("p", {
    className: "header-sub",
    text: "탭 1: 파이프라인 프로세스 | 탭 2: 버전별 업그레이드",
  });
  app.appendChild(title);
  app.appendChild(sub);

  const content = el("div");

  function renderPipelineTab() {
    const stages = data.pipeline_stages || [];
    const flow = data.pipeline_flow || {};
    const selected = stages.find((s) => s.id === state.selectedStageId) || stages[0];

    const card = el("div", { className: "card" });
    card.appendChild(
      renderPipelineFlow(flow, selected?.id, (id) => {
        state.selectedStageId = id;
        repaint();
      })
    );
    card.appendChild(
      el("p", {
        className: "flow-hint",
        text: "화살표는 직각 연결로 표시합니다. 박스를 클릭하면 단계 상세가 아래에 표시됩니다.",
      })
    );

    const buttons = el("div", { className: "stage-button-grid" });
    stages.forEach((s) => {
      buttons.appendChild(
        el(
          "button",
          {
            className: `stage-button ${s.id === selected?.id ? "active" : ""}`,
            onclick: () => {
              state.selectedStageId = s.id;
              repaint();
            },
          },
          s.label
        )
      );
    });

    const detail = el("div", { className: "card", style: "margin-top:12px;" });
    detail.appendChild(el("h3", { text: selected?.label || "-" }));

    const grid = el("div", { className: "stage-grid" });
    grid.appendChild(el("div", {}, [el("div", { className: "kv-title", text: "INPUT" }), el("div", { text: selected?.input || "-" })]));
    grid.appendChild(el("div", {}, [el("div", { className: "kv-title", text: "PROCESS" }), el("div", { text: selected?.process || "-" })]));
    grid.appendChild(el("div", {}, [el("div", { className: "kv-title", text: "OUTPUT" }), el("div", { text: selected?.output || "-" })]));
    detail.appendChild(grid);

    detail.appendChild(el("h4", { style: "margin:14px 0 8px;", text: "핵심 변수 설명" }));
    detail.appendChild(
      renderTable(
        ["변수", "설명"],
        (selected?.variables || []).map((v) => ({ 변수: v.name, 설명: v.desc }))
      )
    );

    if ((selected?.weights || []).length > 0) {
      detail.appendChild(el("h4", { style: "margin:14px 0 8px;", text: "가중치/공식과 배경" }));
      detail.appendChild(
        renderTable(
          ["항목", "공식/가중치", "왜 이 비중인가"],
          selected.weights.map((w) => ({
            항목: w.name,
            "공식/가중치": w.value,
            "왜 이 비중인가": w.why,
          }))
        )
      );
    }

    return el("div", {}, [card, buttons, detail]);
  }

  function renderVersionTab() {
    const history = data.version_history || [];
    const transitions = data.transitions || [];
    const summary = data.summary || {};

    const top = el("div", { className: "card" });
    top.appendChild(el("h3", { text: "버전별 핵심 변화(수치 기반)" }));
    top.appendChild(
      renderTable(
        [
          "버전",
          "평가 케이스",
          "pass_rate",
          "parse_fail",
          "nonempty_rate",
          "genre_hit_rate",
          "precision@k",
          "ndcg@k",
          "hard_violation_rate",
          "reco_fail",
          "해석",
        ],
        history.map((row) => ({
          버전: `v${row.version}`,
          "평가 케이스": row.case_count,
          pass_rate: row.pass_rate,
          parse_fail: row.parse_fail_count,
          nonempty_rate: row.nonempty_rate,
          genre_hit_rate: row.genre_hit_rate,
          "precision@k": row.precision_at_k,
          "ndcg@k": row.ndcg_at_k,
          hard_violation_rate: row.hard_violation_rate,
          reco_fail: row.reco_fail_count,
          해석: row.note,
        }))
      )
    );

    const summaryBox = el("div", { className: "summary-box" });
    summaryBox.appendChild(el("div", { className: "summary-item", text: `- v1~v7: ${summary.v1_to_v7 || "-"}` }));
    summaryBox.appendChild(el("div", { className: "summary-item", text: `- v8: ${summary.v8 || "-"}` }));
    summaryBox.appendChild(el("div", { className: "summary-item", text: `- v9~v10: ${summary.v9_to_v10 || "-"}` }));
    summaryBox.appendChild(el("div", { className: "summary-item", text: `- 한 줄 결론: ${summary.one_liner || "-"}` }));
    top.appendChild(summaryBox);

    const grid = el("div", { className: "transition-grid" });
    transitions.forEach((t) => {
      const card = el("div", { className: "card" });
      card.appendChild(el("h4", { text: t.key }));
      card.appendChild(el("div", { text: `- 무엇이 바뀌었나: ${t.what}` }));
      card.appendChild(el("div", { style: "margin-top:5px;", text: `- 왜 바뀌었나: ${t.why}` }));
      card.appendChild(
        el("div", {
          className: "delta-line",
          text:
            `Δ pass ${fmtDelta(t.deltas.pass_rate)}, nonempty ${fmtDelta(t.deltas.nonempty_rate)}, ` +
            `genre ${fmtDelta(t.deltas.genre_hit_rate)}, hard ${fmtDelta(t.deltas.hard_violation_rate)}, ` +
            `parse_fail ${t.deltas.parse_fail_count >= 0 ? "+" : ""}${t.deltas.parse_fail_count}, ` +
            `reco_fail ${t.deltas.reco_fail_count >= 0 ? "+" : ""}${t.deltas.reco_fail_count}`,
        })
      );
      if (t.case_count_changed) {
        card.appendChild(el("div", { className: "delta-line", text: `평가셋 변경: ${t.case_count_from} -> ${t.case_count_to}` }));
      }
      grid.appendChild(card);
    });

    return el("div", {}, [top, grid]);
  }

  function repaint() {
    const tabRow = el("div", { className: "tab-row" });
    tabRow.appendChild(
      el(
        "button",
        {
          className: `tab-button ${state.activeTab === "pipeline" ? "active" : ""}`,
          onclick: () => {
            state.activeTab = "pipeline";
            repaint();
          },
        },
        "파이프라인 프로세스"
      )
    );
    tabRow.appendChild(
      el(
        "button",
        {
          className: `tab-button ${state.activeTab === "versions" ? "active" : ""}`,
          onclick: () => {
            state.activeTab = "versions";
            repaint();
          },
        },
        "버전별 업그레이드"
      )
    );

    content.innerHTML = "";
    content.appendChild(tabRow);
    content.appendChild(state.activeTab === "pipeline" ? renderPipelineTab() : renderVersionTab());
  }

  app.appendChild(content);
  repaint();
}

function renderError(message) {
  const app = document.getElementById("app");
  app.innerHTML = "";
  app.appendChild(el("div", { className: "card", text: `화면 로딩 실패: ${message}` }));
}

fetch("/api/dashboard-data")
  .then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  })
  .then(renderApp)
  .catch((err) => renderError(String(err)));
