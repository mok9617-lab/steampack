from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .query_parser import parse_query
from .ranker import recommend_games


@dataclass
class EvalCase:
    query: str
    preferred_genres: list[str]
    excluded: list[str]
    must_have: list[str]
    soft_prefs: list[str]
    domain: str
    difficulty: str


def _base_cases() -> list[EvalCase]:
    specs = [
        {
            "domain": "healing_single_rpg",
            "difficulty": "hard",
            "preferred": ["RPG"],
            "excluded": ["Horror"],
            "must": ["Singleplayer"],
            "soft": ["Healing"],
            "queries": [
                "힐링되는 싱글 RPG 추천해줘. 공포는 싫어",
                "편안한 싱글 RPG 추천, 호러는 말고",
                "잔잔한 솔플 RPG 찾고 있어. 공포 제외",
            ],
        },
        {
            "domain": "strategy_simulation",
            "difficulty": "easy",
            "preferred": ["Strategy", "Simulation"],
            "excluded": [],
            "must": [],
            "soft": [],
            "queries": [
                "전략 시뮬레이션 게임 추천해줘",
                "전략+시뮬 게임 추천",
                "전술적인 시뮬레이션 추천 부탁해",
            ],
        },
        {
            "domain": "multiplayer_fps",
            "difficulty": "medium",
            "preferred": ["FPS"],
            "excluded": [],
            "must": ["Multiplayer"],
            "soft": [],
            "queries": [
                "멀티로 할 만한 FPS 추천",
                "친구랑 할 FPS 멀티 게임 추천",
                "협동 가능한 FPS 추천해줘",
            ],
        },
        {
            "domain": "indie_adventure",
            "difficulty": "easy",
            "preferred": ["Indie", "Adventure"],
            "excluded": [],
            "must": [],
            "soft": [],
            "queries": [
                "인디 어드벤처 게임 찾고 있어",
                "인디 모험 게임 추천",
                "어드벤처 느낌의 인디 게임 추천해줘",
            ],
        },
        {
            "domain": "f2p_action",
            "difficulty": "easy",
            "preferred": ["Free To Play", "Action"],
            "excluded": [],
            "must": [],
            "soft": [],
            "queries": [
                "무료 플레이 액션 게임 추천",
                "F2P 액션 게임 뭐가 좋아?",
                "무료로 즐길 액션 게임 추천해줘",
            ],
        },
        {
            "domain": "exclude_horror_survival",
            "difficulty": "medium",
            "preferred": ["Survival"],
            "excluded": ["Horror"],
            "must": [],
            "soft": [],
            "queries": [
                "공포는 제외하고 생존 게임 추천",
                "무서운 건 빼고 서바이벌 추천해줘",
                "호러 말고 생존 장르 게임 추천",
            ],
        },
        {
            "domain": "story_rpg",
            "difficulty": "medium",
            "preferred": ["RPG"],
            "excluded": [],
            "must": [],
            "soft": ["StoryRich"],
            "queries": [
                "스토리 좋은 RPG 부탁해",
                "서사 중심 RPG 추천",
                "이야기 몰입감 있는 RPG 추천해줘",
            ],
        },
        {
            "domain": "fast_action",
            "difficulty": "medium",
            "preferred": ["Action"],
            "excluded": [],
            "must": [],
            "soft": ["FastPaced"],
            "queries": [
                "속도감 있는 액션 게임",
                "템포 빠른 액션 게임 추천",
                "빠르게 몰아치는 액션 게임 추천해줘",
            ],
        },
        {
            "domain": "challenge_rpg",
            "difficulty": "medium",
            "preferred": ["RPG"],
            "excluded": [],
            "must": [],
            "soft": ["Challenge"],
            "queries": [
                "어려운 도전적인 RPG",
                "난이도 높은 RPG 추천해줘",
                "하드코어한 RPG 게임 추천",
            ],
        },
        {
            "domain": "coop_adventure",
            "difficulty": "medium",
            "preferred": ["Adventure"],
            "excluded": [],
            "must": ["Multiplayer"],
            "soft": [],
            "queries": [
                "협동 멀티 어드벤처 게임",
                "친구와 같이 할 어드벤처 추천",
                "멀티 가능한 모험 게임 추천해줘",
            ],
        },
        {
            "domain": "single_sim_healing",
            "difficulty": "hard",
            "preferred": ["Simulation"],
            "excluded": [],
            "must": ["Singleplayer"],
            "soft": ["Healing"],
            "queries": [
                "싱글로 천천히 할 수 있는 시뮬레이션",
                "혼자 편하게 할 시뮬 게임 추천",
                "힐링되는 솔플 시뮬레이션 추천해줘",
            ],
        },
        {
            "domain": "indie_no_horror",
            "difficulty": "medium",
            "preferred": ["Indie"],
            "excluded": ["Horror"],
            "must": [],
            "soft": [],
            "queries": [
                "호러 말고 인디 게임",
                "공포 제외 인디 게임 추천",
                "무서운 요소 없는 인디 게임 추천해줘",
            ],
        },
        {
            "domain": "single_story_adventure",
            "difficulty": "hard",
            "preferred": ["Adventure"],
            "excluded": [],
            "must": ["Singleplayer"],
            "soft": ["StoryRich"],
            "queries": [
                "스토리 좋은 싱글 어드벤처 추천",
                "혼자 몰입해서 할 서사 어드벤처 추천",
                "싱글 플레이 위주 모험 게임 추천해줘",
            ],
        },
        {
            "domain": "hidden_gem_indie",
            "difficulty": "medium",
            "preferred": ["Indie"],
            "excluded": [],
            "must": [],
            "soft": ["HiddenGem"],
            "queries": [
                "숨겨진 명작 느낌의 인디 게임 추천",
                "덜 알려진 인디 꿀겜 추천해줘",
                "마이너하지만 좋은 인디 게임 추천",
            ],
        },
        {
            "domain": "exclude_f2p_rpg",
            "difficulty": "medium",
            "preferred": ["RPG"],
            "excluded": ["Free To Play"],
            "must": [],
            "soft": [],
            "queries": [
                "무료 플레이 말고 RPG 추천",
                "F2P 제외하고 RPG 추천해줘",
                "부분유료화 아닌 RPG 게임 추천",
            ],
        },
        {
            "domain": "single_no_multiplayer_action",
            "difficulty": "hard",
            "preferred": ["Action"],
            "excluded": ["Multiplayer"],
            "must": ["Singleplayer"],
            "soft": [],
            "queries": [
                "멀티는 싫고 싱글 액션 게임 추천",
                "온라인 요소 없는 솔플 액션 추천해줘",
                "혼자 하는 액션 게임, 멀티 제외",
            ],
        },
        {
            "domain": "cozy_simulation",
            "difficulty": "medium",
            "preferred": ["Simulation"],
            "excluded": [],
            "must": [],
            "soft": ["Healing"],
            "queries": [
                "힐링되는 시뮬레이션 추천",
                "편안한 분위기의 시뮬 게임 추천해줘",
                "느긋하게 즐길 시뮬레이션 게임 추천",
            ],
        },
        {
            "domain": "story_action",
            "difficulty": "medium",
            "preferred": ["Action"],
            "excluded": [],
            "must": [],
            "soft": ["StoryRich"],
            "queries": [
                "스토리도 괜찮은 액션 게임 추천",
                "서사 있는 액션 게임 추천해줘",
                "이야기 중심 액션 장르 추천",
            ],
        },
        {
            "domain": "multiplayer_survival",
            "difficulty": "medium",
            "preferred": ["Survival"],
            "excluded": [],
            "must": ["Multiplayer"],
            "soft": [],
            "queries": [
                "친구랑 할 생존 게임 추천",
                "협동 멀티 서바이벌 추천해줘",
                "멀티 가능한 생존 장르 게임 추천",
            ],
        },
        {
            "domain": "hard_mix_rpg",
            "difficulty": "hard",
            "preferred": ["RPG"],
            "excluded": ["Horror", "Free To Play"],
            "must": ["Singleplayer"],
            "soft": ["StoryRich", "Healing"],
            "queries": [
                "싱글 RPG 추천해줘. 공포랑 무료플레이는 싫고 스토리 좋은 힐링 느낌이면 좋겠어",
                "호러/F2P 제외하고 혼자 하는 스토리 RPG 추천",
                "솔플 RPG 중에서 편안하고 서사 좋은 게임 추천, 공포는 빼줘",
            ],
        },
    ]

    out: list[EvalCase] = []
    for spec in specs:
        for q in spec["queries"]:
            out.append(
                EvalCase(
                    query=q,
                    preferred_genres=list(spec["preferred"]),
                    excluded=list(spec["excluded"]),
                    must_have=list(spec["must"]),
                    soft_prefs=list(spec["soft"]),
                    domain=str(spec["domain"]),
                    difficulty=str(spec["difficulty"]),
                )
            )
    return out


def _noisy_variants(query: str) -> list[str]:
    return [
        query,
        query.replace(" ", ""),
        query.replace("추천", "추천좀"),
        f"!!! {query} ???",
    ]


def _contains_all(expected: list[str], actual: list[str]) -> bool:
    aset = set(actual)
    return all(x in aset for x in expected)


def _genre_set(result_item: dict) -> set[str]:
    return {str(g).lower() for g in (result_item.get("genres") or [])}


def _tag_set(result_item: dict) -> set[str]:
    return {str(t).lower() for t in (result_item.get("tags") or [])}


def _has_multiplayer_signal(item: dict) -> bool:
    if "multiplayer_signal" in item:
        return bool(item.get("multiplayer_signal"))
    genres = _genre_set(item)
    tags = _tag_set(item)
    if "multiplayer" in genres or "massively multiplayer" in genres:
        return True
    if "multiplayer" in tags or "co-op" in tags:
        return True
    joined = " ".join(item.get("evidence_reviews", [])).lower()
    signals = ["multiplayer", "co-op", "coop", "online", "멀티", "협동", "파티", "친구랑"]
    return any(s in joined for s in signals)


def _preferred_hit(item: dict, expected_preferred: list[str]) -> bool:
    if not expected_preferred:
        return True
    if int(item.get("preferred_genre_hits", 0)) > 0:
        return True
    genres = _genre_set(item)
    tags = _tag_set(item)
    for pg in expected_preferred:
        lpg = pg.lower()
        if lpg in genres or lpg in tags:
            return True
        if lpg == "free to play" and ("free to play" in genres or "free to play" in tags or "f2p" in tags):
            return True
    return False


def _hard_violation(item: dict, case: EvalCase) -> bool:
    genres = _genre_set(item)
    tags = _tag_set(item)

    if "Horror" in case.excluded and ("horror" in genres or "horror" in tags):
        return True
    if "Free To Play" in case.excluded and ("free to play" in genres or "free to play" in tags or "f2p" in tags):
        return True

    multiplayer_signal = _has_multiplayer_signal(item)
    if "Singleplayer" in case.must_have and multiplayer_signal:
        return True
    if "Multiplayer" in case.must_have and not multiplayer_signal:
        return True

    return False


def _relevance(item: dict, case: EvalCase) -> int:
    if _hard_violation(item, case):
        return 0
    return 1 if _preferred_hit(item, case.preferred_genres) else 0


def _dcg(rels: list[int]) -> float:
    score = 0.0
    for i, rel in enumerate(rels, start=1):
        if rel <= 0:
            continue
        score += (2**rel - 1) / math.log2(i + 1)
    return score


def _new_counter() -> dict[str, float | int]:
    return {
        "parse_total": 0,
        "parse_pass": 0,
        "rec_total": 0,
        "rec_nonempty": 0,
        "rec_hit": 0,
        "rec_pref_hit": 0,
        "rec_rr_sum": 0.0,
        "rec_precision_sum": 0.0,
        "rec_ndcg_sum": 0.0,
        "rec_lists_with_violation": 0,
        "rec_violation_items": 0,
        "rec_items_total": 0,
    }


def _counter_to_metrics(counter: dict[str, float | int]) -> dict[str, float | int]:
    parse_total = int(counter["parse_total"])
    parse_pass = int(counter["parse_pass"])
    rec_total = int(counter["rec_total"])
    rec_nonempty = int(counter["rec_nonempty"])
    rec_hit = int(counter["rec_hit"])
    rec_pref_hit = int(counter["rec_pref_hit"])
    rec_rr_sum = float(counter["rec_rr_sum"])
    rec_precision_sum = float(counter["rec_precision_sum"])
    rec_ndcg_sum = float(counter["rec_ndcg_sum"])
    rec_lists_with_violation = int(counter["rec_lists_with_violation"])
    rec_violation_items = int(counter["rec_violation_items"])
    rec_items_total = int(counter["rec_items_total"])
    return {
        "parser": {
            "total_variants": parse_total,
            "pass_count": parse_pass,
            "pass_rate": round(parse_pass / parse_total, 4) if parse_total else 0.0,
        },
        "recommendation": {
            "total_cases": rec_total,
            "nonempty_rate": round(rec_nonempty / rec_total, 4) if rec_total else 0.0,
            "preferred_genre_hit_rate": round(rec_pref_hit / rec_total, 4) if rec_total else 0.0,
            "hit_at_k": round(rec_hit / rec_total, 4) if rec_total else 0.0,
            "mrr_at_k": round(rec_rr_sum / rec_total, 4) if rec_total else 0.0,
            "precision_at_k": round(rec_precision_sum / rec_total, 4) if rec_total else 0.0,
            "ndcg_at_k": round(rec_ndcg_sum / rec_total, 4) if rec_total else 0.0,
            "hard_constraint_violation_rate": round(rec_lists_with_violation / rec_total, 4) if rec_total else 0.0,
            "hard_violation_item_rate": round(rec_violation_items / rec_items_total, 4) if rec_items_total else 0.0,
        },
    }


def run_evaluation(
    db_path: Path,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    top_k: int = 5,
) -> dict:
    cases = _base_cases()

    parse_total = 0
    parse_pass = 0
    parse_failures: list[dict] = []

    rec_total = 0
    rec_nonempty = 0
    rec_pref_hit = 0
    rec_hit = 0
    rec_rr_sum = 0.0
    rec_precision_sum = 0.0
    rec_ndcg_sum = 0.0
    rec_lists_with_violation = 0
    rec_violation_items = 0
    rec_items_total = 0
    rec_failures: list[dict] = []
    by_domain: dict[str, dict[str, float | int]] = {}
    by_difficulty: dict[str, dict[str, float | int]] = {}

    for case in cases:
        domain_counter = by_domain.setdefault(case.domain, _new_counter())
        diff_counter = by_difficulty.setdefault(case.difficulty, _new_counter())

        for query_variant in _noisy_variants(case.query):
            parse_total += 1
            domain_counter["parse_total"] = int(domain_counter["parse_total"]) + 1
            diff_counter["parse_total"] = int(diff_counter["parse_total"]) + 1
            parsed = parse_query(query_variant)
            ok = (
                _contains_all(case.preferred_genres, parsed.preferred_genres)
                and _contains_all(case.excluded, parsed.excluded_genres_or_moods)
                and _contains_all(case.must_have, parsed.must_have)
                and _contains_all(case.soft_prefs, parsed.soft_preferences)
            )
            if ok:
                parse_pass += 1
                domain_counter["parse_pass"] = int(domain_counter["parse_pass"]) + 1
                diff_counter["parse_pass"] = int(diff_counter["parse_pass"]) + 1
            else:
                parse_failures.append(
                    {
                        "query": query_variant,
                        "expected": {
                            "preferred_genres": case.preferred_genres,
                            "excluded": case.excluded,
                            "must_have": case.must_have,
                            "soft_prefs": case.soft_prefs,
                        },
                        "actual": parsed.to_dict(),
                        "domain": case.domain,
                        "difficulty": case.difficulty,
                    }
                )

        rec_total += 1
        domain_counter["rec_total"] = int(domain_counter["rec_total"]) + 1
        diff_counter["rec_total"] = int(diff_counter["rec_total"]) + 1
        rec = recommend_games(db_path=db_path, query=case.query, top_k=top_k, model_name=model_name)
        results = list(rec.get("results", []))

        if results:
            rec_nonempty += 1
            domain_counter["rec_nonempty"] = int(domain_counter["rec_nonempty"]) + 1
            diff_counter["rec_nonempty"] = int(diff_counter["rec_nonempty"]) + 1
        else:
            rec_failures.append(
                {
                    "query": case.query,
                    "reason": "empty_results",
                    "domain": case.domain,
                    "difficulty": case.difficulty,
                }
            )
            continue

        rels: list[int] = []
        first_rel_rank: int | None = None
        list_has_violation = False
        has_preferred_hit = False

        for rank, item in enumerate(results[:top_k], start=1):
            v = _hard_violation(item, case)
            if v:
                list_has_violation = True
                rec_violation_items += 1
                domain_counter["rec_violation_items"] = int(domain_counter["rec_violation_items"]) + 1
                diff_counter["rec_violation_items"] = int(diff_counter["rec_violation_items"]) + 1

            rel = _relevance(item, case)
            rels.append(rel)
            if rel > 0 and first_rel_rank is None:
                first_rel_rank = rank
            if _preferred_hit(item, case.preferred_genres):
                has_preferred_hit = True

        k_used = len(rels)
        rec_items_total += k_used
        domain_counter["rec_items_total"] = int(domain_counter["rec_items_total"]) + k_used
        diff_counter["rec_items_total"] = int(diff_counter["rec_items_total"]) + k_used

        if first_rel_rank is not None:
            rec_hit += 1
            rec_rr_sum += 1.0 / float(first_rel_rank)
            domain_counter["rec_hit"] = int(domain_counter["rec_hit"]) + 1
            diff_counter["rec_hit"] = int(diff_counter["rec_hit"]) + 1
            domain_counter["rec_rr_sum"] = float(domain_counter["rec_rr_sum"]) + (1.0 / float(first_rel_rank))
            diff_counter["rec_rr_sum"] = float(diff_counter["rec_rr_sum"]) + (1.0 / float(first_rel_rank))

        if has_preferred_hit:
            rec_pref_hit += 1
            domain_counter["rec_pref_hit"] = int(domain_counter["rec_pref_hit"]) + 1
            diff_counter["rec_pref_hit"] = int(diff_counter["rec_pref_hit"]) + 1

        precision_val = (sum(rels) / k_used) if k_used else 0.0
        rec_precision_sum += precision_val
        domain_counter["rec_precision_sum"] = float(domain_counter["rec_precision_sum"]) + precision_val
        diff_counter["rec_precision_sum"] = float(diff_counter["rec_precision_sum"]) + precision_val

        ideal_rels = sorted(rels, reverse=True)
        dcg = _dcg(rels)
        idcg = _dcg(ideal_rels)
        ndcg_val = (dcg / idcg) if idcg > 0 else 0.0
        rec_ndcg_sum += ndcg_val
        domain_counter["rec_ndcg_sum"] = float(domain_counter["rec_ndcg_sum"]) + ndcg_val
        diff_counter["rec_ndcg_sum"] = float(diff_counter["rec_ndcg_sum"]) + ndcg_val

        if list_has_violation:
            rec_lists_with_violation += 1
            domain_counter["rec_lists_with_violation"] = int(domain_counter["rec_lists_with_violation"]) + 1
            diff_counter["rec_lists_with_violation"] = int(diff_counter["rec_lists_with_violation"]) + 1
            rec_failures.append(
                {
                    "query": case.query,
                    "reason": "hard_constraint_violation",
                    "domain": case.domain,
                    "difficulty": case.difficulty,
                }
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "parser_metrics": {
            "total_variants": parse_total,
            "pass_count": parse_pass,
            "pass_rate": round(parse_pass / parse_total, 4) if parse_total else 0.0,
        },
        "recommendation_metrics": {
            "total_cases": rec_total,
            "nonempty_rate": round(rec_nonempty / rec_total, 4) if rec_total else 0.0,
            "preferred_genre_hit_rate": round(rec_pref_hit / rec_total, 4) if rec_total else 0.0,
            "hit_at_k": round(rec_hit / rec_total, 4) if rec_total else 0.0,
            "mrr_at_k": round(rec_rr_sum / rec_total, 4) if rec_total else 0.0,
            "precision_at_k": round(rec_precision_sum / rec_total, 4) if rec_total else 0.0,
            "ndcg_at_k": round(rec_ndcg_sum / rec_total, 4) if rec_total else 0.0,
            "hard_constraint_violation_rate": round(rec_lists_with_violation / rec_total, 4) if rec_total else 0.0,
            "hard_violation_item_rate": round(rec_violation_items / rec_items_total, 4) if rec_items_total else 0.0,
        },
        "segment_metrics": {
            "by_domain": {key: _counter_to_metrics(counter) for key, counter in sorted(by_domain.items())},
            "by_difficulty": {
                key: _counter_to_metrics(counter) for key, counter in sorted(by_difficulty.items())
            },
        },
        "parse_failures": parse_failures[:20],
        "recommendation_failures": rec_failures[:20],
    }
    return report


def save_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
