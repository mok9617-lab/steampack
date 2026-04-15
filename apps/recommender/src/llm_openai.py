from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class OpenAILLM:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.client = self._build_client()
        self.model = model
        self.errors: list[str] = []

    def _build_client(self) -> OpenAI:
        # Add explicit timeout/retry to survive transient transport instability.
        return OpenAI(api_key=self.api_key, timeout=25.0, max_retries=2)

    def _reset_client(self) -> None:
        self.client = self._build_client()

    def _chat(self, system: str, user: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                msg = resp.choices[0].message.content or ""
                return msg.strip()
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "client has been closed" in msg or "cannot send a request" in msg:
                    try:
                        self._reset_client()
                    except Exception:
                        pass
                if attempt < 2:
                    time.sleep(0.35 * (attempt + 1))
                    continue
        err_name = type(last_exc).__name__ if last_exc is not None else "UnknownError"
        self.errors.append(f"chat: {err_name}: {last_exc}")
        return ""

    def _empty_parse_payload(self, with_rewrite: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "preferred_genres": [],
            "excluded_genres_or_moods": [],
            "excluded_terms": [],
            "must_have": [],
            "soft_preferences": [],
            "play_style": [],
            "session_length": [],
            "difficulty": [],
            "focus": [],
        }
        if with_rewrite:
            payload["rewritten_query"] = ""
        return payload

    def _load_json_object(self, raw: str) -> dict[str, Any]:
        match = JSON_RE.search(raw)
        obj = json.loads(match.group(0) if match else raw)
        if isinstance(obj, dict):
            return obj
        return {}

    def rewrite_query_for_recommendation(self, query: str) -> str:
        if not query:
            return ""
        system = (
            "Return only JSON.\n"
            'Schema: {"rewritten_query": ""}\n'
            "Task: Rewrite a Korean user query into a recommendation-engine-friendly Korean query.\n"
            "Keep intent unchanged. Do not invent constraints.\n"
            "Output should be concise and include, when present:\n"
            "- preferred genre/mood\n"
            "- play mode (single/multi)\n"
            "- exclusions (e.g., no horror, no free-to-play)\n"
            "- reference game hint if user mentioned one\n"
        )
        raw = self._chat(system, query)
        if not raw:
            self.errors.append("rewrite_query_for_recommendation: empty_response")
            return ""
        try:
            obj = self._load_json_object(raw)
            return str(obj.get("rewritten_query", "")).strip()
        except Exception as exc:
            self.errors.append(f"rewrite_query_for_recommendation: invalid_json ({exc})")
            return ""

    def rewrite_and_parse_query(self, query: str) -> dict[str, Any]:
        if not query:
            return self._empty_parse_payload(with_rewrite=True)
        system = (
            "Return only JSON.\n"
            "Schema: "
            '{"rewritten_query":"","preferred_genres":[],"excluded_genres_or_moods":[],"excluded_terms":[],"must_have":[],"soft_preferences":[],"play_style":[],"session_length":[],"difficulty":[],"focus":[]}\n'
            "Task:\n"
            "1) Rewrite Korean user query to recommendation-engine-friendly Korean query.\n"
            "2) Parse intent into structured fields.\n"
            "Rules:\n"
            "- Keep intent unchanged and concise.\n"
            "- Preserve user-specific traits; do NOT over-generalize.\n"
            "- Keep concrete hints when present: mood, time-length, difficulty, control burden, social mode.\n"
            "- Do not invent constraints.\n"
            "- preferred_genres allowed: RPG, Action, Adventure, Strategy, Simulation, Survival, FPS, Horror, Indie, Free To Play, Casual\n"
            "- excluded_genres_or_moods allowed: Horror, Free To Play, Multiplayer, RPG, Action, Adventure, Strategy, Simulation, Survival, FPS, Indie, Casual, Racing, Sports, Puzzle, Platformer, Rhythm, Visual Novel, Open World, Crafting, Anime, Card Game, Turn-Based, Violence/Gore, Sexual Content\n"
            "- must_have allowed: Singleplayer, Multiplayer\n"
            "- soft_preferences allowed: StoryRich, Healing, Challenge, FastPaced, HiddenGem\n"
            "- play_style allowed: Relaxed, Competitive, Exploration, BuildCraft, Narrative\n"
            "- session_length allowed: Short, Long\n"
            "- difficulty allowed: Easy, Hard\n"
            "- focus allowed: Combat, Story, Growth, Puzzle, Management\n"
        )
        raw = self._chat(system, query)
        if not raw:
            self.errors.append("rewrite_and_parse_query: empty_response")
            return self._empty_parse_payload(with_rewrite=True)
        try:
            obj = self._load_json_object(raw)
            out = self._empty_parse_payload(with_rewrite=True)
            out["rewritten_query"] = str(obj.get("rewritten_query", "")).strip()
            for key in [
                "preferred_genres",
                "excluded_genres_or_moods",
                "excluded_terms",
                "must_have",
                "soft_preferences",
                "play_style",
                "session_length",
                "difficulty",
                "focus",
            ]:
                out[key] = list(obj.get(key, []))
            return out
        except Exception as exc:
            self.errors.append(f"rewrite_and_parse_query: invalid_json ({exc})")
            return self._empty_parse_payload(with_rewrite=True)

    def parse_query(self, query: str) -> dict[str, list[str]]:
        system = (
            "Return only JSON.\n"
            "Schema: "
            '{"preferred_genres":[],"excluded_genres_or_moods":[],"excluded_terms":[],"must_have":[],"soft_preferences":[],"play_style":[],"session_length":[],"difficulty":[],"focus":[]}\n'
            "Allowed values:\n"
            "- preferred_genres: RPG, Action, Adventure, Strategy, Simulation, Survival, FPS, Horror, Indie, Free To Play, Casual\n"
            "- excluded_genres_or_moods: Horror, Free To Play, Multiplayer, RPG, Action, Adventure, Strategy, Simulation, Survival, FPS, Indie, Casual, Racing, Sports, Puzzle, Platformer, Rhythm, Visual Novel, Open World, Crafting, Anime, Card Game, Turn-Based, Violence/Gore, Sexual Content\n"
            "- excluded_terms: free-form Korean short terms extracted from user exclusions\n"
            "- must_have: Singleplayer, Multiplayer\n"
            "- soft_preferences: StoryRich, Healing, Challenge, FastPaced, HiddenGem\n"
            "- play_style: Relaxed, Competitive, Exploration, BuildCraft, Narrative\n"
            "- session_length: Short, Long\n"
            "- difficulty: Easy, Hard\n"
            "- focus: Combat, Story, Growth, Puzzle, Management\n"
            "Rules:\n"
            "- Keep user intent as-is\n"
            "- If user mentions exclusion in natural language, fill excluded_genres_or_moods and/or excluded_terms\n"
            "- If uncertain, return empty arrays\n"
        )
        raw = self._chat(system, query)
        if not raw:
            self.errors.append("parse_query: empty_response")
            return self._empty_parse_payload(with_rewrite=False)
        try:
            obj = self._load_json_object(raw)
            out = self._empty_parse_payload(with_rewrite=False)
            for key in [
                "preferred_genres",
                "excluded_genres_or_moods",
                "excluded_terms",
                "must_have",
                "soft_preferences",
                "play_style",
                "session_length",
                "difficulty",
                "focus",
            ]:
                out[key] = list(obj.get(key, []))
            return out
        except Exception as exc:
            self.errors.append(f"parse_query: invalid_json ({exc})")
            return self._empty_parse_payload(with_rewrite=False)

    def summarize_reviews_ko(self, query: str, game_name: str, reviews: list[str]) -> list[str]:
        if not reviews:
            return []
        joined = "\n".join(f"- {r}" for r in reviews[:5])
        system = (
            "한국어로만 답하고 주어진 리뷰를 1~3개의 짧은 근거 문장으로 요약하라. "
            "질문과 관련된 내용만 포함하고 JSON 배열만 반환하라."
        )
        user = f"질문: {query}\n게임: {game_name}\n리뷰:\n{joined}"
        raw = self._chat(system, user)
        if not raw:
            self.errors.append("summarize_reviews_ko: empty_response")
            return []
        try:
            match = ARRAY_RE.search(raw)
            arr = json.loads(match.group(0) if match else raw)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()][:3]
            return []
        except Exception as exc:
            self.errors.append(f"summarize_reviews_ko: invalid_json ({exc})")
            return []

    def generate_reason_ko(self, query: str, item: dict[str, Any], evidence_lines: list[str]) -> str:
        if not evidence_lines:
            return ""
        joined = "\n".join(f"- {x}" for x in evidence_lines[:3])
        system = (
            "한국어로만 답하고 과장 없이 2문장 이내로 추천 이유를 작성하라. "
            "반드시 제공된 근거 리뷰만 기반으로 작성하라."
        )
        user = (
            f"질문: {query}\n"
            f"게임: {item.get('name','')}\n"
            f"장르: {', '.join(item.get('genres', []))}\n"
            f"근거:\n{joined}"
        )
        out = self._chat(system, user)
        if not out:
            self.errors.append("generate_reason_ko: empty_response")
        return out

    def summarize_and_reason_ko(
        self,
        query: str,
        game_name: str,
        genres: list[str],
        reviews: list[str],
    ) -> dict[str, Any]:
        if not reviews:
            return {"summaries": [], "reason": ""}
        joined = "\n".join(f"- {r}" for r in reviews[:5])
        system = (
            "한국어로만 답하고 JSON만 반환하라.\n"
            'Schema: {"summaries":[],"reason":""}\n'
            "Rules:\n"
            "- summaries: 질문과 관련된 핵심 근거 1~3개(제공된 리뷰 기반)\n"
            "- reason: 과장 없이 2문장 이내 추천 이유\n"
            "- 제공된 리뷰 근거 밖의 사실은 만들지 말 것\n"
        )
        user = (
            f"질문: {query}\n"
            f"게임: {game_name}\n"
            f"장르: {', '.join(genres)}\n"
            f"리뷰:\n{joined}"
        )
        raw = self._chat(system, user)
        if not raw:
            self.errors.append("summarize_and_reason_ko: empty_response")
            return {"summaries": [], "reason": ""}
        try:
            obj = self._load_json_object(raw)
            summaries = obj.get("summaries", [])
            if not isinstance(summaries, list):
                summaries = []
            summaries = [str(x).strip() for x in summaries if str(x).strip()][:3]
            reason = str(obj.get("reason", "")).strip()
            return {"summaries": summaries, "reason": reason}
        except Exception as exc:
            self.errors.append(f"summarize_and_reason_ko: invalid_json ({exc})")
            return {"summaries": [], "reason": ""}

    def generate_one_liner_ko(
        self,
        query: str,
        game_name: str,
        reason_ko: str,
        caution_notes: list[str],
    ) -> str:
        base_reason = (reason_ko or "").strip()
        notes = [str(x).strip() for x in (caution_notes or []) if str(x).strip()]
        if not base_reason and not notes:
            return ""

        caution_text = "\n".join(f"- {x}" for x in notes[:2]) if notes else "- 없음"
        system = (
            "한국어로만 답하고 한 줄 평 1문장만 작성하라.\n"
            "규칙:\n"
            "- 추천 이유를 반영할 것\n"
            "- 부정 의견이 있으면 완곡하게 단점/주의점을 함께 언급할 것\n"
            "- 과장/허위 금지, 45자 이내 권장\n"
            "- 문장 외 다른 텍스트 금지\n"
        )
        user = (
            f"질문: {query}\n"
            f"게임: {game_name}\n"
            f"추천 이유: {base_reason or '없음'}\n"
            f"부정 의견:\n{caution_text}"
        )
        out = self._chat(system, user).strip()
        if not out:
            self.errors.append("generate_one_liner_ko: empty_response")
            return ""
        return out.splitlines()[0].strip()

    def guess_game_titles(self, hint: str) -> list[str]:
        if not hint:
            return []
        system = (
            "Return only JSON.\n"
            'Schema: {"candidates": []}\n'
            "Task: infer likely official game titles from a short user hint (often Korean alias).\n"
            "Rules:\n"
            "- up to 3 candidates\n"
            "- most likely first\n"
            "- include official English title when possible\n"
            "- if unsure, return empty list\n"
        )
        raw = self._chat(system, hint)
        if not raw:
            self.errors.append("guess_game_titles: empty_response")
            return []
        try:
            match = JSON_RE.search(raw)
            obj = json.loads(match.group(0) if match else raw)
            arr = obj.get("candidates", [])
            if not isinstance(arr, list):
                return []
            out: list[str] = []
            for x in arr:
                s = str(x).strip()
                if s and s.lower() not in {y.lower() for y in out}:
                    out.append(s)
            return out[:3]
        except Exception as exc:
            self.errors.append(f"guess_game_titles: invalid_json ({exc})")
            return []
