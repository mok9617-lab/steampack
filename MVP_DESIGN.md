# Steam Review-Based Game Recommendation MVP Design

## 1) Goal and Scope
- Goal: Accept a Korean user query, recommend Steam Top 5 games, and provide evidence-based reasons in Korean.
- Input: Korean free-text query (example: "힐링되는 싱글 RPG 추천해줘. 공포는 싫어")
- Output:
  - Top 5 recommended games
  - Evidence summary per game (matched constraints + key review evidence)
  - 2-sentence Korean explanation grounded in review evidence
- Data scope:
  - Candidate games: 500
  - Review window: last 1 year
  - Review languages: ko/en

## 2) Finalized Decisions
- Candidate pool size: 500
- Language strategy: Korean query + Korean/English reviews
- Review period: recent 1 year
- Ranking policy: hard filter -> semantic similarity retrieval -> evidence-based re-ranking
- Constraint handling: mixed policy
  - Hard filter for forbidden/must-have constraints
  - Soft preferences used only in re-ranking tie-break
- Explanation: evidence retrieval + LLM summarization
- Evaluation: fixed manual query set

## 3) Architecture
- `collector`: fetch game metadata and reviews from Steam
- `preprocessor`: language filter, cleanup, quality filtering
- `feature_builder`: embeddings, sentiment, keyword/topic features
- `query_parser`: parse user intent into structured constraints
- `ranker`: hard-filter + similarity ranking + evidence re-ranking
- `explainer`: retrieve evidence reviews and generate Korean reasons
- `evaluator`: run fixed query-set and log scores

## 4) Data Model (SQLite)
```sql
CREATE TABLE games (
  app_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  release_date TEXT,
  genres TEXT,              -- JSON array string
  tags TEXT,                -- JSON array string
  positive_ratio REAL,
  review_count INTEGER,
  updated_at TEXT
);

CREATE TABLE reviews (
  review_id TEXT PRIMARY KEY,
  app_id INTEGER NOT NULL,
  language TEXT NOT NULL,       -- 'ko' or 'en'
  review_text TEXT NOT NULL,
  cleaned_text TEXT,
  voted_up INTEGER NOT NULL,    -- 1/0
  votes_up INTEGER,
  weighted_vote_score REAL,
  steam_purchase INTEGER,
  received_for_free INTEGER,
  playtime_forever INTEGER,      -- total playtime minutes
  playtime_at_review INTEGER,    -- playtime at review-post time
  review_date TEXT,
  sentiment_score REAL,
  review_weight REAL,            -- trust weight after preprocessing
  embedding BLOB,
  FOREIGN KEY (app_id) REFERENCES games(app_id)
);

CREATE TABLE game_profiles (
  app_id INTEGER PRIMARY KEY,
  profile_embedding BLOB,
  top_keywords TEXT,            -- JSON array string
  mood_tags TEXT,               -- JSON array string
  recent_review_count INTEGER,  -- number of valid reviews in last 1 year
  positive_ratio_1y REAL,
  median_playtime_1y REAL,
  updated_at TEXT,
  FOREIGN KEY (app_id) REFERENCES games(app_id)
);

CREATE TABLE eval_queries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_text TEXT NOT NULL,
  expected_constraints TEXT,
  created_at TEXT
);

CREATE TABLE eval_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  eval_query_id INTEGER NOT NULL,
  app_id INTEGER NOT NULL,
  rank INTEGER NOT NULL,
  relevance_score INTEGER,      -- 1~5 manual
  reason_quality_score INTEGER, -- 1~5 manual
  notes TEXT,
  created_at TEXT,
  FOREIGN KEY (eval_query_id) REFERENCES eval_queries(id),
  FOREIGN KEY (app_id) REFERENCES games(app_id)
);
```

## 5) Preprocessing Spec v1 (Important)

### A. Column selection
- Required:
  - `app_id, review_id, language, review_text, voted_up, review_date`
  - `playtime_forever, playtime_at_review`
  - `votes_up, weighted_vote_score, steam_purchase, received_for_free`
- Optional:
  - Author profile fields (for later trust tuning)

### B. Hard data filters
- Keep only reviews within last 365 days
- Keep only `ko` and `en`
- Remove very short reviews (less than 5 tokens)
- Remove duplicated or near-duplicated spam patterns

### C. Text normalization
- Common:
  - strip URLs/noise, normalize whitespace
  - preserve game/genre keywords
- Korean:
  - tokenize with Korean analyzer, keep key nouns/adjectives
- English:
  - lowercase + light normalization (avoid over-cleaning)

### D. Review trust policy (includes playtime, no fixed formula)
- Assign review trust labels instead of weighted score:
  - `high`: meaningful playtime + purchase signal + normal text quality
  - `medium`: partially satisfied trust signals
  - `low`: very low playtime or suspected spam/noise
- Usage rule:
  - `low` trust reviews are excluded from evidence extraction
  - profile embedding is built from `high/medium` reviews only
  - if a game has too few valid reviews after filtering, mark as low-confidence candidate
- Trust signals:
  - playtime (`playtime_forever`, `playtime_at_review`)
  - purchase/free flags
  - community votes (`votes_up`, `weighted_vote_score`)
  - text quality heuristics (length/diversity/repetition)

### E. Game-level aggregation
- Build game embedding from valid (`high/medium`) reviews
- Extract top positive/negative keywords
- Store interpretable indicators:
  - `recent_review_count`
  - `positive_ratio_1y`
  - `median_playtime_1y`

## 6) Why Playtime Is Important
- Playtime is one of the strongest reliability signals in game reviews.
- A review from a user with meaningful playtime is generally more trustworthy for recommendation.
- Playtime also helps detect mismatch between hype and actual engagement.

### Playtime handling examples
- Review-level:
  - very low playtime reviews are not used as core evidence
  - medium/high playtime reviews are prioritized in evidence retrieval
- Game-level:
  - use `median_playtime_1y` as engagement indicator (not as hardcoded weighted score)

## 7) Ranking Strategy (no fixed weighted formula)
1) Hard filter:
- remove games violating forbidden constraints
- remove games not meeting must-have constraints

2) Similarity retrieval:
- rank remaining games by semantic similarity between query embedding and game profile embedding

3) Evidence-based re-ranking:
- re-order top candidates by rule-based evidence quality checks:
  - recent review sufficiency
  - playtime reliability of evidence reviews
  - consistency between query intent and extracted review keywords
  - soft preference match as tie-break only

4) Output confidence:
- attach confidence label (`high/medium/low`) based on evidence sufficiency and consistency

## 8) Explanation Generation
- Retrieve top 2~3 evidence reviews per recommended game
- Translate English evidence to Korean only at final response stage
- Generate concise Korean reasons with strict grounding
- Fallback to template if evidence is insufficient

## 9) API Draft
```python
def collect_data(app_ids: list[int], days: int = 365) -> None: ...
def build_profiles() -> None: ...
def parse_query(query_ko: str) -> dict: ...
def recommend(query_ko: str, top_k: int = 5) -> list[dict]: ...
def explain_recommendations(query_ko: str, recs: list[dict]) -> list[dict]: ...
```

## 10) Suggested Project Structure
```text
project_final(re)/
  steam.py
  src/
    collector.py
    preprocess.py
    features.py
    query_parser.py
    ranker.py
    explainer.py
    evaluator.py
    db.py
    config.py
  data/
    steam_mvp.db
  prompts/
    query_parser_prompt.txt
    reason_generator_prompt.txt
  tests/
    test_ranker.py
    test_query_parser.py
```

## 11) Evaluation Plan (Manual Set)
- Build fixed query set of 20~30 prompts
- Metrics per prompt:
  - recommendation relevance (1~5)
  - constraint compliance (pass/fail)
  - reason quality (1~5)
- Targets:
  - compliance >= 95%
  - relevance avg >= 3.5
  - reason quality avg >= 3.5
