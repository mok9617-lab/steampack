# SteamPack 추천모델 보고서 작성 가이드

## 1. 프로젝트 한 줄 요약
- 한국어 자연어 질의를 입력받아 Steam 게임을 추천하고, 근거 리뷰 기반 한국어 추천 사유를 제공하는 하이브리드 추천 시스템입니다.

## 2. 시스템 개요
- 입력: 한국어 자유 질의 (예: `힐링되는 싱글 RPG 추천해줘. 공포는 싫어`)
- 출력:
  - Top-K 게임 추천
  - 근거 리뷰(evidence)
  - 한국어 추천 사유(reason_ko), 한 줄 평(one_liner_ko)
- 핵심 구성:
  - `query_parser`: 질의 정규화/의도 파싱
  - `ranker`: 하드 필터 + 유사도 검색 + 재랭킹
  - `llm_openai`: 질의 재작성, 설명 생성
  - `evaluator`: 정량 평가 및 리포트 생성

## 3. 추천 파이프라인
### Query Mode
1. Query Input  
2. Normalize (인코딩/노이즈 복구 포함)  
3. Parse/Rewrite (LLM + 규칙 파싱)  
4. Similarity Search (임베딩 기반 후보 검색)  
5. Re-rank + Filter (제외/필수조건 반영)  
6. Evidence & Reason (근거/설명 생성)  
7. Final Top-K 반환

### Similar-To Mode
- `~같은/비슷한` 표현에서 기준 게임 매칭 성공 시 활성화
- 질의 벡터 대신 **기준 게임 profile embedding**으로 유사 게임 검색

## 4. 데이터/모델
- 임베딩 모델: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384차원)
- 저장소:
  - 원천/중간 데이터: `data/`
  - 코드: `apps/recommender/src/`
  - 실행 엔트리: `steam.py`, `apps/recommender/steam.py`

## 5. 이번 개선 핵심(요약)
- 모노레포 구조 전환 (`apps/recommender`)
- 평가셋 확장: 12 -> 60케이스, domain/difficulty 분할 집계 추가
- 파서 개선:
  - 제외 의도 패턴 강화 (`싫고`, `빼줘` 등)
  - 제외 문맥 기반 추출 강화
  - 한글 인코딩(모지바케) 복구 로직 강화
  - 선호/제외 충돌 보정 로직 개선

## 6. 성능 결과 (핵심 버전 비교)
| Version | Parse Pass | Non-empty | Genre Hit | Hit@K | Precision@K | NDCG@K | Hard Violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 0.9167 | 0.7500 | 0.7500 | - | - | - | 0.0000 |
| v7 | 1.0000 | 1.0000 | 0.9167 | - | - | - | 0.0000 |
| v8 | 0.7375 | 1.0000 | 0.9333 | 0.9333 | 0.9300 | 0.9317 | 0.0167 |
| v9 | 0.7833 | 1.0000 | 1.0000 | 1.0000 | 0.9967 | 0.9984 | 0.0167 |
| v10 | **0.8167** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** |

출처: `data/eval_report_v1.json`, `data/eval_report_v7.json`, `data/eval_report_v8.json`, `data/eval_report_v9.json`, `data/eval_report_v10.json`

## 7. 세그먼트 결과 (v10)
- 총 케이스: 60
- 도메인 수: 20
- 난이도별:
  - easy: parser 1.0000 / genre hit 1.0000 / hard violation 0.0000
  - medium: parser 0.8611 / genre hit 1.0000 / hard violation 0.0000
  - hard: parser 0.6000 / genre hit 1.0000 / hard violation 0.0000

해석:
- 추천 품질/제약 준수는 거의 완성도 높음
- 남은 과제는 hard 질의의 **파서 안정성** 추가 개선

## 8. 보고서에 넣기 좋은 결론 문장(예시)
- “평가셋을 60케이스로 확장한 뒤 domain/difficulty 단위 진단을 통해 취약 구간을 특정했고, 파서/제약 필터 개선을 반복 적용해 최종 v10에서 hard constraint violation을 0으로 낮췄다.”
- “최종 모델은 추천 결과 반환률, 장르 적중, 순위 품질(Precision/NDCG)에서 모두 1.0을 달성했으며, 향후 개선 타깃은 hard 질의 파싱 강건성이다.”

## 9. 재현 방법
```bash
python steam.py evaluate --top-k 5 --output data/eval_report_v10.json
```

```bash
streamlit run apps/recommender/presentation_streamlit.py --server.fileWatcherType none
```

## 10. 한계 및 다음 단계
- 한계:
  - 일부 hard 질의에서 파서 pass_rate가 아직 완전하지 않음
  - 환경에 따라 한글 인코딩 노이즈 재발 가능성
- 다음 단계:
  - hard 질의 중심 파서 규칙/LLM 파싱 보강
  - similar-to 전용 평가셋 추가 확장
  - 사용자 피드백 기반 온라인 평가 지표 도입
