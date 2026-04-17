# Codex Handover Summary

## 1) 작업 목적 요약
- Steam 인기 게임 리뷰를 수집/분석하고, 추천/비추천 이유를 도출하는 파이프라인 구축
- 결과를 JSON/MD/Excel/Streamlit UI로 제공
- 아키텍처 문서화 및 후속 확장 가능한 구조 정리

## 2) 주요 진행 내역 (순서)
1. Steam `appreviews` 엔드포인트 실검증 완료
2. Crimson Desert(앱ID `3321460`) 리뷰 수집 코드 구현
3. 리뷰 분석 MVP(키워드/비율/리포트) 구축
4. 전체 리뷰 수집 + 시계열 + 플레이타임 구간 분석 확장
5. Streamlit 대시보드 구축
6. SteamSpy 장르 결합 파이프라인 구축
7. Top10/Top100 수집 확장
8. 카테고리별 추천 Top10 + 추천/비추천 이유 도출
9. 결과 한글화(리포트/UI 라벨) 반영

## 3) 모델/분석 버전 현황
- 감정분석 모드(steam.py):
  - `lexicon`
  - `model`
  - `hybrid` (권장)
- 모델 기반 설정:
  - 모델명: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
  - 하이브리드 임계값: `0.55` (모델 confidence)
  - strong lexicon override: `abs(lex_score) >= 2`
- 설치된 주요 패키지:
  - `transformers==5.5.1`
  - `torch==2.11.0`
  - `sentencepiece==0.2.1`
  - `openpyxl==3.1.5`
  - `pandas==2.3.3`

## 4) 핵심 스크립트
- 수집/감정분석(TopN): `steam.py`
- Top100 x 1000 수집(페이지네이션): `collect_top100_reviews.py`
- 단일 게임 텍스트 분석: `text_analysis.py`
- Top10 이유 분석: `top10_reason_analysis.py`
- 카테고리 추천 분석: `analyze_category_top100.py`
- 엑셀 변환:
  - `export_top10_to_excel.py`
  - `export_top10_reasons_to_excel.py`
- UI:
  - 기존 분석 대시보드: `streamlit_app.py`
  - 카테고리 추천 대시보드: `category_recommendation_app.py`

## 5) 주요 산출물
- Top10 감정분석: `top10_steam_korean_reviews_sentiment.json`
- Top100 리뷰 수집: `top100_reviews_1000_each.json`
- 카테고리 추천(Action): 
  - `action_top10_recommendations.json`
  - `action_top10_recommendations.md`
- 이유 요약:
  - `top10_reason_summary.json`
  - `top10_reason_summary.md`
  - `top10_reason_summary.xlsx`
- 종합 엑셀:
  - `top10_steam_korean_reviews_sentiment.xlsx`
  - `top10_steam_korean_reviews_sentiment_text.xlsx`

## 6) 문서
- 아키텍처: `ARCHITECTURE.md`
- 리뷰 데이터 기준 스키마: `REVIEW_DATA_BASELINE.md`

## 7) 현재 이슈/주의사항
- PowerShell 콘솔에서 한글이 깨져 보일 수 있음 (파일 자체 UTF-8 저장)
- `strict` 모드로 Top100x1000 수집 시 일부 앱 리뷰 수 부족으로 실패 가능
  - non-strict로 완료한 데이터가 `top100_reviews_1000_each.json`
- 이유 추출은 여전히 규칙+프레이즈 기반이라 일부 게임에서 generic phrase가 섞일 수 있음

## 8) 재실행 커맨드 모음
- Top100 수집:
```powershell
python collect_top100_reviews.py --top-n 100 --reviews-per-game 1000 --language all --output top100_reviews_1000_each.json
```
- 카테고리 추천 분석:
```powershell
python analyze_category_top100.py --input top100_reviews_1000_each.json --category Action --top-k 10 --output-json action_top10_recommendations.json --output-md action_top10_recommendations.md
```
- 카테고리 추천 UI:
```powershell
streamlit run category_recommendation_app.py
```

## 9) 다음 권장 작업
1. 이유 라벨 생성에 LLM 요약 단계 추가 (`Frequent Topic` 자연어화)
2. 카테고리별 사전 확장(액션/시뮬/전략/스포츠)
3. 다국어 처리 분리(ko/en 우선 + 언어별 stopword/토크나이저)
4. 통합 실행기(`run_batch.py`) 및 run metadata 기록 표준화
