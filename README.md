# 🏭 QC 외경 측정 대시보드

자동차 부품 가공 라인의 **외경 측정값** 하나로,
수강생이 배운 **Streamlit · 추론통계 · 머신러닝 · 시각화**가 어떻게 맞물리는지 보여주는 종합 프로젝트입니다.

## 무엇을 하나요
| 탭 | 내용 | 활용한 학습 |
|---|---|---|
| 📋 데이터 개요 | 측정 데이터 표·KPI·분포/박스플롯 | Streamlit, 시각화 |
| 📐 추론통계 | 신뢰구간, 정규성 검정, t-검정, Cpk, 관리도 | 추론통계, 시각화 |
| 🤖 머신러닝 | 공정 조건으로 불량을 사전 예측(랜덤포레스트) + 실시간 예측 | 머신러닝, 시각화 |
| 🧩 정리 | 네 가지가 하나의 흐름으로 이어지는 방식 | — |

> 데이터는 `data.py`에서 합성 생성합니다(실측 시스템 불필요). 사이드바에서 표본 수·시드·규격을 바꾸면 모든 결과가 즉시 재계산됩니다.

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
브라우저에서 http://localhost:8501 로 접속.

## Streamlit Community Cloud 배포 (무료)
1. 이 폴더를 GitHub 저장소로 push
   ```bash
   git init && git add . && git commit -m "QC ML dashboard"
   git branch -M main
   git remote add origin https://github.com/<사용자명>/<저장소>.git
   git push -u origin main
   ```
2. https://share.streamlit.io 접속 → GitHub 로그인
3. **New app** → 저장소 / 브랜치(`main`) / 메인 파일(`app.py`) 선택
4. **Deploy** → 1~2분 후 공개 URL 발급 (`https://<앱이름>.streamlit.app`)

`requirements.txt`만 있으면 Streamlit Cloud가 의존성을 자동 설치합니다.

## 파일 구조
```
QAQC_ml_project/
├── app.py                # 메인 앱 (탭 4개)
├── data.py               # 합성 측정 데이터 생성
├── requirements.txt      # 배포용 의존성
├── .streamlit/config.toml
└── README.md
```
