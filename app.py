"""QC 외경 측정 대시보드 — 추론통계 · 머신러닝 · 시각화 종합 프로젝트.

수강생이 배운 4가지가 하나의 품질관리 문제에서 어떻게 맞물리는지 보여준다.
  • Streamlit  : 화면 구성 · 위젯 · 레이아웃 · 차트 (이 앱 전체)
  • 추론통계    : 신뢰구간 · 정규성 검정 · 두 라인 평균 비교(t-검정) · 공정능력(Cpk) · 관리도
  • 머신러닝    : 공정 조건으로 불량을 '미리' 예측하는 분류 모델
  • 시각화      : 분포 · 관리도 · 혼동행렬 · ROC · 변수 중요도
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import (
    FEATURES_CAT,
    FEATURES_NUM,
    TARGET,
    TARGET_COL,
    generate_data,
)

st.set_page_config(page_title="QC 외경 측정 대시보드", page_icon="🏭", layout="wide")


# ──────────────────────────────────────────────────────────────────────────
# 데이터 · 모델 (캐시)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(n, seed, lsl, usl):
    return generate_data(n, seed, lsl, usl)


@st.cache_resource
def train_model(n, seed, lsl, usl):
    df = generate_data(n, seed, lsl, usl)
    X = df[FEATURES_CAT + FEATURES_NUM]
    y = df[TARGET_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    clf = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(
            n_estimators=200, random_state=seed, class_weight="balanced")),
    ])
    clf.fit(X_tr, y_tr)
    return clf, X_te, y_te


# ──────────────────────────────────────────────────────────────────────────
# 사이드바 — 입력 위젯 (Streamlit input 위젯 활용)
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 데이터 · 규격 설정")
n = st.sidebar.slider("측정 표본 수", 200, 2000, 600, step=100)
seed = st.sidebar.number_input("랜덤 시드", 0, 9999, 42)
lsl, usl = st.sidebar.slider(
    "규격 한계 (LSL / USL, mm)", 49.5, 50.5, (49.80, 50.20), step=0.01
)
st.sidebar.caption(f"목표값 {TARGET:.2f} mm · 규격 {lsl:.2f}–{usl:.2f} mm")

df = load_data(n, seed, lsl, usl)


# ──────────────────────────────────────────────────────────────────────────
# 헤더
# ──────────────────────────────────────────────────────────────────────────
st.title("🏭 QC 외경 측정 대시보드")
st.markdown(
    "자동차 부품 가공 라인의 **외경 측정값** 하나로 "
    "**추론통계 · 머신러닝 · 시각화**가 어떻게 이어지는지 보여주는 종합 프로젝트입니다."
)

tab_eda, tab_stats, tab_ml, tab_wrap = st.tabs(
    ["📋 데이터 개요", "📐 추론통계", "🤖 머신러닝", "🧩 정리"]
)


# ──────────────────────────────────────────────────────────────────────────
# 1) 데이터 개요 (EDA + 시각화)
# ──────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("측정 데이터 한눈에 보기")

    defect_rate = df["불량"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("측정 수", f"{len(df):,}건")
    c2.metric("합격", f"{(df['불량'] == 0).sum():,}건")
    c3.metric("불량률", f"{defect_rate * 100:.1f}%")
    c4.metric("평균 외경", f"{df['외경'].mean():.3f} mm")

    st.dataframe(
        df.drop(columns="불량"),
        width="stretch",
        hide_index=True,
        height=260,
        column_config={
            "외경": st.column_config.NumberColumn("외경(mm)", format="%.3f"),
            "가공온도": st.column_config.NumberColumn("가공온도(℃)", format="%.1f"),
            "공구마모": st.column_config.ProgressColumn(
                "공구마모(%)", format="%.0f", min_value=0, max_value=100),
        },
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**외경 분포** — 규격선(빨강) 밖이 불량")
        fig = px.histogram(df, x="외경", color="판정", nbins=40,
                           color_discrete_map={"합격": "#2E86DE", "불합격": "#E74C3C"})
        fig.add_vline(x=lsl, line_color="red", line_dash="dash")
        fig.add_vline(x=usl, line_color="red", line_dash="dash")
        fig.add_vline(x=TARGET, line_color="green")
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with col_r:
        st.markdown("**라인별 외경** — 박스플롯으로 분포 비교")
        fig = px.box(df, x="라인", y="외경", color="라인", points="outliers")
        fig.add_hline(y=usl, line_color="red", line_dash="dash")
        fig.add_hline(y=lsl, line_color="red", line_dash="dash")
        fig.update_layout(height=340, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────────────────────────────────
# 2) 추론통계
# ──────────────────────────────────────────────────────────────────────────
with tab_stats:
    x = df["외경"].to_numpy()
    mu, sigma, n_obs = x.mean(), x.std(ddof=1), len(x)

    st.subheader("① 평균 외경의 95% 신뢰구간")
    se = sigma / np.sqrt(n_obs)
    ci_low, ci_high = stats.t.interval(0.95, n_obs - 1, loc=mu, scale=se)
    c1, c2, c3 = st.columns(3)
    c1.metric("표본 평균", f"{mu:.4f} mm")
    c2.metric("95% CI 하한", f"{ci_low:.4f}")
    c3.metric("95% CI 상한", f"{ci_high:.4f}")
    st.caption(
        "표본으로 **모평균이 있을 법한 범위**를 추정 — 추론통계의 핵심. "
        "표본이 커질수록 구간이 좁아집니다(사이드바에서 표본 수를 늘려 확인)."
    )
    st.divider()

    st.subheader("② 정규성 검정 (Shapiro–Wilk)")
    sample = x if n_obs <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
    w_stat, p_norm = stats.shapiro(sample)
    c1, c2 = st.columns(2)
    c1.metric("검정통계량 W", f"{w_stat:.4f}")
    c2.metric("p-value", f"{p_norm:.4f}")
    if p_norm > 0.05:
        st.success("p > 0.05 → 정규분포로 보아도 무리 없음 (Cpk·관리도 가정 충족).")
    else:
        st.warning("p ≤ 0.05 → 정규성 의심 — 관리도/Cpk 해석에 주의.")
    st.divider()

    st.subheader("③ 두 라인 평균 비교 (독립표본 t-검정)")
    a = df.loc[df["라인"] == "A", "외경"]
    b = df.loc[df["라인"] == "B", "외경"]
    t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("A 평균", f"{a.mean():.4f}")
    c2.metric("B 평균", f"{b.mean():.4f}")
    c3.metric("p-value", f"{p_t:.4f}")
    if p_t < 0.05:
        st.error("p < 0.05 → 두 라인의 평균 외경은 **통계적으로 유의하게 다름**.")
    else:
        st.info("p ≥ 0.05 → 두 라인 평균 차이는 통계적으로 유의하지 않음.")
    st.caption("가설검정: '두 라인은 같다(H₀)'를 데이터로 기각할 수 있는지 판단합니다.")
    st.divider()

    st.subheader("④ 공정능력 지수 (Cp / Cpk)")
    cp = (usl - lsl) / (6 * sigma)
    cpk = min(usl - mu, mu - lsl) / (3 * sigma)
    c1, c2 = st.columns(2)
    c1.metric("Cp", f"{cp:.2f}")
    c2.metric("Cpk", f"{cpk:.2f}",
              delta="양호" if cpk >= 1.33 else "개선 필요",
              delta_color="normal" if cpk >= 1.33 else "inverse")
    st.caption("Cpk ≥ 1.33 이면 공정능력 양호. Cp와 Cpk 차이가 크면 평균이 한쪽으로 치우친 것.")
    st.divider()

    st.subheader("⑤ 관리도 (I-Chart, ±3σ)")
    ucl, lcl = mu + 3 * sigma, mu - 3 * sigma
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, mode="lines+markers", name="외경",
                             marker=dict(size=4), line=dict(width=1)))
    fig.add_hline(y=mu, line_color="green", annotation_text="CL")
    fig.add_hline(y=ucl, line_color="orange", line_dash="dash", annotation_text="UCL")
    fig.add_hline(y=lcl, line_color="orange", line_dash="dash", annotation_text="LCL")
    fig.add_hline(y=usl, line_color="red", line_dash="dot", annotation_text="USL")
    fig.add_hline(y=lcl if lcl < lsl else lsl, line_color="red", line_dash="dot")
    out = np.where((x > ucl) | (x < lcl))[0]
    if len(out):
        fig.add_trace(go.Scatter(x=out, y=x[out], mode="markers",
                                 marker=dict(size=9, color="red"), name="관리이탈"))
    fig.update_layout(height=360, margin=dict(t=10, b=10), xaxis_title="측정 순서")
    st.plotly_chart(fig, width="stretch")
    st.caption("시간 순서로 공정이 ±3σ 관리한계를 벗어나는지 모니터링합니다.")


# ──────────────────────────────────────────────────────────────────────────
# 3) 머신러닝
# ──────────────────────────────────────────────────────────────────────────
with tab_ml:
    st.subheader("공정 조건으로 불량을 '미리' 예측하기")
    st.caption("입력: 라인·설비·가공온도·공구마모·스핀들속도 (측정 전 알 수 있는 값) → 출력: 불량 여부")

    clf, X_te, y_te = train_model(n, seed, lsl, usl)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)

    c1, c2, c3 = st.columns(3)
    c1.metric("정확도", f"{acc * 100:.1f}%")
    c2.metric("ROC-AUC", f"{auc:.3f}")
    c3.metric("테스트 표본", f"{len(y_te)}건")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**혼동행렬**")
        cm = confusion_matrix(y_te, pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="예측", y="실제", color="건수"),
                        x=["합격", "불량"], y=["합격", "불량"])
        fig.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with col_r:
        st.markdown("**ROC 곡선**")
        fpr, tpr, _ = roc_curve(y_te, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="gray"), showlegend=False))
        fig.update_layout(height=320, margin=dict(t=10, b=10),
                          xaxis_title="거짓 양성률", yaxis_title="참 양성률")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**변수 중요도** — 어떤 공정 조건이 불량에 가장 큰 영향을 주는가")
    names = clf.named_steps["pre"].get_feature_names_out()
    imp = clf.named_steps["rf"].feature_importances_
    imp_df = pd.DataFrame({"변수": names, "중요도": imp}).sort_values("중요도")
    fig = px.bar(imp_df, x="중요도", y="변수", orientation="h")
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("### 🔮 실시간 예측 — 조건을 바꿔 불량 확률 확인")
    p1, p2, p3 = st.columns(3)
    in_line = p1.selectbox("라인", ["A", "B"])
    in_machine = p1.selectbox("설비", ["M1", "M2", "M3"])
    in_temp = p2.slider("가공온도(℃)", 45.0, 75.0, 60.0, 0.5)
    in_wear = p2.slider("공구마모(%)", 0.0, 100.0, 50.0, 1.0)
    in_speed = p3.slider("스핀들속도(rpm)", 900, 1500, 1200, 10)

    row = pd.DataFrame([{
        "라인": in_line, "설비": in_machine, "가공온도": in_temp,
        "공구마모": in_wear, "스핀들속도": in_speed,
    }])
    p_defect = clf.predict_proba(row)[0, 1]
    st.progress(float(p_defect), text=f"예측 불량 확률 {p_defect * 100:.1f}%")
    if p_defect >= 0.5:
        st.error(f"⚠️ 불량 위험 높음 ({p_defect * 100:.1f}%) — 공구 점검/조건 조정 권장")
    else:
        st.success(f"✅ 양호 ({p_defect * 100:.1f}%) — 현재 조건 유지 가능")


# ──────────────────────────────────────────────────────────────────────────
# 4) 정리
# ──────────────────────────────────────────────────────────────────────────
with tab_wrap:
    st.subheader("배운 내용이 하나의 프로젝트에서 어떻게 맞물리나")
    st.markdown(
        """
| 배운 것 | 이 프로젝트에서의 역할 |
|---|---|
| **Streamlit** | 사이드바 입력 위젯, 탭 레이아웃, metric/dataframe/chart — 앱 전체 |
| **추론통계** | 신뢰구간으로 모평균 추정, t-검정으로 라인 차이 판정, Cpk로 공정능력 평가, 관리도로 이상 감지 |
| **머신러닝** | 공정 조건으로 불량을 *사전* 예측 → 측정 전에 위험 라인을 골라낸다 |
| **시각화** | 분포·박스플롯·관리도·혼동행렬·ROC·변수중요도로 결과를 '보이게' 만든다 |

**흐름**: 측정값을 **통계로 진단** → 원인 변수를 **ML로 학습** → 결과를 **시각화로 전달** → **Streamlit으로 배포**해 누구나 웹에서 사용.
        """
    )
    st.info("사이드바에서 표본 수·시드·규격을 바꾸면 모든 탭의 통계·모델·차트가 즉시 다시 계산됩니다.")
