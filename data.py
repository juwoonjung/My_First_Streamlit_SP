"""자동차 부품 가공 라인 외경 측정 데이터 생성.

실제 측정 시스템이 없으므로, 공정 변수(온도·공구마모·속도 등)가
외경(mm)에 영향을 주는 구조를 합성 데이터로 흉내 낸다.
QC 강의의 모든 예제와 동일하게 '외경 측정값' 도메인을 사용한다.
"""

import numpy as np
import pandas as pd

TARGET = 50.0          # 목표 외경(mm)
LSL_DEFAULT = 49.80    # 규격 하한
USL_DEFAULT = 50.20    # 규격 상한

# 머신러닝 입력 변수 (측정 '전'에 알 수 있는 공정 조건)
FEATURES_CAT = ["라인", "설비"]
FEATURES_NUM = ["가공온도", "공구마모", "스핀들속도"]
TARGET_COL = "불량"


def generate_data(n: int = 600, seed: int = 42,
                  lsl: float = LSL_DEFAULT, usl: float = USL_DEFAULT) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    line = rng.choice(["A", "B"], size=n, p=[0.55, 0.45])
    machine = rng.choice(["M1", "M2", "M3"], size=n)
    temp = rng.normal(60.0, 4.0, n)        # 가공 온도(℃)
    tool_wear = rng.uniform(0, 100, n)     # 공구 마모도(%)
    speed = rng.normal(1200, 80, n)        # 스핀들 속도(rpm)

    # 외경(mm): 공정 조건이 나빠질수록 목표값에서 위로 드리프트
    drift = (
        0.004 * (temp - 60.0)
        + 0.0018 * tool_wear
        + np.where(line == "B", 0.025, 0.0)
        + 0.00005 * (speed - 1200)
    )
    noise = rng.normal(0, 0.03, n)
    diameter = TARGET + drift + noise

    defect = ((diameter < lsl) | (diameter > usl)).astype(int)

    return pd.DataFrame({
        "측정번호": np.arange(1, n + 1),
        "라인": line,
        "설비": machine,
        "가공온도": np.round(temp, 1),
        "공구마모": np.round(tool_wear, 1),
        "스핀들속도": np.round(speed).astype(int),
        "외경": np.round(diameter, 3),
        "판정": np.where(defect == 1, "불합격", "합격"),
        "불량": defect,
    })
