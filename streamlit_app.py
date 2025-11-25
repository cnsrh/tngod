import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. 데이터 정의 (제공된 HTML 파일에서 추출)
# 쓰레기 종류별 10가지 판별 기준 점수 (1~30점 척도)
data = {
    '쓰레기 종류': ['음식물', '플라스틱', '캔', '비닐', '종이', '유리', '일반쓰레기', '스티로폼'],
    '투명도': [5, 20, 2, 23, 3, 27, 3, 10],
    '광택도': [3, 17, 28, 18, 5, 25, 6, 15],
    '반사율': [3, 15, 29, 15, 8, 28, 5, 12],
    '표면 거칠기': [26, 8, 5, 6, 17, 7, 24, 15],
    '색상 단순성': [8, 25, 18, 20, 20, 18, 8, 17],
    '외곽선의 뚜렷함 정도': [5, 20, 24, 18, 21, 24, 9, 18],
    '형태 단일성': [3, 24, 28, 14, 25, 28, 5, 20],
    '질량감': [7, 15, 20, 5, 6, 25, 12, 9],
    '표면 반복성': [3, 20, 25, 18, 22, 21, 6, 15],
    '오염/손상 정도': [24, 12, 15, 18, 15, 8, 28, 12]
}
df = pd.DataFrame(data)

# 특성(X)과 레이블(y) 분리
features = df.columns.drop('쓰레기 종류')
X = df[features]
y = df['쓰레기 종류']

# 2. 모델 학습
# 데이터가 작으므로 학습 데이터 전체를 사용하여 모델 학습
# Streamlit 앱에서는 모델 저장 및 로드 없이 바로 학습하여 사용
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Streamlit 앱 구성 시작
st.set_page_config(
    page_title="쓰레기 종류 예측 AI",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🗑️ 쓰레기 종류 예측 모델 시뮬레이터")
st.markdown("제공된 10가지 판별 기준 점수(1~30점 척도)를 입력하여 쓰레기 종류를 예측합니다.")
st.markdown("---")


# 4. 사용자 입력 받기 (슬라이더)
st.sidebar.header("특성 점수 입력 (1~30점)")

# 각 특성에 대한 슬라이더를 딕셔너리에 저장
user_scores = {}
for feature in features:
    # 점수 범위는 1점에서 30점
    default_value = 15 # 중간값으로 설정
    user_scores[feature] = st.sidebar.slider(
        f"**{feature}**", # 슬라이더 라벨
        min_value=1,
        max_value=30,
        value=default_value,
        step=1
    )

st.sidebar.markdown("---")
st.sidebar.info("왼쪽 사이드바의 슬라이더를 조작하여 예측을 시작하세요.")


# 5. 예측 및 결과 표시
if st.sidebar.button('쓰레기 종류 예측하기'):
    # 사용자 입력 데이터를 DataFrame 형태로 준비
    input_data = pd.DataFrame([user_scores])

    # 5-1. 예측 결과
    prediction = model.predict(input_data)[0]

    # 5-2. 예측 확률 (신뢰도)
    probabilities = model.predict_proba(input_data)
    confidence_scores = dict(zip(model.classes_, probabilities[0]))
    
    # 예측된 클래스의 확률
    predicted_confidence = confidence_scores[prediction]
    
    st.success(f"## 💡 예측 결과: **{prediction}**")
    st.metric(label="예측 신뢰도", value=f"{predicted_confidence*100:.2f}%")
    
    st.markdown("---")
    st.subheader("모델이 학습한 다른 클래스별 확률")
    
    # 확률을 내림차순으로 정렬하여 표시
    sorted_confidence = sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True)

    # 상위 3개 클래스만 표시 (예측된 클래스는 제외)
    top_n = min(len(sorted_confidence), 4) # 최대 4개까지 표시
    
    for i in range(top_n):
        trash_type, prob = sorted_confidence[i]
        
        # 예측된 종류는 이미 위에 표시했으므로 제외
        if trash_type == prediction and predicted_confidence > 0.99:
            continue

        # 진행바 표시
        st.write(f"**{trash_type}**")
        st.progress(prob)
        st.caption(f"확률: {prob*100:.2f}%")
        
else:
    st.info("슬라이더를 조작하거나 '쓰레기 종류 예측하기' 버튼을 눌러 결과를 확인하세요.")

st.markdown("---")
st.subheader("모델 학습 데이터")
st.dataframe(df, use_container_width=True)

# 6. Streamlit Cloud 배포 안내
st.sidebar.markdown("""
---
### 🚀 배포 안내
1.  이 코드를 `streamlit_app.py`로 저장합니다.
2.  `requirements.txt` 파일도 함께 저장합니다.
3.  이 두 파일을 GitHub 저장소에 커밋합니다.
4.  [Streamlit Cloud]에 접속하여 해당 GitHub 저장소를 연결하면 앱이 배포됩니다.
""")
