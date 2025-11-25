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

# 쓰레기 종류별 분리수거 방법 및 이미지 정보 (한국 기준)
# 이미지는 플레이스홀더를 사용합니다. 실제 배포 시에는 사용자에게 저작권이 있는 이미지를 사용해야 합니다.
disposal_info = {
    '음식물': {
        '방법': "물기를 최대한 제거하고 전용 봉투에 넣어 배출합니다. 일반 쓰레기와 혼합되지 않도록 주의해야 합니다.",
        '이미지': [
            "https://placehold.co/200x150/50C878/FFFFFF?text=음식물+쓰레기+분리",
            "https://placehold.co/200x150/50C878/FFFFFF?text=물기+제거",
            "https://placehold.co/200x150/50C878/FFFFFF?text=전용+봉투+사용"
        ]
    },
    '플라스틱': {
        '방법': "내용물을 비우고 깨끗하게 헹군 후, 라벨(뚜껑 등)을 제거하고 압착하여 투명 또는 반투명 봉투에 담아 배출합니다.",
        '이미지': [
            "https://placehold.co/200x150/1E90FF/FFFFFF?text=플라스틱+내용물+비우기",
            "https://placehold.co/200x150/1E90FF/FFFFFF?text=라벨+제거",
            "https://placehold.co/200x150/1E90FF/FFFFFF?text=압착+배출"
        ]
    },
    '캔': {
        '방법': "내용물을 비우고 헹군 후, 캔을 압착하고 철/비철 캔류 수거함에 배출합니다. 부탄가스 등은 구멍을 뚫어 내용물을 완전히 제거 후 배출합니다.",
        '이미지': [
            "https://placehold.co/200x150/FF4500/FFFFFF?text=캔+압착",
            "https://placehold.co/200x150/FF4500/FFFFFF?text=내용물+제거",
            "https://placehold.co/200x150/FF4500/FFFFFF?text=캔류+수거함"
        ]
    },
    '비닐': {
        '방법': "이물질이 묻은 비닐은 물티슈 등으로 닦거나 가볍게 헹구어 비닐류 수거함에 배출합니다. 오염이 심한 경우 일반 쓰레기로 버립니다.",
        '이미지': [
            "https://placehold.co/200x150/8A2BE2/FFFFFF?text=비닐+이물질+제거",
            "https://placehold.co/200x150/8A2BE2/FFFFFF?text=비닐류+분리수거",
            "https://placehold.co/200x150/8A2BE2/FFFFFF?text=심한+오염시+일반쓰레기"
        ]
    },
    '종이': {
        '방법': "물기에 젖지 않도록 묶거나 종이상자에 담아 배출합니다. 테이프나 비닐 코팅된 부분은 제거해야 합니다.",
        '이미지': [
            "https://placehold.co/200x150/D2B48C/FFFFFF?text=종이+묶음+배출",
            "https://placehold.co/200x150/D2B48C/FFFFFF?text=코팅+부분+제거",
            "https://placehold.co/200x150/D2B48C/FFFFFF?text=박스+분리"
        ]
    },
    '유리': {
        '방법': "병뚜껑 등을 제거하고 내용물을 비운 후, 깨끗이 헹궈 색깔별 또는 기타 유리 수거함에 배출합니다. 깨진 유리는 신문지에 싸서 일반쓰레기로 버립니다.",
        '이미지': [
            "https://placehold.co/200x150/98FB98/000000?text=유리병+뚜껑+제거",
            "https://placehold.co/200x150/98FB98/000000?text=내용물+제거+후+배출",
            "https://placehold.co/200x150/98FB98/000000?text=색깔별+분리"
        ]
    },
    '일반쓰레기': {
        '방법': "재활용이 불가능한 모든 쓰레기는 종량제 봉투에 담아 배출합니다. 음식물 쓰레기나 재활용 가능한 품목이 섞이지 않도록 주의합니다.",
        '이미지': [
            "https://placehold.co/200x150/808080/FFFFFF?text=일반쓰레기+종량제",
            "https://placehold.co/200x150/808080/FFFFFF?text=재활용+불가능",
            "https://placehold.co/200x150/808080/FFFFFF?text=봉투+밀봉"
        ]
    },
    '스티로폼': {
        '방법': "내용물을 비우고 테이프, 운송장 등을 완전히 제거 후 깨끗하게 하여 '스티로폼' 전용 수거함에 배출합니다. 오염된 것은 일반쓰레기로 버립니다.",
        '이미지': [
            "https://placehold.co/200x150/F08080/FFFFFF?text=스티로폼+테이프+제거",
            "https://placehold.co/200x150/F08080/FFFFFF?text=스티로폼+수거함",
            "https://placehold.co/200x150/F08080/FFFFFF?text=오염시+일반쓰레기"
        ]
    }
}


# 특성(X)과 레이블(y) 분리
features = df.columns.drop('쓰레기 종류')
X = df[features]
y = df['쓰레기 종류']

# 2. 모델 학습
# 데이터가 작으므로 학습 데이터 전체를 사용하여 모델 학습
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
    
    # --- 추가된 분리수거 정보 표시 ---
    st.subheader(f"✅ {prediction} 분리수거 방법 (한국 기준)")
    info = disposal_info.get(prediction, {'방법': '분리수거 정보가 없습니다.', '이미지': []})
    st.info(info['방법'])

    # 이미지 3개 표시
    if info['이미지']:
        st.markdown("#### 참고 이미지")
        cols = st.columns(len(info['이미지']))
        for i, img_url in enumerate(info['이미지']):
            # Image of <Item> 태그 사용을 위한 임의의 텍스트 추가
            # 실제 Streamlit에서는 st.image(img_url)을 사용합니다.
            cols[i].image(img_url, caption=f"이미지 {i+1}")
            
    st.markdown("---")
    # --- 추가된 분리수거 정보 표시 끝 ---

    st.subheader("모델이 학습한 다른 클래스별 확률")
    
    # 확률을 내림차순으로 정렬하여 표시
    sorted_confidence = sorted(confidence_scores.items(), key=lambda item: item[1], reverse=True)

    # 상위 4개 클래스만 표시
    top_n = min(len(sorted_confidence), 4) 
    
    for i in range(top_n):
        trash_type, prob = sorted_confidence[i]
        
        # 예측된 종류는 이미 위에 표시했으므로 제외 (신뢰도가 99% 이상인 경우)
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
