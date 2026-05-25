import streamlit as st
import pandas as pd
import joblib
import time

# --- 페이지 설정 및 디자인 (Custom CSS) ---
st.set_page_config(page_title="AI 당뇨병 진단 시스템", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .result-card { padding: 20px; border-radius: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf, #2e7bcf); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 모델 및 스케일러 로드 ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('diabete_model_fixed.pkl')
        scaler = joblib.load('diabetes_scaler.pkl')
        
        # [수정] 스케일러가 요구하는 14개 컬럼 순서 그대로 정의합니다.
        columns = ['임신횟수', '혈당', '혈압', '피부두께', '인슐린', 'BMI', '당뇨병가족력지수', '나이', 
                   '건강지표점수', '신체부담도', '혈당인슐린결합', '비만지표', '유전노화지수', '고령']
        return model, scaler, columns
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return None, None, None

model, scaler, feature_columns = load_assets()

# --- 사이드바: 정보 입력 ---
with st.sidebar:
    st.header("📋 환자 정보 입력")
    st.info("정확한 예측을 위해 모든 수치를 입력해주세요.")
    
    preg = st.number_input("임신 횟수", 0, 20, 0)
    glucose = st.slider("혈당 (Glucose)", 0.0, 300.0, 100.0)
    bp = st.slider("혈압 (Blood Pressure)", 0.0, 200.0, 70.0)
    skin = st.number_input("피부 두께", 0.0, 100.0, 20.0)
    insulin = st.number_input("인슐린", 0.0, 900.0, 80.0)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("가족력 지수", 0.0, 3.0, 0.5, format="%.3f")
    age = st.number_input("나이", 0, 120, 30)

# --- 메인 화면 ---
st.title("🩺 AI 당뇨 발병 위험도 분석")
st.write("기계학습 모델을 활용하여 입력된 건강 지표를 기반으로 당뇨 발병 가능성을 예측합니다.")

if st.button("🚀 위험도 분석 시작"):
    if model and scaler:
        with st.spinner('AI가 데이터를 분석 중입니다...'):
            time.sleep(1.5) # 분석 시각 효과
            
            # 1. 기본 데이터프레임 생성
            input_df = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                                    columns=['임신횟수', '혈당', '혈압', '피부두께', '인슐린', 'BMI', '당뇨병가족력지수', '나이'])
            
            # 2. 스케일러를 만족시키기 위해 파생 변수 6개 복구 (총 14개)
            input_df['건강지표점수'] = input_df[['혈당', '혈압', 'BMI']].sum(axis=1)
            input_df['신체부담도'] = input_df['임신횟수'] + input_df['나이']
            input_df['혈당인슐린결합'] = input_df['혈당'] * input_df['인슐린']
            input_df['비만지표'] = input_df['BMI'] + input_df['피부두께']
            input_df['유전노화지수'] = input_df['당뇨병가족력지수'] * input_df['나이']
            input_df['고령'] = (input_df['나이'] >= 50).astype(int)
            
            # 스케일러 학습 당시의 14개 컬럼 순서로 재정렬
            input_df_ordered = input_df[feature_columns]
            
            # 3. 스케일러 통과 (14개 데이터 변환)
            input_scaled = scaler.transform(input_df_ordered)
            
            # 4. [핵심] 모델은 8개만 원하므로, 스케일링된 결과에서 앞의 8개 열만 슬라이싱!
            input_scaled_for_model = input_scaled[:, :8]
            
            # 5. 모델 예측 수행
            prob = model.predict_proba(input_scaled_for_model)[0][1] * 100
            result = 1 if prob > 50 else 0
            
            # 6. 결과 디스플레이
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 분석 요약")
                st.metric("예상 당뇨 확률", f"{prob:.1f}%", delta=f"{prob-50:.1f}%" if prob > 50 else f"{prob-50:.1f}%", delta_color="inverse")
            
            with col2:
                st.subheader("🏥 진단 결과")
                if result == 1:
                    st.error("⚠️ **당뇨 발병 위험군**으로 분류되었습니다.")
                    st.write("전문의와의 상담 및 정밀 검사를 권장합니다.")
                else:
                    st.success("✅ **정상군**으로 분류되었습니다.")
                    st.write("현재의 건강한 생활 습관을 유지해 주세요.")
            
            st.balloons() if result == 0 else st.warning("건강 관리에 주의가 필요합니다.")
    else:
        st.error("모델 또는 스케일러를 불러올 수 없습니다. 파일 상태를 확인해주세요.")