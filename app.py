import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# =========================================================================
# 1. 웹 페이지 기본 스타일 및 반응형 레이아웃 설정
# =========================================================================
st.set_page_config(
    page_title="PROPTY - AI 부동산 시세 분석 엔진", 
    page_icon="🔮",
    layout="wide" 
)

# 📱 반응형 분기점 CSS 적용
# 화면 너비가 800px 이하(모바일)일 때는 사이드바를 아예 숨겨버리고 메인 홈에 집중시킵니다.
st.html("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 3rem; }
    div[data-testid='stMetricValue'] { font-size: 26px; font-weight: 700; }
    
    /* 모바일 기기(화면 폭 800px 이하)일 때 Streamlit의 사이드바 자체를 보이지 않도록 처리 */
    @media (max-width: 800px) {
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        span[data-testid="collapsedControl"] {
            display: none !important;
        }
    }
</style>
""")

# 헤더 영역
st.title("🔮 PROPTY : AI 입지 시세 추론 데이터 랩")
st.caption("서울 25개 구 전체 & 수원 4개 구 실거래 데이터를 병합하여 머신러닝 알고리즘으로 분석하는 지능형 부동산 서비스입니다.")
st.markdown("---")

ENCODING_KEY = "본인의_공공데이터포털_인코딩_인증키_입력"

API_CONFIGS = {
    '아파트': 'http://apis.data.go.kr/1613000/RTMSOBJSvc/getRTMSDataSvcAptTradeDev',
    '오피스텔': 'http://apis.data.go.kr/1613000/RTMSOBJSvc/getRTMSDataSvcOffiTrade',
    '연립다세대': 'http://apis.data.go.kr/1613000/RTMSOBJSvc/getRTMSDataSvcRHTrade',
    '단독주택': 'http://apis.data.go.kr/1613000/RTMSOBJSvc/getRTMSDataSvcSHTrade'
}

REGION_GROUPS = {
    '서울_도심강북권': ['11110', '11140', '11170', '11380', '11410', '11440'],
    '서울_강남동남권': ['11650', '11680', '11710', '11740'],
    '서울_강서서남권': ['11470', '11500', '11530', '11545', '11560', '11590', '11620'],
    '서울_강동동북권': ['11200', '11215', '11230', '11260', '11290', '11305', '11320', '11350'],
    '수원_장안구': ['41111'], 
    '수원_권선구': ['41113'], 
    '수원_팔달구': ['41115'], 
    '수원_영통구': ['41117']
}

# =========================================================================
# 2. 백엔드 AI 모델 멀티 학습 로직
# =========================================================================
@st.cache_resource(show_spinner="전체 행정구역 시세 데이터 최적화 분석 중...")
def train_multi_models():
    integrated_data = []
    current_date = datetime.now()
    TARGET_MONTHS = [current_date.strftime("%Y%m")]
    
    api_available = False
    try:
        test_url = f"{API_CONFIGS['아파트']}?serviceKey={ENCODING_KEY}"
        test_params = {'LAWD_CD': '11110', 'DEAL_YMD': TARGET_MONTHS[0], 'numOfRows': '1'}
        response = requests.get(test_url, params=test_params, timeout=0.5)
        if response.status_code == 200:
            api_available = True
    except Exception:
        api_available = False

    if api_available:
        for home_type, base_url in API_CONFIGS.items():
            for region_name, lawd_cds in REGION_GROUPS.items():
                for lawd_cd in lawd_cds:
                    for deal_ymd in TARGET_MONTHS:
                        request_url = f"{base_url}?serviceKey={ENCODING_KEY}"
                        params = {'LAWD_CD': lawd_cd, 'DEAL_YMD': deal_ymd, 'numOfRows': '50'}
                        try:
                            response = requests.get(request_url, params=params, timeout=0.3)
                            if response.status_code == 200:
                                root = ET.fromstring(response.content)
                                if root.find('.//item') is not None:
                                    for item in root.findall('.//item'):
                                        try:
                                            amount = int(item.find('거래금액').text.strip().replace(',', ''))
                                            area_m2 = float(item.find('연면적').text.strip()) if home_type == '단독주택' else float(item.find('전용면적').text.strip())
                                            area_pyeong = round(area_m2 / 3.3058, 1)
                                            price_per_pyeong = round(amount / area_pyeong, 0)
                                            city_label = '서울' if '서울' in region_name else '수원'
                                            
                                            integrated_data.append({
                                                'city': city_label, 'detail_region': region_name,
                                                'home_type': home_type, 'area_pyeong': area_pyeong, 'price_per_pyeong': price_per_pyeong
                                            })
                                        except: continue
                        except Exception:
                            continue
                
    df = pd.DataFrame(integrated_data)
    
    if df.empty or len(df) < 30:
        sim_list = []
        for r_name in REGION_GROUPS.keys():
            city = '서울' if '서울' in r_name else '수원'
            for t in ['아파트', '오피스텔', '연립다세대', '단독주택']:
                base_p = 6800 if '강남' in r_name else (4800 if '도심' in r_name else (3600 if '서남' in r_name else 2500))
                sim_list.append(pd.DataFrame({
                    'city': [city]*40, 'detail_region': [r_name]*40, 'home_type': [t]*40,
                    'area_pyeong': list(np.random.uniform(12, 52, 40)),
                    'price_per_pyeong': list(np.random.normal(base_p if t=='아파트' else base_p*0.6, 300, 40))
                }))
        df = pd.concat(sim_list, ignore_index=True)
        
    df['home_type_encoded'] = df['home_type'].astype('category').cat.codes
    X = df[['area_pyeong', 'price_per_pyeong', 'home_type_encoded']].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm_city = SVC(kernel='rbf', C=1.5, probability=True).fit(X_scaled, df['city'].values)
    svm_detail = SVC(kernel='rbf', C=1.5, probability=True).fit(X_scaled, df['detail_region'].values)
    knn_city = KNeighborsClassifier(n_neighbors=5).fit(X_scaled, df['city'].values)
    knn_detail = KNeighborsClassifier(n_neighbors=5).fit(X_scaled, df['detail_region'].values)
    
    return svm_city, svm_detail, knn_city, knn_detail, scaler, df

svm_city, svm_detail, knn_city, knn_detail, scaler, df_total = train_multi_models()

# =========================================================================
# 3. 반응형 제어 - 세션 상태를 이용한 모바일 기기 감지 및 입력 위젯 스위칭
# =========================================================================
# Streamlit의 쿼리 파라미터를 사용해 모바일 뷰 강제 전환 테스트용 파라미터 마련 (?device=mobile)
query_params = st.query_params
is_mobile = query_params.get("device", "pc") == "mobile"

# 모바일 전용 토글 위젯 (화면 우측 상단 배치로 유연성 증대)
col_head, col_switch = st.columns([4, 1])
with col_switch:
    device_mode = st.toggle("📱 모바일 화면 모드", value=is_mobile)

# --- 입력 데이터 허브 설정 ---
home_type, area, price_per_pyeong, selected_model = None, None, None, None
submit_btn = False

if device_mode:
    # 📱 [모바일 모드 활성화 시] : 메인 홈 화면 최상단에 위젯 강제 노출
    st.subheader("⚙️ 모바일용 매물 조건 입력")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        home_type = st.selectbox("🏠 주택 유형", ['아파트', '오피스텔', '연립다세대', '단독주택'], key="m_type")
        price_per_pyeong = st.number_input("💰 희망 평