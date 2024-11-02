import requests
import pandas as pd
import streamlit as st
from io import StringIO
from matplotlib import pyplot as plt
from datetime import datetime

# Streamlit 앱 제목
st.title("환경 데이터 그래프")

# 현재 날짜 기준으로 연도와 월을 자동 생성
current_year = datetime.now().year
current_month = datetime.now().month

# 사용 가능한 연도와 월을 생성하는 함수
def generate_year_month_options(start_year=2024):
    years = [str(year) for year in range(start_year, current_year + 1)]
    months = [f"{month:02d}" for month in range(1, 13) if current_year > start_year or month <= current_month]
    return years, months

years, months = generate_year_month_options()

# 사용자에게 연도와 월을 선택하도록 옵션 제공
selected_year = st.selectbox("연도를 선택하세요:", years)
selected_month = st.selectbox("월을 선택하세요:", months)

# GitHub 파일 URL을 생성하는 함수
def get_file_url(year, month):
    return f"https://raw.githubusercontent.com/sungjoon23/sap2024/main/hw8/{year}-{month}/{year}.{month}.Jeonju.csv"

# CSV 파일 URL
url = get_file_url(selected_year, selected_month)

# CSV 파일을 URL에서 직접 가져와 데이터프레임으로 로드
@st.cache_data
def load_data(file_url):
    response = requests.get(file_url)
    response.raise_for_status()  # 요청이 성공했는지 확인
    data = pd.read_csv(StringIO(response.text))
    return data

# 데이터 로드
try:
    df = load_data(url)
    # 데이터 출력 (테이블 형태로)
    st.write("CSV 파일에서 가져온 데이터:")
    st.dataframe(df)

    # 사용자에게 보여줄 첫 번째 데이터 선택
    option1 = st.selectbox(
        'Select first data to plot:',
        ('TEMP', 'HUMI', 'IRRAD', 'WIND', 'RAIN',)
    )

    # 사용자에게 보여줄 두 번째 데이터 선택
    option2 = st.selectbox(
        'Select second data to plot (for secondary axis):',
        ('TEMP', 'HUMI', 'IRRAD', 'WIND', 'RAIN',)
    )

    # 시간(Timestamp)을 인덱스로 설정
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # 선택된 데이터에 따른 그래프 그리기 (두 개의 y축)
    st.write(f"{option1} 데이터 및 {option2} 데이터에 대한 그래프:")

    fig, ax1 = plt.subplots()

    # 첫 번째 y축에 대한 데이터 플로팅 (왼쪽 y축)
    ax1.plot(df.index, df[option1], marker='o', linestyle='-', color='r')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel(option1, color='r')
    ax1.tick_params(axis='y', labelcolor='k')

    # 두 번째 y축 생성 (오른쪽 y축)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df[option2], marker='o', linestyle='-', color='b')
    ax2.set_ylabel(option2, color='b')
    ax2.tick_params(axis='y', labelcolor='k')

    # Streamlit에서 그래프 표시
    st.pyplot(fig)

except requests.exceptions.RequestException:
    st.error("데이터를 불러오는 중 오류가 발생했습니다. URL을 확인해 주세요.")
except pd.errors.EmptyDataError:
    st.error("CSV 파일이 비어있습니다. 다른 파일을 선택해 주세요.")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
