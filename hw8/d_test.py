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
selected_months = st.multiselect("월을 선택하세요 (여러 달 선택 가능):", months, default=["10", "11"])

# GitHub 파일 URL을 생성하는 함수
def get_file_url(year, month):
    return f"https://raw.githubusercontent.com/sungjoon23/sap2024/main/hw8/{year}-{month}/{year}.{month}.Jeonju.csv"

# CSV 파일을 URL에서 직접 가져와 데이터프레임으로 로드
@st.cache_data(ttl=600)  # 캐시 유효 시간을 10분으로 설정
def load_data(file_url):
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # 요청이 성공했는지 확인
        data = pd.read_csv(StringIO(response.text))
        return data
    except requests.exceptions.RequestException:
        return None

# 선택한 모든 월의 데이터를 결합
all_data = []
for month in selected_months:
    url = get_file_url(selected_year, month)
    df = load_data(url)
    if df is not None:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # 날짜 형식 변환
        all_data.append(df)
    else:
        st.warning(f"{month}월 데이터 파일이 존재하지 않거나 로드할 수 없습니다.")

# 모든 데이터를 하나의 데이터프레임으로 결합
if all_data:
    df = pd.concat(all_data)
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)  # 날짜 순서대로 정렬

    # 데이터 출력 (테이블 형태로)
    st.write("CSV 파일에서 가져온 결합된 데이터:")
    st.dataframe(df)

    # 사용자에게 보여줄 첫 번째 데이터 선택
    option1 = st.selectbox(
        'Select first data to plot (왼쪽 Y축):',
        ('TEMP', 'HUMI', 'IRRAD', 'WIND', 'RAIN')
    )

    # 사용자에게 두 번째 y축 표시 여부 선택
    show_secondary_axis = st.checkbox("두 번째 Y축 표시", value=True)

    # 두 번째 y축의 데이터를 선택할 수 있게 하되, 표시 여부에 따라 플롯 설정
    if show_secondary_axis:
        option2 = st.selectbox(
            'Select second data to plot (오른쪽 Y축):',
            ('TEMP', 'HUMI', 'IRRAD', 'WIND', 'RAIN')
        )

    # 선택된 데이터에 따른 그래프 그리기
    st.write(f"{option1} 데이터 및 {option2 if show_secondary_axis else ''} 데이터에 대한 그래프:")

    fig, ax1 = plt.subplots()

    # 첫 번째 y축에 대한 데이터 플로팅 (왼쪽 y축)
    ax1.plot(df.index, df[option1], linestyle='-', color='r', label=option1)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel(option1, color='r')
    ax1.tick_params(axis='y', labelcolor='k')

    # 두 번째 y축 생성 및 표시 여부 결정
    if show_secondary_axis:
        ax2 = ax1.twinx()
        ax2.plot(df.index, df[option2], linestyle='-', color='b', label=option2)
        ax2.set_ylabel(option2, color='b')
        ax2.tick_params(axis='y', labelcolor='k')

    # x축 레이블을 표시합니다.
    ax1.tick_params(axis='x', rotation=45)

    # 그래프 제목과 레이아웃 설정
    fig.tight_layout()
    ax1.legend(loc='upper left')
    if show_secondary_axis:
        ax2.legend(loc='upper right')

    # Streamlit에서 그래프 표시
    st.pyplot(fig)
else:
    st.warning("선택한 월에 대한 데이터가 없습니다.")
