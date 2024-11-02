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
@st.cache_data
def load_data(file_url):
    response = requests.get(file_url)
    response.raise_for_status()  # 요청이 성공했는지 확인
    data = pd.read_csv(StringIO(response.text))
    return data

# 선택한 모든 월의 데이터를 결합
all_data = []
for month in selected_months:
    try:
        url = get_file_url(selected_year, month)
        df = load_data(url)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # 날짜 형식 변환
        all_data.append(df)
    except requests.exceptions.RequestException:
        st.error(f"{month}월 데이터를 불러오는 중 오류가 발생했습니다.")
    except pd.errors.EmptyDataError:
        st.error(f"{month}월 CSV 파일이 비어있습니다.")
    except Exception as e:
        st.error(f"{month}월 데이터를 불러오는 중 오류가 발생했습니다: {e}")

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
        'Select first data to plot:',
        ('TEMP', 'HUMI', 'IRRAD', 'WIND', 'RAIN',)
    )

    # 사용자에게 보여줄 두 번째 데이터 선택
    option2 = st.selectbox(
        'Select second data to plot (for secondary axis):',
        ('TEMP', 'HUMI', 'IRRAD', 'WIND', 'RAIN',)
    )

    # 선택된 데이터에 따른 그래프 그리기 (두 개의 y축)
    st.write(f"{option1} 데이터 및 {option2} 데이터에 대한 그래프:")

    fig, ax1 = plt.subplots()

    # 첫 번째 y축에 대한 데이터 플로팅 (왼쪽 y축)
    ax1.plot(df.index, df[option1], marker='o', linestyle='-', color='r')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel(option1, color='r')
    ax1.tick_params(axis='y', labelcolor='k')
    
    # x축 값 제거
    ax1.tick_params(axis='x', labelbottom=False)

    # 두 번째 y축 생성 (오른쪽 y축)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df[option2], marker='o', linestyle='-', color='b')
    ax2.set_ylabel(option2, color='b')
    ax2.tick_params(axis='y', labelcolor='k')

    # Streamlit에서 그래프 표시
    st.pyplot(fig)
else:
    st.error("선택된 달에 대한 데이터가 없습니다.")
