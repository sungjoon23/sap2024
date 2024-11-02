import requests
import pandas as pd
import streamlit as st
from io import StringIO  # 여기에서 StringIO를 가져옵니다.
from matplotlib import pyplot as plt

# CSV 파일의 GitHub RAW URL
url = "https://raw.githubusercontent.com/sungjoon23/sap2024/main/hw8/2024-10/2024.10.Jeonju.csv"

# Streamlit 앱 제목
st.title("환경 데이터 그래프")

# CSV 파일을 URL에서 직접 가져와 데이터프레임으로 로드
@st.cache_data
def load_data():
    response = requests.get(url)
    response.raise_for_status()  # 요청이 성공했는지 확인
    data = pd.read_csv(StringIO(response.text))  # StringIO를 사용하여 텍스트를 읽습니다.
    return data

# 데이터 로드
df = load_data()

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

