import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


# 데이터 불러오기 (최신 파일을 자동으로 불러옴)
def load_latest_data(city_name):
    files = glob.glob(f"weather_data_{city_name}_*.csv")
    if not files:
        st.error("데이터 파일을 찾을 수 없습니다. 먼저 데이터를 받아오세요.")
        return None
    latest_file = max(files, key=os.path.getctime)
    return pd.read_csv(latest_file)


# 간단한 기온 예측 함수 (이동 평균 사용)
def predict_next_day_temperature(df):
    df['datetime'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M')
    df = df.set_index('datetime')
    daily_avg_temp = df['temperature'].resample('D').mean()

    if len(daily_avg_temp) < 2:
        st.warning("데이터가 충분하지 않아 예측을 할 수 없습니다.")
        return None, None

    # 예측: 전날 기온의 이동 평균으로 다음 날 예측
    next_day = daily_avg_temp.index[-1] + timedelta(days=1)
    predicted_temp = daily_avg_temp.rolling(window=2).mean().iloc[-1]

    return next_day, predicted_temp


# 기온 그래프 그리기
def plot_temperature(df):
    df['datetime'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M')
    plt.figure(figsize=(10, 5))
    plt.plot(df['datetime'], df['temperature'], marker='o')
    plt.title('Temperature Changes Over Time')
    plt.xlabel('Date and Time')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    st.pyplot(plt)


# Streamlit 앱
def main():
    st.title("기온 변화 및 다음날 예측")

    # 도시 선택
    cities = {'서울': '서울', '부산': '부산', '인천': '인천'}
    city_name = st.selectbox("도시를 선택하세요", list(cities.keys()))

    # 최신 데이터 로드
    df = load_latest_data(city_name)

    if df is not None:
        # 데이터 시각화
        st.write(f"{city_name}의 기온 변화")
        plot_temperature(df)

        # 다음 날 기온 예측
        next_day, predicted_temp = predict_next_day_temperature(df)
        if next_day and predicted_temp:
            st.write(f"예측된 {next_day.strftime('%Y-%m-%d')}의 평균 기온: {predicted_temp:.2f}°C")

            # 날씨 예측 결과에 따른 간단한 설명
            if predicted_temp > 30:
                st.write("다음 날은 무더운 날씨가 예상됩니다.")
            elif predicted_temp < 0:
                st.write("다음 날은 추운 날씨가 예상됩니다.")
            else:
                st.write("다음 날은 온화한 날씨가 예상됩니다.")
    else:
        st.error("데이터를 불러오는 데 문제가 발생했습니다.")


if __name__ == "__main__":
    main()