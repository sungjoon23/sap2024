import requests
import pandas as pd
from datetime import datetime
from io import StringIO

# AWS 기상 데이터를 URL에서 받아오는 함수
def fetch_aws_data(site, dev, year, month, day):
    url = f"http://203.239.47.148:8080/dspnet.aspx?Site={site}&Dev={dev}&Year={year}&Mon={month}&Day={day}"
    response = requests.get(url)

    if response.status_code == 200:
        # 데이터가 CSV 형식으로 내려온다고 가정
        data = response.text
        return data
    else:
        return None

# 데이터를 pandas DataFrame으로 변환하는 함수
def convert_to_dataframe(data):
    # 데이터가 CSV 형식이라 가정하고 처리
    df = pd.read_csv(StringIO(data), sep=",")  # 적절한 구분자 지정 필요
    # 시간 열이 있다고 가정하고 변환 (필요한 열 이름에 맞게 수정)
    df['time'] = pd.to_datetime(df['time'])  # 'time' 열 이름을 실제 데이터 열에 맞게 수정
    df.set_index('time', inplace=True)
    return df

# 1시간 단위로 리샘플링하고 평균값을 계산하는 함수
def resample_data_hourly(df):
    df_hourly = df.resample('H').mean()  # 1시간 단위로 리샘플링 후 평균값 계산
    return df_hourly

# 데이터를 CSV 파일로 저장하는 함수 (덮어쓰기)
def save_data(df, city_name):
    # 파일 경로 설정
    file_path = f"weather_data_{city_name}_hourly.csv"
    df.to_csv(file_path, mode='a', header=not pd.io.common.file_exists(file_path))  # 데이터 추가 저장

# 메인 함수
def main():
    site = 85  # 예시 Site ID
    dev = 1  # 예시 Device ID
    city_name = "Jeonju"  # 도시 이름 또는 관측소 이름

    # 오늘 날짜 기준으로 데이터를 요청
    now = datetime.now()
    data = fetch_aws_data(site, dev, now.year, now.month, now.day)

    if data:
        # 데이터를 DataFrame으로 변환
        df = convert_to_dataframe(data)

        # 1시간 단위로 리샘플링하고 평균 계산
        df_hourly = resample_data_hourly(df)

        # 데이터를 CSV 파일로 저장
        save_data(df_hourly, city_name)
        print("1시간 단위 평균 데이터를 저장했습니다.")
    else:
        print("데이터를 가져오지 못했습니다.")

if __name__ == "__main__":
    main()
