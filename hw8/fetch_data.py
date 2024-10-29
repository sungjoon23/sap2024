import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
import os


# AWS 기상 데이터를 URL에서 받아오는 함수
def fetch_aws_data(site, dev, year, month, day):
    url = f"http://203.239.47.148:8080/dspnet.aspx?Site={site}&Dev={dev}&Year={year}&Mon={month}&Day={day}"
    response = requests.get(url)

    if response.status_code == 200:
        # 데이터를 텍스트 형식으로 반환 (예: CSV 형식)
        data = response.text
        return data
    else:
        print(f"Error: Failed to fetch data. Status code: {response.status_code}")
        return None


# 데이터를 pandas DataFrame으로 변환하는 함수
def convert_to_dataframe(data):
    df = pd.read_csv(StringIO(data), sep=",", header=None)  # 데이터를 읽어옴 (header 없을 수 있음)

    # 첫 번째 열이 시간 데이터이므로 그 열을 시간으로 변환
    time_col = 0  # 첫 번째 열이 시간 열
    try:
        df[time_col] = pd.to_datetime(df[time_col])  # 첫 번째 열을 시간 형식으로 변환
        df.set_index(time_col, inplace=True)  # 시간 열을 인덱스로 설정
    except Exception as e:
        print(f"Error: Failed to convert time column. {e}")
        return None

    # 열 이름 확인을 위해서 출력 (필요에 따라 제거 가능)
    print("변환된 데이터프레임:", df.head())

    return df


# 1시간 단위로 리샘플링하고 평균값을 계산하는 함수
def resample_data_hourly(df):
    df_hourly = df.resample('h').mean()  # 1시간 단위로 리샘플링 후 평균값 계산
    return df_hourly


# 새로운 폴더를 만들고 데이터를 CSV로 저장하는 함수
def save_data(df, city_name):
    # 오늘 날짜 정보
    now = datetime.now()
    month_folder = now.strftime("%Y-%m")  # 예: "2024-10"

    # 연월 형식으로 파일 이름 변경 (예: 2024.10.Jeonju.csv)
    file_name = f"{now.strftime('%Y.%m.')}{city_name}.csv"

    # 절대 경로로 파일 저장 경로 설정
    folder_path = os.path.join(os.getcwd(), "hw8", month_folder)
    os.makedirs(folder_path, exist_ok=True)

    # 파일 경로 설정
    file_path = os.path.join(folder_path, file_name)

    # 파일이 이미 존재하면 기존 데이터를 불러오고 병합
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        merged_df = pd.concat([existing_df, df])
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
    else:
        merged_df = df

    # 병합된 데이터를 CSV로 저장
    merged_df.to_csv(file_path)
    print(f"Data has been saved to {file_path}.")


# 메인 함수
def main():
    site = 85  # 예시 Site ID
    dev = 1  # 예시 Device ID
    city_name = "Jeonju"  # 도시 이름 또는 관측소 이름

    # 어제 날짜 기준으로 데이터를 요청
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    data = fetch_aws_data(site, dev, yesterday.year, yesterday.month, yesterday.day)

    if data:
        # 가져온 데이터 미리보기 출력
        print(f"Fetched data preview:\n{data[:500]}")  # 앞 500자를 미리 출력

        # 데이터를 DataFrame으로 변환
        df = convert_to_dataframe(data)

        if df is not None:
            # 1시간 단위로 리샘플링하고 평균 계산
            df_hourly = resample_data_hourly(df)

            # 데이터를 월별 폴더에 저장 (연월 형식으로 파일명 지정)
            save_data(df_hourly, city_name)
            print("Hourly average data saved and merged successfully.")
        else:
            print("Error: Failed to convert data to DataFrame.")
    else:
        print("Error: Failed to fetch data.")


if __name__ == "__main__":
    main()
