import requests
import pandas as pd
from datetime import datetime
from io import StringIO
import os

# AWS 기상 데이터를 URL에서 받아오는 함수
def fetch_aws_data(site, dev, year, month, day):
    url = f"http://203.239.47.148:8080/dspnet.aspx?Site={site}&Dev={dev}&Year={year}&Mon={month}&Day={day}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.text
        if not data.strip():  # 데이터가 비어 있는지 확인
            print("Error: No data fetched from the server.")
            return None
        return data
    else:
        print(f"Error: Failed to fetch data. Status code: {response.status_code}")
        return None

# 데이터를 pandas DataFrame으로 변환하는 함수
def convert_to_dataframe(data):
    try:
        df = pd.read_csv(StringIO(data), sep=",", header=None)
        df = df.rename(columns={0: 'Timestamp', 1: 'TEMP', 2: 'HUMI', 6: 'IRRAD', 7: 'WIND_DIR', 13: 'WIND', 14: 'RAIN',
                                15: 'WIND_MAX', 16: 'VOLTAGE'})
        df = df[['Timestamp', 'TEMP', 'HUMI', 'IRRAD', 'WIND_DIR', 'WIND', 'RAIN', 'WIND_MAX', 'VOLTAGE']]

        # 첫 번째 열을 시간 형식으로 변환
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.set_index('Timestamp', inplace=True)

        # 변환 중 오류가 발생한 경우를 대비해 NaT 또는 NaN이 있는 행을 제거
        df.dropna(how="all", inplace=True)

        if df.empty:
            print("Warning: DataFrame is empty after conversion.")
            return None

        return df
    except pd.errors.EmptyDataError:
        print("Error: No columns to parse from the fetched data.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# 1시간 단위로 리샘플링하고 평균값을 계산하는 함수
def resample_data_hourly(df):
    if df is not None and not df.empty:
        df_hourly = df.resample('h').mean()  # 1시간 단위로 리샘플링 후 평균값 계산
        df_hourly = df_hourly.round(2)  # 소수점 2자리까지만 반올림
        return df_hourly
    else:
        print("Error: Cannot resample an empty DataFrame.")
        return None

# 새로운 폴더를 만들고 데이터를 CSV로 저장하는 함수 (세로로 데이터 누적)
def save_data(df, city_name):
    if df is None or df.empty:
        print("Error: No data to save.")
        return

    now = datetime.now()
    month_folder = now.strftime("%Y-%m")

    file_name = f"{now.strftime('%Y.%m.')}{city_name}.csv"
    folder_path = os.path.join(os.getcwd(), "hw8", month_folder)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        merged_df = pd.concat([existing_df, df], axis=0)
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
    else:
        merged_df = df

    merged_df.sort_index(inplace=True)
    merged_df.to_csv(file_path)
    print(f"Data has been saved to {file_path}.")

# 메인 함수
def main():
    site = 85
    dev = 1
    city_name = "Jeonju"

    now = datetime.now()
    data = fetch_aws_data(site, dev, now.year, now.month, now.day)

    if data:
        df = convert_to_dataframe(data)

        if df is not None:
            df_hourly = resample_data_hourly(df)

            if df_hourly is not None:
                save_data(df_hourly, city_name)
                print("Hourly average data saved and merged successfully.")
            else:
                print("Error: Failed to resample data.")
        else:
            print("Error: Failed to convert data to DataFrame.")
    else:
        print("Error: Failed to fetch data.")

if __name__ == "__main__":
    main()
