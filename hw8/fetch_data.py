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
        # 데이터를 텍스트 형식으로 반환 (예: CSV 형식)
        data = response.text
        return data
    else:
        return None

# 데이터를 pandas DataFrame으로 변환하는 함수
def convert_to_dataframe(data):
    df = pd.read_csv(StringIO(data), sep=",")  # 데이터를 읽어옴

    # 열 이름을 출력해서 확인
    print("열 이름:", df.columns)

    # 시간 열을 찾고 변환 ('time', 'timestamp' 등의 다양한 이름을 지원)
    time_columns = ['time', 'timestamp', 'Time', 'Timestamp']  # 시간 관련 열 이름 후보
    time_col = None
    for col in time_columns:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
    else:
        print("시간 관련 열이 없습니다. 열 이름을 확인하세요.")
        return None
    
    return df

# 1시간 단위로 리샘플링하고 평균값을 계산하는 함수
def resample_data_hourly(df):
    df_hourly = df.resample('H').mean()  # 1시간 단위로 리샘플링 후 평균값 계산
    return df_hourly

# 새로운 폴더를 만들고 데이터를 CSV로 저장하는 함수
def save_data(df, city_name):
    # 오늘 날짜 정보
    now = datetime.now()
    month_folder = now.strftime("%Y-%m")  # 예: "2024-10"
    
    # 연월 형식으로 파일 이름 변경 (예: 2024.10.Jeonju.csv)
    file_name = f"{now.strftime('%Y.%m.')}{city_name}.csv"

    # 해당 월 폴더 경로 생성 (존재하지 않으면 새로 생성)
    folder_path = os.path.join("hw8", month_folder)
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
    print(f"데이터가 {file_path} 파일에 저장되었습니다.")

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

        if df is not None:
            # 1시간 단위로 리샘플링하고 평균 계산
            df_hourly = resample_data_hourly(df)

            # 데이터를 월별 폴더에 저장 (연월 형식으로 파일명 지정)
            save_data(df_hourly, city_name)
            print("1시간 단위 평균 데이터를 저장하고 누적했습니다.")
        else:
            print("데이터프레임 변환에 실패했습니다.")
    else:
        print("데이터를 가져오지 못했습니다.")

if __name__ == "__main__":
    main()
