import requests
import pandas as pd
from datetime import datetime


def fetch_and_save_environment_data(station_id, year, month, day):
    # API URL에 년, 월, 일 파라미터 추가
    url = (f"https://api.taegon.kr/stations/{station_id}/"
           f"?sy={year}&sm={month}&sd={day}"
           f"&ey={year}&em={month}&ed={day}&format=csv")

    # 파일 이름 설정 (리포지토리 내에 데이터를 누적 저장할 파일)
    file_name = "environment_data.csv"

    # API로부터 데이터 가져오기
    response = requests.get(url)

    if response.status_code == 200:
        # 응답 데이터를 pandas로 읽기
        new_data = pd.read_csv(pd.compat.StringIO(response.content.decode('utf-8')))

        # 기존 파일이 있으면 데이터를 덧붙이고, 없으면 새로 생성
        try:
            existing_data = pd.read_csv(file_name)
            updated_data = pd.concat([existing_data, new_data])
        except FileNotFoundError:
            updated_data = new_data

        # 데이터 저장 (덮어쓰기)
        updated_data.to_csv(file_name, index=False)
        print(f"Data for station {station_id} saved and updated in {file_name}.")
    else:
        print(f"Failed to fetch data for station {station_id}. Status Code: {response.status_code}")


if __name__ == "__main__":
    today = datetime.now()
    station_id = 146  
    year = today.year
    month = today.month
    day = today.day

    fetch_and_save_environment_data(station_id, year, month, day)
