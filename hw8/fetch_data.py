import requests
import pandas as pd
from datetime import datetime


# AWS 기상 데이터를 URL에서 받아오는 함수
def fetch_aws_data(site, dev, year, month, day):
    url = f"http://203.239.47.148:8080/dspnet.aspx?Site={site}&Dev={dev}&Year={year}&Mon={month}&Day={day}"
    response = requests.get(url)

    if response.status_code == 200:
        # 필요한 경우 데이터 파싱 (예시: CSV 형식으로 변환 등)
        data = response.text
        return data
    else:
        return None


# 데이터를 저장하는 함수
def save_data(data, city_name):
    # 파일 경로 설정
    file_path = f"weather_data_{city_name}.csv"

    # 데이터가 기존에 존재하는지 확인 후 저장
    with open(file_path, 'a') as f:
        f.write(data)  # 파일 끝에 데이터를 추가


# 메인 함수
def main():
    site = 85  # 예시 Site ID
    dev = 1  # 예시 Device ID
    city_name = "Seoul"  # 도시 이름 또는 관측소 이름

    # 오늘 날짜 기준으로 데이터를 요청
    now = datetime.now()
    data = fetch_aws_data(site, dev, now.year, now.month, now.day)

    if data:
        save_data(data, city_name)
    else:
        print("데이터를 가져오지 못했습니다.")


if __name__ == "__main__":
    main()
