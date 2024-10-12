import pandas as pd
import os
from datetime import datetime, timedelta
import streamlit as st
import folium
from streamlit_folium import folium_static
import plotly.express as px
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# DVR 모델: 발육 속도를 계산하는 함수
def calculate_dvr_new(t, A=107.94, B=0.9, Tc=5.4):
    if t > Tc:
        DVRi = (1 / (A * (B ** t))) * 100  # t에서 Tc 이상만 계산하도록 수정
        return DVRi
    else:
        return 0

def predict_blooming_day_new_dvr(temperature_data, threshold=100, Tc=5.4):
    dvs_sum = 0
    for day, temp in enumerate(temperature_data, start=1):
        dvr = calculate_dvr_new(temp, Tc=Tc)
        dvs_sum += dvr
        if dvs_sum >= threshold:
            return day
    return None


# DVR1 계산: Tc 범위에서 발육속도 계산
def calculate_dvr1(temp_min, temp_max, Tc=5.4):
    if 0 <= temp_min <= Tc or 0 <= temp_max <= Tc:
        # 온도가 0~Tc도 사이에 있는 경우 1씩 더하는 단순 계산
        return 1  # 추가적인 로직이 필요할 수 있음 (예: 온도 범위에 따라 가중치 부여)
    return 0


# DVR2 계산: 평균 기온이 Tc 이상일 때 발육 속도를 계산
def calculate_dvr2(temperature, Tc=5.4):
    if temperature > Tc:
        return (temperature - Tc) / 100  # 더 복잡한 식이 있을 수 있음 (연구 문헌 참고)
    return 0


# 예상 만개일을 계산하는 함수
def predict_blooming_day_mdvr(temp_min_data, temp_max_data, avg_temp_data, Tc=5.4):
    dvr1_sum = 0
    dvr2_sum = 0
    day_of_dvr1_end = None

    # DVR1 누적 계산: DVR1 합계가 2에 도달할 때까지 반복
    for day in range(len(temp_min_data)):
        dvr1_sum += calculate_dvr1(temp_min_data[day], temp_max_data[day], Tc=Tc)
        if dvr1_sum >= 2:
            day_of_dvr1_end = day
            break

    # 내생휴면이 해제되지 않은 경우 None 반환
    if day_of_dvr1_end is None:
        return None

    # DVR2 누적 계산: DVR2 합계가 0.9593에 도달할 때까지 반복
    for day in range(day_of_dvr1_end, len(avg_temp_data)):
        dvr2_sum += calculate_dvr2(avg_temp_data[day], Tc=Tc)
        if dvr2_sum >= 0.9593:
            return day + 1  # 만개일 반환 (1일차는 day 0에서 시작하므로 +1)

    return None  # 만개일을 예측할 수 없으면 None 반환


# 냉각량(Chill Units) 계산 함수
def calculate_chill_units(temp_min, temp_max, Tc=5.4):
    chill_units = 0
    if temp_min < Tc:
        chill_units += Tc - temp_min  # temp_min이 Tc보다 낮을 때의 냉각량 계산
    if temp_max < Tc:
        chill_units += Tc - temp_max  # temp_max가 Tc보다 낮을 때의 냉각량 계산
    return chill_units


# 가온량(Heat Units) 계산 함수
def calculate_heat_units(temp_min, temp_max, Tc=5.4):
    heat_units = 0
    if temp_min > Tc:
        heat_units += temp_min - Tc  # temp_min이 Tc보다 높을 때의 가온량 계산
    if temp_max > Tc:
        heat_units += temp_max - Tc  # temp_max가 Tc보다 높을 때의 가온량 계산
    return heat_units


# 예상 만개일을 계산하는 함수 (Chill Days Model)
def predict_blooming_day_cd(temp_min_data, temp_max_data, Cr=-86.4, Hr=272, Tc=5.4):
    chill_units_sum = 0
    heat_units_sum = 0
    chill_reached = False

    # 저온 요구량(Cr)이 충족될 때까지 냉각량을 계산
    for day in range(len(temp_min_data)):
        chill_units = calculate_chill_units(temp_min_data[day], temp_max_data[day], Tc)
        chill_units_sum -= chill_units  # 누적 냉각량 감소
        if chill_units_sum <= Cr:  # 저온 요구도(Cr)에 도달하면
            chill_reached = True
            break

    # 저온 요구량이 충족되지 않으면 None 반환
    if not chill_reached:
        return None

    # 고온 요구량(Hr)을 계산하여 만개기 예측
    for day in range(day, len(temp_min_data)):  # 내생휴면이 끝난 시점부터 시작
        heat_units = calculate_heat_units(temp_min_data[day], temp_max_data[day], Tc)
        heat_units_sum += heat_units  # 가온량 누적
        if heat_units_sum >= Hr:  # 고온 요구도(Hr)에 도달하면 만개일 반환
            return day + 1  # 1일차는 day 0이므로 +1

    return None  # 만개일을 예측할 수 없으면 None 반환

# 모델 선택 및 실행 함수
def select_model_and_predict(model_choice, temp_min_data, temp_max_data, avg_temp_data):
    if model_choice == 1:
        return predict_blooming_day_new_dvr(avg_temp_data)
    elif model_choice == 2:
        return predict_blooming_day_mdvr(temp_min_data, temp_max_data, avg_temp_data)
    elif model_choice == 3:
        return predict_blooming_day_cd(temp_min_data, temp_max_data)
    else:
        return "잘못된 선택입니다."


def process_csv_file_single_model(file_path, model_choice):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if 'tavg' not in df.columns:
        if 'tmax' in df.columns and 'tmin' in df.columns:
            df['tavg'] = (df['tmax'] + df['tmin']) / 2
        else:
            raise KeyError("데이터셋에 'tavg' 또는 'tmax', 'tmin' 컬럼이 없습니다.")

    avg_temp_data = df['tavg'].values
    temp_min_data = df['tmin'].values
    temp_max_data = df['tmax'].values

    results = {}

    for year in range(2000, 2025):
        year_mask = df['year'] == year
        year_avg_temp = avg_temp_data[year_mask]
        year_min_temp = temp_min_data[year_mask]
        year_max_temp = temp_max_data[year_mask]

        # 선택된 모델에 따른 예측 수행
        blooming_day = select_model_and_predict(model_choice, year_min_temp, year_max_temp, year_avg_temp)

        if isinstance(blooming_day, int):
            start_date = datetime(year, 1, 1)
            bloom_date = start_date + timedelta(days=blooming_day - 1)
            results[year] = bloom_date.strftime('%Y.%m.%d')
        else:
            results[year] = "예측 불가"

    return df, results

# CSV 파일 처리 함수 (모든 모델 예측)
def process_csv_file_all_models(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if 'tavg' not in df.columns:
        if 'tmax' in df.columns and 'tmin' in df.columns:
            df['tavg'] = (df['tmax'] + df['tmin']) / 2
        else:
            raise KeyError("데이터셋에 'tavg' 또는 'tmax', 'tmin' 컬럼이 없습니다.")

    avg_temp_data = df['tavg'].values
    temp_min_data = df['tmin'].values
    temp_max_data = df['tmax'].values

    results = {
        'dvr_bloom_date': {},
        'mdvr_bloom_date': {},
        'cd_bloom_date': {}
    }

    for year in range(2000, 2025):
        year_mask = df['year'] == year
        year_avg_temp = avg_temp_data[year_mask]
        year_min_temp = temp_min_data[year_mask]
        year_max_temp = temp_max_data[year_mask]

        # DVR 모델 예측
        dvr_blooming_day = predict_blooming_day_new_dvr(year_avg_temp)
        if dvr_blooming_day is not None:
            start_date = datetime(year, 1, 1)
            bloom_date = start_date + timedelta(days=dvr_blooming_day - 1)
            results['dvr_bloom_date'][year] = bloom_date.strftime('%Y.%m.%d')
        else:
            results['dvr_bloom_date'][year] = "예측 불가"

        # mDVR 모델 예측
        mdvr_blooming_day = predict_blooming_day_mdvr(year_min_temp, year_max_temp, year_avg_temp)
        if mdvr_blooming_day is not None:
            start_date = datetime(year, 1, 1)
            bloom_date = start_date + timedelta(days=mdvr_blooming_day - 1)
            results['mdvr_bloom_date'][year] = bloom_date.strftime('%Y.%m.%d')
        else:
            results['mdvr_bloom_date'][year] = "예측 불가"

        # CD 모델 예측
        cd_blooming_day = predict_blooming_day_cd(year_min_temp, year_max_temp)
        if cd_blooming_day is not None:
            start_date = datetime(year, 1, 1)
            bloom_date = start_date + timedelta(days=cd_blooming_day - 1)
            results['cd_bloom_date'][year] = bloom_date.strftime('%Y.%m.%d')
        else:
            results['cd_bloom_date'][year] = "예측 불가"

    return df, results

# 모델 비교 함수 추가
def process_csv_file_for_comparison(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    avg_temp_data = df['tavg'].values
    temp_min_data = df['tmin'].values
    temp_max_data = df['tmax'].values

    comparison_results = []

    for year in range(2000, 2025):
        year_mask = df['year'] == year
        year_avg_temp = avg_temp_data[year_mask]
        year_min_temp = temp_min_data[year_mask]
        year_max_temp = temp_max_data[year_mask]

        # 각 모델의 예측 개화일
        dvr_blooming_day = predict_blooming_day_new_dvr(year_avg_temp)
        mdvr_blooming_day = predict_blooming_day_mdvr(year_min_temp, year_max_temp, year_avg_temp)
        cd_blooming_day = predict_blooming_day_cd(year_min_temp, year_max_temp)

        if isinstance(dvr_blooming_day, int):
            start_date = datetime(year, 1, 1)
            dvr_bloom_date = start_date + timedelta(days=dvr_blooming_day - 1)
        else:
            dvr_bloom_date = "예측 불가"

        if isinstance(mdvr_blooming_day, int):
            start_date = datetime(year, 1, 1)
            mdvr_bloom_date = start_date + timedelta(days=mdvr_blooming_day - 1)
        else:
            mdvr_bloom_date = "예측 불가"

        if isinstance(cd_blooming_day, int):
            start_date = datetime(year, 1, 1)
            cd_bloom_date = start_date + timedelta(days=cd_blooming_day - 1)
        else:
            cd_bloom_date = "예측 불가"

        comparison_results.append({
            "Year": year,
            "DVR": dvr_bloom_date,
            "mDVR": mdvr_bloom_date,
            "CD": cd_bloom_date
        })

    return pd.DataFrame(comparison_results)


# 실제 개화일 데이터를 불러오는 함수
def load_actual_bloom_dates(file_path):
    df_avg_blooming = pd.read_csv(file_path)
    df_avg_blooming.columns = df_avg_blooming.columns.str.strip()

    actual_bloom_dates = {}
    for _, row in df_avg_blooming.iterrows():
        actual_bloom_date = datetime.strptime(row['blooming'].strip(), '%Y-%m-%d')
        year = int(row['year'])
        actual_bloom_dates[year] = actual_bloom_date

    return actual_bloom_dates


def calculate_r2_rmse_mae_and_plot(predicted_bloom_dates, actual_bloom_dates, model_name):
    # 실제 개화일과 예측 개화일을 저장할 리스트
    actual_bloom_list = []
    predicted_bloom_list = []

    for year, predicted_date in predicted_bloom_dates.items():
        if predicted_date != "예측 불가" and year in actual_bloom_dates:
            actual_bloom_date = actual_bloom_dates[year]
            predicted_bloom_date = datetime.strptime(predicted_date, '%Y.%m.%d')

            # 실제 개화일과 예측 개화일을 둘 다 날짜 형식으로 변환한 후 추가
            actual_bloom_list.append(actual_bloom_date.timetuple().tm_yday)
            predicted_bloom_list.append(predicted_bloom_date.timetuple().tm_yday)

    if not actual_bloom_list or not predicted_bloom_list:
        st.write(f"{model_name} 모델에 대한 유효한 예측 데이터가 없습니다.")
        return

    # 실제 개화일(Y축)과 예측 개화일(X축)을 NumPy 배열로 변환
    actual_bloom_array = np.array(actual_bloom_list)
    predicted_bloom_array = np.array(predicted_bloom_list)

    # 회귀 모델을 사용하여 R² 값 계산
    model = LinearRegression()
    actual_bloom_array_reshaped = actual_bloom_array.reshape(-1, 1)
    model.fit(actual_bloom_array_reshaped, predicted_bloom_array)
    predicted_values = model.predict(actual_bloom_array_reshaped)

    # R² 값을 계산
    r2 = r2_score(predicted_bloom_array, predicted_values)

    # RMSE와 MAE 계산
    rmse = np.sqrt(mean_squared_error(predicted_bloom_array, predicted_values))
    mae = mean_absolute_error(predicted_bloom_array, predicted_values)

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_bloom_array, predicted_bloom_array, color='blue', label='Predicted vs Actual')
    plt.plot(actual_bloom_array, predicted_values, color='red', label=f'Regression Line (R² = {r2:.2f})')
    plt.xlabel('Actual Blooming Day')
    plt.ylabel('Predicted Blooming Day')
    plt.legend()
    plt.grid(True)

    # 그래프 출력
    st.pyplot(plt)

    # R², RMSE, MAE 값 출력
    st.write(f"**{model_name} 모델의 성능 지표:**")
    st.write(f"- **R² 값:** {r2:.2f}")
    st.write(f"- **RMSE 값:** {rmse:.2f}")
    st.write(f"- **MAE 값:** {mae:.2f}")


# 개화일에 따라 색상을 결정하는 함수 (평균 개화일과 비교)
def get_color(bloom_date, avg_blooming_date):
    if bloom_date == "예측 불가":
        return "gray"
    else:
        # 공백 제거 후 날짜 형식 변환
        bloom_date_dt = datetime.strptime(bloom_date, "%Y.%m.%d")
        avg_bloom_date_dt = datetime.strptime(avg_blooming_date.strip(), "%Y-%m-%d")  # 공백 제거

        if bloom_date_dt > avg_bloom_date_dt:
            return "red"  # 늦은 개화일
        elif bloom_date_dt == avg_bloom_date_dt:
            return "green"  # 같은 날 개화
        else:
            return "yellow"  # 이른 개화일

# GDD (Growing Degree Days) 계산 함수
def calculate_gdd(tmin, tmax, base_temp=5.4):
    # 일일 평균 기온 계산
    tavg = (tmin + tmax) / 2
    # 일일 GDD 계산 (기온이 기준 온도(base_temp)보다 높을 때만)
    gdd = max(tavg - base_temp, 0)
    return gdd

# Streamlit 앱 설정
st.title('지역별 개화 예측')

# 좌측과 우측 레이아웃으로 페이지 구분 (비율 1:2)
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("설정")

    # hw4 디렉토리 내의 CSV 파일 목록 자동 인식
    csv_dir = 'C:/code/pythonProject3/hw4'
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    selected_region = st.selectbox('지역을 선택하세요:', csv_files)
    model_choice = st.selectbox(
        "개화 예측 모델을 선택하세요:",
        (1, 2, 3),
        format_func=lambda x: {1: "DVR 모델", 2: "mDVR 모델", 3: "CD 모델"}[x]
    )

    years = list(range(2000, 2025))
    st.subheader("연도 선택:")
    slider_year = st.slider('', 2000, 2024, 2020)

    details_button = st.button('개화 예측 실행')
    show_temp_button = st.button('온도 데이터 확인')
    compare_button = st.button('모델 결과 비교')
    st_button = st.button("모델 성능 검정")

with right_col:
    file_path = os.path.join(csv_dir, selected_region)
    df, results = process_csv_file_all_models(file_path)

    st.subheader(f"{selected_region} 데이터")
    st.dataframe(df)

    st.subheader(f": {slider_year}년 개화일 예측 결과")
    bloom_date_slider = results['dvr_bloom_date'].get(slider_year, "예측 불가")
    st.write(f"{slider_year}년 개화일: {bloom_date_slider}")

    # 온도 데이터 확인 버튼을 클릭했을 때 tmax, tavg, tmin, 누적 GDD 그래프 표시
    if show_temp_button:
        st.subheader(f"{selected_region}의 {slider_year}년 온도 데이터 및 누적 GDD 그래프")

        # 선택된 연도의 데이터 필터링
        year_data = df[df['year'] == slider_year]

        # date 컬럼 생성
        year_data['date'] = pd.to_datetime(
            year_data['year'].astype(str) + '-' + year_data['month'].astype(str) + '-' + year_data['day'].astype(str))

        # GDD 계산하여 year_data에 추가 (누적합 계산)
        year_data['GDD'] = year_data.apply(lambda row: calculate_gdd(row['tmin'], row['tmax']), axis=1)
        year_data['cumulative_GDD'] = year_data['GDD'].cumsum()  # GDD 누적합 계산

        # tmax, tavg, tmin의 값을 일별로 선 그래프로 그리기
        fig = px.line(year_data, x='date', y=['tmax', 'tavg', 'tmin', 'cumulative_GDD'], labels={
            'value': 'Temperature (°C) & Cumulative GDD',
            'date': 'Date'
        })

        # 그래프 출력
        st.plotly_chart(fig)

    if compare_button:
        # 모델 비교 함수 호출
        comparison_df = process_csv_file_for_comparison(file_path)

        # 시간 부분을 제거하는 코드 (문자열에서 뒷부분 자르기)
        comparison_df['DVR'] = comparison_df['DVR'].astype(str).str.split(' ').str[0]
        comparison_df['mDVR'] = comparison_df['mDVR'].astype(str).str.split(' ').str[0]
        comparison_df['CD'] = comparison_df['CD'].astype(str).str.split(' ').str[0]

        # 모델 비교 결과를 표로 출력
        st.subheader(f"{selected_region}의 2000~2024년 모델별 개화 예측 비교")
        st.table(comparison_df)

    # 모델 성능 검정 버튼 추가
    if st_button:
        st.write("모델 성능 검정 중...")

        avg_blooming_file = 'C:/code/pythonProject3/hw4/wb/wb_avg.csv'
        actual_bloom_dates = load_actual_bloom_dates(avg_blooming_file)

        # Get predictions for each model
        predictions = {
            "DVR 모델": results['dvr_bloom_date'],
            "mDVR 모델": results['mdvr_bloom_date'],
            "CD 모델": results['cd_bloom_date']
        }

        # Divide into three columns
        col1, col2, col3 = st.columns(3)

        # Display the performance of DVR model in the first column
        with col1:
            st.subheader("DVR 모델 성능")
            calculate_r2_rmse_mae_and_plot(predictions["DVR 모델"], actual_bloom_dates, "DVR 모델")

        # Display the performance of mDVR model in the second column
        with col2:
            st.subheader("mDVR 모델 성능")
            calculate_r2_rmse_mae_and_plot(predictions["mDVR 모델"], actual_bloom_dates, "mDVR 모델")

        # Display the performance of CD model in the third column
        with col3:
            st.subheader("CD 모델 성능")
            calculate_r2_rmse_mae_and_plot(predictions["CD 모델"], actual_bloom_dates, "CD 모델")

    if details_button:
        avg_blooming_file = 'C:/code/pythonProject3/hw4/wb/wb_avg.csv'
        df_avg_blooming = pd.read_csv(avg_blooming_file)
        df_avg_blooming.columns = df_avg_blooming.columns.str.strip()
        avg_blooming_date = df_avg_blooming[df_avg_blooming['year'] == slider_year]['blooming'].values[0]
        st.subheader(f"{slider_year}년 개화일 결과")
        st.write(f"{slider_year}년 개화일: {avg_blooming_date}")

        m = folium.Map(location=[36.5, 127.5], zoom_start=7)

        region_coordinates = {
            "wb_cheonan.csv": {'lat': 36.8151, 'lon': 127.1139},
            "wb_incheon.csv": {'lat': 37.4563, 'lon': 126.7052},
            "wb_jeonju.csv": {'lat': 35.8251, 'lon': 127.1480},
            "wb_naju.csv": {'lat': 35.0158, 'lon': 126.7112},
            "wb_sacheon.csv": {'lat': 35.0037, 'lon': 128.0646},
            "wb_sangju.csv": {'lat': 36.4151, 'lon': 128.1599},
            "wb_ulju.csv": {'lat': 35.5624, 'lon': 129.2424},
            "wb_wanju.csv": {'lat': 35.9054, 'lon': 127.1625},
            "wb_yeongcheon.csv": {'lat': 35.9733, 'lon': 128.9381},
        }

        bloom_data = {}
        for csv_file in csv_files:
            file_path = os.path.join(csv_dir, csv_file)
            _, results = process_csv_file_all_models(file_path)
            bloom_data[csv_file] = results['dvr_bloom_date'].get(slider_year, "예측 불가")

        for csv_file, data in region_coordinates.items():
            bloom_date = bloom_data.get(csv_file, "예측 불가")
            folium.CircleMarker(
                location=[data['lat'], data['lon']],
                radius=10,
                color=get_color(bloom_date, avg_blooming_date),  # 평균 개화일과 비교한 색상 결정
                fill=True,
                fill_color=get_color(bloom_date, avg_blooming_date),
                fill_opacity=0.7,
                popup=f"{csv_file}: {bloom_date}"
            ).add_to(m)

        folium_static(m)

# Load the weather data for prediction
file_path = r'C:\code\pythonProject3\hw4\wb_seoul.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Check if 'year', 'month', and 'day' columns exist
if {'year', 'month', 'day'}.issubset(df.columns):
    # Create 'date' column if the necessary columns are present
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
else:
    st.error("The required 'year', 'month', and 'day' columns are missing in the dataset.")
    st.stop()

# Load the actual bloom data
actual_bloom_file_path = r'C:\code\pythonProject3\hw4\wb\actually data.csv'
actual_bloom_data = pd.read_csv(actual_bloom_file_path)
actual_bloom_data['blooming_date'] = pd.to_datetime(actual_bloom_data['blooming'], format='%Y-%m-%d')
actual_bloom_data['DOY_actual'] = actual_bloom_data['blooming_date'].dt.dayofyear

# 벚꽃 개화 모델 버튼 추가 (Unique key added)
new_button = st.button("벚꽃 개화 모델", key="bloom_model_button")

if new_button:
    # Split layout into two columns (left for first model, right for second model)
    col1, col2 = st.columns(2)

    # **First Model: Heat Units-based Prediction**
    with col1:
        st.subheader("생물계절 기반 벚꽃 개화 예측")

        # Function to calculate heat units (GDD concept)
        def calculate_heat_units(tmin, tmax, base_temp=7):
            tavg = (tmin + tmax) / 2
            if tavg > base_temp:
                return tavg - base_temp  # GDD only accumulates when the temperature is above base_temp
            else:
                return 0  # no GDD accumulation below the base temperature

        # Prediction function with dormancy ending on January 31st
        def predict_blooming_day(df, year, heat_requirement=123.5, base_temp=7):
            heat_units_sum = 0
            bloom_day = None

            # Set dormancy release day to January 31st of the following year
            dormancy_release_day = pd.to_datetime(f"{year + 1}-01-31")

            # Filter data starting from February 1st of the next year
            start_date = dormancy_release_day + pd.DateOffset(days=1)
            end_date = pd.to_datetime(f"{year + 1}-12-31")
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            # Accumulate heat units after dormancy release
            for index, row in df_filtered.iterrows():
                tmin = row['tmin']
                tmax = row['tmax']
                date = row['date']

                heat_units_sum += calculate_heat_units(tmin, tmax, base_temp)
                if heat_units_sum >= heat_requirement:
                    bloom_day = date
                    break

            return dormancy_release_day, bloom_day

        # Collect predicted bloom data
        predicted_bloom_data = []
        for year in range(2000, 2024):
            dormancy_release_day, bloom_day = predict_blooming_day(df, year)
            if bloom_day is not None:
                predicted_bloom_data.append({
                    'year': year + 1,
                    'predicted_bloom_date': bloom_day,
                    'DOY_predicted': bloom_day.timetuple().tm_yday
                })

        # 결과를 DataFrame으로 변환
        predicted_bloom_df = pd.DataFrame(predicted_bloom_data)

        # Merge actual and predicted data
        merged_df = pd.merge(actual_bloom_data[['year', 'DOY_actual']], predicted_bloom_df[['year', 'DOY_predicted']], on='year')

        # Linear regression to calculate R²
        actual_bloom_array = merged_df['DOY_actual'].values
        predicted_bloom_array = merged_df['DOY_predicted'].values

        model = LinearRegression()
        actual_bloom_array_reshaped = actual_bloom_array.reshape(-1, 1)
        model.fit(actual_bloom_array_reshaped, predicted_bloom_array)
        predicted_values = model.predict(actual_bloom_array_reshaped)

        # R² value
        r2_value = r2_score(predicted_bloom_array, predicted_values)

        # MAE and RMSE values
        mae_value = mean_absolute_error(predicted_bloom_array, predicted_values)
        rmse_value = np.sqrt(mean_squared_error(predicted_bloom_array, predicted_values))

        # Display results for first model
        st.write(f"**R² Value:** {r2_value:.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae_value:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse_value:.2f}")

        # Plotting actual vs predicted bloom days (scatter and regression line)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(actual_bloom_array, predicted_bloom_array, color='blue', label='Predicted vs Actual', s=50)
        ax.plot(actual_bloom_array, predicted_values, color='red', label=f'Regression Line (R² = {r2_value:.2f})')
        ax.set_xlabel('Actual Blooming Day')
        ax.set_ylabel('Predicted Blooming Day')
        ax.legend(loc='upper left')
        ax.grid(True)

        # Streamlit에 그래프 출력
        st.pyplot(fig)

    # **Second Model: KMA-based Prediction**
    with col2:
        st.subheader("KMA 모델을 이용한 벚꽃 개화 예측")

        # Function to calculate the average temperature for February and March
        def calculate_feb_march_avg_temp(df, year):
            # 2월 데이터 필터링
            feb_data = df[(df['year'] == year) & (df['month'] == 2)]
            feb_avg_temp = feb_data['tavg'].mean()

            # 3월 데이터 필터링
            march_data = df[(df['year'] == year) & (df['month'] == 3)]
            march_avg_temp = march_data['tavg'].mean()

            return feb_avg_temp, march_avg_temp

        # 기상청 예측 모형: 1월 31일 이후 2월과 3월 기온으로 개화일 예측
        def predict_blooming_day_kma(feb_avg_temp, march_avg_temp, base_date='01-31'):
            # 3월 기온에 가중치 1.7배 부여
            weighted_march_avg_temp = 1.7 * march_avg_temp
            # 가중치 적용한 온도 평균값 사용
            predicted_bloom_day = 110.5 - feb_avg_temp - weighted_march_avg_temp
            return predicted_bloom_day

        # Collect predicted bloom data using KMA model
        predicted_bloom_data_kma = []

        for year in range(2000, 2024):
            # 2월과 3월 평균 기온을 계산
            feb_avg_temp, march_avg_temp = calculate_feb_march_avg_temp(df, year)

            if pd.notnull(feb_avg_temp) and pd.notnull(march_avg_temp):
                # 1월 31일 이후 기온으로 개화일 예측
                predicted_bloom_day = predict_blooming_day_kma(feb_avg_temp, march_avg_temp)

                # 결과 저장
                predicted_bloom_data_kma.append({
                    'year': year,
                    'predicted_bloom_day': predicted_bloom_day
                })

        # DataFrame 변환
        predicted_bloom_df_kma = pd.DataFrame(predicted_bloom_data_kma)

        # 실제 개화일과 예측 개화일 병합
        merged_df_kma = pd.merge(actual_bloom_data[['year', 'DOY_actual']], predicted_bloom_df_kma[['year', 'predicted_bloom_day']], on='year')

        # R², MAE, RMSE 계산
        actual_bloom_array_kma = merged_df_kma['DOY_actual'].values
        predicted_bloom_array_kma = merged_df_kma['predicted_bloom_day'].values

        r2_value_kma = r2_score(actual_bloom_array_kma, predicted_bloom_array_kma)
        mae_value_kma = mean_absolute_error(actual_bloom_array_kma, predicted_bloom_array_kma)
        rmse_value_kma = np.sqrt(mean_squared_error(actual_bloom_array_kma, predicted_bloom_array_kma))

        # Display results for second model
        st.write(f"**KMA 모델 R² Value:** {r2_value_kma:.2f}")
        st.write(f"**KMA 모델 Mean Absolute Error (MAE):** {mae_value_kma:.2f}")
        st.write(f"**KMA 모델 Root Mean Squared Error (RMSE):** {rmse_value_kma:.2f}")

        # Plotting actual vs predicted bloom days for KMA model (scatter and regression line)
        fig_kma, ax_kma = plt.subplots(figsize=(8, 6))
        ax_kma.scatter(actual_bloom_array_kma, predicted_bloom_array_kma, color='blue', label='Predicted vs Actual', s=50)
        ax_kma.plot(actual_bloom_array_kma, actual_bloom_array_kma, color='red', label=f'Ideal Line')  # y = x 선 (ideal line)
        ax_kma.set_xlabel('Actual Blooming Day')
        ax_kma.set_ylabel('Predicted Blooming Day')
        ax_kma.legend(loc='upper left')
        ax_kma.grid(True)

        # Streamlit에 그래프 출력
        st.pyplot(fig_kma)

    st.subheader("Old vs New Dataset: Tavg Temperatures for February and March")

    # Load old and new datasets
    file_path_old = r'C:\code\pythonProject3\hw4\old(1955~2004).csv'
    file_path_new = r'C:\code\pythonProject3\hw4\wb_seoul.csv'

    df_old = pd.read_csv(file_path_old)
    df_new = pd.read_csv(file_path_new)

    # Clean column names and filter only February and March data
    df_old.columns = df_old.columns.str.strip()
    df_new.columns = df_new.columns.str.strip()
    df_old_feb_march = df_old[(df_old['month'] == 2) | (df_old['month'] == 3)]
    df_new_feb_march = df_new[(df_new['month'] == 2) | (df_new['month'] == 3)]

    # Combine day and month into a single continuous day number for plotting
    df_old_feb_march['day_of_year'] = (df_old_feb_march['month'] - 1) * 30 + df_old_feb_march['day']
    df_new_feb_march['day_of_year'] = (df_new_feb_march['month'] - 1) * 30 + df_new_feb_march['day']

    # Plot side-by-side for comparison
    col1, col2, col3 = st.columns(3)

    # Plot for the old dataset in the first column
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(df_old_feb_march['day_of_year'], df_old_feb_march['tavg'], color='blue', label='Old Dataset',
                    marker='o')
        ax1.set_xlabel('Day (Feb 1 - Mar 31)')
        ax1.set_ylabel('Tavg (°C)')
        ax1.set_title('Old Dataset: Tavg Temperatures for February and March')
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

    # Plot for the new dataset in the second column
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(df_new_feb_march['day_of_year'], df_new_feb_march['tavg'], color='green', label='New Dataset',
                    marker='x')
        ax2.set_xlabel('Day (Feb 1 - Mar 31)')
        ax2.set_ylabel('Tavg (°C)')
        ax2.set_title('New Dataset: Tavg Temperatures for February and March')
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    # Calculate and display average temperatures in the third column
    with col3:
        feb_avg_old = df_old[df_old['month'] == 2]['tavg'].mean()
        mar_avg_old = df_old[df_old['month'] == 3]['tavg'].mean()
        feb_avg_new = df_new[df_new['month'] == 2]['tavg'].mean()
        mar_avg_new = df_new[df_new['month'] == 3]['tavg'].mean()
        st.subheader("Average Temperatures for February and March")
        # Display the average temperatures with larger font size using markdown
        st.markdown(f"<h4><b>Old Dataset - February Average Tavg:</b> {feb_avg_old:.2f} °C</h4>",
                    unsafe_allow_html=True)
        st.markdown(f"<h4><b>Old Dataset - March Average Tavg:</b> {mar_avg_old:.2f} °C</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4><b>New Dataset - February Average Tavg:</b> {feb_avg_new:.2f} °C</h4>",
                    unsafe_allow_html=True)
        st.markdown(f"<h4><b>New Dataset - March Average Tavg:</b> {mar_avg_new:.2f} °C</h4>", unsafe_allow_html=True)




