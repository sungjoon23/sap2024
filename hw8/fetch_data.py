import requests
import pandas as pd
from datetime import datetime
from io import StringIO
import os

# Function to fetch AWS weather data from the URL
def fetch_aws_data(site, dev, year, month, day):
    url = f"http://203.239.47.148:8080/dspnet.aspx?Site={site}&Dev={dev}&Year={year}&Mon={month:02d}&Day={day:02d}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the response was unsuccessful (e.g., 404, 500)

        data = response.text.strip()
        if data == "NoFile" or not data:
            print("Error: No data available on the server for the specified date.")
            return None

        print("Fetched data preview:", data[:200])  # Print first 200 characters for preview
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None  # Return None if there's an issue with the HTTP request

# Function to convert data to DataFrame
def convert_to_dataframe(data):
    try:
        df = pd.read_csv(StringIO(data), sep=",", header=None)

        # Define expected columns
        expected_columns = {0: 'Timestamp', 1: 'TEMP', 2: 'HUMI', 6: 'IRRAD', 7: 'WIND_DIR',
                            13: 'WIND', 14: 'RAIN', 15: 'WIND_MAX', 16: 'VOLTAGE'}

        if len(df.columns) < max(expected_columns.keys()) + 1:
            print("Error: Fetched data does not match expected format.")
            return None

        # Rename and select columns
        df.rename(columns=expected_columns, inplace=True)
        df = df[['Timestamp', 'TEMP', 'HUMI', 'IRRAD', 'WIND_DIR', 'WIND', 'RAIN', 'WIND_MAX', 'VOLTAGE']]
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')  # Coerce errors into NaT (Not a Time)
        df.set_index('Timestamp', inplace=True)
        df.dropna(how="all", inplace=True)  # Drop rows with all NaN values

        if df.empty:
            print("Warning: DataFrame is empty after conversion.")
            return None

        return df
    except pd.errors.EmptyDataError:
        print("Error: No columns to parse from the fetched data.")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None  # Return None if any unexpected error occurs during the conversion

# Function to resample data hourly and calculate mean
def resample_data_hourly(df):
    try:
        if df is not None and not df.empty:
            df_hourly = df.resample('H').mean()  # Resample data by hour and calculate the mean
            df_hourly = df_hourly.round(2)  # Round to 2 decimal places
            return df_hourly
        else:
            print("Error: Cannot resample an empty DataFrame.")
            return None
    except Exception as e:
        print(f"Error during resampling: {e}")
        return None  # Return None if there's an issue during resampling

# Function to save data to CSV
def save_data(df, city_name):
    try:
        if df is None or df.empty:
            print("Error: No data to save.")
            return

        now = datetime.now()
        month_folder = now.strftime("%Y-%m")
        file_name = f"{now.strftime('%Y.%m.')}{city_name}.csv"
        folder_path = os.path.join("hw8", month_folder)
        os.makedirs(folder_path, exist_ok=True)  # Create folder if it does not exist
        file_path = os.path.join(folder_path, file_name)

        # Check if file already exists and merge data if it does
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            merged_df = pd.concat([existing_df, df]).drop_duplicates(keep='last').sort_index()
        else:
            merged_df = df

        # Save the merged DataFrame to CSV
        merged_df.to_csv(file_path)
        print(f"Data has been saved to {file_path}.")
    except Exception as e:
        print(f"Error during saving data: {e}")

# Main function to orchestrate data fetching, processing, and saving
def main():
    site = 85
    dev = 1
    city_name = "Jeonju"

    now = datetime.now()
    try:
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
    except Exception as e:
        print(f"Unexpected error in main function: {e}")

if __name__ == "__main__":
    main()
