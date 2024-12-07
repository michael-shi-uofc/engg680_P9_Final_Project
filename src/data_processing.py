import pandas as pd
import requests
import Geohash

def fetch_data(api_url):
    """
    Fetch traffic incident data from Calgary Open Data API.
    """
    print("Fetching data...")
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()
    data = response.json()
    return pd.DataFrame(data)

def preprocess_data(df):
    """
    Preprocess data: geohash encoding and extracting temporal features.
    """
    print("Preprocessing data...")
    df['start_dt'] = pd.to_datetime(df['start_dt'])
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df.dropna(subset=['latitude', 'longitude'], inplace=True)

    # Encode geohash (precision = 5 for regional granularity)
    df["geohash"] = df.apply(lambda x: Geohash.encode(x["latitude"], x["longitude"], precision=5), axis=1)

    # Extract temporal features
    df['day'] = df['start_dt'].dt.weekday  # Monday=0, Sunday=6
    df['hour'] = df['start_dt'].dt.hour
    df['month'] = df['start_dt'].dt.month

    # Keep only relevant columns
    df = df[['latitude', 'longitude', 'geohash', 'day', 'hour', 'month']]
    return df

if __name__ == "__main__":
    API_URL = "https://data.calgary.ca/resource/35ra-9556.json?$limit=50000"
    data = fetch_data(API_URL)
    if not data.empty:
        processed_data = preprocess_data(data)
        processed_data.to_csv("../data/processed_data.csv", index=False)
        print("Processed data saved.")
