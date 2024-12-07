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