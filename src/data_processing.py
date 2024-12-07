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
