import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.api.models import load_model

def generate_heatmap(data_path, model_path, day, month, hour, grid_size=50):
    """
    Generate a heatmap for traffic incident probabilities in Calgary based on user inputs.

    Parameters:
        data_path (str): Path to the dataset (CSV file).
        model_path (str): Path to the trained model.
        day (int): Day of the week (0 = Monday, 6 = Sunday).
        month (int): Month of the year (1 = January, 12 = December).
        hour (int): Hour of the day (0-23).
        grid_size (int): Number of bins for latitude and longitude.
    """
    # Load the encoded dataset
    df = pd.read_csv(data_path)

    # Calculate sine and cosine encoding for the input time values
    day_sin = np.sin(2 * np.pi * day / 7)
    day_cos = np.cos(2 * np.pi * day / 7)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Filter the dataset based on the input time values
    filtered_df = df.copy()
    filtered_df['day_sin'] = day_sin
    filtered_df['day_cos'] = day_cos
    filtered_df['hour_sin'] = hour_sin
    filtered_df['hour_cos'] = hour_cos
    filtered_df['month_sin'] = month_sin
    filtered_df['month_cos'] = month_cos

    # Load the trained model
    model = load_model(model_path)

    # Feature columns used for prediction
    feature_columns = [
        'latitude', 'longitude', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ] + [col for col in df.columns if col.startswith('geohash_')]
    features = filtered_df[feature_columns]

    # Predict probabilities
    filtered_df['predicted_probability'] = model.predict(features).ravel()

    # Generate heatmap data for folium
    heatmap_data = filtered_df[['latitude', 'longitude', 'predicted_probability']].values.tolist()

    # Create a folium map centered on Calgary
    calgary_map = folium.Map(location=[51.0447, -114.0719], zoom_start=11)

    # Add heatmap layer
    HeatMap(heatmap_data, radius=15, blur=10, max_zoom=1).add_to(calgary_map)

    # Save map to HTML file
    output_file = f"../output/calgary_heatmap_day{day}_month{month}_hour{hour}.html"
    calgary_map.save(output_file)
    print(f"Heatmap saved to {output_file}")

    # Return the map object for further manipulation if needed
    return calgary_map


if __name__ == "__main__":
    data_path = "../data/encoded_dataset.csv"
    model_path = "../model/traffic_model.keras"

    # User-provided inputs
    day = int(input("Enter the day of the week (0=Monday, 6=Sunday): "))
    month = int(input("Enter the month (1=January, 12=December): "))
    hour = int(input("Enter the hour (0-23): "))

    generate_heatmap(data_path, model_path, day, month, hour)