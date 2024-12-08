import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def encode_features(df):
    """
    Encode features for the classification model.
    """
    print("Encoding features...")
    
    required_columns = ['geohash', 'day', 'hour', 'month']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset is missing required columns: {set(required_columns) - set(df.columns)}")
    
    # Cyclical encoding for temporal features
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # One-hot encoding for geohashes
    encoder = OneHotEncoder()
    geohash_encoded = encoder.fit_transform(df[['geohash']]).toarray()
    geohash_df = pd.DataFrame(geohash_encoded, columns=encoder.get_feature_names_out(['geohash']))
    df = pd.concat([df.reset_index(drop=True), geohash_df.reset_index(drop=True)], axis=1)

    # Drop unused columns
    df.drop(['geohash', 'day', 'hour', 'month'], axis=1, inplace=True)
    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/dataset_with_negatives.csv")
    encoded_df = encode_features(df)
    encoded_df.to_csv("../data/encoded_dataset.csv", index=False)
    print("Encoded dataset saved.")
 