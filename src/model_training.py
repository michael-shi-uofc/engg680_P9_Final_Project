import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.api.layers import Dense, Dropout
from keras.api.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

def train_model(data_path, model_path):
    """
    Train a DNN binary classifier for traffic incidents.
    """
    # Load encoded data
    df = pd.read_csv(data_path)

    if 'label' not in df.columns or df['label'].isna().any():
        raise ValueError("Dataset must contain a 'label' column with no missing values.")

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build DNN model
    model = Sequential([
        Dense(50, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32
    )