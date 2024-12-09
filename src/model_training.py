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
        epochs=50,
        batch_size=32
    )

    print(f"Number of features during training: {X_train.shape[1]}")

    # Evaluate model on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")

    y_pred_proba = model.predict(X_test).ravel()

    # Default threshold F1 score
    y_pred = (y_pred_proba > 0.5).astype(int)
    default_f1 = f1_score(y_test, y_pred)
    print(f"F1-Score: {default_f1:.4f}")


    print("Adjusted Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model.save(model_path.replace('.h5', '.keras'))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model("../data/encoded_dataset.csv", "../model/traffic_model.h5")
