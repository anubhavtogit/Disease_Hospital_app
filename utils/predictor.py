import pickle
import numpy as np
import pandas as pd
import os

MODEL_DIR = "models"

# Load model, scaler, encoder, and X_encoded_columns automatically
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "tree_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "X_columns.pkl"), "rb") as f:
        X_encoded_columns = pickle.load(f)

    return model, scaler, le, X_encoded_columns


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict(input_dict):
    model, scaler, le, X_encoded_columns = load_artifacts()

    # Convert input to DataFrame
    new_input = pd.DataFrame([input_dict])

    # Ensure all columns exist
    for col in X_encoded_columns:
        if col not in new_input.columns:
            new_input[col] = 0
    new_input = new_input[X_encoded_columns]

    # Scale
    new_input_scaled = scaler.transform(new_input)

    # Predict probabilities
    probs = model.predict_proba(new_input_scaled)[0]
    softmax_probs = softmax(probs)

    # Get predicted label
    predicted_label = le.inverse_transform([np.argmax(probs)])[0]

    # Calculate weightages
    weightages = []
    for disease, prob in zip(le.classes_, softmax_probs):
        if prob < 0.05:
            weight = 0
        elif prob < 0.10:
            weight = 1
        elif prob < 0.15:
            weight = 2
        elif prob < 0.20:
            weight = 3
        else:
            weight = 4
        weightages.append((disease, prob * 100, weight))

    return probs, weightages, predicted_label
