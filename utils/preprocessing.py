import pandas as pd

def preprocess_user_input(user_input: dict, feature_columns):
    """
    Convert user input dict into a DataFrame aligned with model feature columns.
    """
    new_input = pd.DataFrame([user_input])

    for col in feature_columns:
        if col not in new_input.columns:
            new_input[col] = 0

    new_input = new_input[feature_columns]
    return new_input
