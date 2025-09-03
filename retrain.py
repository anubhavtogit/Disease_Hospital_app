import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = "models"
DATA_PATH = "data/symptom_classification_dataset.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Replace categorical values with numbers
df.replace({"no": 0, "mild": 25, "severe": 50}, inplace=True)

X = df.drop(columns=['label'])
y = df['label']

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train_scaled, y_train)

# ✅ Save artifacts
with open(os.path.join(MODEL_DIR, "tree_model.pkl"), "wb") as f:
    pickle.dump(tree_model, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

with open(os.path.join(MODEL_DIR, "X_columns.pkl"), "wb") as f:
    pickle.dump(X_encoded.columns.tolist(), f)

print("Model retrained and saved successfully!")
