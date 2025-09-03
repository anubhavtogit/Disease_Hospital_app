import streamlit as st
from utils.predictor import predict
from utils.hospital_recommender import recommend_hospitals
import subprocess

st.title("🩺 Disease Prediction App")

# ========== Input Fields ==========
temperature = st.number_input("Temperature", value=98.6)
cough = st.selectbox("Cough", ["no", "mild", "severe"])
runny_nose = st.selectbox("Runny Nose", ["no", "mild", "severe"])
sore_throat = st.selectbox("Sore Throat", ["no", "mild", "severe"])
headache = st.selectbox("Headache", ["no", "mild", "severe"])
chills = st.selectbox("Chills", ["no", "yes"])
shortness_of_breath = st.selectbox("Shortness of Breath", ["no", "mild", "severe"])
night_sweating = st.selectbox("Night Sweating", ["no", "yes"])
loss_of_taste = st.selectbox("Loss of Taste", ["no", "yes"])
loss_of_smell = st.selectbox("Loss of Smell", ["no", "yes"])

# Convert user inputs into model format
user_input = {
    "temperature": temperature,
    "cough_no": 1 if cough=="no" else 0, "cough_mild": 1 if cough=="mild" else 0, "cough_severe": 1 if cough=="severe" else 0,
    "runny_nose_no": 1 if runny_nose=="no" else 0, "runny_nose_mild": 1 if runny_nose=="mild" else 0, "runny_nose_severe": 1 if runny_nose=="severe" else 0,
    "sore_throat_no": 1 if sore_throat=="no" else 0, "sore_throat_mild": 1 if sore_throat=="mild" else 0, "sore_throat_severe": 1 if sore_throat=="severe" else 0,
    "headache_no": 1 if headache=="no" else 0, "headache_mild": 1 if headache=="mild" else 0, "headache_severe": 1 if headache=="severe" else 0,
    "chills": 1 if chills=="yes" else 0,
    "shortness_of_breath_no": 1 if shortness_of_breath=="no" else 0, "shortness_of_breath_mild": 1 if shortness_of_breath=="mild" else 0, "shortness_of_breath_severe": 1 if shortness_of_breath=="severe" else 0,
    "night_sweating": 1 if night_sweating=="yes" else 0,
    "loss_of_taste": 1 if loss_of_taste=="yes" else 0,
    "loss_of_smell": 1 if loss_of_smell=="yes" else 0,
}

# ========== Prediction ==========
if st.button("Predict Disease"):
    try:
        probs, weightages, predicted_label = predict(user_input)

        st.session_state["prediction_done"] = True
        st.session_state["weightages"] = weightages
        st.session_state["predicted_label"] = predicted_label

        st.subheader("Probabilities")
        for disease, prob, weight in weightages:
            st.write(f"**{disease}**: {prob:.2f}% (Weight {weight})")

        st.success(f"🩺 Predicted Disease: {predicted_label}")

    except FileNotFoundError:
        st.error("Model not found. Please retrain first!")

# ========== Hospital Recommendation ==========
if "prediction_done" in st.session_state and st.session_state["prediction_done"]:
    st.markdown("---")
    st.subheader("🏥 Recommended Hospitals")
    input_city = st.selectbox("Enter your city", ["MetroCity", "RiverTown", "HillVille", "GreenBay", "SunPort"])

    if st.button("Find Best Hospitals"):
        recommended = recommend_hospitals(st.session_state["weightages"], input_city)
        st.dataframe(recommended)

# ========== Retrain Button ==========
st.markdown("---")
st.subheader("Model Maintenance")

if st.button("Retrain Model"):
    with st.spinner("Training new model..."):
        result = subprocess.run(["python", "retrain.py"], capture_output=True, text=True)

    if result.returncode == 0:
        st.success("Model retrained successfully!")
        st.text(result.stdout)
    else:
        st.error("Retraining failed!")
        st.text(result.stderr)
