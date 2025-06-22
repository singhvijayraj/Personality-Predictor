import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("personality_predictor.pkl")

st.title("🧠 Personality Predictor")
st.markdown("Predict whether you're an **Introvert** or **Extrovert**.")

time_spent = st.slider("Time spent alone (0-10)", 0, 10, 5)
events = st.slider("Social event attendance (0-10)", 0, 10, 5)
outside = st.slider("Going outside (0-10)", 0, 10, 5)
friends = st.slider("Friends circle size (0-10)", 0, 10, 5)
posts = st.slider("Post frequency (0-10)", 0, 10, 5)

stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
drained = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])

stage_fear = "Yes" if stage_fear == "Yes" else "No"
drained = "Yes" if drained == "Yes" else "No"

input_data = pd.DataFrame([{
    "Time_spent_Alone": time_spent,
    "Social_event_attendance": events,
    "Going_outside": outside,
    "Friends_circle_size": friends,
    "Post_frequency": posts,
    "Stage_fear": stage_fear,
    "Drained_after_socializing": drained
}])

if st.button("Predict Personality"):
    prob = model.predict_proba(input_data)[0]
    pred_class = model.predict(input_data)[0]

    labels = ["Introvert", "Extrovert"]
    colors = ["blue", "green"]
    percentages = [prob[0] * 100, prob[1] * 100]

    st.markdown(f"### 🧬 Prediction: **{labels[pred_class]}**")
    st.markdown(f"- Introvert: `{percentages[0]:.2f}%`")
    st.markdown(f"- Extrovert: `{percentages[1]:.2f}%`")

    fig, ax = plt.subplots()
    ax.bar(labels, percentages, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    st.pyplot(fig)
