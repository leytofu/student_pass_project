import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Pass Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run: python train_model.py")
    st.stop()


st.title("🎓 Student Pass Prediction System")
st.write("Predict a student's result based on study hours.")

student_name = st.text_input(
    "Student name",
    placeholder="Enter the student's name"
)

study_hours = st.number_input(
    "Study hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Result", type="primary"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        prediction = int(model.predict([[study_hours]])[0])

        if prediction == 1:
            st.success(f"Congratulations, {student_name}! Prediction: PASS")
        else:
            st.error(f"{student_name}, prediction: FAIL")

        st.write(f"Study hours entered: {study_hours:g} hours")

        st.info(
            "This is a prediction from a simple practice model, "
            "not a guarantee of the student's actual result. "
            "The model uses study hours only."
        )

        # Bonus chart: study hours versus predicted class
        hours_range = list(range(0, 11))
        predicted_classes = [
            int(model.predict([[hours]])[0])
            for hours in hours_range
        ]

        chart_data = pd.DataFrame({
            "Study Hours": hours_range,
            "Predicted Class (0 = Fail, 1 = Pass)": predicted_classes
        })

        st.subheader("Bonus: Study Hours vs. Predicted Result")

        fig, ax = plt.subplots()
        ax.plot(
            chart_data["Study Hours"],
            chart_data["Predicted Class (0 = Fail, 1 = Pass)"],
            marker="o"
        )
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Predicted Class (0 = Fail, 1 = Pass)")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Fail", "Pass"])
        ax.set_xticks(range(0, 11))
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)