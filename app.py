import streamlit as st
import joblib
# Load trained models
logistic_model = joblib.load('logistic_regression.pkl')
decision_tree_model = joblib.load('decision_tree.pkl')
knn_model = joblib.load('knn.pkl')
naive_bayes_model = joblib.load('naive_bayes.pkl')
random_forest_model = joblib.load('random_forest.pkl')

# Load scaler and feature names
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Classification")
st.write("Select a machine learning model and enter the required feature values to predict the diagnosis.")

model_name = st.selectbox(
    "Select a Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

st.subheader("Enter Feature Values")

input_values = []

columns = st.columns(3)

for i, feature in enumerate(feature_names):
    with columns[i % 3]:
        value = st.number_input(
            feature,
            value=0.0
        )
        input_values.append(value)

st.subheader("Prediction")

if st.button("Predict Diagnosis"):

    # Convert input values into 2D format
    input_data = [input_values]

    # Scale input for models that require scaling
    input_scaled = scaler.transform(input_data)

    # Select the model
    if model_name == "Logistic Regression":
        prediction = logistic_model.predict(input_scaled)[0]
        probability = logistic_model.predict_proba(input_scaled)[0][1]

    elif model_name == "Decision Tree":
        prediction = decision_tree_model.predict(input_data)[0]
        probability = decision_tree_model.predict_proba(input_data)[0][1]

    elif model_name == "kNN":
        prediction = knn_model.predict(input_scaled)[0]
        probability = knn_model.predict_proba(input_scaled)[0][1]

    elif model_name == "Naive Bayes":
        prediction = naive_bayes_model.predict(input_scaled)[0]
        probability = naive_bayes_model.predict_proba(input_scaled)[0][1]

    elif model_name == "Random Forest":
        prediction = random_forest_model.predict(input_data)[0]
        probability = random_forest_model.predict_proba(input_data)[0][1]

    # Display result
    if prediction == 0:
        st.success("Diagnosis: Benign")
    else:
        st.error("Diagnosis: Malignant")

    st.write(f"Probability of Malignant: {probability:.2%}")