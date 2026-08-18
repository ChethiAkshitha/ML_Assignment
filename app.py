import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

# Load trained models and preprocessing objects
logistic_model = joblib.load("logistic_regression.pkl")
decision_tree_model = joblib.load("decision_tree.pkl")
knn_model = joblib.load("knn.pkl")
naive_bayes_model = joblib.load("naive_bayes.pkl")
random_forest_model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "kNN": knn_model,
    "Naive Bayes": naive_bayes_model,
    "Random Forest": random_forest_model
}
scaled_models = {"Logistic Regression", "kNN", "Naive Bayes"}

st.set_page_config(page_title="Breast Cancer Classification", page_icon="🩺", layout="wide")
st.title("🩺 Breast Cancer Classification")
st.write("Upload the test dataset to evaluate the trained classification models.")

st.subheader("1. Upload Test Data (CSV)")
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

st.subheader("2. Select a Machine Learning Model")
model_name = st.selectbox("Model", list(models.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    missing_features = [f for f in feature_names if f not in test_df.columns]
    if missing_features:
        st.error("Missing required feature columns: " + ", ".join(missing_features))
        st.stop()

    if "diagnosis" not in test_df.columns:
        st.error("The uploaded test data must contain the 'diagnosis' column.")
        st.stop()

    X_test = test_df[feature_names].copy()

    # Normalize diagnosis values so B/M and 0/1 formats are accepted.
    y_raw = test_df["diagnosis"].astype(str).str.strip().str.upper()
    diagnosis_map = {
        "B": 0, "M": 1,
        "0": 0, "1": 1,
        "0.0": 0, "1.0": 1
    }
    y_test = y_raw.map(diagnosis_map)

    if y_test.isna().any():
        invalid_values = sorted(y_raw[y_test.isna()].unique().tolist())
        st.error(
            "The diagnosis column must contain B/M or 0/1 values. "
            f"Invalid value(s): {invalid_values}"
        )
        st.stop()
    y_test = y_test.astype(int)

    st.subheader("3. Model Performance on Test Data")
    results = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        X_used = scaler.transform(X_test) if name in scaled_models else X_test
        y_pred = model.predict(X_used)
        y_prob = model.predict_proba(X_used)[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob

        results.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.format({
        "Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1 Score": "{:.4f}", "MCC": "{:.4f}"
    }), use_container_width=True)

    # Display predictions for the first 30 test patients.
    st.subheader("3A. Predictions on First 30 Test Patients")
    prediction_table = test_df[["id", "diagnosis"]].head(30).copy()
    prediction_table["Actual Diagnosis"] = prediction_table["diagnosis"].map({
        "B": "Benign", "M": "Malignant"
    })
    for name in models:
        prediction_table[f"{name} Prediction"] = [
            "Malignant" if p == 1 else "Benign"
            for p in predictions[name][:30]
        ]
    prediction_table = prediction_table.drop(columns=["diagnosis"])
    st.dataframe(prediction_table, use_container_width=True)

    st.subheader(f"4. Detailed Evaluation - {model_name}")
    selected_pred = predictions[model_name]
    selected_prob = probabilities[model_name]

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_test, selected_pred):.2%}")
    c2.metric("AUC", f"{roc_auc_score(y_test, selected_prob):.2%}")
    c3.metric("Precision", f"{precision_score(y_test, selected_pred, zero_division=0):.2%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", f"{recall_score(y_test, selected_pred, zero_division=0):.2%}")
    c5.metric("F1 Score", f"{f1_score(y_test, selected_pred, zero_division=0):.2%}")
    c6.metric("MCC", f"{matthews_corrcoef(y_test, selected_pred):.4f}")

    st.subheader("5. Confusion Matrix")
    cm = confusion_matrix(y_test, selected_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("6. Classification Report")
    report = classification_report(
        y_test, selected_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"),
                 use_container_width=True)
else:
    st.info("Please upload test_data.csv to view the model predictions, evaluation metrics, confusion matrix, and classification report.")
