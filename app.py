
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification Explorer")

st.write(
    """
    This application compares five machine learning classification models
    trained on the Dry Bean Dataset.

    Upload a test CSV file, select a trained model, and view its
    classification performance.
    """
)

# ---------------------------------------------------------
# Available trained models
# ---------------------------------------------------------
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

# ---------------------------------------------------------
# Model selection
# ---------------------------------------------------------
selected_model_name = st.selectbox(
    "Select a classification model",
    list(model_files.keys())
)

# ---------------------------------------------------------
# CSV upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read uploaded dataset
    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(test_data.head())

    # Check for target column
    if "Class" not in test_data.columns:
        st.error(
            "The uploaded CSV must contain a 'Class' column."
        )

    else:

        # Separate features and actual class
        X_uploaded = test_data.drop(columns=["Class"])
        y_actual = test_data["Class"]

        try:
            # -------------------------------------------------
            # Load SAVED model
            # -------------------------------------------------
            model = joblib.load(
                model_files[selected_model_name]
            )

            # -------------------------------------------------
            # Predictions
            # -------------------------------------------------
            y_pred = model.predict(X_uploaded)
            y_prob = model.predict_proba(X_uploaded)

            # -------------------------------------------------
            # Evaluation metrics
            # -------------------------------------------------
            accuracy = accuracy_score(
                y_actual,
                y_pred
            )

            precision = precision_score(
                y_actual,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_actual,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_actual,
                y_pred,
                average="weighted",
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_actual,
                y_pred
            )

            auc = roc_auc_score(
                y_actual,
                y_prob,
                multi_class="ovr",
                average="weighted",
                labels=model.classes_
            )

            # -------------------------------------------------
            # Display metrics
            # -------------------------------------------------
            st.subheader(
                f"Evaluation Results - {selected_model_name}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            col2.metric(
                "AUC",
                f"{auc:.4f}"
            )

            col3.metric(
                "Precision",
                f"{precision:.4f}"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Recall",
                f"{recall:.4f}"
            )

            col5.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            col6.metric(
                "MCC",
                f"{mcc:.4f}"
            )

            # -------------------------------------------------
            # Confusion matrix
            # -------------------------------------------------
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_actual,
                y_pred,
                labels=model.classes_
            )

            fig, ax = plt.subplots(
                figsize=(9, 7)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                ax=ax
            )

            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            ax.set_title(
                f"Confusion Matrix - {selected_model_name}"
            )

            plt.xticks(rotation=45)
            plt.yticks(rotation=0)

            plt.tight_layout()

            st.pyplot(fig)

            # -------------------------------------------------
            # Classification report
            # -------------------------------------------------
            st.subheader("Classification Report")

            report = classification_report(
                y_actual,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df.round(4)
            )

        except Exception as error:

            st.error(
                f"Unable to evaluate the model: {error}"
            )

else:

    st.info(
        "Upload test_data.csv to begin model evaluation."
    )
