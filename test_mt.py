
import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Research Abstract Classifier",
    layout="wide"
)

st.title("Research Abstract Classifier")
st.write("Classify research abstracts into their respective domains.")


# --------------------------------------------------
# LOAD MODELS AND VECTORIZER
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    vectorizer = joblib.load("mt_tfidf_vectorizer.joblib")

    knn_model = joblib.load("mt_knn_model.joblib")

    lr_model = joblib.load("mt_logistic_regression_model.joblib")

    return vectorizer, knn_model, lr_model


vectorizer, knn_model, lr_model = load_artifacts()


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_abstract(text):

    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.subheader("1. Select Machine Learning Model")

model_name = st.selectbox(
    "Choose a classification model",
    ["KNN", "Logistic Regression"]
)

# Select the corresponding model
if model_name == "KNN":
    model = knn_model

else:
    model = lr_model


# --------------------------------------------------
# PREDICTION MODE SELECTION
# --------------------------------------------------

st.subheader("2. Select Prediction Mode")

prediction_mode = st.radio(
    "Choose prediction mode",
    ["Single Prediction", "Batch Prediction"],
    horizontal=True
)


# --------------------------------------------------
# SINGLE PREDICTION
# --------------------------------------------------

if prediction_mode == "Single Prediction":

    st.subheader("Single Abstract Prediction")

    abstract = st.text_area(
        "Paste the research abstract below:",
        height=250,
        placeholder="Paste your research abstract here..."
    )

    if st.button("Predict Domain"):

        if not abstract.strip():

            st.warning("Please enter an abstract before predicting.")

        else:

            with st.spinner(f"Classifying using {model_name}..."):

                cleaned = clean_abstract(abstract)

                if not cleaned:

                    st.warning("The abstract contains no usable text.")

                else:

                    X_new = vectorizer.transform([cleaned])

                    prediction = model.predict(X_new)[0]

                    st.success("Prediction Completed!")

                    st.subheader("Prediction Result")

                    st.info(f"**Model:** {model_name}")

                    st.success(f"**Predicted Research Domain:** {prediction}")


# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

else:

    st.subheader("Batch Abstract Prediction")

    st.write("Upload a CSV file containing multiple research abstracts.")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            if "Abstract" not in df.columns:

                st.error("The uploaded CSV must contain an 'Abstract' column.")

            else:

                st.write("### Uploaded Dataset")

                st.dataframe(df.head())

                st.write(f"Total records: {len(df)}")

                if st.button("Predict Labels"):

                    with st.spinner(f"Classifying using {model_name}..."):

                        # Clean abstracts
                        cleaned = (
                            df["Abstract"]
                            .fillna("")
                            .astype(str)
                            .apply(clean_abstract)
                        )

                        # Identify non-empty abstracts
                        mask = cleaned.str.strip().ne("")

                        # Preserve original dataset
                        results = df.copy()

                        results["predicted"] = "No abstract provided"

                        # Predict labels
                        if mask.any():

                            X_new = vectorizer.transform(cleaned.loc[mask])

                            predictions = model.predict(X_new)

                            results.loc[mask, "predicted"] = predictions

                        st.success("Predictions Completed!")

                        st.write("### Prediction Results")

                        st.dataframe(results.head(10))

                        # Download predictions
                        st.download_button(
                            label="Download Predictions",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name=f"{model_name.lower().replace(' ', '_')}_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as e:

            st.error(f"Error processing CSV file: {e}")
