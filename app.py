import streamlit as st
import joblib

@st.cache_resource
def load_models():
    nb_model = joblib.load("models/naive_bayes.pkl")
    nb_vectorizer = joblib.load("models/NBvectorizer.pkl")
    svm_model = joblib.load("models/svm.pkl") 
    return nb_model, nb_vectorizer, svm_model

nb_model, nb_vectorizer, svm_model = load_models()

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Type in a movie review and find out if it's **Positive** or **Negative** using different models.")

model_choice = st.radio(
    "Choose a model:",
    ("Naive Bayes", "SVM", "Both"),
    horizontal=True
)

user_input = st.text_area("✍️ Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip():
        results = {}

        if model_choice in ("Naive Bayes", "Both"):
            X_input = nb_vectorizer.transform([user_input])
            nb_pred = nb_model.predict(X_input)[0]
            results["Naive Bayes"] = nb_pred

        if model_choice in ("SVM", "Both"):
            svm_pred = svm_model.predict([user_input])[0]  # raw text is fine
            results["SVM"] = svm_pred

        for model_name, pred in results.items():
            if pred == "pos":
                st.success(f"✅ {model_name} Prediction: Positive Review")
            else:
                st.error(f"❌ {model_name} Prediction: Negative Review")
    else:
        st.warning("⚠️ Please enter a review before clicking Predict.")

st.markdown("---")
st.caption("Built with Streamlit · Naive Bayes · SVM")
