import streamlit as st
import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

@st.cache_resource
def load_models():
    nb_model = joblib.load("models/naive_bayes.pkl")
    nb_vectorizer = joblib.load("models/NBvectorizer.pkl")
    svm_model = joblib.load("models/svm.pkl") 
    bert_model_path = "models/lmdb_bert_model"
    bert_model =  AutoModelForSequenceClassification.from_pretrained(bert_model_path)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    return nb_model, nb_vectorizer, svm_model, bert_model, tokenizer

nb_model, nb_vectorizer, svm_model, bert_model, tokenizer = load_models()

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Type in a movie review and find out if it's **Positive** or **Negative** using different models.")

model_choice = st.multiselect(
    "Choose model(s):",
    options=["Naive Bayes", "SVM", "BERT", "Both"],
)

user_input = st.text_area("✍️ Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip():
        results = {}

        if model_choice in ("Naive Bayes"):
            X_input = nb_vectorizer.transform([user_input])
            nb_pred = nb_model.predict(X_input)[0]
            results["Naive Bayes"] = nb_pred

        if model_choice in ("SVM"):
            svm_pred = svm_model.predict([user_input])[0]  # raw text is fine
            results["SVM"] = svm_pred
        
        if model_choice in ("BERT"):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            outputs = bert_model(**inputs)
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=1).item()
            # Assuming 1 = positive, 0 = negative
            bert_pred = "pos" if pred_class == 1 else "neg"
            results["BERT"] = bert_pred

        for model_name, pred in results.items():
            if pred == "pos":
                st.success(f"✅ {model_name} Prediction: Positive Review")
            else:
                st.error(f"❌ {model_name} Prediction: Negative Review")
    else:
        st.warning("⚠️ Please enter a review before clicking Predict.")

st.markdown("---")
st.caption("Built with Streamlit · Naive Bayes · SVM")
