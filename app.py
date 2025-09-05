import streamlit as st
import joblib
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

@st.cache_resource
# load models func
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
    options=["Naive Bayes", "SVM", "BERT"],
)

st.write("Or upload a text file: ")
uploaded_file = st.file_uploader("", type=["txt"])

# upload txt file
if uploaded_file is not None:
    file_text = uploaded_file.read().decode("utf-8")
else:
    file_text = ""

review_text = st.text_area("✍️ Enter your review here:", value=file_text, height=230).strip()

if st.button("Predict Sentiment"):
    if not model_choice:
        st.warning("⚠️ Please select at least one model.")
    elif not review_text:
        st.warning("⚠️ Please enter or upload a review before clicking Predict.")
    else: 
        results = []
        labels = {"pos": "Positive", "neg": "Negative", 1: "Positive", 0: "Negative"}

        # NB
        if "Naive Bayes" in model_choice:
            X_input = nb_vectorizer.transform([review_text])
            nb_pred = nb_model.predict(X_input)[0]
            if hasattr(nb_model, "predict_proba"):
                nb_prob = nb_model.predict_proba(X_input)[0]
                nb_confidence = max(nb_prob) * 100
            else:
                nb_confidence = None
            results.append({
                "model": "Naive Bayes",
                "prediction": ("✅ " if labels.get(nb_pred) == "Positive" else "❌ ") + labels.get(nb_pred, "N/A"),
                "confidence": f"{nb_confidence:.2f}%" if nb_confidence else "N/A"
            })

        # SVM
        if "SVM" in model_choice:
            svm_pred = svm_model.predict([review_text])[0]
            if hasattr(svm_model, "decision_function"):
                decision_value = svm_model.decision_function([review_text])[0]
                svm_confidence = (1 / (1 + torch.exp(-torch.tensor(decision_value)))).item() * 100
            else:
                svm_confidence = None
            results.append({
                "model": "SVM",
                "prediction": ("✅ " if labels.get(svm_pred) == "Positive" else "❌ ") + labels.get(svm_pred, "N/A"),
                "confidence": f"{svm_confidence:.2f}%" if svm_confidence else "N/A"
            })

        # BERT
        if "BERT" in model_choice:
            inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
            outputs = bert_model(**inputs)
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=1).item()
            probs = torch.nn.functional.softmax(logits, dim=1)
            bert_confidence = probs[0][pred_class].item() * 100
            results.append({
                "model": "BERT",
                "prediction": ("✅ " if labels.get(pred_class) == "Positive" else "❌ ") + labels.get(pred_class, "N/A"),
                "confidence": f"{bert_confidence:.2f}%"
            })

        # tbl
        df = pd.DataFrame(results)
        df.index = df.index + 1
        st.table(df)

st.markdown("---")
st.caption("Built with Streamlit · Naive Bayes · SVM · BERT")
