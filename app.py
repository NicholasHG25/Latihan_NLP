import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="MedBot IndoBERT",
    page_icon="🩺",
    layout="centered"
)


# =========================
# KONFIGURASI MODEL
# =========================

MODEL_NAME = "NicholasHG25/medbot-indobert-final"


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# =========================
# TAMPILAN
# =========================

st.title("🩺 MedBot IndoBERT")

st.write(
    "Masukkan pertanyaan atau keluhan untuk mendapatkan "
    "kategori prediksi dari model."
)

text = st.text_area(
    "Masukkan teks:",
    placeholder="Contoh: Saya sering sakit kepala dan terasa berdenyut..."
)


# =========================
# PREDIKSI
# =========================

if st.button("🔍 Prediksi", type="primary"):

    if not text.strip():

        st.warning("Silakan masukkan teks terlebih dahulu.")

    else:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

            prediction = torch.argmax(
                probabilities,
                dim=-1
            ).item()

            confidence = probabilities[0][prediction].item()


        # Ambil nama label dari config model
        label = model.config.id2label[prediction]
        
        
        st.success("Prediksi berhasil!")
        
        st.subheader("Hasil Prediksi")
        
        st.write(f"**Kategori:** `{label}`")
        
        st.write(
            f"**Confidence:** `{confidence * 100:.2f}%`"
        )
