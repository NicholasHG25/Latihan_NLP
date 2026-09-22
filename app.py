import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import html


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="MedBot — Asisten Kesehatan",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =========================================================
# MODEL
# =========================================================

MODEL_NAME = "NicholasHG25/medbot-indobert-final"

ID2LABEL = {
    0: "demam_anak",
    1: "greeting",
    2: "kehamilan",
    3: "kulit_jerawat",
    4: "maag_asam_lambung",
    5: "mata",
    6: "menstruasi",
    7: "migrain_sakit_kepala",
    8: "nutrisi_diet",
    9: "sinusitis_flu",
    10: "tenggorokan",
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


tokenizer, model, device = load_model()


# =========================================================
# RESPONSE SESUAI NOTEBOOK
# =========================================================

RESPONSE_TEMPLATES = {
    "greeting": (
        "Halo! 👋 Aku MedBot. Ceritakan keluhan kesehatanmu, "
        "nanti aku bantu perkirakan kategorinya ya."
    ),
    "maag_asam_lambung": (
        "Sepertinya ini gejala **maag / asam lambung**. "
        "Coba atur pola makan teratur dan hindari makanan pedas/asam dulu. "
        "Kalau berlanjut, sebaiknya periksa ke dokter ya."
    ),
    "migrain_sakit_kepala": (
        "Sepertinya ini gejala **sakit kepala / migrain**. "
        "Coba istirahat cukup dan kurangi paparan layar dulu. "
        "Kalau nyerinya berat atau sering kambuh, konsultasikan ke dokter."
    ),
    "kehamilan": (
        "Ini kedengarannya terkait **kehamilan**. "
        "Untuk kondisi kehamilan sebaiknya rutin periksa ke bidan/dokter "
        "kandungan supaya lebih aman."
    ),
    "kulit_jerawat": (
        "Kelihatannya ini soal **kulit / jerawat**. "
        "Jaga kebersihan wajah dan hindari memencet jerawat. "
        "Kalau tidak membaik, coba konsultasi ke dokter kulit."
    ),
    "menstruasi": (
        "Ini sepertinya berkaitan dengan **siklus menstruasi**. "
        "Kalau siklusnya sangat tidak teratur atau nyerinya berat, "
        "sebaiknya periksa ke dokter kandungan."
    ),
    "mata": (
        "Sepertinya ini keluhan seputar **kesehatan mata**. "
        "Hindari mengucek mata dan istirahatkan mata dari layar. "
        "Kalau keluhan menetap, sebaiknya periksa ke dokter mata."
    ),
    "sinusitis_flu": (
        "Ini kedengarannya gejala **flu / sinusitis**. "
        "Perbanyak istirahat dan cairan. "
        "Kalau demam tinggi atau lebih dari seminggu, sebaiknya ke dokter."
    ),
    "demam_anak": (
        "Ini sepertinya soal **demam pada anak**. "
        "Pantau suhu tubuhnya dan pastikan cukup cairan. "
        "Kalau demam tinggi/lama, segera bawa ke dokter anak."
    ),
    "nutrisi_diet": (
        "Ini kelihatannya seputar **nutrisi / pola makan**. "
        "Coba jaga pola makan seimbang. Untuk kebutuhan spesifik, "
        "konsultasi ke ahli gizi bisa membantu."
    ),
    "tenggorokan": (
        "Sepertinya ini gejala **radang tenggorokan**. "
        "Perbanyak minum air hangat dan istirahatkan suara. "
        "Kalau makin parah, periksa ke dokter THT."
    ),
}

FALLBACK_MESSAGE = (
    "Maaf, aku belum begitu paham maksudnya 🙏 "
    "Bisa coba dijelaskan dengan kalimat lain?"
)

CONFIDENCE_THRESHOLD = 0.5


# =========================================================
# TOPIK CEPAT — SESUAI NOTEBOOK
# =========================================================

QUICK_TOPICS = [
    ("🤢", "Maag", "perut saya sakit dan perih"),
    ("🤕", "Migrain", "kepala saya pusing sebelah"),
    ("🤰", "Kehamilan", "usia kandungan 8 minggu apakah normal mual terus"),
    ("🧴", "Kulit", "muncul jerawat banyak di wajah"),
    ("🤧", "Flu", "hidung mampet dan pilek terus"),
    ("👁️", "Mata", "mata merah dan gatal"),
]


# =========================================================
# CSS — MENYERUPAI TAMPILAN NOTEBOOK
# =========================================================

st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: #ffffff;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 720px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }

    /* Header */
    .mb-header {
        background: linear-gradient(90deg, #3949ab, #3f51b5);
        padding: 18px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        border-radius: 18px 18px 0 0;
    }

    .mb-logo {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        flex-shrink: 0;
    }

    .mb-title {
        color: white;
        font-weight: 700;
        font-size: 17px;
        line-height: 1.3;
    }

    .mb-badge {
        background: #c8f7d4;
        color: #1a7f37;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
        margin-left: 6px;
        white-space: nowrap;
    }

    .mb-status {
        color: #dfe3fa;
        font-size: 12.5px;
        margin-top: 2px;
    }

    /* Warning */
    .mb-warning {
        background: #fff8e1;
        color: #8a6d3b;
        font-size: 12.5px;
        padding: 9px 20px;
        border-bottom: 1px solid #f0e2b6;
        border-radius: 0 0 14px 14px;
        margin-bottom: 10px;
    }

    /* Card */
    .mb-card {
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border: 1px solid #e2e8f0;
        background: white;
    }

    /* Topic chips */
    div.stButton > button {
        border-radius: 999px;
        border: 1px solid #e4e8f0;
        font-size: 12.5px;
        font-weight: 700;
        padding: 5px 10px;
        min-height: 34px;
        box-shadow: none;
        background: #f5f7fb;
        color: #27324a;
    }

    div.stButton > button:hover {
        border: 1px solid #c8d0e0;
        background: #eef2f9;
        transform: translateY(-1px);
    }

    /* Input */
    div[data-testid="stTextInput"] input {
        border-radius: 10px;
        border: 1px solid #d7dbe8;
        padding: 8px 12px;
        background: #ffffff;
        color: #222222;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #3f51b5;
        box-shadow: 0 0 0 1px #3f51b5;
    }

    /* Tombol Kirim */
    div[data-testid="stFormSubmitButton"] button {
        height: 42px !important;
        min-height: 42px !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 700 !important;
        background: #3f51b5 !important;
        color: white !important;
        margin-top: 0px !important;
    }
    
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #303f9f !important;
        color: white !important;
    }
    
    /* Input chat */
    div[data-testid="stTextInput"] input {
        height: 42px !important;
        min-height: 42px !important;
        border-radius: 10px !important;
        border: 1px solid #d7dbe8;
        padding: 8px 12px;
        background: #ffffff;
        color: #222222;
    }

    /* Chat area */
    .chat-area {
        background: #ffffff;
        padding: 8px 0 4px 0;
    }

    .user-row {
        display: flex;
        justify-content: flex-end;
        margin: 8px 20px;
    }

    .user-bubble {
        background: #3f51b5;
        color: white;
        padding: 11px 15px;
        border-radius: 16px 16px 3px 16px;
        max-width: 75%;
        font-size: 13.5px;
        line-height: 1.5;
        word-wrap: break-word;
    }

    .bot-row {
        display: flex;
        justify-content: flex-start;
        margin: 8px 20px;
        gap: 8px;
    }

    .bot-icon {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #3f51b5;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        flex-shrink: 0;
    }

    .bot-bubble {
        background: #f7f8fc;
        color: #222222;
        padding: 12px 16px;
        border-radius: 16px 16px 16px 3px;
        max-width: 78%;
        font-size: 13.5px;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        word-wrap: break-word;
    }

    /* Hide Streamlit branding/menu for a cleaner app */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FUNGSI PREDIKSI
# =========================================================

def predict_intent(text):
    """
    Sama dengan notebook:
    max_length=64, padding='max_length', lalu softmax + argmax.
    """
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=64,
        padding="max_length",
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(torch.argmax(probs))
    confidence = float(probs[pred_id])

    label = ID2LABEL.get(pred_id, str(pred_id))

    return label, confidence


def chatbot_reply(text):
    label, confidence = predict_intent(text)

    if confidence < CONFIDENCE_THRESHOLD:
        return FALLBACK_MESSAGE, label, confidence

    return RESPONSE_TEMPLATES.get(label, FALLBACK_MESSAGE), label, confidence


# =========================================================
# SESSION STATE
# =========================================================

if "chat_log" not in st.session_state:
    st.session_state.chat_log = [
        (
            "bot",
            "👋 Halo! Aku <b>MedBot</b>, asisten kesehatan berbasis "
            "<b>IndoBERT</b> yang sudah dilatih mengenali 10 kategori "
            "keluhan umum (maag, migrain, kehamilan, kulit, dan lainnya)."
            "<br><br>"
            "Klik salah satu topik di atas, atau langsung ketik keluhanmu "
            "di kolom bawah ya!",
        )
    ]


def send_message(text):
    text = text.strip()

    if not text:
        return

    reply, label, confidence = chatbot_reply(text)

    st.session_state.chat_log.append(("user", html.escape(text)))
    st.session_state.chat_log.append(("bot", reply))


# =========================================================
# HEADER + WARNING
# =========================================================

st.markdown(
    """
    <div class="mb-header">
        <div class="mb-logo">🏥</div>
        <div>
            <div class="mb-title">
                MedBot — Asisten Kesehatan
                <span class="mb-badge">IndoBERT Fine-tuned</span>
            </div>
            <div class="mb-status">
                ● Online · Intent Classification Engine
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mb-warning">
        ⚠️ MedBot hanya untuk edukasi. Bukan pengganti dokter.
        Darurat medis: hubungi 119.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TOPIK CEPAT
# =========================================================

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

topic_cols = st.columns(6)

for col, (emoji, label, example) in zip(topic_cols, QUICK_TOPICS):
    with col:
        if st.button(
            f"{emoji} {label}",
            key=f"topic_{label}",
            use_container_width=True,
        ):
            send_message(example)
            st.rerun()


# =========================================================
# CHAT
# =========================================================

st.markdown(
    "<div class='chat-area'>",
    unsafe_allow_html=True,
)

for role, message in st.session_state.chat_log:
    if role == "user":
        st.markdown(
            f"""
            <div class="user-row">
                <div class="user-bubble">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Markdown sederhana agar **teks tebal** dari template tetap bekerja.
        # Escape hanya karakter HTML berbahaya; tag <b> dari welcome message
        # sengaja dipertahankan.
        st.markdown(
            f"""
            <div class="bot-row">
                <div class="bot-icon">🤖</div>
                <div class="bot-bubble">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# INPUT CHAT
# =========================================================

with st.form("chat_form", clear_on_submit=True):

    input_col, send_col = st.columns(
        [8, 2],
        gap="small",
        vertical_alignment="center"
    )

    with input_col:
        user_text = st.text_input(
            "Input",
            placeholder="Ketik keluhan kesehatan Anda...",
            label_visibility="collapsed",
            key="user_text",
        )

    with send_col:
        send_clicked = st.form_submit_button(
            "Kirim",
            use_container_width=True,
        )

if send_clicked:
    if user_text.strip():
        send_message(user_text)
        st.rerun()
    else:
        st.warning("Silakan ketik keluhan terlebih dahulu.")
