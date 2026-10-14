
import streamlit as st
import torch
import pandas as pd
import json
import html
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO_ID = "Qalbiyah/indobert-quran-multilabel"

st.set_page_config(
    page_title="Klasifikasi Tematik Al-Qur'an",
    page_icon="📖",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #f6fafc;
    color: #183449;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #dff4f0, #e7f1ff);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
    border: 1px solid #b9dedb;
}
.hero h1 {
    color: #16465c;
    font-size: 32px;
    margin-bottom: 8px;
}
.hero p {
    color: #507184;
    font-size: 16px;
}
.panel-title {
    color: #16465c;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 12px;
}
.section-title {
    color: #176e70;
    font-size: 17px;
    font-weight: 700;
    margin: 15px 0 10px;
}
.theme-item {
    background: #ffffff;
    border: 1px solid #b5e3d8;
    border-radius: 10px;
    padding: 12px 15px;
    margin-bottom: 9px;
    color: #185a4e;
    font-weight: 600;
}
.theme-check {
    color: #0b9b76;
    margin-right: 10px;
}
.note {
    margin-top: 24px;
    padding: 14px;
    background: #edf5f8;
    border-radius: 10px;
    color: #486474;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Memuat model IndoBERT...")
def muat_model():
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
    model.eval()

    path_config = hf_hub_download(
        repo_id=REPO_ID,
        filename="konfigurasi_deployment.json"
    )

    with open(path_config, encoding="utf-8") as f:
        config = json.load(f)

    return tokenizer, model, config

st.markdown("""
<div class="hero">
    <h1>📖 Klasifikasi Tematik Terjemahan Ayat Al-Qur'an</h1>
    <p>Menggunakan IndoBERT Multi-Label</p>
</div>
""", unsafe_allow_html=True)

kiri, kanan = st.columns(2, gap="large")

with kiri:
    st.markdown(
        '<div class="panel-title">Halaman Utama</div>',
        unsafe_allow_html=True
    )

    teks = st.text_area(
        "Masukkan terjemahan ayat Al-Qur'an",
        placeholder=(
            "Contoh: Sesungguhnya orang-orang yang beriman "
            "dan mengerjakan amal saleh..."
        ),
        height=230
    )

    tombol = st.button(
        "Klasifikasi",
        type="primary",
        use_container_width=True
    )

with kanan:
    st.markdown(
        '<div class="panel-title">Hasil Klasifikasi</div>',
        unsafe_allow_html=True
    )

    if tombol:
        if not teks.strip():
            st.warning("Masukkan teks terjemahan terlebih dahulu.")
        else:
            try:
                tokenizer, model, config = muat_model()

                inputs = tokenizer(
                    teks.strip(),
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(config["max_length"])
                )

                with torch.inference_mode():
                    output = model(**inputs)
                    probabilitas = torch.sigmoid(
                        output.logits
                    )[0].cpu().numpy()

                kategori = config["kategori"]
                threshold = float(config["threshold"])

                hasil = pd.DataFrame({
                    "Label": [f"C{i}" for i in range(1, 17)],
                    "Kategori": [
                        kategori[str(i)] for i in range(1, 17)
                    ],
                    "Probabilitas": probabilitas
                })

                terdeteksi = hasil[
                    hasil["Probabilitas"] >= threshold
                ]

                st.markdown(
                    '<div class="section-title">Tema yang terdeteksi:</div>',
                    unsafe_allow_html=True
                )

                if terdeteksi.empty:
                    st.info(
                        "Tidak ada kategori yang melewati threshold 0,5."
                    )
                else:
                    for _, baris in terdeteksi.iterrows():
                        nama = html.escape(str(baris["Kategori"]))
                        st.markdown(
                            f'<div class="theme-item">'
                            f'<span class="theme-check">✔</span>'
                            f'{nama}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.markdown(
                    '<div class="section-title">Probabilitas tiap label:</div>',
                    unsafe_allow_html=True
                )

                hasil["Probabilitas"] = hasil[
                    "Probabilitas"
                ].round(4)

                st.dataframe(
                    hasil,
                    hide_index=True,
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Gagal menjalankan klasifikasi: {e}")
    else:
        st.info(
            "Masukkan terjemahan ayat pada panel kiri, "
            "kemudian klik Klasifikasi."
        )

st.markdown("""
<div class="note">
    <b>Catatan:</b> Kategori yang ditampilkan merupakan
    prediksi model berdasarkan terjemahan bahasa Indonesia,
    bukan penetapan tafsir definitif ayat Al-Qur'an.
</div>
""", unsafe_allow_html=True)
