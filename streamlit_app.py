import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import json
import io
from ocr_similarity_utils import extract_text_from_file, calculate_bert_similarity, calculate_tfidf_similarity

# Set UKM Theme Colors
UKM_RED = "#E60000"
UKM_BLUE = "#0066B3"
UKM_YELLOW = "#FFD700"
UKM_WHITE = "#FFFFFF"

st.set_page_config(
    page_title="UKM Transfer Credit Checker",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Title and Logo ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://raw.githubusercontent.com/khaliesahazmin/DataExtraction/main/ukm_logo.png/logo_UKM.png", width=80)
with col2:
    st.markdown(f"<h1 style='color:{UKM_RED};'>Transfer Credit Checker System</h1>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color:{UKM_BLUE};'>Universiti Kebangsaan Malaysia</h5>", unsafe_allow_html=True)

st.markdown("---")

# --- File Upload Section ---
st.markdown(f"<h3 style='color:{UKM_RED};'>📄 Upload Syllabus Document</h3>", unsafe_allow_html=True)
uploaded_file1 = st.file_uploader("Upload First Syllabus from UKM (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'])
uploaded_file2 = st.file_uploader("Upload Second Syllabus from another institute/diploma program (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'])

text1 = extract_text(uploaded_file1) if uploaded_file1 else None
text2 = extract_text(uploaded_file2) if uploaded_file2 else None

if text1 and text2:
    st.markdown("### 📊 Document Similarity Analysis")
    bert_score = calculate_bert_similarity(text1, text2)
    tfidf_score = calculate_tfidf_similarity(text1, text2)
    st.write(f"**BERT-based Similarity:** {bert_score}%")
    st.write(f"**TF-IDF Cosine Similarity:** {tfidf_score}%")
else:
    st.warning("Please upload both documents to compare.")

# --- Preprocess Function ---
def preprocess_image(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    denoised = cv2.medianBlur(binary, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sharpened = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)
    return Image.fromarray(sharpened)

# --- Extract Text from File ---
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        images = convert_from_bytes(file.read(), 400)
    else:
        image = Image.open(file)
        images = [image]

    for page in images:
        processed = preprocess_image(page)
        text += pytesseract.image_to_string(processed)
    return text

# --- Detect Institution Type Based on Keywords ---
UKM_KEYWORDS = ["Universiti Kebangsaan Malaysia", "UKM", "Fakulti", "Program", "Kod Kursus"]
OTHER_KEYWORDS = ["Kolej", "Politeknik", "Universiti Teknologi", "Diploma", "Institute", "Akademi", "MARA"]

def classify_document(text):
    ukm_hits = sum([1 for word in UKM_KEYWORDS if word.lower() in text.lower()])
    other_hits = sum([1 for word in OTHER_KEYWORDS if word.lower() in text.lower()])
    if ukm_hits > other_hits:
        return "✅ Detected as UKM Syllabus"
    elif other_hits > ukm_hits:
        return "🏫 Detected as Other Institute/Diploma Syllabus"
    else:
        return "⚠️ Institution Type Could Not Be Determined"

# --- Process Uploaded Files ---
if uploaded_file1 and uploaded_file2:
    with st.spinner("🔍 Extracting text from documents..."):
        text1 = extract_text(uploaded_file1)
        text2 = extract_text(uploaded_file2)

        classification1 = classify_document(text1)
        classification2 = classify_document(text2)

        # Save as JSON
        with open("output1.json", "w") as f:
            json.dump({"extracted_text": text1}, f)

        with open("output2.json", "w") as f:
            json.dump({"extracted_text": text2}, f)

        st.success("✅ Text extracted successfully!")

        # --- Results Section ---
        st.markdown("### 📝 Extracted Text - First Document (UKM Expected)")
        st.info(classification1)
        st.text_area("Text from first document (UKM):", text1, height=200)

        st.markdown("### 📝 Extracted Text - Second Document (Other Institute Expected)")
        st.info(classification2)
        st.text_area("Text from second document (Other Institute):", text2, height=200)

else:
    st.info("📥 Please upload both syllabus documents to begin.")

# --- Similarity Calculation ---
st.markdown("### 📊 Document Similarity Analysis")
bert_score = calculate_bert_similarity(text1, text2)
tfidf_score = calculate_tfidf_similarity(text1, text2)

st.write(f"**BERT-based Similarity:** {bert_score}%")
st.write(f"**TF-IDF + Cosine Similarity:** {tfidf_score}%")

# --- Footer ---
st.markdown("---")
st.markdown(f"<p style='text-align:center;color:{UKM_BLUE};'>© 2025 Universiti Kebangsaan Malaysia | Transfer Credit Checker</p>", unsafe_allow_html=True)
