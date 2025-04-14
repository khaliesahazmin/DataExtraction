import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
import cv2
import json

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
uploaded_file1 = st.file_uploader("Upload First Syllabus (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'])
uploaded_file2 = st.file_uploader("Upload Second Syllabus (PDF/Image)", type=['pdf', 'png', 'jpg', 'jpeg'])

# --- Preprocess Image ---
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

# --- Identify Syllabus Source ---
def classify_syllabus(text):
    ukm_keywords = ["universiti kebangsaan malaysia", "ukm", "fakulti", "kod kursus"]
    other_keywords = ["politeknik", "kolej", "universiti teknologi", "institute", "diploma"]

    text_lower = text.lower()
    ukm_score = sum(kw in text_lower for kw in ukm_keywords)
    other_score = sum(kw in text_lower for kw in other_keywords)

    if ukm_score > other_score:
        return "✅ UKM Syllabus"
    elif other_score > ukm_score:
        return "🏫 Other Institution/Diploma Syllabus"
    else:
        return "❓ Unable to determine (Insufficient keywords)"

# --- Main Logic ---
if uploaded_file1 and uploaded_file2:
    with st.spinner("🔍 Extracting text and classifying documents..."):
        text1 = extract_text(uploaded_file1)
        text2 = extract_text(uploaded_file2)

        result1 = classify_syllabus(text1)
        result2 = classify_syllabus(text2)

        # Save to JSON (optional)
        with open("output1.json", "w") as f:
            json.dump({"extracted_text": text1}, f)
        with open("output2.json", "w") as f:
            json.dump({"extracted_text": text2}, f)

        st.success("✅ Text extracted and classified successfully!")

        # Display classification results
        st.markdown("### 📝 First Document Result")
        st.info(f"🔍 Classification: {result1}")
        st.text_area("Extracted Text from First Document:", text1, height=200)

        st.markdown("### 📝 Second Document Result")
        st.info(f"🔍 Classification: {result2}")
        st.text_area("Extracted Text from Second Document:", text2, height=200)

else:
    st.info("Please upload both syllabus documents to begin.")

# --- Footer ---
st.markdown("---")
st.markdown(f"<p style='text-align:center;color:{UKM_BLUE};'>© 2025 Universiti Kebangsaan Malaysia | Transfer Credit Checker</p>", unsafe_allow_html=True)
