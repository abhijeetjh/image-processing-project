import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Image Processing & Pattern Analysis", layout="centered")
st.title("🧠 Image Processing & Pattern Analysis")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, caption='Original Image', use_column_width=True)

    operation = st.selectbox("Select Image Processing Operation", [
        "Grayscale", "Pixelate", "Box Filter", "Histogram Equalization", "Edge Detection"])

    if operation == "Grayscale":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        st.image(gray, caption="Grayscale Image", use_column_width=True, clamp=True)

    elif operation == "Pixelate":
        pixel_size = st.slider("Pixel Size", 1, 50, 10)
        h, w = img_array.shape[:2]
        temp = cv2.resize(img_array, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        st.image(pixelated, caption="Pixelated Image", use_column_width=True)

    elif operation == "Box Filter":
        kernel = st.slider("Kernel Size", 1, 20, 5)
        box = cv2.blur(img_array, (kernel, kernel))
        st.image(box, caption="Box Filtered Image", use_column_width=True)

    elif operation == "Histogram Equalization":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        eq = cv2.equalizeHist(gray)
        st.image(eq, caption="Histogram Equalized Image", use_column_width=True, clamp=True)

    elif operation == "Edge Detection":
        low = st.slider("Lower Threshold", 0, 255, 50)
        high = st.slider("Upper Threshold", 0, 255, 150)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        st.image(edges, caption="Edge Detected Image", use_column_width=True, clamp=True)

    if st.button("Download Processed Image"):
        buf = io.BytesIO()
        if operation in ["Grayscale", "Histogram Equalization", "Edge Detection"]:
            result = Image.fromarray(eval(operation.lower().replace(" ", "_")))
        else:
            result = Image.fromarray(eval(operation.lower().replace(" ", "")))
        result.save(buf, format="PNG")
        st.download_button(label="Download as PNG", data=buf.getvalue(), file_name="processed_image.png", mime="image/png")
