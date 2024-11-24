import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
import cv2
import tempfile

# Initialize Google Generative AI with API Key
GEMINI_API_KEY = " "  # Replace with your valid API key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Set up Streamlit page
st.set_page_config(page_title="Vision Assist", layout="wide")
st.title("VisionAid - AI Assistant for Visually Impaired")
st.sidebar.title("Features")
st.sidebar.markdown("""
- Scene Understanding
- Text-to-Speech
- Object & Obstacle Detection
""")


def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    text = pytesseract.image_to_string(image)
    return text


def text_to_speech(text):
    """Converts the given text to speech."""
    engine.say(text)
    engine.runAndWait()


def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text


def input_image_setup(uploaded_file):
    """Prepares the uploaded image for processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")


# Function to capture image from webcam
def capture_and_describe():
    """Captures an image from the webcam and processes it."""
    # Start video capture (from the webcam)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if not ret:
        st.write("Failed to capture image.")
        cap.release()
        return

    # Save the captured image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
        img_path = tmpfile.name
        cv2.imwrite(img_path, frame)
        st.image(frame, caption="Captured Image", use_container_width=True)

        # Process the image
        image = Image.open(img_path)
        describe_image(image)
    cap.release()


# Function to describe the uploaded or captured image
def describe_image(image):
    """Generates and describes the contents of an image."""
    text = extract_text_from_image(image)
    st.subheader("Extracted Text")
    st.write(text)

    # Use Google Generative AI for scene description
    input_prompt = """
    You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
    1. List of items detected in the image with their purpose.
    2. Overall description of the image.
    3. Suggestions for actions or precautions for the visually impaired.
    """
    image_data = input_image_setup(uploaded_file=None)
    scene_description = generate_scene_description(input_prompt, image_data)

    st.subheader("Scene Description")
    st.write(scene_description)

    # Text-to-Speech for Scene Description
    text_to_speech(scene_description)
    st.success("Scene Description has been read aloud!")


# Main app functionality
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# Buttons for functionalities
col1, col2, col3 = st.columns(3)
scene_button = col1.button("🔍 Describe Scene")
ocr_button = col2.button("📝 Extract Text")
tts_button = col3.button("🔊 Text-to-Speech")
camera_button = col1.button("📸 Capture Image from Camera")

# Input Prompt for AI Scene Understanding
input_prompt = """
You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
1. List of items detected in the image with their purpose.
2. Overall description of the image.
3. Suggestions for actions or precautions for the visually impaired.
"""

# Process based on user interaction
if uploaded_file:
    image_data = input_image_setup(uploaded_file)

    if scene_button:
        with st.spinner("Generating scene description..."):
            response = generate_scene_description(input_prompt, image_data)
            st.subheader("Scene Description")
            st.write(response)

            # Text-to-Speech for Scene Description
            text_to_speech(response)  # Added TTS for scene description

    if ocr_button:
        with st.spinner("Extracting text from image..."):
            text = extract_text_from_image(image)
            st.subheader("Extracted Text")
            st.write(text)

    if tts_button:
        with st.spinner("Converting text to speech..."):
            text = extract_text_from_image(image)
            if text.strip():
                text_to_speech(text)
                st.success("Text-to-Speech Conversion Completed!")
            else:
                st.warning("No text found in the image.")

# Capture Image from Camera
if camera_button:
    capture_and_describe()

