import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import base64
import io

# Set background with animation
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/gif;base64,{b64_encoded});
            background-size: cover;
            animation: backgroundAnimation 10s infinite alternate;
        }}
        @keyframes backgroundAnimation {{
            0% {{ filter: brightness(0.8); }}
            100% {{ filter: brightness(1.2); }}
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Classify image
def classify(image, model, class_names):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def create_round_image(image, size=(300, 300)):
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    output = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    output.putalpha(mask)
    return output

def display_precautions(class_name):
    precautions = {
        'dew': 'Wear warm clothing to avoid catching a cold.',
        'fogy': 'Drive carefully and use fog lights.',
        'frost': 'Cover your plants to protect them from frost.',
        'glaze': 'Be cautious of slippery roads and sidewalks.',
        'hail': 'Stay indoors and avoid driving if possible.',
        'lightning': 'Avoid open fields and stay indoors.',
        'rain': 'Carry an umbrella and wear waterproof clothing.',
        'rainbow': 'No precautions needed, enjoy the view!',
        'rime': 'Be cautious of icy roads and sidewalks.',
        'sandstrom': 'Wear a mask and protect your eyes.',
        'snow': 'Wear warm clothing and be careful on icy roads.',
        'cloudy': 'Donot Forget to take the Umberalla,Rain is comming.',
        'sunshine': 'Wear sunscreen and stay hydrated.'
    }
    return precautions.get(class_name.lower(), 'No precautions available.')

# Set the background
set_background('wetxx.gif')

# Title and header with animation
st.markdown("""
    <style>
    @keyframes titleAnimation {
        0% { color: white; }
        50% { color: #f9c74f; }
        100% { color: white; }
    }
    @keyframes headerAnimation {
        0% { color: white; }
        50% { color: #f94144; }
        100% { color: white; }
    }
    @keyframes resultAnimation {
        0% { transform: scale(0.9); color: red; }
        50% { transform: scale(1.1); color: green; }
        100% { transform: scale(1); color: blue; }
    }
    @keyframes textBlink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    @keyframes rgbBlink {
        0% { border-color: red; }
        33% { border-color: green; }
        66% { border-color: blue; }
        100% { border-color: red; }
    }
    .round-image {
        animation: rgbBlink 2s infinite;
        border-width: 5px;
        border-style: solid;
        border-radius: 50%;
        width: 350px;
        height: 350px;
        object-fit: cover;
    }
    </style>
    <h1 style="text-align: center; color: white; animation: titleAnimation 2s infinite;">WEATHER ANALYSIS APP</h1>
    <h2 style="text-align: center; color: white; animation: headerAnimation 2s infinite;">UPLOAD OR CAPTURE WEATHER IMAGES</h2>
""", unsafe_allow_html=True)

# File uploader and camera input
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
upload_option = st.selectbox("Select Input Method", ["Upload Image", "Use Camera"], index=0, key="input_method")
if upload_option == "Upload Image":
    file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"], label_visibility="collapsed")
    camera_image = None
elif upload_option == "Use Camera":
    file = None
    camera_image = st.camera_input("Take a picture")
st.markdown("</div>", unsafe_allow_html=True)

# Load classifier
model = load_model('keras_model.h5')

# Load class names
with open('labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f]

# Display image and classify
image = None

if file is not None:
    image = Image.open(file).convert('RGB')
elif camera_image is not None:
    image = Image.open(camera_image).convert('RGB')

if image:
    round_image = create_round_image(image)
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    buffered = io.BytesIO()
    round_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(f'<img src="data:image/png;base64,{img_str}" class="round-image"/>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)
    
    # Display results with animation
    st.markdown(f"""
        <div style="text-align: center;">
            <h3 style="animation: resultAnimation 2s infinite; background-color: #4QPF50; color: Blue;">Predicted class: {class_name}</h3>
            <p style="animation: textBlink 1s infinite; background-color: White; color: Green;">Confidence score: {conf_score:.2%}</p>
        </div>
    """, unsafe_allow_html=True)

    # Display precautions in a table
    precautions = display_precautions(class_name)
    st.markdown(f"""
        <div style="text-align: center;">
            <table style="width:100%; margin-top:20px; border-collapse: collapse; border: 1px solid #ddd;">
                <tr>
                    <th style="text-align:left; padding: 8px; background-color: #4CAF50; color: Red;">Precautions</th>
                </tr>
                <tr>
                    <td style="padding: 8px; background-color: white; color: Orange;">{precautions}</td>
                </tr>
            </table>
        </div>
    """, unsafe_allow_html=True)
