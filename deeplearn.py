import streamlit as st

st.write("ISOM5240")
st.write("hello megan")
st.write("What do you want to do?")

import streamlit as st
from transformers import pipeline

# Load the text classification model pipeline
classifier = pipeline("text-classification",model='isom5240ust/bert-base-uncased-emotion')

# Streamlit application title
st.title("Text Classification for you")
st.write("Classification for 6 emotions: sadness, joy, love, anger, fear, surprise")

# Text input for user to enter the text to classify
text = st.text_area("Enter the text to classify", "")

# Perform text classification when the user clicks the "Classify" button
if st.button("Classify"):
    # Perform text classification on the input text
    result = classifier(text)[0]

    # Display the classification result
    st.write("Text:", text)
    st.write("Label:", result['score'])
    st.write("Score:", result['label'])


import streamlit as st
from PIL import Image
import time

# App title
st.title("Streamlit Demo on Hugging Face")

# Write some text
st.write("Welcome to a demo app showcasing basic Streamlit components!")

# File uploader for image and audio
uploaded_image = st.file_uploader("Upload an image",
                                  type=["jpg", "jpeg", "png"])

# Display image with spinner
if uploaded_image is not None:
    with st.spinner("Loading image..."):
        time.sleep(1)  # Simulate a delay
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Button interaction
if st.button("Click Me"):
    st.write("🎉 You clicked the button!")
