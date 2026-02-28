import streamlit as st

st.write("ISOM5240")
st.write("hello megan")
st.write("What do you want to do?")

import streamlit as st
from transformers import pipeline

# Load the text classification model pipeline
classifier = pipeline("text-classification",model='j-hartmann/emotion-english-distilroberta-base')


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
        st.image(image, caption="Uploaded Image", use_column_width=True) #show the image

# Button interaction
if st.button("Click Me"):
    st.write("🎉 You clicked the button!")



# Program title: Storytelling App

# import part
import streamlit as st
from transformers import pipeline

# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story_text = pipe(text)[0]['generated_text']
    return story_text

# text2audio
def text2audio(story_text):
    pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = pipe(story_text)
    return audio_data


def main():
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    uploaded_file = st.file_uploader("Select an Image...")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


        #Stage 1: Image to Text
        st.text('Processing img2text...')
        scenario = img2text(uploaded_file.name)
        st.write(scenario)

        #Stage 2: Text to Story
        st.text('Generating a story...')
        story = text2story(scenario)
        st.write(story)

        #Stage 3: Story to Audio data
        st.text('Generating audio data...')
        audio_data =text2audio(story)

        # Play button
        if st.button("Play Audio"):
            # Get the audio array and sample rate
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]

            # Play audio directly using Streamlit
            st.audio(audio_array,
                     sample_rate=sample_rate)


if __name__ == "__main__":
    main()



from transformers import pipeline
from PIL import Image
import streamlit as st

#def function
def main():

        
        # Streamlit UI
        st.title("Title: Age Classification using ViT")
        
        # Load the age classification pipeline
        # The code below should be placed in the main part of the program
        age_classifier = pipeline("image-classification",
                                  model="dini-r-a/image_age_classification")
        
        image_name = "middleagedWoman.jpg"
        image_name = Image.open(image_name).convert("RGB")
        
        # Classify age
        age_predictions = age_classifier(image_name)
        st.write(age_predictions)
        age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
        
        # Display results
        st.write("Predicted Age Range:")
        st.write(f"Age range: {age_predictions[0]['label']}")



if __name__ == "__main__":
    main()



def ageclassifier(imagefilename):
    # The code below should be placed in the main part of the program
        age_classifier = pipeline("image-classification",
                                  model="dini-r-a/image_age_classification")
        
        image_name = imagefilename
        image_name = Image.open(image_name).convert("RGB")
        
        # Classify age
        age_predictions = age_classifier(image_name)
return age_predictions


def main():
      # Streamlit UI
        st.title("Title: Age Classification using ViT")
    
        age_predictions = age_classifier("middleagedWoman.jpg")
        
    
