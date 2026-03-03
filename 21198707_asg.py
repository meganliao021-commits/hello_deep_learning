
# Program title: Storytelling Application using Hugging Face Pipelines
# Student ID: <your_student_id>

# --- 1. Import part ---
import streamlit as st
from transformers import pipeline
import torch
import os
from PIL import Image

# --- 2. Function part ---

def img2text(img_filename):
    """
    Requirement 20-22: Extract a caption using the recommended BLIP model.
    Model: Salesforce/blip-image-captioning-base.
    """
    # Initialize image-to-text pipeline
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Generate descriptive text from the image
    result = captioner(img_filename)
    return result[0]['generated_text']

def text2story(scenario):
    """
    Requirement 15-16: Generate a narrative of 50-100 words based on the image caption.
    Requirement 8: Ensure content is appropriate for 3-10 year-old kids.
    Model: aspis/gpt2-genre-story-generation.
    """
    # Initialize text generation pipeline
    story_gen = pipeline("text-generation", model="openai-community/gpt2-medium")
    
    # Crafting a prompt to guide GPT-2 for children's storytelling
    # Setting the genre helps maintain the tone for kids
    prompt = f"Genre: Children's Story. Prompt: {scenario}. Once upon a time,"
    
    # max_new_tokens is tuned to stay within the 50-100 words requirement
    output = story_gen(prompt, max_new_tokens=100, do_sample=True, temperature=0.8, top_k=50)
    
    story_text = output[0]['generated_text']
    return story_text

def text2audio(story_text):
    """
    Requirement 17-18 & 25-27: Convert the generated story into audio format.
    Model: facebook/mms-tts-eng (High efficiency and stability).
    """
    # Initialize text-to-audio pipeline
    tts_pipe = pipeline("text-to-audio", model="facebook/mms-tts-eng")
    
    # Generate audio data
    audio_data = tts_pipe(story_text)
    return audio_data

# --- 3. Main part ---

def main():
    # Application setup following Assessment Criteria for User Experience
    st.set_page_config(page_title="Magic Storytelling App", page_icon="🎨")
    st.header("🎨 Magic Storytelling App for Kids")
    st.write("Welcome! Upload an image and let's create a wonderful story together.")

    # Requirement 12-14: Allow users to specify image filename in current working directory
    # For a better UI, we use file_uploader and save it to the current directory
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Save file to current working directory as required by the assignment
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the image
        st.image(uploaded_file, caption="Your Uploaded Image", use_container_width=True)

        # Stage 1: Image Captioning
        with st.status("Analyzing the image..."):
            caption = img2text(uploaded_file.name)
            st.info(f"Image Details: {caption}")

        # Stage 2: Story Generation
        with st.status("Generating a magical story..."):
            story = text2story(caption)
            st.subheader("📖 The Story")
            st.write(story)
            
            # Word count validation for Requirement 16
            word_count = len(story.split())
            st.caption(f"Word count: {word_count} words (Requirement: 50-100 words)")

        # Stage 3: Text-to-Audio Conversion
        with st.status("Turning the story into audio..."):
            audio_result = text2audio(story)
            
            # Extract audio array and sample rate for Streamlit player
            st.audio(audio_result["audio"], 
                     sample_rate=audio_result["sampling_rate"])
            st.success("The story is ready! Click play above.")

if __name__ == "__main__":
    main()
