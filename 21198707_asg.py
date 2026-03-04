
    
# Program title: Storytelling Application using Hugging Face Pipelines
# Student ID: 21198707 LIAO WEIXI

# --- 1. Import part ---
import streamlit as st
from transformers import pipeline
import torch
import os
from PIL import Image
    
# --- 2. Function part ---

def img2text(img_filename):
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    result = captioner(img_filename)
    return result[0]['generated_text']

def text2story(scenario):
    story_gen = pipeline("text-generation", model="roneneldan/TinyStories-1M")
    prompt = f"Genre: Children's Story. Prompt: {scenario}. Once upon a time,"
    
    # 1. 稍微多给一点 token (比如 150) 确保它能写出结尾
    output = story_gen(prompt, max_new_tokens=150, do_sample=True, temperature=0.8)
    full_text = output[0]['generated_text']
    
    # 2. 提取故事正文
    story_body = "Once upon a time," + full_text.split("Once upon a time,")[-1]
    
    # 3. --- 核心修复：只保留到最后一个句号 ---
    # 寻找最后一个句号的位置
    last_period = story_body.rfind('.')
    if last_period != -1:
        # 只截取到最后一个句号，丢弃后面没写完的断句
        final_story = story_body[:last_period + 1]
    else:
        final_story = story_body

    return final_story

def text2audio(story_text):
    tts_pipe = pipeline("text-to-audio", model="facebook/mms-tts-eng")
    audio_data = tts_pipe(story_text)
    return audio_data

# --- 3. Main part ---

def main():
    st.set_page_config(page_title="Magic Storytelling App", page_icon="🎨")
    st.header("🎨 Magic Storytelling App for Kids")
    st.write("Welcome! Upload an image and let's create a wonderful story together.")

    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
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
