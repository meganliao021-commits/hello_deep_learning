
import streamlit as st
from transformers import pipeline

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import numpy as np

st.head("testing app")
# Testing with the saved model
model2 = AutoModelForSequenceClassification.from_pretrained("CustomModel_yelp",
                                                            num_labels=5)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenized testing data
label = 4 # label = 4
text = "dr. goldberg offers everything i look for in a general practitioner. he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first. really, what more do you need? i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank."
inputs = tokenizer(text,
                   padding=True,
                   truncation=True,
                   return_tensors='pt')

outputs = model2(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.cpu().detach().numpy()

# Get the index of the largest output value
max_index = np.argmax(predictions)

st.write("The label is {label} and the predicted label is {max_index}")

def main():
    sentiment_pipeline = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    st.title("Sentiment Analysis with HuggingFace Spaces")
    st.write("Enter a sentence to analyze its sentiment:")

    user_input = st.text_input("")
    if user_input:
        result = sentiment_pipeline(user_input)
        sentiment = result[0]["label"]
        confidence = result[0]["score"]

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
