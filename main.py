import os
import numpy as np
import tensorflow as tf

from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import load_model
import streamlit as st


# ---- Load Model ----
def load_sentiment_model():
    if os.path.exists("simple_rnn_imdb.h5"):
        return load_model("simple_rnn_imdb.h5", compile=False)
    elif os.path.exists("simple_rnn_imdb_savedmodel"):
        return tf.keras.models.load_model("simple_rnn_imdb_savedmodel", compile=False)
    else:
        raise FileNotFoundError("Model file not found. Please check your path.")


model = load_sentiment_model()
model.summary()

# ---- Load IMDB word index ----
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


# ---- Helper: decode review ----
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])


# ---- Helper: preprocess user input ----
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) for word in words]  # 2 = "unknown"
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review


# ---- Streamlit App ----
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as **Positive** or **Negative**.")

# User input
user_input = st.text_area("✍️ Movie Review")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before classifying.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        st.success(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {prediction[0][0]:.4f}")
else:
    st.info("⌛ Awaiting input...")
