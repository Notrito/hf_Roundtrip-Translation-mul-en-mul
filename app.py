import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

def load_model():
    """Load translation model and tokenizer."""
    try:
        model_name = "Helsinki-NLP/opus-mt-mul-en"
        st.write(f"Loading model: {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        st.write("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def translate(text, model, tokenizer):
    """Translate input text using the loaded model."""
    if not model or not tokenizer:
        return "Error: Model not loaded."
    
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**inputs, max_length=50)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Streamlit UI
st.title("Simple Translation App")
model, tokenizer = load_model()

text_input = st.text_area("Enter text to translate:")
if st.button("Translate"):
    if text_input:
        translation = translate(text_input, model, tokenizer)
        st.write("**Translation:**", translation)
    else:
        st.warning("Please enter text to translate.")
