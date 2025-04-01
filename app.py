import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

def load_models():
    """Load translation models and tokenizers for multilingual translation."""
    try:
        model_to_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        tokenizer_to_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        
        model_from_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
        tokenizer_from_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
        
        return model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def translate(text, model, tokenizer):
    """Translate text using the given model and tokenizer."""
    if model is None or tokenizer is None:
        return "Model not loaded."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    translated_tokens = model.generate(**inputs, max_length=200)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def roundtrip_translate(text, model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en):
    """Perform round-trip translation using two models."""
    if None in (model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en):
        return {
            "original": text,
            "forward_translation": "Error: Models not loaded.",
            "back_translation": "Error: Models not loaded."
        }
    
    forward_translation = translate(text, model_to_en, tokenizer_to_en)
    back_translation = translate(forward_translation, model_from_en, tokenizer_from_en)
    
    return {
        "original": text,
        "forward_translation": forward_translation,
        "back_translation": back_translation
    }

# Load models and tokenizers
model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en = load_models()

# Streamlit UI
st.title("Multilingual Round-Trip Translation App")
st.write("Translate text to English and back to its original language to test translation quality.")

text_input = st.text_area("Enter text to translate:")
if st.button("Translate"):
    if text_input:
        results = roundtrip_translate(text_input, model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en)
        st.write("**Forward Translation (to English):**", results["forward_translation"])
        st.write("**Back Translation (to Original Language):**", results["back_translation"])
    else:
        st.warning("Please enter text to translate.")
