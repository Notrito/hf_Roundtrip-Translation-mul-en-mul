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

def translate(text, model, tokenizer, src_lang, tgt_lang):
    """Translate text using the given model and tokenizer."""
    if model is None or tokenizer is None:
        return "Model not loaded."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    translated_tokens = model.generate(**inputs, max_length=200)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def roundtrip_translate(text, model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en, src_lang, tgt_lang):
    """Perform round-trip translation using two models."""
    if None in (model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en):
        return {
            "original": text,
            "forward_translation": "Error: Models not loaded.",
            "back_translation": "Error: Models not loaded."
        }
    
    forward_translation = translate(text, model_to_en, tokenizer_to_en, src_lang, tgt_lang)
    back_translation = translate(forward_translation, model_from_en, tokenizer_from_en, tgt_lang, src_lang)
    
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

language_options = {
    "spa": "Spanish",
    "fra": "French",
    "deu": "German",
    "ita": "Italian",
    "por": "Portuguese",
    "rus": "Russian",
    "cmn": "Chinese (Simplified)",
    "jpn": "Japanese",
    "kor": "Korean",
    "ara": "Arabic",
    "hin": "Hindi",
    "eng": "English"
}

src_lang = st.selectbox("Select source language:", options=list(language_options.keys()), format_func=lambda x: language_options[x])
tgt_lang = "eng"
text_input = st.text_area("Enter text to translate:")
if st.button("Translate"):
    if text_input:
        results = roundtrip_translate(text_input, model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en, src_lang, tgt_lang)
        st.write("**Forward Translation (to English):**", results["forward_translation"])
        st.write("**Back Translation (to Original Language):**", results["back_translation"])
    else:
        st.warning("Please enter text to translate.")