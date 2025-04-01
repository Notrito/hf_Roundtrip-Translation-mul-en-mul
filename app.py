import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

def load_models():
    """Load translation models and tokenizers for multilingual translation."""
    try:
        # Models for translation to and from English
        model_to_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        tokenizer_to_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        
        model_from_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
        tokenizer_from_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
        
        return model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def translate(text, model, tokenizer, src_lang=None, tgt_lang=None):
    """Translate text using the given model and tokenizer."""
    if model is None or tokenizer is None:
        return "Model not loaded."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # The Helsinki-NLP OPUS-MT models don't require src_lang and tgt_lang to be set on the tokenizer
    # These models are specialized for specific language pairs
    # So we remove these lines that might be causing issues:
    # tokenizer.src_lang = src_lang
    # tokenizer.tgt_lang = tgt_lang
    
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
    
    # For these OPUS-MT models, we don't need to pass src_lang and tgt_lang parameters
    # They're already trained for specific language directions
    forward_translation = translate(text, model_to_en, tokenizer_to_en)
    back_translation = translate(forward_translation, model_from_en, tokenizer_from_en)
    
    return {
        "original": text,
        "forward_translation": forward_translation,
        "back_translation": back_translation
    }

# Streamlit UI
st.title("Multilingual Round-Trip Translation App")
st.write("Translate text to English and back to its original language to test translation quality.")

# Define language options that OPUS-MT supports for translation to/from English
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

# Add Streamlit cache to avoid reloading models on every interaction
@st.cache_resource
def get_cached_models():
    return load_models()

# Load models and tokenizers using the cached function
model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en = get_cached_models()

# Select source language
src_lang = st.selectbox("Select source language:", 
                      options=list(language_options.keys()), 
                      format_func=lambda x: language_options[x])
tgt_lang = "eng"  # English is always the target for the first translation

# Get input text
text_input = st.text_area("Enter text to translate:")

# Add a check to display model information
if st.checkbox("Show model information"):
    st.write("**Translation to English Model:**", "Helsinki-NLP/opus-mt-mul-en")
    st.write("**Translation from English Model:**", "Helsinki-NLP/opus-mt-en-mul")

# Translate button
if st.button("Translate"):
    if text_input:
        with st.spinner("Translating..."):
            # Show a progress indicator
            results = roundtrip_translate(
                text_input, model_to_en, tokenizer_to_en, 
                model_from_en, tokenizer_from_en, src_lang, tgt_lang
            )
            
            # Display results in separate containers
            st.subheader("Translation Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Text:**")
                st.write(results["original"])
            
            with col2:
                st.write("**Forward Translation (to English):**")
                st.write(results["forward_translation"])
            
            st.write("**Back Translation (to Original Language):**")
            st.write(results["back_translation"])
    else:
        st.warning("Please enter text to translate.")