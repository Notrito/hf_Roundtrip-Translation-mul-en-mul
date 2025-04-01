import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

@st.cache_resource
def load_model(model_name="facebook/nllb-200-distilled-600M"):
    """Load a single NLLB model that can handle multiple languages."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def translate(text, model, tokenizer, source_lang, target_lang):
    """Translate text using NLLB model."""
    if model is None or tokenizer is None:
        return "Model not loaded."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Format language codes for NLLB
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=200
    )
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def roundtrip_translate(text, model, tokenizer, src_lang, tgt_lang="eng_Latn"):
    """Perform round-trip translation using NLLB model."""
    if model is None or tokenizer is None:
        return {
            "original": text,
            "forward_translation": "Error: Model not loaded.",
            "back_translation": "Error: Model not loaded."
        }
    
    # First translation: source -> English
    forward_translation = translate(text, model, tokenizer, src_lang, tgt_lang)
    
    # Second translation: English -> source
    back_translation = translate(forward_translation, model, tokenizer, tgt_lang, src_lang)
    
    return {
        "original": text,
        "forward_translation": forward_translation,
        "back_translation": back_translation
    }

# NLLB language codes
language_options = {
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "zho_Hans": "Chinese (Simplified)",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "ara_Arab": "Arabic",
    "hin_Deva": "Hindi",
    "eng_Latn": "English"
}

# Streamlit UI
st.title("Multilingual Round-Trip Translation App")
st.write("Translate text to English and back to its original language to test translation quality.")

# Load NLLB model and tokenizer
model, tokenizer = load_model()

# Select source language
src_lang = st.selectbox("Select source language:", 
                      options=list(language_options.keys()), 
                      format_func=lambda x: language_options[x])
tgt_lang = "eng_Latn"  # English is always the target for the first translation

# Get input text
text_input = st.text_area("Enter text to translate:")

# Add a check to display model information
if st.checkbox("Show model information"):
    st.write("**Model:** facebook/nllb-200-distilled-600M")
    st.write("**Supported languages:** 200+ languages")

# Translate button
if st.button("Translate"):
    if text_input:
        with st.spinner("Translating..."):
            # Show a progress indicator
            results = roundtrip_translate(
                text_input, model, tokenizer, src_lang, tgt_lang
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