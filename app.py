import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st
import time
from streamlit_lottie import st_lottie
import requests

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
    """Translate text using NLLB model with correct BOS token handling."""
    if model is None or tokenizer is None:
        return "Model not loaded."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set the source language
    tokenizer.src_lang = source_lang
    
    # Create input tensors
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Get the correct forced BOS token ID for the target language
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
    
    # Generate translation
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
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

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

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

# Set page config
st.set_page_config(
    page_title="Round-Trip Translation",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .translation-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .arrow-container {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 24px;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .language-indicator {
        font-size: 16px;
        font-style: italic;
        color: #666;
    }
    .title-text {
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5 0%, #9C27B0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title with custom styling
st.markdown('<div class="title-text">Round-Trip Translation</div>', unsafe_allow_html=True)
st.write("Translate text to English and back to its original language to visualize translation quality.")

# Load lottie animation for translation flow
# lottie_translation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_uwWgICRjvr.json")

# Side panel for settings
with st.sidebar:
    st.header("Settings")
    
    # Load NLLB model and tokenizer
    model, tokenizer = load_model()
    
    # Select source language
    src_lang = st.selectbox(
        "Select source language:", 
        options=list(language_options.keys()), 
        format_func=lambda x: language_options[x],
        index=list(language_options.keys()).index("spa_Latn") if "spa_Latn" in language_options else 0
    )
    tgt_lang = "eng_Latn"  # English is always the target for the first translation
    
    # Add a check to display model information
    if st.checkbox("Show model information"):
        st.info("**Model:** facebook/nllb-200-distilled-600M\n\n**Supported languages:** 200+ languages")
    
    # Memory cleanup button
    if st.button("Clear Memory Cache"):
        try:
            import gc
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            st.success("Memory cache cleared!")
        except Exception as e:
            st.error(f"Error clearing memory: {e}")

# Main area
text_input = st.text_area("Enter text to translate:", height=100)

# Translate button
if st.button("Translate", type="primary"):
    if text_input:
        # Container for visualization
        translation_container = st.container()
        
        with translation_container:
            # Original text
            st.markdown(f"<div class='translation-box'><b>Original Text</b> <span class='language-indicator'>({language_options[src_lang]})</span><hr>{text_input}</div>", unsafe_allow_html=True)
            
            # First translation - animated arrow
            col1, col2, col3 = st.columns([0.4, 0.2, 0.4])
            with col2:
                st.markdown("<div class='arrow-container'>↓<br>Translating...</div>", unsafe_allow_html=True)
            
            # Perform first translation with a visual delay
            progress_text = "First translation in progress..."
            first_progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Short delay for visual effect
                first_progress_bar.progress(i + 1)
            
            # Get results
            results = roundtrip_translate(text_input, model, tokenizer, src_lang, tgt_lang)
            
            # Show forward translation
            st.markdown(f"<div class='translation-box'><b>Forward Translation</b> <span class='language-indicator'>(English)</span><hr>{results['forward_translation']}</div>", unsafe_allow_html=True)
            
            # Second translation - animated arrow
            col1, col2, col3 = st.columns([0.4, 0.2, 0.4])
            with col2:
                st.markdown("<div class='arrow-container'>↓<br>Translating back...</div>", unsafe_allow_html=True)
            
            # Perform second translation with a visual delay
            progress_text = "Second translation in progress..."
            second_progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Short delay for visual effect
                second_progress_bar.progress(i + 1)
            
            # Show back translation
            st.markdown(f"<div class='translation-box'><b>Back Translation</b> <span class='language-indicator'>({language_options[src_lang]})</span><hr>{results['back_translation']}</div>", unsafe_allow_html=True)
            
            # Quality assessment
            st.subheader("Roundtrip Quality Assessment")
            
            # Compare original and back translation
            if results['original'].lower() == results['back_translation'].lower():
                st.success("Perfect match! The meaning was fully preserved in the round-trip translation.")
            else:
                # Calculate a very simple similarity score (percentage of words that match)
                original_words = set(results['original'].lower().split())
                back_words = set(results['back_translation'].lower().split())
                common_words = original_words.intersection(back_words)
                similarity = len(common_words) / max(len(original_words), len(back_words)) * 100
                
                if similarity > 80:
                    st.success(f"Good translation! About {similarity:.1f}% of the meaning was preserved.")
                elif similarity > 50:
                    st.warning(f"Moderate translation quality. About {similarity:.1f}% of the meaning was preserved.")
                else:
                    st.error(f"Poor translation quality. Only about {similarity:.1f}% of the meaning was preserved.")
                
                # Show differences
                st.write("**Original:** ", results['original'])
                st.write("**Roundtrip:** ", results['back_translation'])
    else:
        st.warning("Please enter text to translate.")

# Footer
st.markdown("---")
st.markdown("Translation powered by NLLB (No Language Left Behind) - 200 language support")