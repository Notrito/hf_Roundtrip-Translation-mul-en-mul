import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st
import time
from streamlit_lottie import st_lottie
import requests

# Función para cargar el modelo y el tokenizador
@st.cache_resource
def load_model(model_name="facebook/nllb-200-distilled-600M"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Función para traducir
def translate(text, model, tokenizer, source_lang, target_lang):
    if model is None or tokenizer is None:
        return "Model not loaded."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tokenizer.src_lang = source_lang
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
    
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=200
    )
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Traducción de ida y vuelta
def roundtrip_translate(text, model, tokenizer, src_lang, tgt_lang="eng_Latn"):
    if model is None or tokenizer is None:
        return {
            "original": text,
            "forward_translation": "Error: Model not loaded.",
            "back_translation": "Error: Model not loaded."
        }
    
    forward_translation = translate(text, model, tokenizer, src_lang, tgt_lang)
    back_translation = translate(forward_translation, model, tokenizer, tgt_lang, src_lang)
    
    return {
        "original": text,
        "forward_translation": forward_translation,
        "back_translation": back_translation
    }

def calculate_exact_match_percentage(original, back_translated):
    original_words = original.lower().split()
    back_words = back_translated.lower().split()
    
    # Contar palabras coincidentes en la misma posición
    matches = sum(1 for o, b in zip(original_words, back_words) if o == b)
    
    # Calcular porcentaje
    total_words = len(original_words)
    percentage = (matches / total_words) * 100 if total_words > 0 else 0
    
    return percentage
    
# Diccionario de idiomas
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

# Configuración de la página
st.set_page_config(
    page_title="Round-Trip Translation",
    page_icon="🌐",
    layout="wide"
)

# Estilos CSS para mejorar la visualización
st.markdown("""
<style>
    .translation-box {
        background-color: #f0f2f6;
        color: black;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        font-size: 18px;
    }
    .arrow {
        font-size: 32px;
        text-align: center;
        color: #1E88E5;
        font-weight: bold;
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

# Título
st.markdown('<div class="title-text">Round-Trip Translation</div>', unsafe_allow_html=True)
st.write("Translate text to English and back to its original language to visualize translation quality.")

# Cargar modelo
with st.sidebar:
    st.header("Settings")
    model, tokenizer = load_model()
    
    src_lang = st.selectbox(
        "Select source language:", 
        options=list(language_options.keys()), 
        format_func=lambda x: language_options[x],
        index=list(language_options.keys()).index("spa_Latn") if "spa_Latn" in language_options else 0
    )
    
    if st.checkbox("Show model information"):
        st.info("**Model:** facebook/nllb-200-distilled-600M\n\n**Supported languages:** 200+ languages")
    
    if st.button("Clear Memory Cache"):
        try:
            import gc
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            st.success("Memory cache cleared!")
        except Exception as e:
            st.error(f"Error clearing memory: {e}")

# Entrada de texto
text_input = st.text_area("Enter text to translate:", height=100)

# Botón de traducción
if st.button("Translate", type="primary"):
    if text_input:
        # Realizar traducción
        results = roundtrip_translate(text_input, model, tokenizer, src_lang, "eng_Latn")
        
        # Diseño en columnas
        col1, col2 = st.columns([0.45, 0.45])  # Dos columnas para Original y Forward Translation
        
        with col1:
            st.markdown(f"<div class='translation-box'><b>Original Text</b> ({language_options[src_lang]})<hr>{results['original']}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div class='translation-box'><b>Forward Translation</b> (English)<hr>{results['forward_translation']}</div>", unsafe_allow_html=True)

        # Flecha central
        st.markdown("<div class='arrow'>↓</div>", unsafe_allow_html=True)

        # Segunda fila: Back Translation
        col3, _ = st.columns([0.45, 0.45])  # Segunda fila con una sola columna a la izquierda
        with col3:
            st.markdown(f"<div class='translation-box'><b>Back Translation</b> ({language_options[src_lang]})<hr>{results['back_translation']}</div>", unsafe_allow_html=True)

         # Cálculo de porcentaje de palabras exactas
        exact_match_percentage = calculate_exact_match_percentage(results['original'], results['back_translation'])
        
        st.subheader("Roundtrip Quality Assessment")
        st.info(f"**Exact Word Match Percentage:** {exact_match_percentage:.1f}%")

        if exact_match_percentage > 80:
            st.success("Great translation! Most words remained unchanged.")
        elif exact_match_percentage > 50:
            st.warning("Moderate quality. Some words changed.")
        else:
            st.error("Low quality. Many words were altered.")

        st.write("**Original:** ", results['original'])
        st.write("**Roundtrip:** ", results['back_translation'])
    else:
        st.warning("Please enter text to translate.")

# Footer
st.markdown("---")
st.markdown("Translation powered by NLLB (No Language Left Behind) - 200 language support")
