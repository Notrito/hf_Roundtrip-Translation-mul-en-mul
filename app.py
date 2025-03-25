import streamlit as st
from transformers import pipeline

# Load the text-to-speech model
@st.cache_resource
def load_tts_model():
    return pipeline("text-to-speech", model="./models/kakao-enterprise/vits-ljs")

narrator = load_tts_model()

# Streamlit UI
st.title("Text-to-Speech (TTS) with Hugging Face")

# User input
text = st.text_area("Enter text to convert to speech:", "Hello, welcome to this Streamlit app!")

# Generate speech on button click
if st.button("Generate Speech"):
    if text.strip():
        with st.spinner("Generating audio..."):
            narrated_text = narrator(text)

        # Play the audio
        st.audio(narrated_text["audio"][0], format="audio/wav", sample_rate=narrated_text["sampling_rate"])
        st.success("Speech generation complete!")
    else:
        st.error("Please enter some text to generate speech.")



