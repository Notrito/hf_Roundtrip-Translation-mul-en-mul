import streamlit as st
from transformers import VitsModel, AutoTokenizer
import torch
import torchaudio

# Load the text-to-speech model
@st.cache_resource
def load_model():
    model = VitsModel.from_pretrained("kakao-enterprise/vits-ljs")
    tokenizer = AutoTokenizer.from_pretrained("kakao-enterprise/vits-ljs")
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("Text-to-Speech (TTS) with Hugging Face")

# User input
text = st.text_area("Put voice to your text 🎙️:", "Hello, world!")

# Generate speech on button click
if st.button("Generate Speech"):
    if text.strip():
        with st.spinner("Generating audio..."):
            inputs = tokenizer(text, return_tensors="pt", phonemize=False)

            # Generate speech waveform
            with torch.no_grad():
                output = model(**inputs).waveform

            # Save and play audio
            audio_path = "generated_speech.wav"
            torchaudio.save(audio_path, output, sample_rate=22050)
            st.audio(audio_path, format="audio/wav")
            st.success("Speech generation complete!")

             # Option to download
            with open(audio_path, "rb") as f:
                st.download_button("Download Audio", f, file_name="speech.wav", mime="audio/wav")

    else:
        st.error("Please enter some text to generate speech.")

