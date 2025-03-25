import streamlit as st
import torch
import soundfile as sf
from kokoro import KPipeline

# Define options for the dropdown
options = ['Bella', 'Nicole', 'Sarah']

# Create a dropdown menu
selected_option = st.selectbox("Select an option", options)
if selected_option == 'Bella':
    selected_option = 'af_bella'
if selected_option == 'Nicole':
    selected_option = 'af_nicole'
if selected_option == 'Sarah':
    selected_option = 'af_sarah'
# Load the text-to-speech model
@st.cache_resource
def load_pipeline():
    return KPipeline(lang_code='a')

pipeline = load_pipeline()

# Streamlit UI
st.title("Text-to-Speech (TTS) with Kokoro")

# User input
text = st.text_area("Put voice to your text 🎙️:", "Hello, world!")

# Generate speech on button click
if st.button("Generate Speech"):
    if text.strip():
        with st.spinner("Generating audio..."):
            generator = pipeline(text, voice=selected_option)

            audio_data = None
            for i, (gs, ps, audio) in enumerate(generator):
                audio_data = audio  # Save last audio chunk

            if audio_data is not None:
                # Save and play audio
                audio_path = "generated_speech.wav"
                sf.write(audio_path, audio_data, 24000)
                st.audio(audio_path, format="audio/wav")
            
                st.success("Speech generation complete!")
    
                # Option to download
                with open(audio_path, "rb") as f:
                    st.download_button("Download Audio", f, file_name="speech.wav", mime="audio/wav")
            else:
                st.error("Failed to generate audio.")

    else:
        st.error("Please enter some text to generate speech.")

