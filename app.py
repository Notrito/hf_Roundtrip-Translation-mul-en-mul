import streamlit as st
import torch
import soundfile as sf
from kokoro import KPipeline
from kokoro.pipeline import LANG_CODES  # Import available language codes & voices

from huggingface_hub import list_repo_files  # Import function to fetch files from HF Hub

# Hugging Face repo details
REPO_ID = "hexgrad/Kokoro-82M"
VOICE_DIR = "voices/"

# Function to fetch available voices from the repo
def get_available_voices():
    try:
        files = list_repo_files(REPO_ID)  # Fetch all files in the repo
        voices = [file.replace(VOICE_DIR, "").replace(".json", "") 
                  for file in files if file.startswith(VOICE_DIR)]
        return voices
    except Exception as e:
        st.error(f"Error fetching voices: {e}")
        return []

# Fetch available voices dynamically
available_voices = get_available_voices()

# Convert LANG_CODES dictionary to a usable format for Streamlit
lang_options = {f"{name} ({code})": code for code, name in LANG_CODES.items()}

# Dropdown for language selection
selected_lang_name = st.selectbox("Select language", list(lang_options.keys()))
selected_lang = lang_options[selected_lang_name]  # Convert selection to language code

# Show second dropdown only if voices exist
if available_voices:
    selected_voice = st.selectbox("Select a voice", available_voices)
    
    st.write(f"🔹 Selected Language Code: `{selected_lang}`")
    st.write(f"🔹 Selected Voice: `{selected_voice}`")
else:
    st.warning("No voices available for this language.")

# Load the text-to-speech model
@st.cache_resource
def load_pipeline():
    return KPipeline(lang_code=selected_lang)

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

