import streamlit as st
import torch
import soundfile as sf
from kokoro import KPipeline
from kokoro.pipeline import LANG_CODES, VOICES  # Import available language codes & voices

# Convert LANG_CODES dictionary to a usable format for Streamlit
lang_options = {f"{name} ({code})": code for code, name in LANG_CODES.items()}

# Dropdown for language selection
selected_lang_name = st.selectbox("Select language", list(lang_options.keys()))
selected_lang = lang_options[selected_lang_name]  # Convert selection to language code

# Fetch available voices for selected language
available_voices = VOICES.get(selected_lang, {})

# Show second dropdown only if voices exist
if available_voices:
    voice_options = {name: voice_code for voice_code, name in available_voices.items()}
    selected_voice_name = st.selectbox("Select a voice", list(voice_options.keys()))
    selected_voice = voice_options[selected_voice_name]

    st.write(f"🔹 Selected Language Code: `{selected_lang}`")
    st.write(f"🔹 Selected Voice Code: `{selected_voice}`")
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

