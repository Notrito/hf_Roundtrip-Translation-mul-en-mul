import streamlit as st
import torch
import soundfile as sf
from kokoro import KPipeline

# Define language options
lang_options = ["a", "b"]


lang_options = {
    "English (USA)" : "a",
    "English (Britain)" : "b"
}
selected_lang = st.selectbox("Select language", list(lang_options.keys()))
# Define voice options based on selected language
voice_options = {
    "a": {"Bella": "af_bella", "Nicole": "af_nicole", "Sarah": "af_sarah"},
    "b": {"Alice": "bf_alice", "Emma": "bf_emma"},
}

# Get available voices for selected language
available_voices = voice_options.get(selected_lang, {})

# Show second dropdown only if the first selection is valid
if available_voices:
    selected_voice = st.selectbox("Select a voice", list(available_voices.keys()))
    mapped_voice = available_voices[selected_voice]
    
    st.write(f"🔹 Selected Voice Code: `{mapped_voice}`")

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

