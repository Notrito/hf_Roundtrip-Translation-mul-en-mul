import streamlit as st
import torch
import soundfile as sf
from kokoro import KPipeline
from kokoro.pipeline import LANG_CODES
from huggingface_hub import hf_hub_download, list_repo_files
import os

# Hugging Face repo details
REPO_ID = "hexgrad/Kokoro-82M"
VOICE_DIR = "voices/"

def get_available_voices_and_languages():
    try:
        # Fetch all voice files
        files = list_repo_files(REPO_ID)
        
        # Extract voice files and their languages
        voices = [
            file.replace(VOICE_DIR, "").replace(".pt", "") 
            for file in files 
            if file.startswith(VOICE_DIR) and file.endswith('.pt')
        ]
        
        # Create a dictionary to map languages to their voices
        lang_voices = {}
        for voice in voices:
            # Split the voice name to get language code
            parts = voice.split('_', 1)
            if len(parts) == 2:
                lang_code, voice_name = parts
                if lang_code not in lang_voices:
                    lang_voices[lang_code] = []
                lang_voices[lang_code].append(voice)
        
        return lang_voices
    except Exception as e:
        st.error(f"Error fetching voices: {e}")
        return {}
    
def download_voice_file(voice_name):
    try:
        # Ensure voices directory exists
        os.makedirs("voices", exist_ok=True)
        
        # Download the voice file
        voice_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=f"voices/{voice_name}.pt",
            local_dir="voices",
            local_dir_use_symlinks=False
        )
        return voice_path
    except Exception as e:
        st.error(f"Error downloading voice file {voice_name}: {e}")
        return None

# Get available voices organized by language
lang_voices = get_available_voices_and_languages()

# Convert LANG_CODES dictionary to a usable format for Streamlit
lang_options = {f"{name} ({code})": code for code, name in LANG_CODES.items() if code in lang_voices}

def main():
    st.title("Text-to-Speech (TTS) with Kokoro")

    # Language selection
    selected_lang_name = st.selectbox("Select language", list(lang_options.keys()))
    selected_lang = lang_options[selected_lang_name]
    
    # Voice selection based on selected language
    if selected_lang in lang_voices:
        # Get voices for the selected language
        lang_specific_voices = lang_voices[selected_lang]
        
        # Voice selection dropdown
        selected_voice = st.selectbox("Select a voice", lang_specific_voices)
        
        st.write(f"🔹 Selected Language Code: `{selected_lang}`")
        st.write(f"🔹 Selected Voice: `{selected_voice}`")
    else:
        st.warning(f"No voices available for {selected_lang_name}.")
        return
    # User input
    text = st.text_area("Put voice to your text 🎙️:", "Hello, world!")

    # Generate speech on button click
    if st.button("Generate Speech"):
        if text.strip():
            with st.spinner("Preparing voice model..."):
                # Download voice file if not already present
                voice_file_path = download_voice_file(selected_voice)
                
                if voice_file_path:
                    try:
                        # Load the text-to-speech model with the specific voice
                        pipeline = KPipeline(lang_code=selected_lang)
                        
                        with st.spinner("Generating audio..."):
                            generator = pipeline(text, voice=selected_voice)
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
                    except Exception as e:
                        st.error(f"Error in speech generation: {e}")
                else:
                    st.error("Failed to download voice file.")
        else:
            st.error("Please enter some text to generate speech.")

if __name__ == "__main__":
    main()