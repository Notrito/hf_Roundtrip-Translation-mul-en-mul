import streamlit as st
import torch
import soundfile as sf
from kokoro import KPipeline
from kokoro.pipeline import LANG_CODES


# Streamlit UI
st.title("Kokoro-TTS all languages")
st.markdown("## Text to Speech")
st.markdown("Mix text and selected languages for some funny accents")

# Create language options dictionary
lang_options = {f"{name}": code for code, name in LANG_CODES.items()}

# Streamlit language selection
selected_lang = st.selectbox("Select language", list(lang_options.keys()))
code_lang = lang_options[selected_lang]

# Print language confirmation
# st.write(f"🔹 Selected Language: {selected_lang}")
# st.write(f"🔹 Language Code: `{code_lang}`")

# Define voice options based on selected language
voice_options = {
    "a": {
        "Heart": "af_heart", "Alloy": "af_alloy", "Aoede": "af_aoede",
        "Bella": "af_bella", "Jessica": "af_jessica", "Kore": "af_kore",
        "Nicole": "af_nicole", "Nova": "af_nova", "River": "af_river",
        "Sarah": "af_sarah", "Sky": "af_sky",
        "Adam": "am_adam", "Echo": "am_echo", "Eric": "am_eric",
        "Fenrir": "am_fenrir", "Liam": "am_liam", "Michael": "am_michael",
        "Onyx": "am_onyx", "Puck": "am_puck", "Santa": "am_santa"
    },
    "b": {
        "Alice": "bf_alice", "Emma": "bf_emma", "Isabella": "bf_isabella", "Lily": "bf_lily",
        "Daniel": "bm_daniel", "Fable": "bm_fable", "George": "bm_george", "Lewis": "bm_lewis"
    },
    "j": {
        "Alpha": "jf_alpha", "Gongitsune": "jf_gongitsune", "Nezumi": "jf_nezumi", 
        "Tebukuro": "jf_tebukuro", "Kumo": "jm_kumo"
    },
    "z": {
        "Xiaobei": "zf_xiaobei", "Xiaoni": "zf_xiaoni", "Xiaoxiao": "zf_xiaoxiao", "Xiaoyi": "zf_xiaoyi",
        "Yunjian": "zm_yunjian", "Yunxi": "zm_yunxi", "Yunxia": "zm_yunxia", "Yunyang": "zm_yunyang"
    },
    "e": {
        "Dora": "ef_dora", "Alex": "em_alex", "Santa": "em_santa"
    },
    "f": {
        "Siwis": "ff_siwis"
    },
    "h": {
        "Alpha": "hf_alpha", "Beta": "hf_beta", "Omega": "hm_omega", "Psi": "hm_psi"
    },
    "i": {
        "Sara": "if_sara", "Nicola": "im_nicola"
    },
    "p": {
        "Dora": "pf_dora", "Alex": "pm_alex", "Santa": "pm_santa"
    }
}


# Get available voices for selected language
available_voices = voice_options.get(code_lang, {})

# Show second dropdown only if the first selection is valid
if available_voices:
    name_voice = st.selectbox("Select a voice", list(available_voices.keys()))
    selected_voice = available_voices[name_voice] # VOICE

    # Print voice confirmation
    # st.write(f"🔹 Selected Voice Code: `{selected_voice}`")
else:
    st.error("No voices available for the selected language.")
    
# Load the text-to-speech model
@st.cache_resource
def load_pipeline():
    return KPipeline(lang_code=code_lang)  # LANGUAGE

pipeline = load_pipeline()



# User input
text = st.text_area("Put voice to your text 🎙️:", "Here your text")

# Generate speech on button click
if st.button("Generate Speech"):
    if text.strip():
        with st.spinner("Generating audio..."):
            generator = pipeline(text, voice=selected_voice )  # CHANGE VOICE

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