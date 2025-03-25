import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch


# Load the Stable Diffusion model
model_id = "sd-legacy/stable-diffusion-v1-5"

pipe = pipe.to("cpu")

# Streamlit user input for the prompt
st.title("Stable Diffusion with Streamlit")
st.write("Generate an image based on a text prompt.")

# Prompt input field
prompt = st.text_input("Enter your prompt:", "a photo of a beautiful smile")

if st.button("Generate Image"):
    # Generate the image
    with st.spinner("Generating..."):
        image = pipe(prompt).images[0]  
    
    # Display the generated image
    st.image(image, caption="Generated Image", use_column_width=True)

