import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the Stable Diffusion model
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Streamlit user input for the prompt
st.title("Stable Diffusion with Streamlit")
st.write("Generate an image based on a text prompt.")

# Prompt input field
prompt = st.text_input("Enter your prompt:", "a photo of an astronaut riding a horse on mars")

if st.button("Generate Image"):
    # Generate the image
    with st.spinner("Generating..."):
        image = pipe(prompt).images[0]  
    
    # Display the generated image
    st.image(image, caption="Generated Image", use_column_width=True)

