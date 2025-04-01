import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

def load_models():
    """Load translation models and tokenizers for multilingual translation"""
    print("Loading models...")
    
    model_to_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    tokenizer_to_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    
    model_from_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
    tokenizer_from_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_to_en = model_to_en.to(device)
    model_from_en = model_from_en.to(device)
    
    print(f"Using device: {device}")
    
    return model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en

def translate(text, model, tokenizer):
    """Translate text using the given model and tokenizer"""
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    translated_tokens = model.generate(**inputs, max_length=200)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def roundtrip_translate(text, model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en):
    """Perform round-trip translation using two models"""
    forward_translation = translate(text, model_to_en, tokenizer_to_en)
    back_translation = translate(forward_translation, model_from_en, tokenizer_from_en)
    
    return {
        "original": text,
        "forward_translation": forward_translation,
        "back_translation": back_translation
    }

# Load models and tokenizers
model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en = load_models()

def main():
    with gr.Blocks(title="Multilingual Round-Trip Translation") as demo:
        gr.Markdown("# Multilingual Translation Round-Trip Demo")
        gr.Markdown("Translate text to English and back to its original language to test translation quality.")
        
        input_text = gr.Textbox(label="Original Text", placeholder="Enter text to translate...", lines=3)
        translate_btn = gr.Button("Translate")
        
        with gr.Row():
            with gr.Column():
                target_translation = gr.Textbox(label="Translation to English", lines=3)
            with gr.Column():
                back_translation = gr.Textbox(label="Back-Translation", lines=3)
        
        def process_translation(text):
            results = roundtrip_translate(text, model_to_en, tokenizer_to_en, model_from_en, tokenizer_from_en)
            return results["forward_translation"], results["back_translation"]
        
        translate_btn.click(
            fn=process_translation,
            inputs=[input_text],
            outputs=[target_translation, back_translation]
        )
        
        demo.launch(share=True)
