import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

def load_model(model_name="facebook/nllb-200-distilled-600M"):
    """Load translation model and tokenizer with simple configuration"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Simple model loading without advanced parameters
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None
        )
    except ImportError:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # MODEL
        if torch.cuda.is_available():
            model = model.to("cuda")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    return model, tokenizer

def translate(text, source_lang, target_lang, model, tokenizer):
    """Translate text from source language to target language"""
    tokenizer.src_lang = source_lang
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=200
    )
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def roundtrip_translate(text, source_lang, target_lang, model, tokenizer):
    """Perform round-trip translation and return all results"""
    # First translation: source -> target
    forward_translation = translate(text, source_lang, target_lang, model, tokenizer)
    
    # Second translation: target -> source (back translation)
    back_translation = translate(forward_translation, target_lang, source_lang, model, tokenizer)
    
    return {
        "original": text,
        "forward_translation": forward_translation,
        "back_translation": back_translation
    }

# Language codes and names for NLLB
language_options = {
    "eng_Latn": "English",
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "cmn_Hans": "Chinese (Simplified)",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "ara_Arab": "Arabic",
    "hin_Deva": "Hindi"
}

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Create Gradio interface
    with gr.Blocks(title="Round-Trip Translation Demo") as demo:
        gr.Markdown("# Translation Round-Trip Demo")
        gr.Markdown("See how well the model preserves meaning by translating text to another language and back.")
        
        with gr.Row():
            source_lang = gr.Dropdown(
                choices=list(language_options.items()), 
                label="Source Language",
                value="eng_Latn"
            )
            target_lang = gr.Dropdown(
                choices=list(language_options.items()), 
                label="Target Language",
                value="spa_Latn"
            )
        
        input_text = gr.Textbox(
            label="Original Text", 
            placeholder="Enter text to translate...",
            lines=3
        )
        
        translate_btn = gr.Button("Translate")
        
        with gr.Row():
            with gr.Column():
                target_translation = gr.Textbox(label="Translation", lines=3)
            with gr.Column():
                back_translation = gr.Textbox(label="Back-Translation", lines=3)
        
        def process_translation(text, src_lang, tgt_lang):
            results = roundtrip_translate(text, src_lang, tgt_lang, model, tokenizer)
            return results["forward_translation"], results["back_translation"]
        
        translate_btn.click(
            fn=process_translation,
            inputs=[input_text, source_lang, target_lang],
            outputs=[target_translation, back_translation]
        )
        
        # Example inputs
        examples = [
            ["Hello, how are you today? I hope you're doing well.", "eng_Latn", "spa_Latn"],
            ["The quick brown fox jumps over the lazy dog.", "eng_Latn", "fra_Latn"],
            ["Translation models have improved significantly in recent years.", "eng_Latn", "deu_Latn"],
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[input_text, source_lang, target_lang]
        )
    
    # Launch the interface
    demo.launch(share=False)