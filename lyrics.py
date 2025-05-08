# lyrics.py
import re
from transformers import pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_cache = {}

def load_lyrics_model(model_name):
    if model_name not in model_cache:
        model_cache[model_name] = pipeline("text-generation", model=model_name, device=device)
    return model_cache[model_name]

def clean_lyrics(text, prompt):
    text = text.replace(prompt, "").strip()
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'^\d+\.\s*', '', line)
        if len(line.split()) >= 3:
            clean_line = re.sub(r'[^a-zA-Z0-9\s\'\,\.\!\?]', '', line)
            clean_lines.append(clean_line)
        if len(clean_lines) >= 10:
            break
    if not clean_lines:
        clean_lines = ["I'm unmatched, the greatest you'll see,"] * 10
    while len(clean_lines) < 10:
        clean_lines.append("I'm unmatched, the greatest you'll see,")
    return "\n".join(clean_lines[:10])

def generate_lyrics(model_name, part1, part2):
    generator = load_lyrics_model(model_name)

    prompt1 = f"This is a rap battle. I'm a {part1}, better than a {part2}. Here are exactly 10 rhyming rap lines explaining clearly why I'm superior:\n"
    prompt2 = f"This is a rap battle. I'm a {part2}, better than a {part1}. Here are exactly 10 rhyming rap lines explaining clearly why I'm superior:\n"

    rap1 = generator(prompt1, max_length=250, temperature=0.8, top_p=0.9, repetition_penalty=1.5, do_sample=True, pad_token_id=generator.tokenizer.eos_token_id)[0]["generated_text"]
    rap2 = generator(prompt2, max_length=250, temperature=0.8, top_p=0.9, repetition_penalty=1.5, do_sample=True, pad_token_id=generator.tokenizer.eos_token_id)[0]["generated_text"]

    return clean_lyrics(rap1, prompt1), clean_lyrics(rap2, prompt2)
