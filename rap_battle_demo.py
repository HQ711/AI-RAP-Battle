import gradio as gr
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import torch
import tempfile
import os
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from bark import preload_models, generate_audio
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

############################################################################################
#                               1. Lyric generation
############################################################################################
lyrics_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=device)

generated_lyrics = {"rap1": "", "rap2": ""}

model_cache = {}

def load_lyrics_model(model_name):
    if model_name in model_cache:
        return model_cache[model_name]

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
        return "\n".join(["I'm unmatched, the greatest you'll see,"] * 10)
    while len(clean_lines) < 10:
        clean_lines.append("I'm unmatched, the greatest you'll see,")
    return "\n".join(clean_lines[:10])

def generate_lyrics(prompt):
    output = lyrics_generator(
        prompt,
        max_length=250,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.5,
        pad_token_id=lyrics_generator.tokenizer.eos_token_id,
    )[0]["generated_text"]
    return clean_lyrics(output, prompt)

def generate_battle_lyrics(part1, part2, model_name):
        global lyrics_generator
        lyrics_generator = load_lyrics_model(model_name)

        part1 = part1.strip() or "cat"
        part2 = part2.strip() or "dog"

        prompt1 = (
            f"This is a rap battle. I'm a {part1}, better than a {part2}. "
            f"Here are exactly 10 rhyming rap lines explaining clearly why I'm superior:\n"
        )

        prompt2 = (
            f"This is a rap battle. I'm a {part2}, better than a {part1}. "
            f"Here are exactly 10 rhyming rap lines explaining clearly why I'm superior:\n"
        )

        rap1 = generate_lyrics(prompt1)
        rap2 = generate_lyrics(prompt2)

        generated_lyrics["rap1"] = rap1
        generated_lyrics["rap2"] = rap2

        return rap1, rap2, None, None

############################################################################################
#                                   2. Music generation
############################################################################################
music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

def generate_melody(lyrics, duration=20):
    short_lyrics = lyrics[:200]
    inputs = music_processor(
        text=[f"hiphop instrumental with lyrics: {short_lyrics}"],
        padding=True,
        return_tensors="pt",
    ).to(device)
    audio = music_model.generate(**inputs, max_new_tokens=int(duration * 50))
    melody_path = tempfile.mktemp(suffix=".wav")
    wavfile.write(melody_path, 32000, audio[0, 0].cpu().numpy())
    return melody_path

def generate_vocal_with_bark(lyrics):
    audio_array = generate_audio(lyrics, history_prompt="v2/en_speaker_6")
    vocal_path = tempfile.mktemp(suffix=".wav")
    wavfile.write(vocal_path, rate=24000, data=audio_array)
    return vocal_path

def generate_rap_song(lyrics):
    lyrics = lyrics[:200]
    try:
        melody_path = generate_melody(lyrics)
        vocal_path = generate_vocal_with_bark(lyrics)

        music = AudioSegment.from_wav(melody_path).set_frame_rate(24000) - 12
        vocal = AudioSegment.from_wav(vocal_path)

        duration = min(len(vocal), len(music))
        mixed = vocal[:duration].overlay(music[:duration])

        song_path = tempfile.mktemp(suffix="_song.wav")
        mixed.export(song_path, format="wav")
        return song_path
    except Exception as e:
        print(f"Fail to generate: {e}")
        return None

def generate_song():
    rap1 = generated_lyrics["rap1"]
    rap2 = generated_lyrics["rap2"]

    if not rap1 or not rap2:
        return "Please generate lyrics first.", "Please generate lyrics first."

    song1 = generate_rap_song(rap1) or "Failed to generate"
    song2 = generate_rap_song(rap2) or "Failed to generate"

    return song1, song2


############################################################################################
#                                           Gradio UI
############################################################################################
with gr.Blocks(title="AI RAP Battle") as demo:
    gr.Markdown("# AI RAP Battle")

    with gr.Row():
        topic1 = gr.Textbox(label="Topic 1", placeholder="default: cat", value="")
        vs_mark = gr.Markdown("<h2 style='text-align:center;'>VS</h2>")
        topic2 = gr.Textbox(label="Topic 2", placeholder="default: dog", value="")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                ["EleutherAI/gpt-neo-1.3B", "dzionek/distilgpt2-rap", "gpt2"],
                value="EleutherAI/gpt-neo-1.3B",
                label="Choose Lyric Generation Model",
            )
        with gr.Column(scale=1):
            generate_lyrics_btn = gr.Button("Generate Lyrics", variant="primary")


    with gr.Row():
        with gr.Column():
            gr.Markdown("### Song 1")
            lyrics1 = gr.Textbox(label="Lyrics")
        with gr.Column():
            gr.Markdown("### Song 2")
            lyrics2 = gr.Textbox(label="Lyrics")
    
    with gr.Row():
        generate_song_btn = gr.Button("Generate Song", variant="primary")
    
    with gr.Row():
        with gr.Column():
            song1 = gr.Audio(label="song", type="filepath")
        with gr.Column():
            song2 = gr.Audio(label="song", type="filepath")
    
    generate_lyrics_btn.click(
        fn=generate_battle_lyrics,
        inputs=[topic1, topic2, model_dropdown],
        outputs=[lyrics1, lyrics2]
    )

    generate_song_btn.click(
        fn=generate_song,
        inputs=None,
        outputs=[song1, song2]
    )

# Cleanup function to remove temporary files
def cleanup():
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.endswith((".wav", ".mp3")):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass

if __name__ == "__main__":
    try:
        demo.launch(server_port=7860)
    finally:
        cleanup()