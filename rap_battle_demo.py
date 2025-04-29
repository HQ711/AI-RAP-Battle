import gradio as gr
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import torch
import tempfile
import os
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from gtts import gTTS

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Lyric generation
lyrics_generator = pipeline("text-generation", model="gpt2", device=device)

# 2. Music generation
music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

def generate_melody(lyrics, duration=15):
    """Generate melody using MusicGen"""
    inputs = music_processor(
        text=[f"hiphop instrumental with lyrics: {lyrics}"],
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    audio = music_model.generate(**inputs, max_new_tokens=int(duration*50))
    melody_path = tempfile.mktemp(suffix=".wav")
    wavfile.write(melody_path, 32000, audio[0,0].cpu().numpy())
    return melody_path

def generate_vocal(lyrics):
    """generate vocal using gTTS"""
    vocal_path = tempfile.mktemp(suffix=".mp3")
    tts = gTTS(text=lyrics, lang='en')
    tts.save(vocal_path)
    
    # Convert mp3 to wav
    wav_path = tempfile.mktemp(suffix=".wav")
    AudioSegment.from_mp3(vocal_path).export(wav_path, format="wav")
    os.remove(vocal_path)
    return wav_path

def generate_rap_song(lyrics):
    """generate rap song"""
    try:
        # 1. generate melody
        melody_path = generate_melody(lyrics)
        
        # 2. generate vocal
        vocal_path = generate_vocal(lyrics)
        
        # 3. mix vocal and melody
        vocal = AudioSegment.from_wav(vocal_path)
        music = AudioSegment.from_wav(melody_path) - 12  # 背景音乐降音量
        
        # 4. overlay
        duration = min(len(vocal), len(music))
        mixed = vocal[:duration].overlay(music[:duration])
        
        song_path = tempfile.mktemp(suffix="_song.wav")
        mixed.export(song_path, format="wav")
        return song_path
    
    except Exception as e:
        print(f"生成失败: {e}")
        return None

def generate_rap_battle(topic):
    # generate rap lyrics
    rap1 = lyrics_generator(
        f"Write a rap verse about {topic}, make it rhyme:",
        max_length=50,
        pad_token_id=lyrics_generator.tokenizer.eos_token_id,
        truncation=True
    )[0]["generated_text"]
    
    rap2 = lyrics_generator(
        f"Write a counter rap verse about {topic}, make it rhyme:",
        max_length=50,
        pad_token_id=lyrics_generator.tokenizer.eos_token_id,
        truncation=True
    )[0]["generated_text"]
    
    # generate rap songs
    song1 = generate_rap_song(rap1) or "Failed to generate"
    song2 = generate_rap_song(rap2) or "Failed to generate"
    
    return rap1, rap2, song1, song2

# Gradio UI
with gr.Blocks(title="AI RAP Battle") as demo:
    gr.Markdown("# AI RAP Battle")
    with gr.Row():
        topic = gr.Textbox(label="Battle Topic", placeholder="e.g., cats vs dogs")
        btn = gr.Button("Generate", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Song 1")
            lyrics1 = gr.Textbox(label="Lyrics")
            song1 = gr.Audio(label="song", type="filepath")
        with gr.Column():
            gr.Markdown("### Song 2")
            lyrics2 = gr.Textbox(label="Lyrics")
            song2 = gr.Audio(label="song", type="filepath")
    
    btn.click(
        fn=generate_rap_battle,
        inputs=topic,
        outputs=[lyrics1, lyrics2, song1, song2]
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