# -*- coding: utf-8 -*-
"""Final_project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dcnwxohFuRy-Hfz7kx3Ijsj7GSQ8ptFs
"""

import gradio as gr
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import torch
import tempfile
import os
import tempfile
import torch
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import gradio as gr

# ✅ Fix for PyTorch >=2.6 + Bark compatibility
from torch import serialization
from numpy.core.multiarray import scalar
serialization.add_safe_globals([scalar])

# ⛑ Monkey patch for torch.load
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

# ✅ Bark load after patching
from bark import preload_models, generate_audio
preload_models()

# ✅ Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ GPT-neo for rhymed lyrics
lyrics_generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-1.3B",
    device=0 if device == "cuda" else -1
)

# ✅ 背景音乐模型（MusicGen）
music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

# ✅ 生成歌词（押韵 + 风格化）
def generate_rap_verse(topic):
    prompt = f"Write a rhyming rap verse about {topic}:\n"
    output = lyrics_generator(
        prompt,
        max_length=120,
        pad_token_id=lyrics_generator.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )[0]["generated_text"]

    generated = output.replace(prompt, "").strip()
    lines = generated.split("\n")
    clean_lines = []
    for line in lines:
        if line.strip() == "" or len(line.strip()) < 3:
            break
        clean_lines.append(line.strip())
    return "\n".join(clean_lines).strip()

# ✅ Bark 语音合成
def generate_vocal_with_bark(lyrics):
    audio_array = generate_audio(lyrics, history_prompt="v2/en_speaker_6")
    vocal_path = tempfile.mktemp(suffix=".wav")
    wavfile.write(vocal_path, rate=24000, data=audio_array)
    return vocal_path

# ✅ MusicGen 背景音乐合成
def generate_melody(lyrics, duration=15):
    inputs = music_processor(
        text=[f"hiphop instrumental with lyrics: {lyrics}"],
        padding=True,
        return_tensors="pt",
    ).to(device)

    audio = music_model.generate(**inputs, max_new_tokens=int(duration * 50))
    melody_path = tempfile.mktemp(suffix=".wav")
    wavfile.write(melody_path, 32000, audio[0, 0].cpu().numpy())
    return melody_path

# ✅ 混音人声和伴奏
def generate_rap_song(lyrics):
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
        print(f"生成失败: {e}")
        return None

# ✅ 对抗逻辑（歌词 + 歌曲）
def generate_rap_battle(topic):
    rap1 = generate_rap_verse(topic + " - part 1")
    rap2 = generate_rap_verse(topic + " - part 2")

    song1 = generate_rap_song(rap1) or "Failed to generate"
    song2 = generate_rap_song(rap2) or "Failed to generate"

    return rap1, rap2, song1, song2

# ✅ Gradio 前端 UI
with gr.Blocks(title="AI RAP BATTLE") as demo:
    gr.Markdown("# 🎤 AI RAP BATTLE\nGenerate lyrics, vocals (Bark), and background music (MusicGen)")

    with gr.Row():
        topic = gr.Textbox(label="Rap Battle Topic", placeholder="e.g., AI vs Humans")
        btn = gr.Button("🔥 Generate Battle", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧠 AI Rapper 1")
            lyrics1 = gr.Textbox(label="Lyrics")
            song1 = gr.Audio(label="Rap Track 1", type="filepath")
        with gr.Column():
            gr.Markdown("### 🤖 AI Rapper 2")
            lyrics2 = gr.Textbox(label="Lyrics")
            song2 = gr.Audio(label="Rap Track 2", type="filepath")

    btn.click(
        fn=generate_rap_battle,
        inputs=topic,
        outputs=[lyrics1, lyrics2, song1, song2]
    )

# ✅ 清理临时音频文件
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
        demo.launch()
    finally:
        cleanup()

pip install git+https://github.com/suno-ai/bark.git

pip install gradio

