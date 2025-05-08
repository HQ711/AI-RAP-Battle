# music.py
import torch
import tempfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.io import wavfile
from pydub import AudioSegment
from bark import preload_models, generate_audio

device = "cuda" if torch.cuda.is_available() else "cpu"

# preload bark models at the start
preload_models()

music_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)

def generate_melody(lyrics, duration=20):
    short_lyrics = lyrics[:200]
    inputs = music_processor(text=[f"hiphop instrumental with lyrics: {short_lyrics}"], padding=True, return_tensors="pt").to(device)
    audio = music_model.generate(**inputs, max_new_tokens=int(duration * 50))
    melody_path = tempfile.mktemp(suffix=".wav")
    wavfile.write(melody_path, 32000, audio[0, 0].cpu().numpy())
    return melody_path

def generate_vocal_with_bark(lyrics):
    audio_array = generate_audio(lyrics[:200], history_prompt="v2/en_speaker_6")
    vocal_path = tempfile.mktemp(suffix=".wav")
    wavfile.write(vocal_path, rate=24000, data=audio_array)
    return vocal_path

def generate_rap_song(lyrics):
    lyrics = lyrics[:200]
    melody_path = generate_melody(lyrics)
    vocal_path = generate_vocal_with_bark(lyrics)

    music = AudioSegment.from_wav(melody_path).set_frame_rate(24000) - 12
    vocal = AudioSegment.from_wav(vocal_path)

    duration = min(len(vocal), len(music))
    mixed = vocal[:duration].overlay(music[:duration])

    song_path = tempfile.mktemp(suffix="_song.wav")
    mixed.export(song_path, format="wav")
    return song_path
