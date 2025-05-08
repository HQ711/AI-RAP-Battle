# main.py
import ui
import lyrics
import music
import tempfile, os

generated_lyrics = {"rap1": "", "rap2": ""}

def generate_battle_lyrics(part1, part2, model_name):
    rap1, rap2 = lyrics.generate_lyrics(model_name, part1, part2)
    generated_lyrics["rap1"], generated_lyrics["rap2"] = rap1, rap2
    return rap1, rap2

def generate_song():
    rap1 = generated_lyrics["rap1"]
    rap2 = generated_lyrics["rap2"]

    if not rap1 or not rap2:
        return "Please generate lyrics first.", "Please generate lyrics first."

    song1 = music.generate_rap_song(rap1)
    song2 = music.generate_rap_song(rap2)

    return song1, song2

def cleanup():
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.endswith((".wav", ".mp3")):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass

if __name__ == "__main__":
    demo = ui.create_ui(generate_battle_lyrics, generate_song)
    try:
        demo.launch(server_port=7860)
    finally:
        cleanup()
