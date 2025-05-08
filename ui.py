# ui.py
import gradio as gr

def create_ui(generate_lyrics_fn, generate_song_fn):
    with gr.Blocks(title="AI RAP Battle") as demo:
        gr.Markdown("# 🎤 AI RAP Battle")

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
            lyrics1 = gr.Textbox(label="Lyrics 1")
            lyrics2 = gr.Textbox(label="Lyrics 2")

        generate_song_btn = gr.Button("Generate Song", variant="primary")

        with gr.Row():
            song1 = gr.Audio(label="song 1", type="filepath")
            song2 = gr.Audio(label="song 2", type="filepath")

        generate_lyrics_btn.click(
            fn=generate_lyrics_fn,
            inputs=[topic1, topic2, model_dropdown],
            outputs=[lyrics1, lyrics2]
        )

        generate_song_btn.click(
            fn=generate_song_fn,
            inputs=None,
            outputs=[song1, song2]
        )

    return demo
