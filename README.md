#  AI Rap Battle Generator

This project creates interactive **AI Rap Battles**, enabling users to generate rhymed lyrics and synthesize complete rap songs (vocals and background music) using advanced AI models.

---

##  Features

- **Automatic Lyrics Generation**:
  - Provide two topics, and AI generates rhymed rap lyrics.
  - Multiple selectable AI lyric-generation models.

- **Realistic Vocal Synthesis**:
  - Convert generated lyrics to realistic AI-generated vocals (using **Bark**).

- **AI-generated Background Music**:
  - Original hip-hop instrumentals based on generated lyrics (using **MusicGen**).

- **Interactive UI**:
  - User-friendly web interface built with **Gradio**.

---

##  Tools, Models, and APIs

### **AI Models**

- **Lyrics**:
  - [EleutherAI GPT-Neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B)
  - [dzionek/distilgpt2-rap](https://huggingface.co/dzionek/distilgpt2-rap)
  - [OpenAI GPT-2](https://huggingface.co/gpt2)

- **Vocals**:
  - [Bark (Suno AI)](https://github.com/suno-ai/bark)

- **Music Generation**:
  - [MusicGen Small (Meta AI)](https://huggingface.co/facebook/musicgen-small)

### **Frameworks and Libraries**

- [Gradio](https://gradio.app/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch](https://pytorch.org/)
- [PyDub](https://github.com/jiaaro/pydub)
- [SciPy](https://scipy.org/)

---

##  Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


### Step 2: Set Up Python Environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

### Step 3: Install Dependencies
pip install gradio transformers torch scipy pydub bark


##  Run the App
python main.py

The app will launch a web interface accessible at:
http://127.0.0.1:7860


##  Usage
Enter two opposing rap battle topics.

Select your preferred lyrics-generation model from the dropdown.

Click Generate Lyrics to create unique rap lyrics.

Once lyrics are generated, click Generate Song to synthesize full rap battle songs.

Listen and enjoy!