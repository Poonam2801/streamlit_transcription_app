# 🕉️ Spiritual Content Transcriber

> Transcribe YouTube spiritual discourses in **Hindi · English · Sanskrit** — powered by OpenAI Whisper, accelerated on Apple Silicon.

---

## What It Does

Paste a YouTube link, hit **Transcribe**, and get a clean transcript of any spiritual talk, satsang, pravachan, or discourse — even when the speaker mixes Hindi, English, and Sanskrit shlokas mid-sentence.

The app understands the domain. It's seeded with hundreds of Sanskrit and spiritual terms so Whisper doesn't mishear *Bhagavad Gita* as *Bhagavad Geeta* or *Samadhi* as *Sama Dhi*.

---

## Features

- 🎙️ **Multilingual** — handles Hindi, English, and Sanskrit in the same audio stream
- 🍎 **M2 accelerated** — uses `mlx-whisper` on Apple Silicon for 8–10x faster transcription vs CPU
- 📄 **Three download formats** — plain `.txt`, subtitles `.srt` (uploadable to YouTube), and timestamped `.txt`
- 🔍 **Timestamped segments** — every line tagged with `[MM:SS]` so you can find any moment instantly
- 📿 **Sanskrit corrections** — post-processing dictionary fixes common Whisper mishearings of spiritual terms
- ⚙️ **Model selector** — choose from `large-v3` down to `tiny` depending on your speed/accuracy needs
- 🔁 **Auto language detection** — no need to specify the language for mixed-language content

---

## Requirements

- Python 3.10+
- ffmpeg installed on your system
- Apple Silicon Mac recommended (M1/M2/M3)

---

## Installation

**1. Clone or download the project**

```bash
cd your-project-folder
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Install ffmpeg** (required for audio extraction)

```bash
brew install ffmpeg
```

---

## Running the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. Paste a YouTube URL into the input box
2. Choose your model and language settings in the sidebar
3. Click **🎙️ Transcribe**
4. Wait for the three steps to complete (download → load model → transcribe)
5. Read the transcript in the code block
6. Download as `.txt`, `.srt`, or timestamped `.txt`

---

## Model Guide

| Model | Speed on M2 | Speed on CPU | Best For |
|-------|-------------|--------------|----------|
| `large-v3` | ~4 min / hr | ~45 min / hr | Best accuracy, Sanskrit shlokas |
| `medium` | ~2 min / hr | ~20 min / hr | Good balance for most talks |
| `small` | ~1 min / hr | ~8 min / hr | Quick drafts, mostly English |
| `tiny` | ~30 sec / hr | ~3 min / hr | Testing only |

**Recommendation:** Use `large-v3` with language set to **Auto-detect** for the best results on spiritual content.

---

## Language Settings

| Setting | When to Use |
|---------|-------------|
| 🔍 Auto-detect | Mixed Hindi + English + Sanskrit (recommended) |
| 🇮🇳 Hindi | Pure Hindi discourses |
| 🇬🇧 English | Pure English talks |
| 📿 Sanskrit | Chanting or Vedic recitation |

Enable **Translate to English** to get an English translation of any Hindi or Sanskrit content.

---

## Output Formats

### Plain `.txt`
Full transcript as clean running text. Best for reading, editing, or pasting into documents.

### Subtitles `.srt`
Standard subtitle format with precise timestamps. Can be uploaded directly to YouTube as captions under **YouTube Studio → Subtitles**.

### Timestamped `.txt`
Each segment prefixed with `[MM:SS]` — useful for referencing specific moments in a talk or creating study notes.

---

## Apple Silicon Speed Setup

The app auto-detects if you're on Apple Silicon. For maximum speed make sure `mlx-whisper` is installed:

```bash
pip install mlx-whisper mlx
```

You'll see **🍎 Apple M2 (mlx)** in the caption bar when hardware acceleration is active. If it shows **💻 CPU**, run the above install command and restart the app.

---

## Project Structure

```
your-project/
├── app.py              # Streamlit UI + transcription logic
├── menubar.py          # Mac menu bar background app (optional)
├── run.sh              # One-click launcher script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Troubleshooting

**`zsh: command not found: python`**
Use `python3` instead. On Mac, `python` is not aliased by default.

**`ffmpeg not found` error**
Install it with: `brew install ffmpeg`

**Download fails for a video**
The video may be private, age-restricted, or region-locked. Try a different video to confirm the setup works.

**Transcription is slow**
Make sure `mlx-whisper` is installed and the app shows the 🍎 badge. If you're on CPU, switch to `medium` model for faster results.

**Sanskrit terms are incorrect**
The corrections dictionary covers the most common mishearings. You can add your own corrections in `app.py` in the `SANSKRIT_CORRECTIONS` dictionary at the top of the file.

---

## Running as a Menu Bar App (Optional)

To keep the transcriber running silently in your Mac menu bar:

```bash
pip install rumps
python menubar.py
```

A 🕉️ icon appears in your menu bar. Click it to open the transcriber or start/stop the server.

**Auto-start at login:**
```bash
python menubar.py --install
```

---

## Built With

- [Streamlit](https://streamlit.io) — UI framework
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Apple Silicon accelerated Whisper
- [openai-whisper](https://github.com/openai/whisper) — CPU fallback
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube audio download
- [ffmpeg](https://ffmpeg.org) — Audio processing
- [rumps](https://github.com/jaredks/rumps) — Mac menu bar integration

---

*Built for transcribing spiritual content — may your words be heard clearly.* 🙏
