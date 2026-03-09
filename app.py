"""
YouTube Spiritual Content Transcriber — Streamlit UI
=====================================================
Run:
    streamlit run app.py

Install requirements first:
    pip install streamlit openai-whisper yt-dlp torch
    # ffmpeg must also be installed on your system:
    # macOS:   brew install ffmpeg
    # Ubuntu:  sudo apt install ffmpeg
    # Windows: https://ffmpeg.org/download.html
"""

import os
import re
import json
import tempfile
import threading
from pathlib import Path

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spiritual Transcriber",
    page_icon="🕉️",
    layout="centered",
)

st.title("🕉️ Spiritual Content Transcriber")
st.caption(
    "Transcribes YouTube videos in **Hindi · English · Sanskrit** using OpenAI Whisper large-v3"
)

# ─────────────────────────────────────────────────────────────────────────────
# Spiritual post-processing dictionary
# ─────────────────────────────────────────────────────────────────────────────

SANSKRIT_CORRECTIONS = {
    "bhagavad geeta": "Bhagavad Gita",
    "bhagwad geeta": "Bhagavad Gita",
    "bhagvad gita": "Bhagavad Gita",
    "upnishad": "Upanishad",
    "upnishads": "Upanishads",
    "prana yama": "Pranayama",
    "pranayam": "Pranayama",
    "sama dhi": "Samadhi",
    "sam sara": "Samsara",
    "nirvan": "Nirvana",
    "brhaman": "Brahman",
    "brahmand": "Brahmand",
    "aatman": "Atman",
    "ishwar": "Ishwara",
    "om shanti": "Om Shanti",
    "om shakti": "Om Shakti",
    "satsangh": "Satsang",
    "kirtan": "Kirtan",
    "aarti": "Aarti",
}

SPIRITUAL_PROMPT = (
    "This is a spiritual discourse in Hindi and English, often mixing Sanskrit shlokas, "
    "mantras, and verses from the Bhagavad Gita, Upanishads, and Vedas. "
    "Common Sanskrit terms: Om, Brahman, Atman, Dharma, Karma, Moksha, Guru, Shakti, "
    "Pranayama, Samadhi, Samsara, Nirvana, Ahimsa, Satya, Shanti, Ananda, Maya, "
    "Shloka, Mantra, Namaste, Namaskar, Ishwara, Bhakti, Jnana, Vairagya, Viveka, "
    "Sadhana, Tapas, Seva, Satsang, Kirtan, Aarti, Ashram, Guruji."
)


def post_process(text: str) -> str:
    for wrong, right in SANSKRIT_CORRECTIONS.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Audio download
# ─────────────────────────────────────────────────────────────────────────────

def download_audio(youtube_url: str, output_dir: str) -> tuple[str, str]:
    """
    Download best audio from YouTube.
    Returns (audio_path, video_title).
    Raises RuntimeError on failure.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id", "audio")
        title = info.get("title", "transcript")

    # Locate the downloaded wav
    audio_path = os.path.join(output_dir, f"{video_id}.wav")
    if not os.path.exists(audio_path):
        wavs = list(Path(output_dir).glob("*.wav"))
        if not wavs:
            raise RuntimeError(
                "Audio download failed: no .wav file found. "
                "Make sure ffmpeg is installed on your system."
            )
        audio_path = str(wavs[0])

    return audio_path, title


# ─────────────────────────────────────────────────────────────────────────────
# Whisper transcription
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_size: str):
    """Cache the model so it's only loaded once per session."""
    try:
        import whisper
    except ImportError:
        raise RuntimeError("openai-whisper is not installed. Run: pip install openai-whisper")
    return whisper.load_model(model_size)


def run_transcription(model, audio_path: str, language: str | None, translate: bool) -> dict:
    """Run Whisper with settings tuned for spiritual mixed-language content."""
    options = {
        "task": "translate" if translate else "transcribe",
        "initial_prompt": SPIRITUAL_PROMPT,
        "word_timestamps": True,
        "verbose": False,
        "beam_size": 5,
        "best_of": 5,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
    }
    if language and language != "auto":
        options["language"] = language

    return model.transcribe(audio_path, **options)


# ─────────────────────────────────────────────────────────────────────────────
# Output formatters
# ─────────────────────────────────────────────────────────────────────────────

def to_plain_txt(result: dict) -> str:
    """Full transcript as plain text."""
    return post_process(result.get("text", "").strip())


def to_srt(result: dict) -> str:
    """Convert Whisper segments to SRT subtitle format."""
    lines = []
    for i, seg in enumerate(result.get("segments", []), start=1):
        start = seg["start"]
        end = seg["end"]
        text = post_process(seg["text"].strip())

        def fmt(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int(round((t - int(t)) * 1000))
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        lines.append(str(i))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def to_timestamped_txt(result: dict) -> str:
    """Plain text with [MM:SS] timestamps per segment."""
    lines = []
    for seg in result.get("segments", []):
        start = seg["start"]
        m = int(start // 60)
        s = int(start % 60)
        text = post_process(seg["text"].strip())
        lines.append(f"[{m:02d}:{s:02d}]  {text}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_size = st.selectbox(
        "Whisper Model",
        options=["large-v3", "large-v2", "medium", "small", "base", "tiny"],
        index=0,
        help=(
            "large-v3: best accuracy for Sanskrit/Hindi (slow on CPU)\n"
            "medium: good balance\n"
            "small/base/tiny: faster, less accurate"
        ),
    )

    language = st.selectbox(
        "Language",
        options=["auto", "hi", "en", "sa"],
        index=0,
        format_func=lambda x: {
            "auto": "🔍 Auto-detect (recommended for mixed content)",
            "hi": "🇮🇳 Hindi",
            "en": "🇬🇧 English",
            "sa": "📿 Sanskrit",
        }[x],
    )

    translate_to_english = st.toggle(
        "Translate to English",
        value=False,
        help="Force Whisper to output English translation regardless of source language",
    )

    st.divider()
    st.markdown(
        "**Tips**\n"
        "- Use `large-v3` for best Sanskrit accuracy\n"
        "- Leave language on Auto for Hindi+English mixed talks\n"
        "- SRT file can be uploaded to YouTube as subtitles"
    )

# ── Main area ─────────────────────────────────────────────────────────────────

url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    label_visibility="visible",
)

run_btn = st.button("🎙️ Transcribe", type="primary", use_container_width=True)

if run_btn:
    if not url.strip():
        st.error("Please enter a YouTube URL.")
        st.stop()

    # Validate URL shape
    if "youtube.com" not in url and "youtu.be" not in url:
        st.error("That doesn't look like a YouTube URL. Please check and try again.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Step 1: Download ──────────────────────────────────────────────────
        with st.status("⬇️ Downloading audio …", expanded=True) as status:
            try:
                audio_path, video_title = download_audio(url.strip(), tmpdir)
                status.update(label=f"✅ Downloaded: **{video_title}**", state="complete")
            except Exception as e:
                status.update(label="❌ Download failed", state="error")
                st.error(f"**Download error:** {e}")
                st.info(
                    "Common fixes:\n"
                    "- Make sure `ffmpeg` is installed (`brew install ffmpeg` / `sudo apt install ffmpeg`)\n"
                    "- Make sure `yt-dlp` is installed (`pip install yt-dlp`)\n"
                    "- The video may be age-restricted or private"
                )
                st.stop()

        # ── Step 2: Load model ────────────────────────────────────────────────
        with st.status(f"⏳ Loading Whisper **{model_size}** …", expanded=True) as status:
            try:
                model = load_model(model_size)
                status.update(label=f"✅ Model **{model_size}** ready", state="complete")
            except Exception as e:
                status.update(label="❌ Model load failed", state="error")
                st.error(f"**Model error:** {e}")
                st.stop()

        # ── Step 3: Transcribe ────────────────────────────────────────────────
        with st.status("🎙️ Transcribing … (this may take several minutes)", expanded=True) as status:
            try:
                lang_arg = None if language == "auto" else language
                result = run_transcription(model, audio_path, lang_arg, translate_to_english)
                detected_lang = result.get("language", "unknown")
                status.update(
                    label=f"✅ Transcription complete — detected language: **{detected_lang}**",
                    state="complete",
                )
            except Exception as e:
                status.update(label="❌ Transcription failed", state="error")
                st.error(f"**Transcription error:** {e}")
                st.stop()

        # ── Step 4: Show results ──────────────────────────────────────────────
        st.divider()
        st.subheader("📄 Transcript")

        plain_txt = to_plain_txt(result)
        srt_txt = to_srt(result)
        timestamped_txt = to_timestamped_txt(result)

        # Code block display
        st.code(plain_txt, language=None)

        # Download buttons
        st.subheader("⬇️ Download")
        col1, col2, col3 = st.columns(3)

        safe_title = re.sub(r"[^\w\s-]", "", video_title)[:50].strip().replace(" ", "_")

        with col1:
            st.download_button(
                label="📄 Plain .txt",
                data=plain_txt.encode("utf-8"),
                file_name=f"{safe_title}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                label="🎬 Subtitles .srt",
                data=srt_txt.encode("utf-8"),
                file_name=f"{safe_title}.srt",
                mime="text/plain",
                use_container_width=True,
            )

        with col3:
            st.download_button(
                label="🕐 Timestamped .txt",
                data=timestamped_txt.encode("utf-8"),
                file_name=f"{safe_title}_timestamped.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # Segment details expander
        with st.expander("🔍 View timestamped segments"):
            st.text(timestamped_txt)

        st.success(
            f"✅ Done!  |  Language detected: `{detected_lang}`  |  "
            f"Segments: `{len(result.get('segments', []))}`"
        )
