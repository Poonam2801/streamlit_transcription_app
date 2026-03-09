"""
YouTube Spiritual Content Transcriber — M2 Accelerated
=======================================================
Uses mlx-whisper (Apple Silicon GPU/Neural Engine) for fast transcription.
Falls back to openai-whisper on non-Apple hardware automatically.

Setup:
    pip install -r requirements.txt
    brew install ffmpeg

Run:
    streamlit run app.py
"""

import os
import re
import tempfile
import sys
from pathlib import Path

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Detect Apple Silicon and available backend
# ─────────────────────────────────────────────────────────────────────────────

def detect_backend() -> str:
    """
    Returns 'mlx' if running on Apple Silicon with mlx-whisper installed,
    otherwise 'openai-whisper'.
    """
    if sys.platform != "darwin":
        return "openai-whisper"
    try:
        import mlx_whisper  # noqa: F401
        import mlx.core     # noqa: F401
        return "mlx"
    except ImportError:
        return "openai-whisper"

BACKEND = detect_backend()

# MLX model name mapping (mlx-whisper uses HuggingFace repo strings)
MLX_MODEL_MAP = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "tiny":     "mlx-community/whisper-tiny-mlx",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spiritual Transcriber",
    page_icon="🕉️",
    layout="centered",
)

st.title("🕉️ Spiritual Content Transcriber")

backend_badge = "🍎 Apple M2 (mlx)" if BACKEND == "mlx" else "💻 CPU (openai-whisper)"
st.caption(
    f"Hindi · English · Sanskrit  |  Powered by Whisper  |  Running on **{backend_badge}**"
)

# ─────────────────────────────────────────────────────────────────────────────
# Spiritual domain constants
# ─────────────────────────────────────────────────────────────────────────────

SPIRITUAL_PROMPT = (
    "This is a spiritual discourse in Hindi and English, often mixing Sanskrit shlokas, "
    "mantras, and verses from the Bhagavad Gita, Upanishads, and Vedas. "
    "Common Sanskrit terms: Om, Brahman, Atman, Dharma, Karma, Moksha, Guru, Shakti, "
    "Pranayama, Samadhi, Samsara, Nirvana, Ahimsa, Satya, Shanti, Ananda, Maya, "
    "Shloka, Mantra, Namaste, Namaskar, Ishwara, Bhakti, Jnana, Vairagya, Viveka, "
    "Sadhana, Tapas, Seva, Satsang, Kirtan, Aarti, Ashram, Guruji, Paramatma, "
    "Chaitanya, Kundalini, Chakra, Prana, Nadi, Shuddhi, Veda, Purana, Gita."
)

SANSKRIT_CORRECTIONS = {
    "bhagavad geeta": "Bhagavad Gita",
    "bhagwad geeta":  "Bhagavad Gita",
    "bhagvad gita":   "Bhagavad Gita",
    "upnishad":       "Upanishad",
    "upnishads":      "Upanishads",
    "prana yama":     "Pranayama",
    "pranayam":       "Pranayama",
    "sama dhi":       "Samadhi",
    "sam sara":       "Samsara",
    "nirvan":         "Nirvana",
    "brhaman":        "Brahman",
    "aatman":         "Atman",
    "ishwar":         "Ishwara",
    "paramatman":     "Paramatma",
    "om shanti":      "Om Shanti",
    "om shakti":      "Om Shakti",
    "satsangh":       "Satsang",
    "aarti":          "Aarti",
    "kirtan":         "Kirtan",
    "ashram":         "Ashram",
}

def post_process(text: str) -> str:
    for wrong, right in SANSKRIT_CORRECTIONS.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Audio download
# ─────────────────────────────────────────────────────────────────────────────

def download_audio(youtube_url: str, output_dir: str):
    """Download best audio from YouTube. Returns (wav_path, video_title)."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed — run: pip install yt-dlp")

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
        title    = info.get("title", "transcript")

    audio_path = os.path.join(output_dir, f"{video_id}.wav")
    if not os.path.exists(audio_path):
        wavs = list(Path(output_dir).glob("*.wav"))
        if not wavs:
            raise RuntimeError(
                "Audio download failed — no .wav found.\n"
                "Make sure ffmpeg is installed: brew install ffmpeg"
            )
        audio_path = str(wavs[0])

    return audio_path, title

# ─────────────────────────────────────────────────────────────────────────────
# Model loading — cached so it loads ONCE per session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_mlx_model(model_size: str):
    """Validate mlx-whisper is available and return the HF model path string."""
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "mlx-whisper not installed — run: pip install mlx-whisper mlx"
        )
    return MLX_MODEL_MAP.get(model_size, MLX_MODEL_MAP["large-v3"])


@st.cache_resource(show_spinner=False)
def load_openai_model(model_size: str):
    """Load and cache openai-whisper model in memory."""
    try:
        import whisper
    except ImportError:
        raise RuntimeError(
            "openai-whisper not installed — run: pip install openai-whisper"
        )
    return whisper.load_model(model_size)

# ─────────────────────────────────────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_mlx(model_path: str, audio_path: str, language, translate: bool) -> dict:
    """Run transcription using mlx-whisper (Apple Silicon accelerated)."""
    import mlx_whisper

    kwargs = {
        "path_or_hf_repo": model_path,
        "initial_prompt": SPIRITUAL_PROMPT,
        "word_timestamps": True,
        "verbose": False,
        "task": "translate" if translate else "transcribe",
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
    }
    if language and language != "auto":
        kwargs["language"] = language

    return mlx_whisper.transcribe(audio_path, **kwargs)


def transcribe_openai(model, audio_path: str, language, translate: bool) -> dict:
    """Run transcription using openai-whisper (CPU fallback)."""
    options = {
        "task": "translate" if translate else "transcribe",
        "initial_prompt": SPIRITUAL_PROMPT,
        "word_timestamps": True,
        "verbose": False,
        "beam_size": 5,
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
    return post_process(result.get("text", ""))


def srt_timestamp(t: float) -> str:
    h  = int(t // 3600)
    m  = int((t % 3600) // 60)
    s  = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(result: dict) -> str:
    lines = []
    for i, seg in enumerate(result.get("segments", []), start=1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}")
        lines.append(post_process(seg["text"].strip()))
        lines.append("")
    return "\n".join(lines)


def to_timestamped_txt(result: dict) -> str:
    lines = []
    for seg in result.get("segments", []):
        m = int(seg["start"] // 60)
        s = int(seg["start"] % 60)
        lines.append(f"[{m:02d}:{s:02d}]  {post_process(seg['text'].strip())}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    if BACKEND == "mlx":
        st.success("🍎 Apple Silicon detected\nUsing mlx-whisper (GPU accelerated)")
    else:
        st.info("💻 Running on CPU\nInstall mlx-whisper for M2 speed:\npip install mlx-whisper mlx")

    st.divider()

    model_size = st.selectbox(
        "Whisper Model",
        options=["large-v3", "large-v2", "medium", "small", "base", "tiny"],
        index=0,
        help=(
            "large-v3 → best accuracy for Sanskrit/Hindi\n"
            "medium   → 2x faster, still very good\n"
            "small    → quick drafts"
        ),
    )

    language = st.selectbox(
        "Language",
        options=["auto", "hi", "en", "sa"],
        index=0,
        format_func=lambda x: {
            "auto": "🔍 Auto-detect (best for mixed content)",
            "hi":   "🇮🇳 Hindi",
            "en":   "🇬🇧 English",
            "sa":   "📿 Sanskrit",
        }[x],
    )

    translate_to_english = st.toggle(
        "Translate to English",
        value=False,
        help="Output English translation regardless of source language",
    )

    st.divider()
    st.markdown(
        "**Tips**\n"
        "- `large-v3` + Auto = best for Satsang/Pravachan\n"
        "- `.srt` file uploads directly to YouTube captions\n"
        "- Model downloads once, cached forever after"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

run_btn = st.button("🎙️ Transcribe", type="primary", use_container_width=True)

if run_btn:
    url = url.strip()
    if not url:
        st.error("Please enter a YouTube URL.")
        st.stop()
    if "youtube.com" not in url and "youtu.be" not in url:
        st.error("That doesn't look like a valid YouTube URL. Please check and try again.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── Step 1: Download audio ────────────────────────────────────────────
        with st.status("⬇️  Downloading audio from YouTube …", expanded=True) as status:
            try:
                audio_path, video_title = download_audio(url, tmpdir)
                status.update(
                    label=f"✅  Downloaded: **{video_title}**",
                    state="complete",
                )
            except Exception as e:
                status.update(label="❌  Download failed", state="error")
                st.error(f"**Download error:** {e}")
                st.info(
                    "Common fixes:\n"
                    "- Install ffmpeg: `brew install ffmpeg`\n"
                    "- Install yt-dlp: `pip install yt-dlp`\n"
                    "- Video may be private or age-restricted"
                )
                st.stop()

        # ── Step 2: Load model ────────────────────────────────────────────────
        with st.status(f"⏳  Loading **{model_size}** model …", expanded=True) as status:
            try:
                if BACKEND == "mlx":
                    model_ref = load_mlx_model(model_size)
                    label = f"✅  Model **{model_size}** ready (mlx — Apple Silicon)"
                else:
                    model_ref = load_openai_model(model_size)
                    label = f"✅  Model **{model_size}** ready (CPU)"
                status.update(label=label, state="complete")
            except Exception as e:
                status.update(label="❌  Model load failed", state="error")
                st.error(f"**Model error:** {e}")
                st.stop()

        # ── Step 3: Transcribe ────────────────────────────────────────────────
        speed_note = "3-5 min for a 1hr video on M2" if BACKEND == "mlx" else "may take 20-40 min on CPU"
        with st.status(
            f"🎙️  Transcribing … ({speed_note})",
            expanded=True,
        ) as status:
            try:
                lang_arg = None if language == "auto" else language

                if BACKEND == "mlx":
                    result = transcribe_mlx(model_ref, audio_path, lang_arg, translate_to_english)
                else:
                    result = transcribe_openai(model_ref, audio_path, lang_arg, translate_to_english)

                detected_lang = result.get("language", "unknown")
                n_segments    = len(result.get("segments", []))
                status.update(
                    label=f"✅  Done — language: **{detected_lang}** · {n_segments} segments",
                    state="complete",
                )
            except Exception as e:
                status.update(label="❌  Transcription failed", state="error")
                st.error(f"**Transcription error:** {e}")
                st.stop()

        # ── Step 4: Results ───────────────────────────────────────────────────
        st.divider()

        plain_txt       = to_plain_txt(result)
        srt_txt         = to_srt(result)
        timestamped_txt = to_timestamped_txt(result)
        safe_title      = re.sub(r"[^\w\s-]", "", video_title)[:50].strip().replace(" ", "_")

        st.subheader("📄 Transcript")
        st.code(plain_txt, language=None)

        st.subheader("⬇️  Download")
        col1, col2, col3 = st.columns(3)

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

        with st.expander("🔍 View with timestamps"):
            st.text(timestamped_txt)

        st.success(
            f"✅  Complete  |  Language: `{detected_lang}`  |  "
            f"Segments: `{n_segments}`  |  Backend: `{BACKEND}`"
        )
