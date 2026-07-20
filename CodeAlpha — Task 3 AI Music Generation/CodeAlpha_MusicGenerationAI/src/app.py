from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import torch

try:
    from .generate import (
        load_vocab,
        get_special_ids,
        load_seed_sequence,
        load_model,
        generate_tokens,
        tokens_to_music21_stream,
        find_soundfont,
        render_audio,
    )
except ImportError:
    from generate import (
        load_vocab,
        get_special_ids,
        load_seed_sequence,
        load_model,
        generate_tokens,
        tokens_to_music21_stream,
        find_soundfont,
        render_audio,
    )

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "outputs" / "checkpoints" / "best_model.pt"
GENERATED_MIDI_DIR = BASE_DIR / "outputs" / "generated_midi"

st.set_page_config(page_title="AI Piano Music Generator", page_icon="🎹")


class DefaultArgs:
    embedding_dim = 256
    hidden_size = 512
    num_layers = 3
    dropout = 0.3


@st.cache_resource(show_spinner="Loading model (only happens once per session)...")
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_to_idx, idx_to_token = load_vocab(BASE_DIR)
    special_ids = get_special_ids(token_to_idx)
    model = load_model(CHECKPOINT_PATH, len(token_to_idx), device, DefaultArgs())
    soundfont_path = find_soundfont(BASE_DIR, None)
    return device, token_to_idx, idx_to_token, special_ids, model, soundfont_path


def render_to_wav_bytes(midi_path: Path, soundfont_path):
    """Attempts to render a .mid to .wav bytes. Returns None if unavailable
    or if rendering fails/times out — never raises, so a bad render never
    crashes the app."""
    if soundfont_path is None:
        return None
    tmp_wav = Path(tempfile.mkdtemp()) / "rendered.wav"
    rendered = render_audio(midi_path, tmp_wav, soundfont_path)
    if rendered and tmp_wav.exists():
        return tmp_wav.read_bytes()
    return None


def display_result(record: dict, key_prefix: str):
    """Displays one generated/loaded sample: label, download buttons, and
    audio player if a wav render is available."""
    st.markdown(f"**{record['label']}**")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download MIDI (.mid)",
            data=record["midi_bytes"],
            file_name=record["midi_filename"],
            mime="audio/midi",
            key=f"{key_prefix}_midi_dl",
        )

    if record["wav_bytes"] is not None:
        st.audio(record["wav_bytes"], format="audio/wav")
        with col2:
            st.download_button(
                "⬇️ Download Audio (.wav)",
                data=record["wav_bytes"],
                file_name=record["midi_filename"].replace(".mid", ".wav"),
                mime="audio/wav",
                key=f"{key_prefix}_wav_dl",
            )
    else:
        st.info("Audio playback unavailable for this one (no soundfont, or rendering failed/timed out). MIDI download still works.")


st.title("🎹 AI Piano Music Generator")
st.caption("CodeAlpha Task 3 — LSTM trained on the MAESTRO piano dataset")

if not CHECKPOINT_PATH.exists():
    st.error(f"No trained model found at:\n{CHECKPOINT_PATH}\n\nTrain the model first (see README).")
    st.stop()

device, token_to_idx, idx_to_token, special_ids, model, soundfont_path = load_everything()

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, newest first

st.sidebar.header("Settings")
temperature = st.sidebar.slider(
    "Temperature (creativity)", min_value=0.5, max_value=1.5, value=1.0, step=0.1,
    help="Lower = safer, more repetitive. Higher = more varied, riskier.",
)
num_tokens = st.sidebar.slider(
    "Length (generated tokens)", min_value=100, max_value=500, value=300, step=50,
)

if soundfont_path is None:
    st.sidebar.warning("No soundfont found — downloads still work, playback won't.")
else:
    st.sidebar.success(f"Soundfont loaded: {soundfont_path.name}")

if st.session_state.history:
    if st.sidebar.button("🗑️ Clear all results"):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------
# Section 1 — Generate new music
# ---------------------------------------------------------

st.header("🎼 Generate New Music")

if st.button("🎵 Generate Music", type="primary"):
    with st.spinner("Composing..."):
        seed_indices, seed_source = load_seed_sequence(BASE_DIR, seed_length=50, seed_file=None)

        if seed_indices is None:
            fallback_token = "NOTE_60_1.0"
            seed_indices = [token_to_idx.get(fallback_token, 2)]
            seed_source = "fixed_fallback"

        generated_ids = generate_tokens(
            model=model, seed_indices=seed_indices, num_new_tokens=num_tokens,
            temperature=temperature, device=device, special_token_ids=special_ids,
        )
        generated_tokens_str = [idx_to_token.get(i, "<UNK>") for i in generated_ids]
        midi_stream = tokens_to_music21_stream(generated_tokens_str)

        tmp_dir = Path(tempfile.mkdtemp())
        filename = f"generated_temp{temperature}_{len(st.session_state.history)}.mid"
        midi_path = tmp_dir / filename
        midi_stream.write("midi", fp=str(midi_path))
        midi_bytes = midi_path.read_bytes()

    with st.spinner("Rendering audio (up to ~90s, will not hang forever)..."):
        wav_bytes = render_to_wav_bytes(midi_path, soundfont_path)

    st.session_state.history.insert(0, {
        "label": f"Temp {temperature} · {len(generated_ids)} tokens · seed: {seed_source}",
        "midi_filename": filename,
        "midi_bytes": midi_bytes,
        "wav_bytes": wav_bytes,
    })
    st.rerun()

if st.session_state.history:
    st.subheader(f"Generated results ({len(st.session_state.history)})")
    for idx, record in enumerate(st.session_state.history):
        display_result(record, key_prefix=f"gen_{idx}")
        st.divider()
else:
    st.caption("Nothing generated yet this session.")

# ---------------------------------------------------------
# Section 2 — Listen to an existing MIDI file
# ---------------------------------------------------------

st.header("🎧 Listen to an Existing MIDI File")

existing_files = sorted(GENERATED_MIDI_DIR.glob("*.mid")) if GENERATED_MIDI_DIR.exists() else []

if existing_files:
    chosen_name = st.selectbox("Pick a previously generated file", options=[f.name for f in existing_files])
    if st.button("▶️ Load & Play Selected File"):
        chosen_path = GENERATED_MIDI_DIR / chosen_name
        with st.spinner("Rendering audio (up to ~90s, will not hang forever)..."):
            wav_bytes = render_to_wav_bytes(chosen_path, soundfont_path)
        record = {
            "label": chosen_name,
            "midi_filename": chosen_name,
            "midi_bytes": chosen_path.read_bytes(),
            "wav_bytes": wav_bytes,
        }
        display_result(record, key_prefix="saved_current")
else:
    st.caption(f"No saved files found in {GENERATED_MIDI_DIR}")

st.caption("Or upload any .mid file from your computer:")
uploaded = st.file_uploader("Upload a MIDI file", type=["mid", "midi"], label_visibility="collapsed")

if uploaded is not None and st.button("▶️ Play Uploaded File"):
    tmp_dir = Path(tempfile.mkdtemp())
    uploaded_path = tmp_dir / uploaded.name
    uploaded_path.write_bytes(uploaded.getbuffer())
    with st.spinner("Rendering audio (up to ~90s, will not hang forever)..."):
        wav_bytes = render_to_wav_bytes(uploaded_path, soundfont_path)
    record = {
        "label": uploaded.name,
        "midi_filename": uploaded.name,
        "midi_bytes": uploaded_path.read_bytes(),
        "wav_bytes": wav_bytes,
    }
    display_result(record, key_prefix="uploaded_current")

st.divider()
st.caption(
    "Model: 3-layer LSTM (~8.5M parameters) trained on the MAESTRO v3.0.0 piano dataset. "
    "Note: dynamics/velocity aren't part of the tokenization scheme, so every note plays "
    "at a fixed volume."
)