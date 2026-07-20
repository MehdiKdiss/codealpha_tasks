from __future__ import annotations

import io

import streamlit as st
from deep_translator import GoogleTranslator

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


st.set_page_config(page_title="Language Translator", page_icon="🌐")


@st.cache_data(show_spinner=False)
def get_supported_languages() -> dict:
    """Returns {language_name: language_code}, fetched once and cached."""
    return GoogleTranslator().get_supported_languages(as_dict=True)


def translate_text(text: str, source: str, target: str) -> str:
    return GoogleTranslator(source=source, target=target).translate(text)


def text_to_speech_bytes(text: str, lang_code: str):
    """Generates speech audio for the translated text. Returns None if TTS
    isn't available or the language isn't supported by gTTS — never raises,
    so a TTS failure never breaks translation itself."""
    if not TTS_AVAILABLE:
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang_code).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


st.title("🌐 Language Translation Tool")
st.caption("CodeAlpha Task 1 — enter text, pick languages, translate.")

languages = get_supported_languages()
language_names = sorted(languages.keys())

col1, col2, col3 = st.columns([5, 1, 5])

with col1:
    source_name = st.selectbox(
        "From",
        options=["auto (detect)"] + language_names,
        index=0,
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    swap = st.button("🔄", help="Swap languages")

with col3:
    target_default = language_names.index("english") if "english" in language_names else 0
    target_name = st.selectbox("To", options=language_names, index=target_default)

if swap and source_name != "auto (detect)":
    st.session_state["_swap_pending"] = (target_name, source_name)
    st.rerun()

if "_swap_pending" in st.session_state:
    new_source, new_target = st.session_state.pop("_swap_pending")
    source_name, target_name = new_source, new_target

input_text = st.text_area("Text to translate", height=150, placeholder="Type or paste text here...")

MAX_CHARS = 4900  # GoogleTranslator (via deep-translator) errors above ~5000 chars per call

char_count = len(input_text)
if char_count > MAX_CHARS:
    st.warning(f"Text is {char_count} characters — the translation API has a ~5000 character limit per request. Please shorten it.")

if st.button("🔁 Translate", type="primary", disabled=not input_text.strip() or char_count > MAX_CHARS):
    source_code = "auto" if source_name == "auto (detect)" else languages[source_name]
    target_code = languages[target_name]

    with st.spinner("Translating..."):
        try:
            translated = translate_text(input_text, source_code, target_code)
            st.session_state["last_translation"] = translated
            st.session_state["last_target_code"] = target_code
        except Exception as exc:
            st.error(f"Translation failed: {exc}")
            st.session_state.pop("last_translation", None)

if "last_translation" in st.session_state:
    st.subheader("Translation")
    st.code(st.session_state["last_translation"], language=None)  # st.code has a built-in copy button

    if TTS_AVAILABLE:
        with st.spinner("Generating audio..."):
            audio_bytes = text_to_speech_bytes(
                st.session_state["last_translation"],
                st.session_state["last_target_code"],
            )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                "⬇️ Download audio (.mp3)",
                data=audio_bytes,
                file_name="translation.mp3",
                mime="audio/mpeg",
            )
        else:
            st.caption("🔇 Text-to-speech isn't available for this language.")
    else:
        st.caption("Install gTTS (`pip install gTTS`) to enable text-to-speech playback.")