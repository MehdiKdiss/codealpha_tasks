# 🌐 Language Translation Tool — CodeAlpha Task 1

A web-based translation tool with source/target language selection, built as part of the CodeAlpha Artificial Intelligence Internship.

## Features

- Enter any text, pick source language (or auto-detect) and target language
- Swap languages with one click
- Translated result shown clearly, with a built-in one-click copy button
- Text-to-speech playback of the translation, with MP3 download
- Graceful handling of empty input and text exceeding the API's length limit

## How it works

Translation is powered by `deep-translator`'s `GoogleTranslator`, which uses Google Translate's public web interface rather than the official paid Google Cloud Translation API — same underlying translation quality, no API key or billing setup required. Text-to-speech uses `gTTS` (Google Text-to-Speech).

**Known limitation:** the translation backend has an approximate 5,000-character limit per request; the app checks this before sending and warns the user rather than letting the request fail with an unclear error.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Easiest:** double-click `run_app.py` — opens the app in your browser, no terminal typing required after the one-time install above.

**Or manually:**
```bash
python -m streamlit run app.py
```

## Requirements

- Internet connection (both translation and text-to-speech call external services)
- Python 3.9+

## Credits

Built as part of the [CodeAlpha](https://www.codealpha.tech) Artificial Intelligence Internship — Task 1: Language Translation Tool.
