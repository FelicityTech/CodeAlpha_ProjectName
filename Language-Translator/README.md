# FelicityTech — Language Translation Tool

A single-file HTML app: enter text, pick a source and target language, and
get a translation — with copy and text-to-speech support. No build step,
no backend, no API key required to run it.

## Run it

Just open `language-translator.html` in any modern browser (double-click it,
or drag it into a browser window). That's it — everything runs client-side.

## Features

- Text input with a live character counter (4,500 char limit)
- Source language dropdown with auto-detect, plus a target language dropdown
  (37 languages)
- Swap button to flip source ↔ target and re-translate instantly
- Copy-to-clipboard button for the translated text
- Text-to-speech for both the original and translated text (via the
  browser's built-in Web Speech API)
- `Ctrl+Enter` keyboard shortcut to translate

## How it works

Translation is done client-side with two free, no-key APIs:

1. **Primary:** the unofficial free Google Translate endpoint
   (`translate.googleapis.com/translate_a/single`)
2. **Fallback:** if that fails or is unreachable, it automatically retries
   with the [MyMemory Translation API](https://mymemory.translated.net/)

Whichever one succeeds, the response updates the output panel and shows
which service was used ("via Google Translate" / "via MyMemory API").

## Known limitation — not production-grade translation

Both APIs used here are free tiers meant for light/personal use:

- No official rate-limit guarantee or SLA
- The Google endpoint is unofficial and undocumented — it can change or get
  blocked without notice
- Not suitable for high-volume or commercial use

**For production**, swap in the official **Google Cloud Translation API**
or **Microsoft Translator API**. Both require an API key, and since a
browser-side app can't safely hold a secret key, that means adding a small
backend (e.g. a Flask/FastAPI endpoint or a serverless function) that the
frontend calls instead of hitting the translation API directly. The UI code
itself wouldn't need to change — only the `translateViaGoogle()` /
`translateViaMyMemory()` functions would be swapped for a call to your own
backend.

## File

- `language-translator.html` — the entire app (HTML, CSS, and JS in one file)