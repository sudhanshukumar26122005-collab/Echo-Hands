"""
ECHO HANDS — Text-to-Speech Engine (Phase 2)

Features:
- Browser-native Web Speech API (`speechSynthesis`) integration with natural pronunciation.
- Local desktop TTS fallback (`pyttsx3` / macOS `say`) for non-browser runners.
- SpeechManager:
    * Prevents repeated speech for held words.
    * Waits for stable recognition before uttering words.
    * Automatically triggers when a new word is formed.
    * Supports voice enable/disable toggle.
    * Supports replaying the last spoken word.
    * Comprehensive error handling without crashing.
"""

import json
import time
import threading
import queue
import subprocess
import sys


class SpeechManager:
    """
    Manages speech timing, stabilization, and debounce state.
    Ensures words are spoken naturally once stabilized, without repetition.
    """
    def __init__(self, min_stable_seconds=0.8):
        self.enabled = True
        self.min_stable_seconds = min_stable_seconds
        self.last_spoken_word = ""
        self.candidate_word = ""
        self.candidate_start_time = None
        self.last_status_message = "Ready"
        self.last_error = ""

    def process_word(self, current_word: str) -> bool:
        """
        Evaluates the recognized word.
        Returns True if the word is stable and ready to be spoken for the first time.
        """
        if not self.enabled:
            self.last_status_message = "🔇 Voice output muted"
            return False

        word = (current_word or "").strip()

        if not word:
            self.candidate_word = ""
            self.candidate_start_time = None
            self.last_status_message = "Waiting for signs..."
            return False

        # If already spoken and hasn't changed, do not repeat
        if word == self.last_spoken_word:
            self.last_status_message = f"🔊 Spoken: \"{self.last_spoken_word}\""
            return False

        now = time.time()

        # If the word changed, reset candidate stabilization timer
        if word != self.candidate_word:
            self.candidate_word = word
            self.candidate_start_time = now
            self.last_status_message = f"Stabilizing \"{word}\"..."
            return False

        # Word has been sustained: check if stability threshold is met
        elapsed = now - (self.candidate_start_time or now)
        if elapsed >= self.min_stable_seconds:
            self.last_spoken_word = word
            self.last_status_message = f"🔊 Spoke: \"{word}\""
            return True
        else:
            pct = int((elapsed / self.min_stable_seconds) * 100)
            self.last_status_message = f"Stabilizing \"{word}\" ({pct}%)..."
            return False

    def can_replay(self) -> bool:
        """Returns True if there is a previously spoken word available to replay."""
        return bool(self.last_spoken_word and self.last_spoken_word.strip())

    def reset(self):
        """Resets candidate and last spoken state."""
        self.candidate_word = ""
        self.candidate_start_time = None
        self.last_spoken_word = ""
        self.last_status_message = "Ready"
        self.last_error = ""


def format_natural_word(text: str) -> str:
    """
    Formats the word so speech synthesizers pronounce it naturally as a whole word
    rather than spelling individual letters (e.g. 'BED' -> 'Bed', 'HELLO' -> 'Hello').
    """
    clean = (text or "").strip()
    if not clean:
        return ""
    # Single letter 'A' or 'I' are naturally spoken words
    if len(clean) == 1:
        return clean.upper()
    # If all-caps word, title-case it so TTS doesn't treat it as an acronym
    if clean.isupper() and clean.isalpha():
        return clean.capitalize()
    return clean


def generate_browser_speech_html(text: str, voice_lang: str = "en-US", rate: float = 0.95) -> str:
    """
    Generates browser Web Speech API JavaScript snippet.
    Embeds seamlessly in Streamlit via st.components.v1.html to speak audio directly
    through the user's browser.
    """
    natural_word = format_natural_word(text)
    if not natural_word:
        return ""

    escaped_word = json.dumps(natural_word)
    # Include unique timestamp nonce to force DOM evaluation on identical text triggers
    nonce = int(time.time() * 1000)

    html_code = f"""
    <div id="tts-carrier-{nonce}" style="display:none;"></div>
    <script>
    (function() {{
        try {{
            const word = {escaped_word};
            if (!word) return;

            // Check for Web Speech API in current window or parent window
            const synth = window.speechSynthesis || (window.parent && window.parent.speechSynthesis);
            if (!synth) {{
                console.warn("Web Speech API is not supported in this browser.");
                return;
            }}

            synth.cancel(); // Stop any pending utterances

            const utter = new SpeechSynthesisUtterance(word);
            utter.lang = '{voice_lang}';
            utter.rate = {rate};
            utter.pitch = 1.0;

            function selectVoiceAndSpeak() {{
                try {{
                    const voices = synth.getVoices();
                    if (voices && voices.length > 0) {{
                        // Prefer natural sounding English voices (e.g. Google, Samantha, Daniel, Natural)
                        const natural = voices.find(v => 
                            (v.name.includes("Natural") || v.name.includes("Google") || v.name.includes("Samantha") || v.name.includes("Daniel") || v.name.includes("Karen")) &&
                            v.lang.startsWith("en")
                        );
                        if (natural) {{
                            utter.voice = natural;
                        }} else {{
                            const anyEnglish = voices.find(v => v.lang.startsWith("en"));
                            if (anyEnglish) utter.voice = anyEnglish;
                        }}
                    }}
                    synth.speak(utter);
                }} catch (err) {{
                    console.error("SpeechSynthesis speak error:", err);
                }}
            }}

            if (synth.getVoices().length > 0) {{
                selectVoiceAndSpeak();
            }} else {{
                synth.onvoiceschanged = selectVoiceAndSpeak;
                setTimeout(selectVoiceAndSpeak, 120);
            }}
        }} catch (e) {{
            console.error("ECHO HANDS TTS error:", e);
        }}
    }})();
    </script>
    """
    return html_code


# -----------------------------------------------------------------------------
# Desktop Offline Speech Worker (Fallback for OpenCV desktop runner main.py)
# -----------------------------------------------------------------------------
_speech_queue = queue.Queue()
_worker_thread = None
_lock = threading.Lock()


def _desktop_worker():
    # Attempt to initialize pyttsx3
    engine = None
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
    except Exception:
        engine = None

    while True:
        text = _speech_queue.get()
        if text is None:
            break
        word = format_natural_word(text)
        if word:
            # 1. Try pyttsx3 if available
            spoken = False
            if engine:
                try:
                    engine.say(word)
                    engine.runAndWait()
                    spoken = True
                except Exception:
                    spoken = False

            # 2. macOS native fallback via 'say' binary if pyttsx3 failed
            if not spoken and sys.platform == "darwin":
                try:
                    subprocess.run(["say", "-r", "160", word], check=False, capture_output=True)
                except Exception:
                    pass

        _speech_queue.task_done()


def speak_desktop_async(text: str):
    """
    Speaks text asynchronously via desktop audio engine (for main.py).
    Does not block camera loop.
    """
    if not text or not str(text).strip():
        return

    global _worker_thread
    with _lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_desktop_worker, daemon=True)
            _worker_thread.start()

    _speech_queue.put(str(text))


# Alias for legacy compatibility
speak_async = speak_desktop_async
