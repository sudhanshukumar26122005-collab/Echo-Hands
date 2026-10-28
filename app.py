import os
import sys

# Auto-activate local .venv if executed with system Python
_base_dir = os.path.dirname(os.path.abspath(__file__))
_venv_dir = os.path.join(_base_dir, ".venv")
_venv_python = os.path.join(_venv_dir, "bin", "python")
if os.path.isdir(_venv_dir) and sys.prefix != _venv_dir:
    os.execv(_venv_python, [_venv_python] + sys.argv)

import streamlit as st

# If executed directly with `python app.py`, start Streamlit server
if not st.runtime.exists():
    try:
        from streamlit.runtime.credentials import Credentials, _Activation
        creds = Credentials.get_current()
        creds._conf_file = os.path.join(_base_dir, ".streamlit", "credentials.toml")
        creds.activation = _Activation(email="", is_valid=True)
    except Exception:
        pass

    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

# Safe matplotlib cache and Streamlit directory
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_base_dir, ".cache", "matplotlib"))
os.environ.setdefault("STREAMLIT_CONFIG_DIR", os.path.join(_base_dir, ".streamlit"))

# Streamlit Page Setup
st.set_page_config(
    page_title="ECHO HANDS — Sign Language Recognition & Speech",
    page_icon="🤟",
    layout="wide"
)

import time
import cv2
import numpy as np
import importlib

import hand_detector
import recognizer
import word_builder
import tts

importlib.reload(hand_detector)
importlib.reload(recognizer)
importlib.reload(word_builder)
importlib.reload(tts)

from hand_detector import HandDetector, HAND_CONNECTIONS
from recognizer import recognize_sign, SignStabilizer
from word_builder import WordBuilder
from tts import SpeechManager, generate_browser_speech_html, format_natural_word
import streamlit.components.v1 as components


# ---------- Camera Helper ----------
def open_working_camera(preferred_index=0):
    """
    Attempts to open a working webcam on macOS using AVFoundation or default backend.
    """
    indices_to_try = [preferred_index] + [i for i in [0, 1, 2] if i != preferred_index]

    for idx in indices_to_try:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap is not None and cap.isOpened():
            for _ in range(6):
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    return cap, idx
                time.sleep(0.04)
            cap.release()

        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            for _ in range(6):
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    return cap, idx
                time.sleep(0.04)
            cap.release()

    return None, preferred_index


@st.cache_resource
def get_detector():
    return HandDetector(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)


# Initialize session states
if "word_builder" not in st.session_state:
    st.session_state.word_builder = WordBuilder(hold_seconds=0.7)
if "stabilizer" not in st.session_state:
    st.session_state.stabilizer = SignStabilizer(window_size=5, consensus_ratio=0.6)
if "speech_manager" not in st.session_state:
    st.session_state.speech_manager = SpeechManager(min_stable_seconds=0.8)

# Main Header
st.title("🤟 ECHO HANDS — Sign Language Recognition & Speech")
st.caption("Phase 2: Accurate sign recognition with natural text-to-speech pronunciation.")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Camera & Audio Settings")
    cam_option = st.selectbox(
        "Camera Device",
        options=["Auto-detect", "Camera 0", "Camera 1", "Camera 2"],
        index=0,
        help="Select your webcam device index."
    )
    cam_index_map = {"Auto-detect": 0, "Camera 0": 0, "Camera 1": 1, "Camera 2": 2}
    preferred_cam = cam_index_map[cam_option]

    st.divider()
    st.subheader("🔊 Speech Options")
    speech_rate = st.slider("Speech Rate", min_value=0.7, max_value=1.3, value=0.95, step=0.05, help="Speed of spoken word pronunciation")
    st.markdown(f"**Current Voice Status:**\n- {'🟢 Active' if st.session_state.speech_manager.enabled else '🔴 Muted'}")
    st.markdown(f"- Last Spoken: **`{st.session_state.speech_manager.last_spoken_word or 'None'}`**")

    st.divider()
    st.markdown("**Recognition Capabilities:**\n✔ Complete A–Z ASL Alphabet (26 letters)\n✔ Dynamic Gestures: J (hook) & Z (zigzag)\n✔ Fist Disambiguation: A, E, S, T, N, M\n✔ Natural Text-to-Speech synthesis\n✔ Anti-repetition hold protection\n✔ Manual Replay & Mute toggle")

# Supported Gestures Reference
with st.expander("📖 Complete ASL Alphabet Reference Guide (A–Z Supported)"):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✊ Fist Family",
        "✌️ Dual & Cross",
        "☝️ Single Finger & Hook",
        "↔️ Horizontal & Downward",
        "⭕ Loops, 3-Finger & Dynamic"
    ])
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.markdown("**A** — Fist, thumb upright alongside index")
        c1.markdown("**E** — Fist, thumb folded flat under curled fingertips")
        c2.markdown("**S** — Fist, thumb wrapped across front of fingers")
        c2.markdown("**T** — Fist, thumb tucked under index finger only")
        c3.markdown("**N** — Fist, thumb tucked under index & middle fingers")
        c3.markdown("**M** — Fist, thumb tucked under index, middle, ring")

    with tab2:
        c1, c2 = st.columns(2)
        c1.markdown("**U** — Index & middle upright together (no gap)")
        c1.markdown("**V** — Index & middle upright spread apart ('peace')")
        c2.markdown("**R** — Index & middle upright and crossed")
        c2.markdown("**K** — Index up, middle angled forward, thumb nestled between")

    with tab3:
        c1, c2 = st.columns(2)
        c1.markdown("**D** — Index extended up, others curled, thumb folded")
        c1.markdown("**I** — Pinky extended up, others curled, thumb in")
        c2.markdown("**L** — Thumb & index at 90° angle forming an 'L'")
        c2.markdown("**X** — Index finger bent at knuckle into hook, others curled")

    with tab4:
        c1, c2 = st.columns(2)
        c1.markdown("**G** — Index pointing horizontally, thumb parallel")
        c1.markdown("**H** — Index & middle extended horizontally together")
        c2.markdown("**P** — K-handshape pointing downward towards floor")
        c2.markdown("**Q** — G-handshape pointing downward towards floor")

    with tab5:
        c1, c2 = st.columns(2)
        c1.markdown("**C** — All fingers curved in an open 'C' arch")
        c1.markdown("**O** — All fingertips touch thumb tip in closed circle")
        c1.markdown("**F** — Thumb & index touching in circle, 3 fingers up")
        c2.markdown("**W** — Index, middle, ring fingers up, pinky curled")
        c2.markdown("**B** — 4 fingers straight up, thumb in across palm")
        c2.markdown("**Y** — Thumb & pinky extended wide ('shaka')")
        c2.markdown("**J** — Pinky traces dynamic hook curve downward & up *(Motion)*")
        c2.markdown("**Z** — Index traces dynamic 3-stroke zigzag in the air *(Motion)*")


# Action Controls
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns([1.5, 1.2, 1, 1, 0.9, 0.9])
run_camera = ctrl_col1.checkbox("▶ Start Live Camera", key="camera_toggle")
voice_on = ctrl_col2.checkbox("🔊 Voice Output", value=st.session_state.speech_manager.enabled, key="voice_toggle")
st.session_state.speech_manager.enabled = voice_on

current_word = st.session_state.word_builder.word.strip()
replay_btn = ctrl_col3.button("🔁 Replay Word", disabled=not (st.session_state.speech_manager.can_replay() or bool(current_word)))
speak_now_btn = ctrl_col4.button("🗣️ Speak Word", disabled=not bool(current_word))
if ctrl_col5.button("␣ Space"):
    st.session_state.word_builder.add_space()
if ctrl_col6.button("🗑️ Clear"):
    st.session_state.word_builder.clear()
    st.session_state.stabilizer.reset()
    st.session_state.speech_manager.reset()

# Layout: Camera Feed (left) & Recognition + Speech Card (right)
col_cam, col_info = st.columns([1.7, 1.3])

FRAME_WINDOW = col_cam.image([])
status_card = col_info.empty()
word_card = col_info.empty()
speech_status_card = col_info.empty()
SPEECH_CARRIER = col_info.empty()

# Handle manual buttons outside loop
if replay_btn:
    replay_target = st.session_state.speech_manager.last_spoken_word or current_word
    if replay_target:
        speech_code = generate_browser_speech_html(replay_target, rate=speech_rate)
        with SPEECH_CARRIER:
            components.html(speech_code, height=0)
        speech_status_card.success(f"🔁 Replaying: **`\"{format_natural_word(replay_target)}\"`**")

if speak_now_btn:
    if current_word:
        st.session_state.speech_manager.last_spoken_word = current_word
        speech_code = generate_browser_speech_html(current_word, rate=speech_rate)
        with SPEECH_CARRIER:
            components.html(speech_code, height=0)
        speech_status_card.success(f"🔊 Spoken: **`\"{format_natural_word(current_word)}\"`**")

# Display initial word state
word_card.markdown(
    f"### 🧾 Recognized Word\n## **`{current_word if current_word else '[Waiting for signs...]'}`**"
)

if not run_camera:
    status_card.info("💡 Check **'▶ Start Live Camera'** above to begin recognizing signs and speaking words.")
    if st.session_state.speech_manager.enabled:
        speech_status_card.caption(f"🎙️ **Voice:** {st.session_state.speech_manager.last_status_message}")
    else:
        speech_status_card.warning("🔇 **Voice Output Muted**")

if run_camera:
    try:
        detector = get_detector()
    except Exception as e:
        status_card.error(f"Failed to initialize hand detector: {e}")
        st.stop()

    status_card.info("🔄 Connecting to camera...")
    cap, active_cam_idx = open_working_camera(preferred_cam)

    if cap is None:
        status_card.error(
            "⚠️ **Camera could not be opened.**\n\n"
            "Please check:\n"
            "1. macOS System Settings → Privacy & Security → Camera (allow Terminal/Python).\n"
            "2. Make sure other apps (Zoom, FaceTime, main.py) are closed.\n"
            "3. Try selecting Camera 1 or Camera 2 in the sidebar."
        )
    else:
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass

        stabilizer = st.session_state.stabilizer
        word_builder = st.session_state.word_builder
        speech_manager = st.session_state.speech_manager

        last_status_key = None
        last_word_key = None
        last_speech_key = None
        fail_count = 0

        try:
            while run_camera:
                frame_start_time = time.time()

                ret, frame = cap.read()
                if not ret or frame is None:
                    fail_count += 1
                    if fail_count > 30:
                        status_card.warning("Camera connection dropped. Reconnecting...")
                        break
                    time.sleep(0.02)
                    continue

                fail_count = 0

                # Mirror frame horizontally for natural user experience
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                hands = detector.find_hands(frame)

                detected_letter = ""
                confidence = 0.0
                reason = ""
                handedness = "Right"
                fingers = [0, 0, 0, 0, 0]

                if hands:
                    lm = hands[0]
                    try:
                        detector.draw_landmarks(frame, lm)
                    except Exception:
                        for p1, p2 in HAND_CONNECTIONS:
                            if p1 < len(lm) and p2 < len(lm):
                                cv2.line(frame, (int(lm[p1][1]), int(lm[p1][2])), (int(lm[p2][1]), int(lm[p2][2])), (0, 255, 0), 2)
                        for p in lm:
                            cv2.circle(frame, (int(p[1]), int(p[2])), 5, (0, 0, 255), -1)
                            cv2.circle(frame, (int(p[1]), int(p[2])), 6, (255, 255, 255), 1)

                    handedness = detector.get_handedness(0)

                    # Compute geometric sign classification
                    raw_letter, raw_conf, raw_reason, fingers = recognize_sign(lm, handedness=handedness)

                    # Stabilize prediction over rolling consensus window
                    stable_letter, stable_conf = stabilizer.update(raw_letter, raw_conf)

                    detected_letter = stable_letter
                    confidence = stable_conf if stable_letter else raw_conf
                    reason = raw_reason

                    # Update word accumulator
                    word_builder.update(detected_letter)
                    hold_pct = int(word_builder.progress * 100)

                    # On-screen HUD on camera frame
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                    if detected_letter:
                        hud_text = f"SIGN: {detected_letter} ({int(confidence * 100)}%)"
                        hud_color = (60, 240, 80) # Green
                    elif raw_letter:
                        hud_text = f"HOLD STEADY: {raw_letter}..."
                        hud_color = (60, 220, 240) # Yellow-cyan
                    else:
                        hud_text = "SIGN: [Unrecognized / Transition]"
                        hud_color = (180, 180, 180)

                    cv2.putText(frame, hud_text, (15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, hud_color, 2, cv2.LINE_AA)

                    # On-screen hold progress bar
                    if hold_pct > 0:
                        bar_w, bar_h = 130, 16
                        bx, by = w - bar_w - 20, 20
                        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), -1)
                        fill_w = int(bar_w * (hold_pct / 100.0))
                        cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), (60, 240, 80), -1)
                        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (200, 200, 200), 1)

                else:
                    stabilizer.reset()
                    word_builder.update("")
                    hold_pct = 0

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
                    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                    cv2.putText(frame, "No Hand Detected", (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (80, 80, 240), 2, cv2.LINE_AA)

                # Send frame to Streamlit (resize to 540px width and convert to RGB for instant websocket delivery)
                if frame.shape[1] > 540:
                    scale = 540.0 / frame.shape[1]
                    display_frame = cv2.resize(frame, (540, int(frame.shape[0] * scale)))
                else:
                    display_frame = frame

                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(rgb_frame, channels="RGB")

                # Selective UI update to prevent re-rendering overhead
                status_key = f"{bool(hands)}:{detected_letter}:{int(confidence*100)}:{hold_pct}"
                if status_key != last_status_key:
                    last_status_key = status_key
                    if not hands:
                        status_card.warning("👋 **No Hand Detected**\nPlease show your hand clearly to the camera.")
                    elif detected_letter:
                        hold_badge = f"**(Hold to commit: {hold_pct}%)**" if 0 < hold_pct < 100 else ("✅ *Committed to word*" if hold_pct >= 100 else "")
                        conf_pct = int(confidence * 100)
                        status_card.success(
                            f"### 🅰️ Recognized Sign: **`{detected_letter}`**\n\n"
                            f"- **Confidence:** `{conf_pct}%`\n"
                            f"- **Handedness:** `{handedness}`\n"
                            f"- **Details:** {reason}\n"
                            f"- **Status:** {hold_badge}"
                        )
                    else:
                        status_card.info(
                            "❓ **Gesture Unrecognized / Transitioning**\n\n"
                            "The model does not display confident predictions for ambiguous poses.\n"
                            "Please form one of the supported signs (A, B, D, E, F, I, L, V, W, Y) and hold steady."
                        )

                # Word display update
                current_word = word_builder.word.strip()
                if current_word != last_word_key:
                    last_word_key = current_word
                    word_card.markdown(
                        f"### 🧾 Recognized Word\n## **`{current_word if current_word else '[Waiting for signs...]'}`**"
                    )

                # -------------------------------------------------------------
                # Phase 2: Autonomous Text-to-Speech Processing
                # -------------------------------------------------------------
                if current_word:
                    try:
                        if speech_manager.process_word(current_word):
                            speech_html = generate_browser_speech_html(current_word, rate=speech_rate)
                            with SPEECH_CARRIER:
                                components.html(speech_html, height=0)
                    except Exception as err:
                        speech_manager.last_error = str(err)

                # Speech status card update
                speech_key = f"{speech_manager.enabled}:{speech_manager.last_status_message}:{speech_manager.last_spoken_word}"
                if speech_key != last_speech_key:
                    last_speech_key = speech_key
                    if not speech_manager.enabled:
                        speech_status_card.warning("🔇 **Voice Output Muted** (Check '🔊 Voice Output' above to enable)")
                    elif speech_manager.last_spoken_word:
                        speech_status_card.info(f"🔊 **Voice:** {speech_manager.last_status_message}")
                    else:
                        speech_status_card.caption(f"🎙️ **Voice:** {speech_manager.last_status_message}")

                # Regulate frame rate to smooth ~25-28 FPS to prevent WebSocket queue congestion and browser freeze
                elapsed = time.time() - frame_start_time
                target_interval = 0.036
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

        finally:
            cap.release()
