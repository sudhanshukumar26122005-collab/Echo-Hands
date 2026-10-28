"""
ECHO HANDS — Real-Time Sign Language Recognition & Speech (OpenCV Desktop Runner)
Phase 2: Accurate Sign Language Recognition with Text-to-Speech
"""

import os
import sys

# Auto-activate local .venv if executed with system Python
_base_dir = os.path.dirname(os.path.abspath(__file__))
_venv_dir = os.path.join(_base_dir, ".venv")
_venv_python = os.path.join(_venv_dir, "bin", "python")
if os.path.isdir(_venv_dir) and sys.prefix != _venv_dir:
    os.execv(_venv_python, [_venv_python] + sys.argv)

import cv2
import time
import math

os.environ.setdefault("MPLCONFIGDIR", os.path.join(_base_dir, ".cache", "matplotlib"))

from hand_detector import HandDetector
from recognizer import recognize_sign, SignStabilizer
from word_builder import WordBuilder
from tts import SpeechManager, speak_desktop_async, format_natural_word


def main():
    detector = HandDetector(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Please check: macOS System Settings -> Privacy & Security -> Camera -> allow Terminal / Python.")
        return

    word_builder = WordBuilder(hold_seconds=0.7)
    stabilizer = SignStabilizer(window_size=5, consensus_ratio=0.6)
    speech_manager = SpeechManager(min_stable_seconds=0.8)

    print("=" * 65)
    print("ECHO HANDS — Sign Recognition & Speech Started (Phase 2)")
    print("Supported signs: A, B, D, E, F, I, L, V, W, Y")
    print("Controls:")
    print("  'Space'     : Insert space")
    print("  'Backspace' : Delete last letter")
    print("  'r'         : Replay last spoken word")
    print("  'v'         : Toggle voice output ON / OFF")
    print("  'c'         : Clear text")
    print("  'q'         : Quit")
    print("=" * 65)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        # Flip horizontally for natural mirror view
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        hands = detector.find_hands(img)

        raw_letter = ""
        conf = 0.0
        reason = ""
        fingers = [0, 0, 0, 0, 0]

        if hands:
            lm = hands[0]
            detector.draw_landmarks(img, lm)
            handedness = detector.get_handedness(0)

            # Recognize sign using scale- and rotation-invariant 3D geometric engine
            raw_letter, conf, reason, fingers = recognize_sign(lm, handedness=handedness)

            # Stabilize prediction over rolling temporal window to prevent jitter
            stable_letter, stable_conf = stabilizer.update(raw_letter, conf)

            # Update word builder with stabilized letter
            word = word_builder.update(stable_letter)
            hold_pct = int(word_builder.progress * 100)

            # Phase 2: Autonomous Text-to-Speech for stabilized word
            if word and speech_manager.process_word(word):
                speak_desktop_async(word)

            # --- HUD Overlay ---
            # Header bar background
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 115), (20, 20, 20), -1)
            cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

            # Top Line 1: Detected Hand & Sign
            if stable_letter:
                sign_text = f"Sign: {stable_letter} ({int(conf * 100)}%)"
                sign_color = (80, 240, 100) # Green
            elif raw_letter:
                sign_text = f"Stabilizing: {raw_letter}..."
                sign_color = (80, 200, 240) # Yellow-cyan
            else:
                sign_text = "Sign: [Uncertain / Hold sign steady]"
                sign_color = (180, 180, 180)

            cv2.putText(img, f"Hand: {handedness}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(img, sign_text, (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, sign_color, 2)

            # Top Line 2: Details & Voice status
            voice_label = speech_manager.last_status_message if speech_manager.enabled else "MUTED ('v' to unmute)"
            cv2.putText(img, f"Voice: {voice_label}", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1)

            # Hold Progress Bar
            if hold_pct > 0:
                bar_x, bar_y, bar_w, bar_h = w - 160, 15, 140, 18
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                fill_w = int(bar_w * (hold_pct / 100.0))
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (80, 240, 100), -1)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
                cv2.putText(img, f"Hold {hold_pct}%", (bar_x + 20, bar_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Finger states
            cv2.putText(img, f"Fingers [T,I,M,R,P]: {fingers} | {reason[:30]}", (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 220, 255), 1)

            # Bottom Line: Recognized Word
            word_str = f"WORD: {word}" if word else "WORD: [Waiting for letters...]"
            cv2.putText(img, word_str, (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 100), 2)

        else:
            stabilizer.reset()
            word_builder.update("")

            # No hand indication overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

            cv2.putText(img, "No Hand Detected — Show hand to camera", (15, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 240), 2)

            word_str = f"WORD: {word_builder.word}" if word_builder.word else "WORD: "
            cv2.putText(img, word_str, (15, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 100), 2)

        cv2.imshow("ECHO HANDS — Sign Recognition & Speech", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word_builder.clear()
            speech_manager.reset()
        elif key == ord('r'): # Replay
            target = speech_manager.last_spoken_word or word_builder.word.strip()
            if target:
                speak_desktop_async(target)
        elif key == ord('v'): # Toggle voice
            speech_manager.enabled = not speech_manager.enabled
        elif key == 8: # Backspace
            word_builder.backspace()
        elif key == 32: # Spacebar
            word_builder.add_space()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
