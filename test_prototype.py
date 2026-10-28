"""
ECHO HANDS — Geometric Sign Recognition Prototype & Quick Test Runner
"""

from recognizer import recognize_sign, compute_hand_geometry
from tests.test_phase1 import build_synthetic_hand, set_finger_curl

if __name__ == "__main__":
    print("Running quick test on synthetic sign 'A' and sign 'E'...")

    # Sign A
    pts_A = build_synthetic_hand("Right")
    for f in [5, 9, 13, 17]:
        set_finger_curl(pts_A, f, curl=True)
    pts_A[4] = [0.28, 0.88, 0.0]
    letter_A, conf_A, reason_A, _ = recognize_sign(pts_A)
    print(f"Result A -> Letter: {letter_A}, Confidence: {conf_A:.2f}, Reason: {reason_A}")

    # Sign E
    pts_E = build_synthetic_hand("Right")
    for f in [5, 9, 13, 17]:
        set_finger_curl(pts_E, f, curl=True)
    pts_E[4] = [-0.08, 0.42, 0.0]
    letter_E, conf_E, reason_E, _ = recognize_sign(pts_E)
    print(f"Result E -> Letter: {letter_E}, Confidence: {conf_E:.2f}, Reason: {reason_E}")
