import os
import math
import cv2
import numpy as np

# Standard hand landmarks connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"

class HandDetector:
    """
    Universal Hand Detector that supports both legacy MediaPipe solutions
    and modern MediaPipe Tasks (v0.10.30+ / v1.0+).
    """
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.backend = None

        import mediapipe as mp
        self.mp = mp

        # Check if legacy mp.solutions is available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            # Fall back to MediaPipe Tasks API
            self.backend = "tasks"
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
            if not os.path.exists(model_path):
                self._download_model(model_path)

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

        self.last_handedness = []

    def _download_model(self, dest_path):
        import urllib.request
        try:
            urllib.request.urlretrieve(MODEL_URL, dest_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download MediaPipe HandLandmarker model: {e}")

    def find_hands(self, frame):
        """
        Processes a BGR frame and returns a list of hands,
        where each hand is a list of 21 landmark tuples: (id, x_pixel, y_pixel, z_norm).
        Maintains backward compatibility with (id, x, y) indexing.
        """
        h, w, _ = frame.shape
        hands_landmarks = []
        self.last_handedness = []

        if self.backend == "solutions":
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                if results.multi_handedness:
                    for hd in results.multi_handedness:
                        label = hd.classification[0].label if hd.classification else "Right"
                        self.last_handedness.append(label)
                else:
                    self.last_handedness = ["Right"] * len(results.multi_hand_landmarks)

                for handLms in results.multi_hand_landmarks:
                    lm = []
                    for id, p in enumerate(handLms.landmark):
                        lm.append((id, int(p.x * w), int(p.y * h)))
                    hands_landmarks.append(lm)

        elif self.backend == "tasks":
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgRGB = np.ascontiguousarray(imgRGB)
            mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=imgRGB)
            results = self.detector.detect(mp_img)
            if results.handedness:
                for hd in results.handedness:
                    label = hd[0].category_name if hd else "Right"
                    self.last_handedness.append(label)
            else:
                self.last_handedness = ["Right"] * len(results.hand_landmarks or [])

            if results.hand_landmarks:
                for hand in results.hand_landmarks:
                    lm = []
                    for id, p in enumerate(hand):
                        lm.append((id, int(p.x * w), int(p.y * h)))
                    hands_landmarks.append(lm)

        return hands_landmarks

    def get_handedness(self, hand_idx=0):
        """Returns detected handedness ('Left' or 'Right') for given hand index."""
        if 0 <= hand_idx < len(self.last_handedness):
            return self.last_handedness[hand_idx]
        return "Right"

    def draw_landmarks(self, frame, lm, connection_color=(0, 255, 0), point_color=(0, 0, 255)):
        """
        Draws 21 hand landmarks and skeleton connections onto the frame.
        """
        # Draw skeleton lines
        for p1, p2 in HAND_CONNECTIONS:
            if p1 < len(lm) and p2 < len(lm):
                pt1 = (lm[p1][1], lm[p1][2])
                pt2 = (lm[p2][1], lm[p2][2])
                cv2.line(frame, pt1, pt2, connection_color, 2)

        # Draw landmark points
        for p in lm:
            x, y = int(p[1]), int(p[2])
            cv2.circle(frame, (x, y), 5, point_color, -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)

        return frame
