"""
ECHO HANDS — Complete ASL Alphabet Sign Recognizer (A–Z)
Supports all 26 English letters in American Sign Language (ASL).

Features:
- Complete 26-letter recognition: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z.
- Scale-, rotation-, and translation-invariant 3D hand geometry.
- Disambiguation of visually similar families:
    * Fist family: A, E, S, T, N, M
    * Two-finger family: U, V, R, K, D
    * Horizontal & Downward families: G, H, Q, P
    * Arc & Loop family: C, O, F
    * Hook: X
    * Pinky & Shaka: I, Y
- Dynamic gesture recognition for J and Z with TrajectoryTracker.
- Prediction debouncing with SignStabilizer.
"""

from collections import deque
import math
import time
import numpy as np


class TrajectoryTracker:
    """
    Tracks fingertip motion over rolling time window (last 20-30 frames)
    to recognize dynamic ASL gestures:
    - 'J': Pinky tip traces a downward dip followed by an upward hook.
    - 'Z': Index tip traces a 3-stroke zigzag: right -> down-left -> right.
    """
    def __init__(self, max_frames=25):
        self.pinky_trail = deque(maxlen=max_frames)
        self.index_trail = deque(maxlen=max_frames)
        self.last_detected_gesture = ""
        self.cooldown_until = 0.0

    def update(self, pts, palm_len, u_up, u_thumb):
        now = time.time()
        if now < self.cooldown_until:
            return ""

        if pts is None or len(pts) < 21:
            self.reset()
            return ""

        # Extract normalized 2D projection in world plane
        wrist = pts[0][:2]
        idx_tip = pts[8][:2]
        pnk_tip = pts[20][:2]

        # Record position relative to wrist in units of palm_len
        rel_idx = (idx_tip - wrist) / max(palm_len, 1e-4)
        rel_pnk = (pnk_tip - wrist) / max(palm_len, 1e-4)

        self.index_trail.append((now, float(rel_idx[0]), float(rel_idx[1])))
        self.pinky_trail.append((now, float(rel_pnk[0]), float(rel_pnk[1])))

        # 1. Check for dynamic 'J' (requires at least 10 frames)
        if len(self.pinky_trail) >= 10:
            if self._check_j_motion():
                self.cooldown_until = now + 1.2
                self.reset()
                return "J"

        # 2. Check for dynamic 'Z' (requires at least 12 frames)
        if len(self.index_trail) >= 12:
            if self._check_z_motion():
                self.cooldown_until = now + 1.2
                self.reset()
                return "Z"

        return ""

    def _check_j_motion(self):
        """
        Detects 'J' hook motion in world coordinates:
        Starts high (+y) -> moves downward (y drops to min_y) -> sweeps upward (y increases to end).
        """
        pts = list(self.pinky_trail)
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]

        min_y = min(ys)
        max_y = max(ys)
        y_range = max_y - min_y

        if y_range < 0.18:
            return False

        # Deepest dip of hook in world coordinates
        dip_idx = ys.index(min_y)
        if dip_idx < 2 or dip_idx > len(pts) - 2:
            return False

        # First segment: downward movement from start to dip
        down_dy = ys[0] - min_y
        # Second segment: upward recovery (hook) from dip to end
        up_dy = ys[-1] - min_y

        if down_dy > 0.15 and up_dy > 0.08:
            return True

        return False

    def _check_z_motion(self):
        """
        Detects 'Z' zigzag motion:
        Stroke 1: Moves right (x increases)
        Stroke 2: Moves diagonally down-left (x decreases, y decreases in world coordinates)
        Stroke 3: Moves right (x increases)
        """
        pts = list(self.index_trail)
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]

        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        if x_span < 0.16 or y_span < 0.10:
            return False

        p0 = pts[0]
        p_end = pts[-1]

        # Find peak right in first half (end of stroke 1)
        first_half = pts[:len(pts)//2]
        p_top_right = max(first_half, key=lambda p: p[1])

        # Find lowest/leftmost turn in second half (end of stroke 2)
        second_half = pts[len(pts)//3:]
        p_bot_left = min(second_half, key=lambda p: p[1])

        stroke1_dx = p_top_right[1] - p0[1]
        stroke2_dx = p_bot_left[1] - p_top_right[1]
        stroke2_dy = p_bot_left[2] - p_top_right[2]
        stroke3_dx = p_end[1] - p_bot_left[1]

        # Zigzag criteria: right -> down-left -> right
        if stroke1_dx > 0.10 and stroke2_dx < -0.10 and stroke2_dy < -0.06 and stroke3_dx > 0.10:
            return True

        return False

    def reset(self):
        self.pinky_trail.clear()
        self.index_trail.clear()


class SignStabilizer:
    """
    Temporal debouncer and smoothing buffer for 26-letter alphabet recognition.
    """
    def __init__(self, window_size=5, consensus_ratio=0.6):
        self.window_size = window_size
        self.consensus_ratio = consensus_ratio
        self.history = deque(maxlen=window_size)
        self.last_stable_letter = ""
        self.last_stable_conf = 0.0

    def update(self, letter, confidence):
        if not letter or confidence < 0.55:
            self.history.append("")
        else:
            self.history.append(letter)

        if not self.history:
            return "", 0.0

        counts = {}
        for l in self.history:
            if l:
                counts[l] = counts.get(l, 0) + 1

        if not counts:
            self.last_stable_letter = ""
            self.last_stable_conf = 0.0
            return "", 0.0

        best_letter, best_count = max(counts.items(), key=lambda item: item[1])
        ratio = best_count / len(self.history)

        if ratio >= self.consensus_ratio:
            self.last_stable_letter = best_letter
            self.last_stable_conf = confidence
            return best_letter, confidence
        else:
            return "", 0.0

    def reset(self):
        self.history.clear()
        self.last_stable_letter = ""
        self.last_stable_conf = 0.0


# Module-level trajectory tracker instance
_global_trajectory_tracker = TrajectoryTracker()


def compute_hand_geometry(landmarks, coord_system="auto"):
    """
    Extracts comprehensive 3D invariant hand features across all 26 alphabet letters.
    Supports camera image coordinates (Y-down) and Cartesian world coordinates (Y-up) seamlessly.
    """
    if isinstance(landmarks, np.ndarray):
        pts = landmarks.astype(np.float32).copy()
    else:
        pts = []
        for p in landmarks:
            if len(p) >= 4:
                pts.append([p[1], p[2], p[3]])
            elif len(p) == 3:
                # Detect (id, x, y) vs [x, y, z]
                if isinstance(p[0], (int, np.integer)) and p[0] in range(21) and len(landmarks) == 21 and landmarks[1][0] == 1:
                    pts.append([p[1], p[2], 0.0])
                else:
                    pts.append([p[0], p[1], p[2]])
            else:
                pts.append([p[0], p[1], 0.0])
        pts = np.array(pts, dtype=np.float32)

    # Determine coordinate convention:
    # Camera image coordinates have Y increasing downward (0 at top of frame, H at bottom).
    # Cartesian coordinates have Y increasing upward (ceiling).
    if coord_system == "image":
        is_image = True
    elif coord_system == "cartesian":
        is_image = False
    else:
        # Camera pixel coordinates have large values or non-origin wrist
        is_image = bool(np.linalg.norm(pts[0]) > 0.05 or np.max(pts[:, 1]) > 5.0)

    # Convert to unified World Space (+Y is UP towards ceiling, -Y is DOWN towards floor)
    if is_image:
        pts[:, 1] = -pts[:, 1]

    wrist = pts[0]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    ring_mcp = pts[13]
    pinky_mcp = pts[17]

    palm_len = float(np.linalg.norm(middle_mcp - wrist))
    if palm_len < 1e-4:
        palm_len = 1.0

    # Axes
    v_up = middle_mcp - wrist
    norm_up = np.linalg.norm(v_up)
    u_up = v_up / (norm_up if norm_up > 1e-5 else 1.0)

    # In World Space: +y points UP towards ceiling, -y points DOWN towards floor
    is_pointing_down = bool(u_up[1] < -0.40) # Hand pointing downward
    is_pointing_up = bool(u_up[1] > 0.40)   # Hand pointing upward
    is_horizontal = bool(abs(u_up[0]) > 0.65) # Hand pointing sideways/horizontal

    v_thumb = index_mcp - pinky_mcp
    norm_thumb = np.linalg.norm(v_thumb)
    u_thumb = v_thumb / (norm_thumb if norm_thumb > 1e-5 else 1.0)

    # 4 Fingers: Index(8), Middle(12), Ring(16), Pinky(20)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    finger_mcps = [5, 9, 13, 17]

    extended = []
    curled = []
    curved = []
    hooked = []

    for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
        d_tip_wrist = np.linalg.norm(pts[tip] - wrist) / palm_len
        d_pip_wrist = np.linalg.norm(pts[pip] - wrist) / palm_len
        d_tip_mcp = np.linalg.norm(pts[tip] - pts[mcp]) / palm_len
        proj_up = np.dot(pts[tip] - pts[mcp], u_up) / palm_len

        is_ext = bool((d_tip_wrist > d_pip_wrist + 0.10) and (proj_up > 0.35) and (d_tip_mcp > 0.55))
        is_crl = bool(d_tip_mcp < 0.45 or (d_tip_wrist <= d_pip_wrist + 0.05 and proj_up < 0.22))
        is_crv = bool(0.40 <= d_tip_mcp <= 0.85 and 0.10 <= proj_up <= 0.50 and not is_ext and not is_crl)

        # Hook shape (specifically for X): PIP is extended, but DIP/TIP is bent at 90°
        d_pip_mcp = np.linalg.norm(pts[pip] - pts[mcp]) / palm_len
        d_tip_pip = np.linalg.norm(pts[tip] - pts[pip]) / palm_len
        is_hook = bool(proj_up > 0.10 and d_pip_mcp > 0.18 and d_tip_pip < 0.42 and not is_ext)

        extended.append(is_ext)
        curled.append(is_crl)
        curved.append(is_crv)
        hooked.append(is_hook)

    # Thumb Metrics
    d_thumb_wrist = float(np.linalg.norm(pts[4] - wrist) / palm_len)
    d_thumb_mcp = float(np.linalg.norm(pts[4] - pts[2]) / palm_len)
    thumb_proj_up = float(np.dot(pts[4] - pts[2], u_up) / palm_len)
    thumb_proj_thumb = float(np.dot(pts[4] - pts[2], u_thumb) / palm_len)
    thumb_tip_proj_up = float(np.dot(pts[4] - pts[5], u_up) / palm_len)

    # Distances from thumb tip to knuckles & fingertips
    d_thumb_index_tip = float(np.linalg.norm(pts[4] - pts[8]) / palm_len)
    d_thumb_middle_tip = float(np.linalg.norm(pts[4] - pts[12]) / palm_len)
    d_thumb_ring_tip = float(np.linalg.norm(pts[4] - pts[16]) / palm_len)
    d_thumb_pinky_tip = float(np.linalg.norm(pts[4] - pts[20]) / palm_len)

    d_thumb_index_mcp = float(np.linalg.norm(pts[4] - index_mcp) / palm_len)
    d_thumb_middle_mcp = float(np.linalg.norm(pts[4] - middle_mcp) / palm_len)
    d_thumb_ring_mcp = float(np.linalg.norm(pts[4] - ring_mcp) / palm_len)
    d_thumb_pinky_mcp = float(np.linalg.norm(pts[4] - pinky_mcp) / palm_len)

    # Inter-finger relationships
    d_index_middle_tip = float(np.linalg.norm(pts[8] - pts[12]) / palm_len)

    # Finger crossing: check if middle finger crossed over index finger (Sign R)
    crossing_metric = float(np.dot(pts[12] - pts[8], u_thumb) / palm_len)
    is_crossed = bool(crossing_metric > 0.05 and d_index_middle_tip < 0.35)

    thumb_is_extended = bool(thumb_proj_thumb > 0.35 or (thumb_proj_up > 0.20 and thumb_tip_proj_up > -0.20))
    binary_fingers = [1 if thumb_is_extended else 0] + [1 if e else 0 for e in extended]

    return {
        "palm_len": palm_len,
        "u_up": u_up,
        "u_thumb": u_thumb,
        "is_pointing_down": is_pointing_down,
        "is_pointing_up": is_pointing_up,
        "is_horizontal": is_horizontal,
        "extended": extended, # [index, middle, ring, pinky]
        "curled": curled,
        "curved": curved,
        "hooked": hooked,
        "thumb": {
            "d_thumb_wrist": d_thumb_wrist,
            "d_thumb_mcp": d_thumb_mcp,
            "thumb_proj_up": thumb_proj_up,
            "thumb_proj_thumb": thumb_proj_thumb,
            "thumb_tip_proj_up": thumb_tip_proj_up,
            "d_thumb_index_tip": d_thumb_index_tip,
            "d_thumb_middle_tip": d_thumb_middle_tip,
            "d_thumb_ring_tip": d_thumb_ring_tip,
            "d_thumb_pinky_tip": d_thumb_pinky_tip,
            "d_thumb_index_mcp": d_thumb_index_mcp,
            "d_thumb_middle_mcp": d_thumb_middle_mcp,
            "d_thumb_ring_mcp": d_thumb_ring_mcp,
            "d_thumb_pinky_mcp": d_thumb_pinky_mcp,
            "is_extended": thumb_is_extended
        },
        "d_index_middle_tip": d_index_middle_tip,
        "is_crossed": is_crossed,
        "binary_fingers": binary_fingers,
        "pts": pts
    }


def recognize_sign(landmarks, handedness="Right", tracker=None, coord_system="auto"):
    """
    Recognizes any of the 26 ASL alphabet signs (A–Z) from 21 MediaPipe hand landmarks.
    Handles static hand geometry, visual family disambiguation, and dynamic gestures (J, Z).

    Returns:
        letter (str): Recognized letter 'A'..'Z' or "" (unrecognized / uncertain)
        confidence (float): 0.0 - 1.0 confidence score
        explanation (str): Human-readable geometric rationale
        binary_fingers (list): [thumb, index, middle, ring, pinky] binary states
    """
    if landmarks is None or len(landmarks) < 21:
        return "", 0.0, "No hand landmarks detected", [0, 0, 0, 0, 0]

    geo = compute_hand_geometry(landmarks, coord_system=coord_system)
    ext = geo["extended"] # [index, middle, ring, pinky]
    crl = geo["curled"]
    crv = geo["curved"]
    hok = geo["hooked"]
    th = geo["thumb"]
    num_ext = sum(ext)
    binary_fingers = geo["binary_fingers"]

    # Use provided tracker or fallback to global tracker
    trk = tracker or _global_trajectory_tracker

    # -------------------------------------------------------------------------
    # 0. DYNAMIC GESTURE CHECK (J and Z)
    # -------------------------------------------------------------------------
    dynamic_letter = trk.update(geo["pts"], geo["palm_len"], geo["u_up"], geo["u_thumb"])
    # Dynamic 'J' requires pinky finger to be extended (I-handshape motion)
    if dynamic_letter == "J" and ext[3]:
        return "J", 0.95, "Dynamic 'J' hook motion with pinky completed", binary_fingers
    # Dynamic 'Z' requires index finger to be extended (Z-tracing motion)
    elif dynamic_letter == "Z" and ext[0]:
        return "Z", 0.95, "Dynamic 'Z' zigzag motion with index completed", binary_fingers

    # -------------------------------------------------------------------------
    # 1. DOWNWARD ORIENTED SIGNS: P and Q
    # -------------------------------------------------------------------------
    if geo["is_pointing_down"]:
        # Sign P: Same handshape as K (Index extended, Middle angled forward/down, thumb nestled), pointed DOWN
        if ext[0] and (ext[1] or not crl[1]) and crl[2] and crl[3]:
            return "P", 0.92, "K-handshape pointing downward ('P')", binary_fingers
        # Sign Q: Same handshape as G (Index and Thumb pointing parallel), pointed DOWN
        elif ext[0] and crl[1] and crl[2] and crl[3]:
            return "Q", 0.92, "G-handshape pointing downward ('Q')", binary_fingers
        else:
            return "", 0.40, "Hand pointing downward (unrecognized sign)", binary_fingers

    # -------------------------------------------------------------------------
    # 2. HORIZONTAL ORIENTED SIGNS: G and H
    # -------------------------------------------------------------------------
    if geo["is_horizontal"]:
        # Sign H: Index and Middle extended horizontally together, ring & pinky curled
        if ext[0] and ext[1] and not ext[2] and not ext[3]:
            return "H", 0.94, "Index and middle extended horizontally together ('H')", binary_fingers
        # Sign G: Index extended horizontally, thumb parallel, other 3 fingers curled
        elif ext[0] and not ext[1] and not ext[2] and not ext[3]:
            return "G", 0.94, "Index and thumb extended horizontally ('G')", binary_fingers

    # -------------------------------------------------------------------------
    # 3. CIRCULAR / ARCH SIGNS: C and O
    # -------------------------------------------------------------------------
    is_fist_thumb_upright = bool(th["thumb_proj_up"] > 0.10 and th["thumb_tip_proj_up"] > -0.25 and th["d_thumb_pinky_mcp"] > 0.60)
    all_curled_fist = bool(crl[0] and crl[1] and crl[2] and crl[3])

    # Check O: All fingertips curve together to touch thumb tip (closed loop with hollow inside)
    if (th["d_thumb_index_tip"] < 0.28 and th["d_thumb_middle_tip"] < 0.38 
            and not ext[0] and not ext[1] 
            and not is_fist_thumb_upright 
            and not all_curled_fist):
        return "O", 0.95, "All fingertips touching thumb tip in closed 'O'", binary_fingers

    # Check C: Fingers curved in an arch with wide open gap between thumb and fingers
    if sum(crv) >= 2 and not ext[0] and not ext[1]:
        if 0.30 <= th["d_thumb_index_tip"] <= 0.90 and not is_fist_thumb_upright and not all_curled_fist:
            return "C", 0.93, "Hand curved forming an open 'C' shape", binary_fingers

    # -------------------------------------------------------------------------
    # 4. CLOSED FIST FAMILY: A, E, S, T, N, M (0 extended non-thumb fingers)
    # -------------------------------------------------------------------------
    if num_ext == 0:
        # Check X Hook variation if index is half-bent and others curled
        if hok[0] and crl[1] and crl[2] and crl[3] and not ext[1] and not ext[2] and not ext[3]:
            return "X", 0.93, "Index finger bent into hook shape ('X')", binary_fingers

        # Disambiguate A, E, S, T, N, M:
        # Sign A: Thumb is upright alongside the index finger (pointing up)
        if th["thumb_proj_up"] > 0.10 and th["thumb_tip_proj_up"] > -0.30 and th["d_thumb_pinky_mcp"] > 0.60:
            return "A", 0.95, "Thumb upright alongside index, 4 fingers curled ('A')", [1, 0, 0, 0, 0]

        # Sign S: Thumb wrapped across the FRONT of curled fingers (resting over knuckles)
        if geo["pts"][4][2] > geo["pts"][9][2] + 0.04 and th["d_thumb_middle_mcp"] < 0.50:
            return "S", 0.93, "Thumb wrapped across front of curled fingers ('S')", [0, 0, 0, 0, 0]

        # Sign M: Thumb tucked under 3 fingers (tip closest to pinky MCP)
        if th["d_thumb_pinky_mcp"] <= th["d_thumb_middle_mcp"] and th["d_thumb_pinky_mcp"] < 0.35:
            return "M", 0.92, "Thumb tucked under 3 fingers ('M')", [0, 0, 0, 0, 0]

        # Sign N: Thumb tucked under 2 fingers (tip closest to middle MCP)
        if th["d_thumb_middle_mcp"] < th["d_thumb_index_mcp"] and th["d_thumb_middle_mcp"] < 0.35:
            return "N", 0.92, "Thumb tucked under 2 fingers ('N')", [0, 0, 0, 0, 0]

        # Sign T: Thumb tucked under index finger (tip closest to index MCP)
        if th["d_thumb_index_mcp"] < th["d_thumb_pinky_mcp"] and th["d_thumb_index_mcp"] < 0.38:
            return "T", 0.92, "Thumb tucked under index finger ('T')", [0, 0, 0, 0, 0]

        # Sign E: Thumb folded horizontally across the palm tucked under curled fingertips
        return "E", 0.94, "Thumb folded across palm under curled fingers ('E')", [0, 0, 0, 0, 0]

    # -------------------------------------------------------------------------
    # 5. SINGLE FINGER EXTENDED: D, I, L, X
    # -------------------------------------------------------------------------
    # Only Index extended
    if ext[0] and not ext[1] and not ext[2] and not ext[3]:
        # Sign L: Thumb extended outward at ~90 degrees
        if th["thumb_proj_thumb"] > 0.35 and th["d_thumb_index_tip"] > 0.55:
            return "L", 0.96, "Thumb and index extended in 'L' shape ('L')", [1, 1, 0, 0, 0]
        # Sign D: Index up, thumb touching middle finger
        else:
            return "D", 0.93, "Index finger up, others curled ('D')", [0, 1, 0, 0, 0]

    # Only Pinky extended
    if not ext[0] and not ext[1] and not ext[2] and ext[3]:
        # Sign Y: Thumb also extended outward wide ('shaka')
        if (th["thumb_proj_thumb"] > 0.25 or th["d_thumb_index_mcp"] > 0.45) and th["d_thumb_mcp"] > 0.38:
            return "Y", 0.96, "Thumb and pinky extended out ('shaka' / 'Y')", [1, 0, 0, 0, 1]
        # Sign I: Only pinky up, thumb folded in
        else:
            return "I", 0.95, "Pinky finger up, others curled ('I')", [0, 0, 0, 0, 1]

    # -------------------------------------------------------------------------
    # 6. TWO FINGERS EXTENDED: U, V, R, K
    # -------------------------------------------------------------------------
    if ext[0] and ext[1] and not ext[2] and not ext[3]:
        # Sign R: Index and Middle are crossed over each other
        if geo["is_crossed"]:
            return "R", 0.94, "Index and middle fingers crossed ('R')", [0, 1, 1, 0, 0]

        # Sign K: Middle forward, thumb nestled between index and middle
        if (th["d_thumb_index_mcp"] < 0.45 or th["d_thumb_middle_mcp"] < 0.45) and th["thumb_proj_up"] > 0.15:
            return "K", 0.93, "Index up, middle forward, thumb nestled ('K')", [0, 1, 1, 0, 0]

        # Sign U: Index and Middle held tightly together
        if geo["d_index_middle_tip"] < 0.17:
            return "U", 0.94, "Index and middle fingers up held together ('U')", [0, 1, 1, 0, 0]

        # Sign V: Index and Middle spread apart in 'V' / peace sign
        else:
            return "V", 0.95, "Index and middle fingers up spread apart ('V')", [0, 1, 1, 0, 0]

    # -------------------------------------------------------------------------
    # 7. THREE FINGERS EXTENDED: F, W
    # -------------------------------------------------------------------------
    # Sign F: Middle, Ring, Pinky extended (3 fingers), Thumb and Index tips touch
    if not ext[0] and ext[1] and ext[2] and ext[3]:
        if th["d_thumb_index_tip"] < 0.40:
            return "F", 0.96, "Thumb & index touching, 3 fingers up ('F')", [0, 0, 1, 1, 1]
        else:
            return "F", 0.88, "3 fingers up (F variation)", [0, 0, 1, 1, 1]

    # Sign W: Index, Middle, Ring extended, Pinky curled
    if ext[0] and ext[1] and ext[2] and not ext[3]:
        return "W", 0.95, "Index, middle, and ring up ('W')", [0, 1, 1, 1, 0]

    # -------------------------------------------------------------------------
    # 8. FOUR FINGERS EXTENDED: B
    # -------------------------------------------------------------------------
    if num_ext == 4:
        if th["thumb_proj_thumb"] < 0.42:
            return "B", 0.96, "4 fingers extended straight, thumb in ('B')", [0, 1, 1, 1, 1]
        else:
            return "", 0.40, "Open hand (neutral / non-sign)", [1, 1, 1, 1, 1]

    # -------------------------------------------------------------------------
    # 9. UNRECOGNIZED / TRANSITIONAL POSE
    # -------------------------------------------------------------------------
    return "", 0.0, "Unrecognized hand gesture", binary_fingers


def recognize_letter(fingers):
    """
    Legacy compatibility function for 5-finger binary tuples.
    """
    letter_map = {
        (1, 0, 0, 0, 0): "A",
        (0, 1, 1, 1, 1): "B",
        (0, 1, 0, 0, 0): "D",
        (0, 0, 0, 0, 0): "E",
        (0, 0, 1, 1, 1): "F",
        (0, 0, 0, 0, 1): "I",
        (1, 1, 0, 0, 0): "L",
        (0, 1, 1, 0, 0): "V",
        (0, 1, 1, 1, 0): "W",
        (1, 0, 0, 0, 1): "Y"
    }
    return letter_map.get(tuple(fingers), "")

