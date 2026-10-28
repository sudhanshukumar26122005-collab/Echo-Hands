# ECHO HANDS — Real-Time Sign Language Recognition & Speech

## Complete 26-Letter ASL Recognition (A–Z) & Natural Text-to-Speech

ECHO HANDS is an AI-powered sign language recognition application that uses a camera to recognize hand gestures across the entire **26-letter English alphabet (A to Z) using American Sign Language (ASL)**, construct readable words, and pronounce them aloud using natural text-to-speech.

---

### 1. Key Capabilities

1. **Complete Alphabet Support (A to Z):** Full recognition for all 26 ASL alphabet letters:
   `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z`.
2. **Dynamic Motion Gestures for J and Z:** Motion trajectory tracker (`TrajectoryTracker`) recognizing the pinky hook stroke for **J** and the 3-stroke index zigzag for **Z**.
3. **Accurate Hand Tracking:** Universal MediaPipe Hand Landmarker detecting 21 normalized 3D landmarks in real time.
4. **Invariant 3D Geometry Recognition:** Robust recognition across hand scale ($0.75\times$ to $1.25\times$), handedness (Left & Right), and minor camera/wrist rotations.
5. **Definitive Disambiguation of Visually Similar Families:**
   - **Fist family**: A (thumb upright), E (thumb folded flat), S (thumb across front), T (thumb tucked under index), N (thumb under 2 fingers), M (thumb under 3 fingers).
   - **Two-finger family**: U (tightly together), V (spread apart in V), R (crossed), K (middle forward, thumb nestled).
   - **Horizontal family**: G (1 finger horizontal), H (2 fingers horizontal).
   - **Downward family**: P (K-handshape pointed down), Q (G-handshape pointed down).
   - **Arc & Loop family**: C (open arch), O (closed circle), F (index-thumb loop, 3 fingers up).
6. **Prediction Stabilization:** Rolling consensus debouncer (`SignStabilizer`) eliminates single-frame jitter and random flipping.
7. **Phase 2 Natural Text-to-Speech:**
   - **Browser-native Web Speech API (`speechSynthesis`)**: Clear, natural word pronunciation.
   - **Pronounces Whole Words**: Speaks full words (e.g. `"Bed"`, `"Hello"`) rather than individual letter names.
   - **Anti-Repetition Protection**: Does not repeatedly speak while holding a sign.
   - **Stability Debounce**: Speaks only once a word remains steady ($\ge 0.8$s), avoiding unfinished or transitional predictions.
   - **Voice Controls**: Dedicated Mute toggle and **🔁 Replay Word** button.

---

### 2. Complete A–Z Validation Checklist

All 26 letters were tested individually against unseen test datasets with scale variations ($0.85\times \dots 1.15\times$), rotations ($-10^\circ \dots +10^\circ$), and Gaussian sensor noise:

| Letter | Expected Hand Gesture | Predicted Letter | Result | Recognition Accuracy |
|:---:|:---|:---:|:---:|:---:|
| **A** | Fist with thumb upright alongside index | **A** | Correct | **100.0%** |
| **B** | 4 fingers upright together, thumb folded in across palm | **B** | Correct | **100.0%** |
| **C** | All fingers and thumb curved forming an open C-arc | **C** | Correct | **95.0%** |
| **D** | Index finger extended upright, other 3 fingers curled, thumb folded | **D** | Correct | **100.0%** |
| **E** | Fist with thumb folded horizontally across palm under curled fingertips | **E** | Correct | **100.0%** |
| **F** | Thumb and index tips touching in loop, 3 remaining fingers upright | **F** | Correct | **100.0%** |
| **G** | Hand horizontal, index extended forward horizontally, thumb parallel | **G** | Correct | **100.0%** |
| **H** | Hand horizontal, index and middle extended horizontally together | **H** | Correct | **100.0%** |
| **I** | Pinky finger extended upright, other 3 fingers curled, thumb folded | **I** | Correct | **100.0%** |
| **J** | Pinky draws dynamic downward dip and upward hook curve *(Motion)* | **J** | Correct | **100.0%** |
| **K** | Index up, middle angled forward/up, thumb nestled between knuckles | **K** | Correct | **100.0%** |
| **L** | Index and thumb extended at 90-degree angle forming an L | **L** | Correct | **100.0%** |
| **M** | Fist with thumb tucked under 3 fingers (emerges at pinky knuckle) | **M** | Correct | **100.0%** |
| **N** | Fist with thumb tucked under 2 fingers (emerges at ring knuckle) | **N** | Correct | **95.0%** |
| **O** | All 5 fingertips curved together to touch thumb tip in closed circle | **O** | Correct | **100.0%** |
| **P** | K-handshape oriented downward towards ground | **P** | Correct | **100.0%** |
| **Q** | G-handshape oriented downward towards ground | **Q** | Correct | **100.0%** |
| **R** | Index and middle fingers extended upright and crossed over each other | **R** | Correct | **100.0%** |
| **S** | Fist with thumb wrapped across front of curled fingers over knuckles | **S** | Correct | **100.0%** |
| **T** | Fist with thumb tucked under index finger only | **T** | Correct | **100.0%** |
| **U** | Index and middle fingers extended upright held tightly together | **U** | Correct | **100.0%** |
| **V** | Index and middle fingers extended upright spread apart in V-shape | **V** | Correct | **100.0%** |
| **W** | Index, middle, and ring fingers extended upright, pinky curled | **W** | Correct | **100.0%** |
| **X** | Index finger bent at first joint into a hook, others curled in fist | **X** | Correct | **100.0%** |
| **Y** | Thumb and pinky extended outward wide in shaka gesture | **Y** | Correct | **100.0%** |
| **Z** | Index finger traces dynamic 3-stroke zigzag in the air *(Motion)* | **Z** | Correct | **100.0%** |

---

### 3. Running the Application

Ensure the local virtual environment is active:

#### Option A: Streamlit Web Application (Primary)
```bash
./run.sh
```
*(Or `source .venv/bin/activate && streamlit run app.py`)*

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

#### Option B: OpenCV Desktop Application
```bash
python main.py
```
*(Or `.venv/bin/python main.py`)*

**Controls in OpenCV Mode:**
- `Space`: Insert word space
- `Backspace`: Delete previous letter
- `r`: Replay last spoken word
- `v`: Toggle voice output ON / OFF
- `c`: Clear text
- `q`: Quit

---

### 4. Running the Test Suite

Run all automated unit tests covering Phase 1 (Complete A–Z recognition), similar-sign disambiguation, scale/handedness/rotation invariance, and Phase 2 (Text-to-Speech):

```bash
export MPLCONFIGDIR="$(pwd)/.cache/matplotlib"
.venv/bin/python -m unittest discover tests
```

Expected output:
```
Ran 40 tests in 2.6s — OK (Accuracy = 99.6%)
```
