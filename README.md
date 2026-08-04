# Eye-Gaze Virtual Keyboard — Technical Documentation

## 1. Project Summary

This project is an **accessibility-focused, eye-gaze-controlled on-screen keyboard**. A webcam tracks the user's iris position and head orientation in real time, maps that gaze onto screen coordinates, and lets the user "type" by dwelling (fixating) on a virtual key for a set amount of time — no hands, mouse, or physical keyboard required.

It is built entirely on:
- **Pygame** — window, camera capture, UI rendering, input/event loop
- **MediaPipe Tasks (Face Landmarker)** — 478-point face mesh + iris landmarks
- **NumPy** — vector math and a hand-written ridge-regression calibration model

The repository contains one main application plus supporting scaffolding/test scripts:

| File | Role |
|---|---|
| `gaze_keyboard.py` | The full application (tracking thread, calibration, UI, keyboard, main loop) |
| `face_landmarker.task` | Pretrained MediaPipe face-mesh model (auto-downloaded if missing) |
| `gaze_calibration.json` | Persisted calibration model from a previous session |
| `test_cam.py` | Minimal script to verify the webcam is detected and can grab a frame |
| `test_mp.py` | Minimal script to verify MediaPipe can detect a face in a single captured frame |
| `cv2.py` | A **dummy stub** module (not real OpenCV — see §7) |
| `LICENSE` | GNU GPLv3 |

---

## 2. High-Level Architecture

The application runs **two concurrent loops** that communicate through a thread-safe shared dictionary:

```
┌─────────────────────────────┐          ┌──────────────────────────────┐
│   BACKGROUND THREAD          │          │        MAIN THREAD            │
│   run_tracker()               │          │        main()                 │
│                               │  shared  │                                │
│  Camera → MediaPipe →         │  state   │  Read gaze → Map to screen →  │
│  eye-ratio / head-pose        │  (dict + │  Smooth (One-Euro) → Hover /  │
│  extraction → blink/outlier   │  lock)   │  dwell detection → Keyboard   │
│  filtering → rolling median   │ ───────► │  actions → Pygame draw/flip   │
│                               │          │                                │
└─────────────────────────────┘          └──────────────────────────────┘
```

`shared_state` holds: `h`, `v` (raw horizontal/vertical iris ratios), `yaw`, `pitch` (head pose), `found` (is a usable reading available this frame), and `pip` (a small preview surface for the on-screen webcam picture-in-picture). A `threading.Lock` guards all reads/writes.

---

## 3. Workflow — Step by Step

### Step 1 — Startup
1. `main()` initializes Pygame and creates a 1024×768 window.
2. The `face_landmarker.task` model is downloaded from Google's public MediaPipe model bucket if it isn't already present locally.
3. The background tracker thread (`run_tracker`) is spawned; it owns the camera and the MediaPipe detector for the rest of the program's life.
4. A `GazeCalibrator` is constructed. Its constructor immediately tries to **load `gaze_calibration.json`** from disk — if screen resolution matches, a returning user skips calibration entirely.
5. Two keyboard layouts are built: a classic single-tier `VirtualKeyboard` (full QWERTY grid) and a two-tier `ZoomKeyboard` (zone-then-letter). The zoom layout is selected by default because its hit targets are ~8× larger.

### Step 2 — Per-Frame Tracking (background thread)
For every camera frame:
1. Grab a frame from the webcam via `pygame.camera`, horizontally flip it (mirror effect), and convert it into a MediaPipe `Image`.
2. Run `FaceLandmarker.detect_for_video()` to get 478 face landmarks.
3. `get_eye_ratios()` computes, for each eye independently:
   - **Iris center** — the mean of 5 dedicated iris landmarks (more stable than a single point).
   - **Horizontal/vertical ratio (`h`, `v`)** — iris center position relative to the eye-corner midpoint, normalized by eye width/height.
   - **Eye-aspect-ratio (`ear`)** — used to detect blinks (a closed eye collapses this ratio).
   - The two eyes' `h`/`v` values are averaged, but if they **disagree** beyond a threshold (a sign one eye's landmarks are unreliable — glare, occlusion, extreme angle), the frame's reading is flagged unusable.
4. `get_head_pose()` estimates **yaw** and **pitch** from the nose tip's position relative to the eye-corner midpoint (yaw) and forehead-chin midpoint (pitch). This is fed to the calibration model as separate features rather than being subtracted from the eye ratio by a fixed formula — letting the regression learn the true head-pose relationship per user/camera.
5. **Blink/occlusion handling:** if the eyes look closed or the two-eye readings disagree too much, the frame is marked not-OK. Rather than immediately reporting "gaze lost" (which would freeze or hide the cursor on every blink), the tracker **holds the last known-good reading** for up to 15 frames. Beyond that, tracking is genuinely reported lost.
6. **Rolling median filter:** the last 5 valid `h`, `v`, `yaw`, `pitch` readings are kept in `deque` buffers and median-filtered before being published, suppressing single-frame spikes.
7. The landmark points are drawn onto a small preview image (`pip`) for on-screen feedback, colored green (tracking) or amber (blink).
8. Results are written into `shared_state` under the lock.

### Step 3 — Calibration
Calibration is required before gaze can reliably control the cursor (mouse mode bypasses it).

- **13 spatial points**: a 3×3 core grid (15%/50%/85% of width/height) plus 4 true edge points (top/bottom/left/right centers), so the regression has real data support near screen boundaries instead of extrapolating.
- **4 head-pose sweep points**: the user keeps looking at the screen center while turning their head left/right/up/down. This is the *only* step that gives the yaw/pitch regression terms real variance to learn from — during the spatial grid the head is essentially still, so without this step the model would (correctly, but unhelpfully) learn to ignore head pose.
- For each point: the user presses **SPACEBAR**, the system discards the first `SETTLE_FRAMES` (10) samples (fixation settling), then collects `SAMPLES_PER_POINT` (20) samples. If within-point jitter (`std`) exceeds a threshold, the point is **automatically re-collected** (up to 3 retries) rather than accepting noisy ground truth.
- Once all 17 points are collected, `compute_mapping()` fits the model (see §4) and estimates accuracy via **leave-one-out cross-validation** in pixels, displayed to the user for 5 seconds after calibration ("Good" <40px, "Fair" <80px, "Poor" otherwise).
- The fitted model (`ax`, `ay` coefficients + normalization stats + screen size) is **persisted to `gaze_calibration.json`**.

### Step 4 — Live Cursor Mapping (main thread, every frame)
1. Read the latest `h`, `v`, `yaw`, `pitch`, `found` from `shared_state`.
2. If `mouse_mode` is on, the cursor simply follows the physical mouse (useful for testing/comparison).
3. Otherwise, `calibrator.map(h, v, yaw, pitch)` converts the standardized feature vector into raw screen `(x, y)` pixels via the fitted regression.
4. The raw mapped point is smoothed through a **One-Euro Filter** (adaptive low-pass filter — heavier smoothing when nearly still, lighter/faster response when moving quickly) to balance jitter suppression against input lag.

### Step 5 — Hover / Dwell-Click Detection
1. **Hit-testing** uses each key's `hit_rect` (the *full* grid cell, edge-to-edge — no dead zones between padded, smaller visual key rectangles).
2. **Hysteresis**: once a key is hovered, the cursor must leave a slightly shrunk version of that key's hit rect before hover switches — prevents flicker at boundaries.
3. **Gravity well**: if the cursor misses every key (e.g., slight overshoot past the keyboard edge), the nearest key within 60px is still captured, rather than losing the dwell attempt.
4. If the same key stays hovered for `dwell_limit` seconds (default 1.2s, adjustable 0.1–3.0s via on-screen −/+ buttons), it "clicks": the letter is typed, or the mapped action executes (SPACE, BACKSPACE, CLEAR, CALIBRATE, LAYOUT toggle, or zone navigation).
5. **Implicit recalibration**: every successful dwell-click (on a letter or a zone tile) is also fed back into the calibrator as a new training sample — the key's center is assumed to be where the user was actually looking. Outlier rejection discards a sample if the model's current prediction already misses that key by more than 220px (likely a mis-hit, not genuine fixation). Every 5 new implicit samples, the model is **automatically refit and re-saved**, so the mapping keeps improving the more the user types.

### Step 6 — Rendering
Each frame redraws: the typed-text panel (word-wrapped, last 3 lines), the mode/dwell/smoothing control buttons (with their own dwell-clickable hover state), the webcam PiP with a "NOT CALIBRATED" warning if applicable, the active keyboard layout with per-key hover/dwell progress bars, the calibration overlay (dots, progress ring, instructions) while calibrating, the post-calibration accuracy readout, and the gaze cursor itself.

### Step 7 — Shutdown
`ESC` or window close sets `shared_state['running'] = False`, which stops both the main loop and the tracker thread; the tracker thread is joined before Pygame quits.

---

## 4. The Calibration Model in Detail

`GazeCalibrator` fits **two independent ridge-regularized quadratic regressions** — one predicting screen X, one predicting screen Y — from an 8-term standardized feature vector:

```
features = [1, h, v, h², v², h·v, yaw, pitch]   (bias + linear + quadratic + cross + head pose)
```

Key design choices (documented directly in the code's docstrings):
- **Standardization** (zero mean, unit std per feature) before fitting, so ridge regularization penalizes all coefficients comparably.
- **Ridge regression (λ = 0.6)**, closed-form (`(AᵀA + λI)⁻¹ Aᵀb`, bias term left unpenalized) instead of plain least squares — with only ~13–17 calibration points feeding 8 parameters, unregularized least squares is close to interpolating noise; ridge keeps the fit stable.
- **Weighted combination of explicit and implicit data**: explicit calibration-point samples are weighted 3× higher than inferred (implicit) dwell-click samples, since the latter's "the user was looking at the key center" assumption is less certain.
- **Leave-one-out RMS accuracy estimate** (pixels), computed only over the clean explicit points, shown to the user immediately after calibration.
- **Disk persistence** (`gaze_calibration.json`): stores `ax`, `ay` coefficient arrays, the normalization stats, and the screen size the model was fit for (a resolution change invalidates the saved model on load).

`gaze_calibration.json` in this repo is a concrete example of a saved model — 8 coefficients each for `ax`/`ay`, normalization means/stds for `h`, `v`, `yaw`, `pitch`, and a `1024×768` screen size.

---

## 5. Supporting Subsystems

### One-Euro Filter (`OneEuroFilter`)
A standard adaptive filter (Casiez et al.) applied independently to cursor X and Y. Its cutoff frequency increases with the estimated speed of movement — heavy smoothing while the eye is nearly still (reduces jitter), lighter smoothing during fast movement (reduces lag). The `smoothing_alpha` UI control adjusts `min_cutoff` live.

### Two Keyboard Layouts
- **`VirtualKeyboard`** — classic single-tier full QWERTY + punctuation + action row (SPACE / BACKSPACE / CLEAR / CALIBRATE / LAYOUT), sized to use ~58% of the vertical screen.
- **`ZoomKeyboard`** — a two-stage layout: first dwell selects one of 6 "zones" of ~5 letters (shown as large tiles, e.g. "Q W E R T"); that dwell swaps to a second screen showing just those letters enlarged to fill most of the keyboard area, plus a BACK tile. This trades one extra dwell per character for roughly 3× the target area — worthwhile once gaze-mapping accuracy is no longer the dominant error source and hit-target size is.

Both layouts share the `VirtualKey` primitive, which separates a smaller **visual rect** (what's drawn, with padding gaps) from a larger, gapless **hit rect** (what dwell logic tests against), and tracks its own `hover_time`, `locked`, and click-animation state.

---

## 6. Data Flow Diagram

```
Webcam frame
   │
   ▼
MediaPipe FaceLandmarker (478 landmarks incl. iris)
   │
   ▼
get_eye_ratios()  ──► (h, v, yaw, pitch, ear, ok)
   │
   ├─ ok=False (blink/disagreement) → hold last-good reading (≤15 frames) or report lost
   ▼
Rolling median filter (last 5 frames)
   │
   ▼
shared_state  (thread-safe handoff)
   │
   ▼
GazeCalibrator.map(h, v, yaw, pitch)  ──► raw screen (x, y)
   │
   ▼
One-Euro Filter (adaptive smoothing)  ──► cursor (x, y)
   │
   ▼
Hit-test against active keyboard (hysteresis + gravity well)
   │
   ▼
Dwell timer ≥ dwell_limit? ──► Yes ──► Key action (type char / navigate / toggle)
   │                                        │
   No → keep drawing progress bar           ▼
                                   GazeCalibrator.refine() (implicit recalibration)
```

---

## 7. Notes on the Auxiliary / Stub Files

- **`cv2.py`**: This is **not the real OpenCV library** — it's a lightweight dummy module that intercepts `import cv2` and returns harmless no-op objects (a fake `VideoCapture` that never opens, and constants like `COLOR_*`/`FONT_*` resolving to `0`). It exists purely so that any code path referencing `cv2` doesn't crash with an `ImportError` in an environment where real OpenCV isn't installed — `gaze_keyboard.py` itself does not actually use OpenCV; all camera/image work goes through `pygame.camera`.
- **`test_cam.py`**: A standalone sanity check — lists available cameras via `pygame.camera` and confirms a single frame can be captured.
- **`test_mp.py`**: A standalone sanity check that chains the camera capture into a single `FaceLandmarker.detect()` call (static-image mode) to confirm the model downloads and a face can be found — this is essentially a minimal, one-shot version of what `run_tracker()` does continuously in `VIDEO` mode.
- **`LICENSE`**: GNU General Public License v3.0.

---

## 8. Dependencies & Setup

```bash
pip install pygame mediapipe numpy
python gaze_keyboard.py
```

- On first run, `face_landmarker.task` (~a few MB) is downloaded automatically from Google's public MediaPipe model storage if not already present in the working directory.
- A working webcam accessible to `pygame.camera` is required.
- Recommended: run calibration (`CALIBRATE` key/button) in consistent, even lighting; poor lighting is the most common cause of a "Poor" accuracy readout.

---

## 9. Summary of the Design's Key Robustness Features

| Problem | Mitigation |
|---|---|
| Single-eye tracking noise | Average both eyes + iris 5-point averaging |
| One eye occluded/unreliable | Two-eye disagreement check discards the frame |
| Blinks | EAR threshold detection + short hold of last-good reading |
| Frame-to-frame jitter | Rolling median (pre-filter) + One-Euro filter (post-mapping) |
| Head movement corrupting gaze estimate | Yaw/pitch as explicit regression features, calibrated via a dedicated head-tilt sweep |
| Overfitting with few calibration points | Feature standardization + ridge regularization |
| Small/hard-to-hit keys | Gapless hit-rects, hysteresis, gravity-well capture, and an optional zoom two-tier layout |
| Calibration drift over a session | Implicit recalibration from dwell-clicks, weighted lower than explicit points, with outlier rejection |
| Re-launching the app each time | Calibration model persisted to/loaded from `gaze_calibration.json` |
| Unknown calibration quality | Leave-one-out pixel-accuracy estimate shown to the user |