import os
import sys
import threading
import time
import math
import urllib.request
import numpy as np

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pygame.camera
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------------------------------------------------------
# 1. Download Model if missing
# ------------------------------------------------------------------
MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Face Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# ------------------------------------------------------------------
# 2. Eye Ratio Calculation (Enhanced: 5-iris-landmark avg + head-pose)
# ------------------------------------------------------------------
def get_head_pose(landmarks, img_w, img_h):
    """Estimate head yaw and pitch from stable face landmarks."""
    def pt3d(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w])
    
    nose_tip = pt3d(1)
    chin = pt3d(152)
    left_eye_outer = pt3d(33)
    right_eye_outer = pt3d(263)
    forehead = pt3d(10)
    
    # Horizontal: compare nose-tip x relative to midpoint of eye corners
    eye_mid_x = (left_eye_outer[0] + right_eye_outer[0]) / 2.0
    eye_width = abs(right_eye_outer[0] - left_eye_outer[0])
    if eye_width < 1.0:
        eye_width = 1.0
    yaw = (nose_tip[0] - eye_mid_x) / eye_width  # positive = looking right
    
    # Vertical: compare nose-tip y relative to midpoint of forehead-chin
    face_mid_y = (forehead[1] + chin[1]) / 2.0
    face_height = abs(chin[1] - forehead[1])
    if face_height < 1.0:
        face_height = 1.0
    pitch = (nose_tip[1] - face_mid_y) / face_height  # positive = looking down
    
    return yaw, pitch

def get_eye_ratios(landmarks, img_w, img_h):
    """Returns (h, v, yaw, pitch, ear, ok).

    h, v      : raw horizontal/vertical iris ratios (NOT head-pose corrected here).
                Head pose is instead passed through as separate features (yaw, pitch)
                so the calibration regression can learn the true relationship between
                head movement and screen position, instead of relying on a hand-tuned
                fixed subtraction coefficient that only holds for one camera/user.
    ear       : eye-aspect-ratio (avg of both eyes) used to detect blinks/occlusion.
    ok        : False if eyes look closed, landmarks look unreliable, or the two eyes'
                independent readings disagree too much (a strong signal that one eye's
                landmarks are compromised) -> caller should discard this frame.
    """
    def get_pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * img_w, lm.y * img_h])
    
    def get_iris_center(indices):
        """Average all 5 iris landmarks for a stable center."""
        pts = [get_pt(i) for i in indices]
        return np.mean(pts, axis=0)
        
    def calc_ratio(outer, inner, top, bottom, iris):
        C = (outer + inner) / 2.0
        W = np.linalg.norm(outer - inner)
        H = np.linalg.norm(top - bottom)
        if W < 1.0 or H < 1.0: return 0.0, 0.0, 0.0
        h = (iris[0] - C[0]) / W
        v = (iris[1] - C[1]) / H
        ear = H / W  # eye-aspect-ratio proxy; drops sharply when eye closes
        return h, v, ear
        
    L_outer = get_pt(33); L_inner = get_pt(133); L_top = get_pt(159); L_bottom = get_pt(145)
    L_iris = get_iris_center([468, 469, 470, 471, 472])
    
    R_inner = get_pt(362); R_outer = get_pt(263); R_top = get_pt(386); R_bottom = get_pt(374)
    R_iris = get_iris_center([473, 474, 475, 476, 477])
    
    hl, vl, earl = calc_ratio(L_outer, L_inner, L_top, L_bottom, L_iris)
    hr, vr, earr = calc_ratio(R_outer, R_inner, R_top, R_bottom, R_iris)
    
    avg_h = (hl + hr) / 2.0
    avg_v = (vl + vr) / 2.0
    avg_ear = (earl + earr) / 2.0
    # If the two eyes' independently-computed ratios disagree a lot, blindly averaging
    # them produces a garbage midpoint that isn't where either eye is actually looking.
    # Large disagreement is a reliable tell for one eye being occluded, glare-affected,
    # or foreshortened by an extreme head angle.
    disagreement = math.hypot(hl - hr, vl - vr)
    
    yaw, pitch = get_head_pose(landmarks, img_w, img_h)

    # Reject frames where eyes look closed/near-closed (blink), landmarks are
    # degenerate (e.g. extreme angle collapsing eye width to near-zero), or the
    # two eyes disagree too much to trust the averaged reading.
    EAR_BLINK_THRESHOLD = 0.12
    DISAGREEMENT_THRESHOLD = 0.18  # empirical starting point; h/v ratios span roughly ±0.3
    ok = (avg_ear > EAR_BLINK_THRESHOLD) and (disagreement < DISAGREEMENT_THRESHOLD)

    return avg_h, avg_v, yaw, pitch, avg_ear, ok

# ------------------------------------------------------------------
# 3. Background Tracker Thread
# ------------------------------------------------------------------
def run_tracker(shared_state):
    from collections import deque
    pygame.camera.init()
    cameras = pygame.camera.list_cameras()
    if not cameras:
        print("No camera found!")
        return
        
    cam = pygame.camera.Camera(cameras[0], (640, 480))
    cam.start()
    
    # Rolling median buffer for pre-filtering (rejects residual glitch spikes)
    MEDIAN_SIZE = 5
    h_buffer = deque(maxlen=MEDIAN_SIZE)
    v_buffer = deque(maxlen=MEDIAN_SIZE)
    yaw_buffer = deque(maxlen=MEDIAN_SIZE)
    pitch_buffer = deque(maxlen=MEDIAN_SIZE)

    # Last known-good reading, held through blinks/occlusion instead of
    # emitting garbage or freezing shared_state['found'] to False every blink.
    last_good = {'h': 0.0, 'v': 0.0, 'yaw': 0.0, 'pitch': 0.0}
    BLINK_HOLD_MAX_FRAMES = 15  # ~0.15-0.5s depending on fps; beyond this, admit we lost tracking
    blink_hold_count = 0
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1)
        
    with vision.FaceLandmarker.create_from_options(options) as detector:
        while shared_state['running']:
            img_surf = cam.get_image()
            if img_surf is None:
                time.sleep(0.01)
                continue
                
            img_surf = pygame.transform.flip(img_surf, True, False)
            w, h = img_surf.get_size()
            
            img_array = pygame.surfarray.array3d(img_surf)
            img_array = np.transpose(img_array, (1, 0, 2))
            img_array = np.ascontiguousarray(img_array)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
            
            timestamp_ms = int(time.time() * 1000)
            try:
                result = detector.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                result = None
                
            h_ratio, v_ratio, yaw, pitch = 0.0, 0.0, 0.0, 0.0
            found = False
            blinked = False

            if result and result.face_landmarks:
                landmarks = result.face_landmarks[0]
                raw_h, raw_v, raw_yaw, raw_pitch, ear, eyes_open = get_eye_ratios(landmarks, w, h)

                if eyes_open:
                    h_ratio, v_ratio, yaw, pitch = raw_h, raw_v, raw_yaw, raw_pitch
                    found = True
                    blink_hold_count = 0

                    # Apply rolling median pre-filter (rejects residual single-frame spikes)
                    h_buffer.append(h_ratio); v_buffer.append(v_ratio)
                    yaw_buffer.append(yaw); pitch_buffer.append(pitch)
                    if len(h_buffer) >= 3:
                        h_ratio = float(np.median(h_buffer))
                        v_ratio = float(np.median(v_buffer))
                        yaw = float(np.median(yaw_buffer))
                        pitch = float(np.median(pitch_buffer))

                    last_good = {'h': h_ratio, 'v': v_ratio, 'yaw': yaw, 'pitch': pitch}
                else:
                    # Likely a blink: hold the last good gaze estimate briefly instead
                    # of reporting "not found" (which would freeze/hide the cursor) or
                    # feeding a closed-eye reading into tracking (which would be noise).
                    blinked = True
                    blink_hold_count += 1
                    if blink_hold_count <= BLINK_HOLD_MAX_FRAMES:
                        h_ratio, v_ratio = last_good['h'], last_good['v']
                        yaw, pitch = last_good['yaw'], last_good['pitch']
                        found = True
                    else:
                        found = False  # eyes closed too long; genuinely lost

                # Draw all iris + eye landmarks on PiP for feedback
                iris_ids = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
                eye_ids = [33, 133, 159, 145, 362, 263, 386, 374]
                dot_color = (255, 200, 0) if blinked else (0, 255, 0)
                for idx in eye_ids:
                    lm = landmarks[idx]
                    pygame.draw.circle(img_surf, dot_color, (int(lm.x*w), int(lm.y*h)), 2)
                for idx in iris_ids:
                    lm = landmarks[idx]
                    pygame.draw.circle(img_surf, (0, 200, 255), (int(lm.x*w), int(lm.y*h)), 2)
            
            pip_surf = pygame.transform.scale(img_surf, (160, 120))
            
            with shared_state['lock']:
                if found:
                    shared_state['h'] = h_ratio
                    shared_state['v'] = v_ratio
                    shared_state['yaw'] = yaw
                    shared_state['pitch'] = pitch
                    shared_state['found'] = True
                else:
                    shared_state['found'] = False
                shared_state['pip'] = pip_surf
            
            time.sleep(0.01)
    cam.stop()

# ------------------------------------------------------------------
# 4. Calibration & Keyboard Classes
# ------------------------------------------------------------------
class GazeCalibrator:
    """Enhanced calibrator: 13-point grid (3x3 core + 4 true edge points), multi-sample
    averaging with a per-point steadiness check, ridge-regularized quadratic regression
    over standardized gaze + head-pose features, weighted to blend explicit calibration
    with implicit recalibration from dwell-click data, disk persistence, and a reported
    leave-one-out accuracy estimate.

    Why ridge + standardization: with only ~13 calibration points feeding an 8-parameter
    model (h, v, h^2, v^2, h*v, yaw, pitch + bias), and yaw/pitch barely varying while the
    user's head stays still during calibration, a plain least-squares fit is very close to
    just interpolating noise -- small measurement jitter produces wildly different, unstable
    coefficients and a mapping that looks fine on the calibration points themselves but is
    wrong everywhere else. Standardizing each feature and adding L2 regularization keeps
    coefficients bounded and shrinks the weight given to features (like yaw/pitch) that the
    calibration session didn't actually provide enough variation to pin down reliably.
    """
    SAMPLES_PER_POINT = 20       # frames to collect per calibration point
    SETTLE_FRAMES = 10           # frames to discard while user fixates
    IMPLICIT_BUFFER_SIZE = 50    # rolling buffer of implicit calibration samples
    REFIT_INTERVAL = 5           # refit mapping every N new implicit samples
    EXPLICIT_WEIGHT = 3.0        # trust deliberate calibration-point fixations more...
    IMPLICIT_WEIGHT = 1.0        # ...than inferred "they were probably looking at the key" data
    OUTLIER_REJECT_PX = 220      # skip an implicit sample if current model already misses by more than this
    CALIB_FILE = "gaze_calibration.json"
    N_FEATURES = 8                # [1, h, v, h^2, v^2, h*v, yaw, pitch]
    RIDGE_LAMBDA = 0.6              # L2 penalty in standardized feature space; tuned via
                                     # simulation against plain least-squares (see notes below) —
                                     # too low reverts to lstsq-like instability, too high underfits
    STD_FLOOR = 0.02               # prevents divide-by-near-zero when a feature barely varied
    MAX_POINT_STD = 0.045          # if raw h/v jitter within a point exceeds this, refixation is required
    MAX_POINT_STD_TILT = 0.075     # more lenient during head-tilt points: some fixation drift while
                                     # turning the head is normal (imperfect vestibulo-ocular reflex)
    MAX_POINT_RETRIES = 3          # give up and accept noisy data after this many re-collections

    def __init__(self, w, h):
        from collections import deque
        self.w, self.h_screen = w, h
        # 13-point layout: 3x3 core grid + 4 true edge/corner-adjacent points, so the
        # fit has real support near the screen boundaries instead of only extrapolating
        # a quadratic curve fit from points that stop at 15%/85% of the screen.
        self.points = []
        for vy in [0.15, 0.5, 0.85]:
            for vx in [0.15, 0.5, 0.85]:
                self.points.append((int(w * vx), int(h * vy)))
        for (vx, vy) in [(0.5, 0.04), (0.5, 0.96), (0.04, 0.5), (0.96, 0.5)]:
            self.points.append((int(w * vx), int(h * vy)))
        self.n_spatial_points = len(self.points)  # 13

        # Head-pose calibration sweep: same on-screen target (center), but the user is
        # asked to keep their eyes fixed on it while gently turning their head. This is
        # the only way to give the yaw/pitch regression terms real, decoupled variance to
        # learn from -- during the spatial grid above the head is essentially stationary,
        # so ridge regression correctly (but unhelpfully) shrinks those coefficients to
        # ~0, meaning head-pose compensation does nothing without this extra step.
        self.tilt_prompts = [
            "Turn your head slightly LEFT",
            "Turn your head slightly RIGHT",
            "Tilt your head slightly UP",
            "Tilt your head slightly DOWN",
        ]
        center = (int(w * 0.5), int(h * 0.5))
        for _ in self.tilt_prompts:
            self.points.append(center)
        
        self.calibrating = False
        self.point_idx = 0
        self.point_retries = 0
        self.needs_refixation = False  # UI flag: last collection window was too noisy
        self.recorded_h = []
        self.recorded_v = []
        self.recorded_yaw = []
        self.recorded_pitch = []
        self.recorded_sx = []
        self.recorded_sy = []
        # Feature vector: [1, h, v, h^2, v^2, h*v, yaw, pitch] (standardized before use)
        self.ax = np.zeros(self.N_FEATURES); self.ax[0] = w / 2; self.ax[1] = 5000
        self.ay = np.zeros(self.N_FEATURES); self.ay[0] = h / 2; self.ay[2] = 5000
        self.is_calibrated = False
        self.calib_rms_px = None  # leave-one-out accuracy estimate, set after calibration
        # Default (identity) feature normalization until a real fit provides better stats
        self._norm = dict(h_mean=0.0, h_std=1.0, v_mean=0.0, v_std=1.0,
                           yaw_mean=0.0, yaw_std=1.0, pitch_mean=0.0, pitch_std=1.0)
        
        # Multi-sample collection state
        self.collecting = False
        self.sample_buffer_h = []
        self.sample_buffer_v = []
        self.sample_buffer_yaw = []
        self.sample_buffer_pitch = []
        self.settle_counter = 0
        
        # Implicit recalibration state: (h, v, yaw, pitch, screen_x, screen_y)
        self.implicit_buffer = deque(maxlen=self.IMPLICIT_BUFFER_SIZE)
        self.implicit_count = 0  # counts samples since last refit

        self._try_load()

    def start(self):
        self.calibrating = True
        self.point_idx = 0
        self.point_retries = 0
        self.needs_refixation = False
        self.recorded_h = []
        self.recorded_v = []
        self.recorded_yaw = []
        self.recorded_pitch = []
        self.recorded_sx = []
        self.recorded_sy = []
        self.collecting = False
        self.sample_buffer_h = []
        self.sample_buffer_v = []
        self.sample_buffer_yaw = []
        self.sample_buffer_pitch = []
        self.settle_counter = 0
        
    def begin_collect(self):
        """Start collecting samples for the current point."""
        self.collecting = True
        self.sample_buffer_h = []
        self.sample_buffer_v = []
        self.sample_buffer_yaw = []
        self.sample_buffer_pitch = []
        self.settle_counter = 0
        
    def feed_sample(self, h, v, yaw=0.0, pitch=0.0):
        """Feed a gaze sample during collection. Returns progress 0.0-1.0, or -1 if done."""
        if not self.collecting:
            return 0.0
            
        self.settle_counter += 1
        if self.settle_counter <= self.SETTLE_FRAMES:
            return 0.0  # still settling, discard
            
        self.sample_buffer_h.append(h)
        self.sample_buffer_v.append(v)
        self.sample_buffer_yaw.append(yaw)
        self.sample_buffer_pitch.append(pitch)
        
        progress = len(self.sample_buffer_h) / self.SAMPLES_PER_POINT
        
        if len(self.sample_buffer_h) >= self.SAMPLES_PER_POINT:
            std_h = float(np.std(self.sample_buffer_h))
            std_v = float(np.std(self.sample_buffer_v))
            is_tilt_point = self.point_idx >= self.n_spatial_points
            limit = self.MAX_POINT_STD_TILT if is_tilt_point else self.MAX_POINT_STD
            unsteady = (std_h > limit or std_v > limit)

            if unsteady and self.point_retries < self.MAX_POINT_RETRIES:
                # The gaze signal wandered too much during this window -- likely the
                # user hadn't settled on the dot yet, glanced away mid-collection, or
                # blinked. Recollect rather than bake noisy ground truth into the fit.
                self.point_retries += 1
                self.needs_refixation = True
                self.sample_buffer_h = []
                self.sample_buffer_v = []
                self.sample_buffer_yaw = []
                self.sample_buffer_pitch = []
                self.settle_counter = 0
                return 0.0

            self.needs_refixation = False
            self.point_retries = 0

            # Average the collected samples (reject outliers with median)
            avg_h = float(np.median(self.sample_buffer_h))
            avg_v = float(np.median(self.sample_buffer_v))
            avg_yaw = float(np.median(self.sample_buffer_yaw))
            avg_pitch = float(np.median(self.sample_buffer_pitch))
            
            sx, sy = self.points[self.point_idx]
            self.recorded_h.append(avg_h)
            self.recorded_v.append(avg_v)
            self.recorded_yaw.append(avg_yaw)
            self.recorded_pitch.append(avg_pitch)
            self.recorded_sx.append(sx)
            self.recorded_sy.append(sy)
            
            self.collecting = False
            self.point_idx += 1
            
            if self.point_idx >= len(self.points):
                self.compute_mapping()
                self.calibrating = False
                self._save()
            return -1.0  # done with this point
            
        return progress
        
    def record_point(self, h, v, yaw=0.0, pitch=0.0):
        """Legacy single-call trigger (spacebar) that kicks off multi-frame collection
        for the current calibration point; the point isn't finalized until
        SAMPLES_PER_POINT frames later inside feed_sample."""
        if not self.calibrating: return
        self.begin_collect()
            
    def _fit_norm_stats(self, all_h, all_v, all_yaw, all_pitch):
        def stats(arr):
            arr = np.array(arr, dtype=float)
            return float(arr.mean()), float(max(arr.std(), self.STD_FLOOR))
        h_mean, h_std = stats(all_h)
        v_mean, v_std = stats(all_v)
        yaw_mean, yaw_std = stats(all_yaw)
        pitch_mean, pitch_std = stats(all_pitch)
        return dict(h_mean=h_mean, h_std=h_std, v_mean=v_mean, v_std=v_std,
                    yaw_mean=yaw_mean, yaw_std=yaw_std, pitch_mean=pitch_mean, pitch_std=pitch_std)

    def _feature_vec(self, h, v, yaw, pitch):
        n = self._norm
        hn = (h - n['h_mean']) / n['h_std']
        vn = (v - n['v_mean']) / n['v_std']
        yn = (yaw - n['yaw_mean']) / n['yaw_std']
        pn = (pitch - n['pitch_mean']) / n['pitch_std']
        return np.array([1, hn, vn, hn * hn, vn * vn, hn * vn, yn, pn])

    def _ridge_solve(self, A, b, lam):
        """Closed-form ridge regression, bias term (column 0) left unpenalized."""
        reg = lam * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        return np.linalg.solve(A.T @ A + reg, A.T @ b)

    def compute_mapping(self, source="calibration"):
        """Fit a ridge-regularized quadratic mapping (over standardized gaze ratios +
        head yaw/pitch) from all available data, weighting deliberate calibration
        fixations more than inferred implicit samples."""
        all_h, all_v, all_yaw, all_pitch = list(self.recorded_h), list(self.recorded_v), list(self.recorded_yaw), list(self.recorded_pitch)
        all_sx, all_sy = list(self.recorded_sx), list(self.recorded_sy)
        weights = [self.EXPLICIT_WEIGHT] * len(all_h)

        for (ih, iv, iyaw, ipitch, isx, isy) in self.implicit_buffer:
            all_h.append(ih); all_v.append(iv); all_yaw.append(iyaw); all_pitch.append(ipitch)
            all_sx.append(isx); all_sy.append(isy)
            weights.append(self.IMPLICIT_WEIGHT)
        
        n = len(all_h)
        if n < 4:
            return  # need at least a handful of points; ridge tolerates n < N_FEATURES

        # Recompute normalization from the current pool of data so standardized
        # features stay well-scaled as implicit samples come in over time.
        self._norm = self._fit_norm_stats(all_h, all_v, all_yaw, all_pitch)

        A = np.zeros((n, self.N_FEATURES))
        Bx = np.zeros(n)
        By = np.zeros(n)
        for i in range(n):
            A[i] = self._feature_vec(all_h[i], all_v[i], all_yaw[i], all_pitch[i])
            Bx[i] = all_sx[i]
            By[i] = all_sy[i]

        # Weighted ridge regression: scale rows by sqrt(weight) then regularize.
        w_sqrt = np.sqrt(np.array(weights))
        Aw = A * w_sqrt[:, None]
        Bxw = Bx * w_sqrt
        Byw = By * w_sqrt

        try:
            self.ax = self._ridge_solve(Aw, Bxw, self.RIDGE_LAMBDA)
            self.ay = self._ridge_solve(Aw, Byw, self.RIDGE_LAMBDA)
            self.is_calibrated = True
            if source == "calibration":
                self.calib_rms_px = self._estimate_accuracy_px()
                msg = f"Calibration complete! {n} points, ridge fit."
                if self.calib_rms_px is not None:
                    msg += f" Estimated accuracy: ~{self.calib_rms_px:.0f}px."
                print(msg)
            else:
                print(f"Implicit recalibration: {n} total points ({len(self.implicit_buffer)} implicit).")
        except np.linalg.LinAlgError:
            print("Calibration failed due to singular matrix.")

    def _estimate_accuracy_px(self):
        """Leave-one-out estimate of calibration accuracy in pixels, computed only over
        the explicit calibration points (clean ground truth). This is the number shown
        to the user so a bad calibration is visible instead of silently trusted."""
        n = len(self.recorded_h)
        if n < 6:
            return None
        errors = []
        for leave_i in range(n):
            idx = [i for i in range(n) if i != leave_i]
            hs = [self.recorded_h[i] for i in idx]; vs = [self.recorded_v[i] for i in idx]
            yaws = [self.recorded_yaw[i] for i in idx]; pitches = [self.recorded_pitch[i] for i in idx]
            sxs = [self.recorded_sx[i] for i in idx]; sys_ = [self.recorded_sy[i] for i in idx]

            A = np.array([self._feature_vec(hs[i], vs[i], yaws[i], pitches[i]) for i in range(len(hs))])
            try:
                ax_loo = self._ridge_solve(A, np.array(sxs), self.RIDGE_LAMBDA)
                ay_loo = self._ridge_solve(A, np.array(sys_), self.RIDGE_LAMBDA)
            except np.linalg.LinAlgError:
                continue

            vec = self._feature_vec(self.recorded_h[leave_i], self.recorded_v[leave_i],
                                     self.recorded_yaw[leave_i], self.recorded_pitch[leave_i])
            pred_x, pred_y = np.dot(ax_loo, vec), np.dot(ay_loo, vec)
            errors.append(math.hypot(pred_x - self.recorded_sx[leave_i], pred_y - self.recorded_sy[leave_i]))
        return float(np.mean(errors)) if errors else None
    
    def refine(self, h, v, screen_x, screen_y, yaw=0.0, pitch=0.0):
        """Add an implicit calibration sample from a dwell-triggered keystroke.
        The key center (screen_x, screen_y) is the assumed screen target.

        Outlier-rejected: if the model's *current* prediction for this gaze reading
        already lands far from the key that was actually typed, the sample is more
        likely a mis-hit or a fast corrective glance than genuine sustained fixation,
        so it's dropped rather than allowed to drag the mapping off course.
        """
        if not self.is_calibrated:
            return  # don't refine before initial calibration
        
        pred_x, pred_y = self.map(h, v, yaw, pitch)
        dist = math.hypot(pred_x - screen_x, pred_y - screen_y)
        if dist > self.OUTLIER_REJECT_PX:
            return

        self.implicit_buffer.append((h, v, yaw, pitch, screen_x, screen_y))
        self.implicit_count += 1
        
        # Refit every REFIT_INTERVAL new samples to avoid excessive computation
        if self.implicit_count >= self.REFIT_INTERVAL:
            self.implicit_count = 0
            self.compute_mapping(source="implicit")
            self._save()
            
    def map(self, h, v, yaw=0.0, pitch=0.0):
        vec = self._feature_vec(h, v, yaw, pitch)
        return int(np.dot(self.ax, vec)), int(np.dot(self.ay, vec))

    # -- Persistence -------------------------------------------------
    def _save(self):
        """Persist the fitted model so returning users skip full recalibration."""
        try:
            import json
            data = {
                'ax': self.ax.tolist(), 'ay': self.ay.tolist(),
                'norm': self._norm,
                'w': self.w, 'h_screen': self.h_screen,
            }
            with open(self.CALIB_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: could not save calibration ({e})")

    def _try_load(self):
        """Load a previous session's model if the screen resolution matches."""
        try:
            import json
            if not os.path.exists(self.CALIB_FILE):
                return
            with open(self.CALIB_FILE, 'r') as f:
                data = json.load(f)
            if data.get('w') != self.w or data.get('h_screen') != self.h_screen:
                return  # resolution changed; stale model would misbehave
            self.ax = np.array(data['ax'])
            self.ay = np.array(data['ay'])
            if 'norm' in data:
                self._norm = data['norm']
            self.is_calibrated = True
            print("Loaded calibration from previous session.")
        except Exception as e:
            print(f"Warning: could not load saved calibration ({e})")

# ------------------------------------------------------------------
# One-Euro Filter for adaptive smoothing
# ------------------------------------------------------------------
class OneEuroFilter:
    """
    Attempt to reduce jitter when stationary while staying responsive 
    when moving. Based on the 1€ Filter paper by Casiez et al.
    """
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
        
    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
        
    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
            
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1.0 / 60.0
        self.t_prev = t
        
        # Estimate derivative
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

class VirtualKey:
    def __init__(self, text, rect, hit_rect=None):
        self.text = text
        self.rect = pygame.Rect(rect)          # visual (padded) rect -- what gets drawn
        # hit_rect is what dwell/hover collision actually tests against. It defaults to
        # the visual rect, but VirtualKeyboard passes the *full* grid cell (no padding
        # gap) so the ~10-20px gaps between drawn keys aren't dead zones where a
        # slightly-off gaze registers nothing at all.
        self.hit_rect = pygame.Rect(hit_rect) if hit_rect is not None else self.rect
        self.hover_time = 0.0
        self.locked = False
        self.anim_timer = 0.0
        # optional metadata other components (e.g. zoom keyboard) can use for routing
        self.key_type = 'letter'
        self.payload = None
        
    def draw(self, surface, font, is_hovered, dwell_ratio, dt):
        color = (40, 40, 45)
        text_color = (255, 255, 255)
        
        if self.anim_timer > 0:
            self.anim_timer -= dt
            
        is_clicked = self.locked or self.anim_timer > 0
        draw_rect = self.rect.copy()
        
        if is_clicked:
            color = (0, 230, 118) # Neon Green for typed
            text_color = (0, 0, 0)
            draw_rect.inflate_ip(-8, -8) # Shrink animation
        elif is_hovered:
            color = (0, 210, 255) # Neon Blue
            text_color = (0, 0, 0)
            
        pygame.draw.rect(surface, color, draw_rect, border_radius=10)
        pygame.draw.rect(surface, (60, 60, 65), draw_rect, width=2, border_radius=10)
        
        txt_surf = font.render(self.text, True, text_color)
        txt_rect = txt_surf.get_rect(center=draw_rect.center)
        surface.blit(txt_surf, txt_rect)
        
        # Dwell progress bar at bottom
        if is_hovered and not is_clicked and dwell_ratio > 0:
            prog_rect = pygame.Rect(draw_rect.left + 5, draw_rect.bottom - 8, (draw_rect.width - 10) * dwell_ratio, 4)
            pygame.draw.rect(surface, (255, 50, 50), prog_rect, border_radius=2)

class VirtualKeyboard:
    def __init__(self, screen_w, screen_h):
        self.keys = []
        rows = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '?'],
            ['SPACE', 'BACKSPACE', 'CLEAR', 'CALIBRATE', 'LAYOUT']
        ]
        
        # Use more of the vertical space for the keyboard (was exactly half the
        # screen) so each key is physically bigger relative to typical gaze error.
        keyboard_y_start = int(screen_h * 0.42)
        keyboard_h = screen_h - keyboard_y_start
        row_h = keyboard_h // len(rows)
        
        for r_idx, row in enumerate(rows):
            n_keys = len(row)
            key_w = (screen_w - 40) // n_keys
            for c_idx, char in enumerate(row):
                x = 20 + c_idx * key_w
                y = keyboard_y_start + r_idx * row_h
                # Visual rect: small padding so keys look distinct.
                visual = (x + 5, y + 10, key_w - 10, row_h - 20)
                # Hit rect: the FULL cell, edge-to-edge with neighbors -- no dead zone.
                hit = (x, y, key_w, row_h)
                k = VirtualKey(char, visual, hit_rect=hit)
                k.key_type = 'action' if char in ('SPACE', 'BACKSPACE', 'CLEAR', 'CALIBRATE', 'LAYOUT') else 'letter'
                self.keys.append(k)

class ZoomKeyboard:
    """Two-tier ('zoom') keyboard: first dwell picks a zone of ~5 letters shown as a
    few large tiles; that dwell then swaps to a second screen showing just those
    letters blown up to fill most of the keyboard area, plus a BACK tile. This trades
    one extra dwell per character for roughly 3x the target area of the single-tier
    layout -- worthwhile once single-tier selection is still unreliable even after
    calibration and dead-zone fixes, since target size is now the dominant error
    source rather than gaze-mapping accuracy.
    """
    ZONES = [
        ['Q', 'W', 'E', 'R', 'T'],
        ['Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G'],
        ['H', 'J', 'K', 'L', ','],
        ['Z', 'X', 'C', 'V', 'B'],
        ['N', 'M', '.', '?'],
    ]
    ACTIONS = ['SPACE', 'BACKSPACE', 'CLEAR', 'CALIBRATE', 'LAYOUT']

    def __init__(self, screen_w, screen_h):
        self.w, self.h = screen_w, screen_h
        self.mode = 'zones'
        self.current_zone = None
        self.keys = []
        self._build_zone_view()

    def _action_row_top(self):
        return self.h - 90

    def _add_action_row(self):
        n = len(self.ACTIONS)
        y = self._action_row_top()
        cell_w = (self.w - 40) // n
        for i, label in enumerate(self.ACTIONS):
            x = 20 + i * cell_w
            visual = (x + 5, y + 10, cell_w - 10, 55)
            hit = (x, y, cell_w, 80)
            k = VirtualKey(label, visual, hit_rect=hit)
            k.key_type = 'action'
            self.keys.append(k)

    def _build_zone_view(self):
        self.mode = 'zones'
        self.current_zone = None
        self.keys = []

        area_top = int(self.h * 0.42)
        area_bottom = self._action_row_top()
        area_h = area_bottom - area_top
        cols, rows = 3, 2
        cell_w = (self.w - 40) // cols
        cell_h = area_h // rows

        for i, zone in enumerate(self.ZONES):
            r, c = divmod(i, cols)
            x = 20 + c * cell_w
            y = area_top + r * cell_h
            visual = (x + 6, y + 8, cell_w - 12, cell_h - 16)
            hit = (x, y, cell_w, cell_h)
            label = " ".join(zone)  # e.g. "Q W E R T" -- readable at a glance
            k = VirtualKey(label, visual, hit_rect=hit)
            k.key_type = 'zone'
            k.payload = i
            self.keys.append(k)

        self._add_action_row()

    def _build_letter_view(self, zone_idx):
        self.mode = 'letters'
        self.current_zone = zone_idx
        self.keys = []

        letters = self.ZONES[zone_idx]
        area_top = int(self.h * 0.42)
        area_bottom = self._action_row_top()
        area_h = area_bottom - area_top

        total_cells = len(letters) + 1  # + BACK tile
        cell_w = (self.w - 40) // total_cells

        for i, ch in enumerate(letters):
            x = 20 + i * cell_w
            visual = (x + 6, area_top + 8, cell_w - 12, area_h - 16)
            hit = (x, area_top, cell_w, area_h)
            k = VirtualKey(ch, visual, hit_rect=hit)
            k.key_type = 'letter'
            self.keys.append(k)

        # BACK tile, same size as the letters so it's just as easy/hard to hit
        x = 20 + len(letters) * cell_w
        visual = (x + 6, area_top + 8, cell_w - 12, area_h - 16)
        hit = (x, area_top, cell_w, area_h)
        back_k = VirtualKey("BACK", visual, hit_rect=hit)
        back_k.key_type = 'back'
        self.keys.append(back_k)

        self._add_action_row()

    def enter_zone(self, zone_idx):
        self._build_letter_view(zone_idx)

    def back_to_zones(self):
        self._build_zone_view()

# ------------------------------------------------------------------
# 5. Main App
# ------------------------------------------------------------------
def main():
    pygame.init()
    W, H = 1024, 768
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Eye-Gaze Virtual Keyboard")
    
    font_large = pygame.font.SysFont("segoeui", 48)
    font_keys = pygame.font.SysFont("segoeui", 32, bold=True)
    font_ui = pygame.font.SysFont("segoeui", 24)
    
    # State
    shared_state = {
        'running': True, 'lock': threading.Lock(),
        'h': 0.0, 'v': 0.0, 'yaw': 0.0, 'pitch': 0.0, 'found': False, 'pip': None
    }
    
    tracker_thread = threading.Thread(target=run_tracker, args=(shared_state,))
    tracker_thread.start()
    
    calibrator = GazeCalibrator(W, H)
    keyboard_classic = VirtualKeyboard(W, H)
    keyboard_zoom = ZoomKeyboard(W, H)
    use_zoom_layout = True  # zoom's tiles are ~8x the hit area of the classic layout
    keyboard = keyboard_zoom
    
    typed_text = ""
    dwell_limit = 1.2
    smoothing_alpha = 0.15  # controls One-Euro min_cutoff (lower = smoother)
    mouse_mode = False
    
    # One-Euro adaptive filters for cursor smoothing
    filter_x = OneEuroFilter(min_cutoff=smoothing_alpha, beta=0.007)
    filter_y = OneEuroFilter(min_cutoff=smoothing_alpha, beta=0.007)
    cursor_x, cursor_y = W / 2, H / 2
    last_hovered_key = None
    calib_progress = 0.0  # progress of current calibration point collection
    calib_result_timer = 0.0  # counts down while showing the post-calibration accuracy readout
    was_calibrating = False
    
    clock = pygame.time.Clock()
    
    ui_anim_timers = {'MODE': 0.0, 'DM': 0.0, 'DP': 0.0, 'SM': 0.0, 'SP': 0.0}
    
    # UI Rects
    btn_mode = pygame.Rect(20, 220, 160, 40)
    btn_dwell_minus = pygame.Rect(200, 220, 40, 40)
    btn_dwell_plus = pygame.Rect(400, 220, 40, 40)
    btn_smooth_minus = pygame.Rect(460, 220, 40, 40)
    btn_smooth_plus = pygame.Rect(660, 220, 40, 40)
    
    while shared_state['running']:
        dt = clock.tick(60) / 1000.0
        
        if calib_result_timer > 0:
            calib_result_timer -= dt
        if was_calibrating and not calibrator.calibrating:
            calib_result_timer = 5.0  # show the accuracy readout for 5s after finishing
        was_calibrating = calibrator.calibrating
        
        for k_tag in ui_anim_timers:
            if ui_anim_timers[k_tag] > 0: ui_anim_timers[k_tag] -= dt
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shared_state['running'] = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    shared_state['running'] = False
                elif event.key == pygame.K_SPACE and calibrator.calibrating:
                    # Spacebar to record calibration point
                    with shared_state['lock']:
                        h, v = shared_state['h'], shared_state['v']
                        yaw, pitch = shared_state['yaw'], shared_state['pitch']
                    calibrator.record_point(h, v, yaw, pitch)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Allow clicking keys in mouse mode
                    mx, my = pygame.mouse.get_pos()
                    for k in keyboard.keys:
                        if k.hit_rect.collidepoint(mx, my):
                            kt = getattr(k, 'key_type', 'letter')
                            if kt == 'zone': keyboard.enter_zone(k.payload)
                            elif kt == 'back': keyboard.back_to_zones()
                            elif k.text == 'BACKSPACE': typed_text = typed_text[:-1]
                            elif k.text == 'SPACE': typed_text += " "
                            elif k.text == 'CLEAR': typed_text = ""
                            elif k.text == 'CALIBRATE': calibrator.start()
                            elif k.text == 'LAYOUT':
                                use_zoom_layout = not use_zoom_layout
                                keyboard = keyboard_zoom if use_zoom_layout else keyboard_classic
                            else: typed_text += k.text
                            k.locked = True
                            k.anim_timer = 0.15
                    # Check UI buttons
                    if btn_mode.collidepoint(mx, my): 
                        mouse_mode = not mouse_mode
                        ui_anim_timers['MODE'] = 0.15
                    if btn_dwell_minus.collidepoint(mx, my): 
                        dwell_limit = max(0.1, dwell_limit - 0.1)
                        ui_anim_timers['DM'] = 0.15
                    if btn_dwell_plus.collidepoint(mx, my): 
                        dwell_limit = min(3.0, dwell_limit + 0.1)
                        ui_anim_timers['DP'] = 0.15
                    if btn_smooth_minus.collidepoint(mx, my): 
                        smoothing_alpha = max(0.1, smoothing_alpha - 0.05)
                        ui_anim_timers['SM'] = 0.15
                    if btn_smooth_plus.collidepoint(mx, my): 
                        smoothing_alpha = min(1.0, smoothing_alpha + 0.05)
                        ui_anim_timers['SP'] = 0.15
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    for k in keyboard.keys: k.locked = False
        
        # Get raw gaze
        with shared_state['lock']:
            raw_h = shared_state['h']
            raw_v = shared_state['v']
            raw_yaw = shared_state['yaw']
            raw_pitch = shared_state['pitch']
            found = shared_state['found']
            pip_surf = shared_state['pip']
            
        if mouse_mode:
            mx, my = pygame.mouse.get_pos()
            cursor_x, cursor_y = mx, my
        else:
            if found:
                # Feed samples to calibrator during auto-collection
                if calibrator.calibrating and calibrator.collecting:
                    calib_progress = calibrator.feed_sample(raw_h, raw_v, raw_yaw, raw_pitch)
                
                mapped_x, mapped_y = calibrator.map(raw_h, raw_v, raw_yaw, raw_pitch)
                # Update One-Euro filter min_cutoff from slider
                filter_x.min_cutoff = smoothing_alpha
                filter_y.min_cutoff = smoothing_alpha
                now = time.time()
                cursor_x = filter_x(mapped_x, now)
                cursor_y = filter_y(mapped_y, now)
            
        # Handle Hover Logic
        # Hysteresis: if the cursor is still inside a shrunk version of the currently-
        # hovered key's hit_rect, keep hovering it rather than re-scanning from scratch.
        # Uses hit_rect (the full, gapless cell) not the smaller visual rect.
        HYSTERESIS_MARGIN = 12
        GRAVITY_CAPTURE_RADIUS = 60  # px: how far outside every key a near-miss still snaps in

        def _dist_to_rect(rect, x, y):
            cx = min(max(x, rect.left), rect.right)
            cy = min(max(y, rect.top), rect.bottom)
            return math.hypot(x - cx, y - cy)

        hovered_key = None
        if last_hovered_key is not None and hasattr(last_hovered_key, 'hit_rect'):
            sticky_rect = last_hovered_key.hit_rect.inflate(-HYSTERESIS_MARGIN, -HYSTERESIS_MARGIN)
            if sticky_rect.collidepoint(cursor_x, cursor_y):
                hovered_key = last_hovered_key
        if hovered_key is None:
            for k in keyboard.keys:
                if k.hit_rect.collidepoint(cursor_x, cursor_y):
                    hovered_key = k
                    break
        if hovered_key is None:
            # Gravity well: the cursor missed every key, but if it's only just outside
            # one (e.g. overshot past the keyboard's outer edge), snap to the nearest
            # one anyway rather than registering "no key" and losing the dwell attempt.
            best_k, best_d = None, GRAVITY_CAPTURE_RADIUS
            for k in keyboard.keys:
                d = _dist_to_rect(k.hit_rect, cursor_x, cursor_y)
                if d < best_d:
                    best_d = d
                    best_k = k
            hovered_key = best_k
                
        # Also check UI controls for dwell
        controls = [(btn_mode, 'MODE'), (btn_dwell_minus, 'DM'), (btn_dwell_plus, 'DP'),
                   (btn_smooth_minus, 'SM'), (btn_smooth_plus, 'SP')]
        for r, tag in controls:
            if r.collidepoint(cursor_x, cursor_y):
                # Fake a key for dwell logic
                class FakeKey:
                    def __init__(self): 
                        self.text = tag
                        self.locked = False
                        self.hover_time = 0
                        self.anim_timer = 0
                if last_hovered_key and last_hovered_key.text == tag:
                    hovered_key = last_hovered_key
                else:
                    hovered_key = FakeKey()
                break

        if hovered_key:
            if hovered_key != last_hovered_key:
                if last_hovered_key: last_hovered_key.hover_time = 0.0
                hovered_key.hover_time = 0.0
                hovered_key.locked = False
                last_hovered_key = hovered_key
            else:
                if not hovered_key.locked:
                    hovered_key.hover_time += dt
                    if hovered_key.hover_time >= dwell_limit:
                        hovered_key.locked = True
                        hovered_key.anim_timer = 0.15
                        # Trigger Action
                        txt = hovered_key.text
                        key_type = getattr(hovered_key, 'key_type', 'letter')
                        is_keyboard_key = False
                        if key_type == 'zone':
                            keyboard.enter_zone(hovered_key.payload)
                            last_hovered_key = None  # keys list was just rebuilt; avoid stale hysteresis
                        elif key_type == 'back':
                            keyboard.back_to_zones()
                            last_hovered_key = None
                        elif txt == 'BACKSPACE': typed_text = typed_text[:-1]; is_keyboard_key = True
                        elif txt == 'SPACE': typed_text += " "; is_keyboard_key = True
                        elif txt == 'CLEAR': typed_text = ""; is_keyboard_key = True
                        elif txt == 'CALIBRATE': calibrator.start()
                        elif txt == 'LAYOUT':
                            use_zoom_layout = not use_zoom_layout
                            keyboard = keyboard_zoom if use_zoom_layout else keyboard_classic
                            last_hovered_key = None  # switched to a different keys list entirely
                        elif txt == 'MODE': 
                            mouse_mode = not mouse_mode
                            ui_anim_timers['MODE'] = 0.15
                        elif txt == 'DM': 
                            dwell_limit = max(0.1, dwell_limit - 0.1)
                            ui_anim_timers['DM'] = 0.15
                        elif txt == 'DP': 
                            dwell_limit = min(3.0, dwell_limit + 0.1)
                            ui_anim_timers['DP'] = 0.15
                        elif txt == 'SM': 
                            smoothing_alpha = max(0.1, smoothing_alpha - 0.05)
                            ui_anim_timers['SM'] = 0.15
                        elif txt == 'SP': 
                            smoothing_alpha = min(1.0, smoothing_alpha + 0.05)
                            ui_anim_timers['SP'] = 0.15
                        else: typed_text += txt; is_keyboard_key = True
                        
                        # Implicit recalibration: use this dwell as a data point. Zone
                        # tiles count too -- a zone dwell is still a deliberate fixation
                        # on a known screen location, even though it doesn't type a letter.
                        if (is_keyboard_key or key_type == 'zone') and not mouse_mode and hasattr(hovered_key, 'rect'):
                            key_cx = hovered_key.rect.centerx
                            key_cy = hovered_key.rect.centery
                            calibrator.refine(raw_h, raw_v, key_cx, key_cy, raw_yaw, raw_pitch)
        else:
            if last_hovered_key:
                last_hovered_key.hover_time = 0.0
                last_hovered_key.locked = False
                last_hovered_key = None
                
        # Draw Background
        screen.fill((15, 15, 18))
        
        # Draw Text Area
        text_rect = pygame.Rect(20, 20, W - 220, 180)
        pygame.draw.rect(screen, (30, 30, 35), text_rect, border_radius=15)
        # Render text with word wrap (simple)
        words = typed_text.split(" ")
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if font_large.size(test_line)[0] > text_rect.width - 20:
                lines.append(current_line)
                current_line = word + " "
            else:
                current_line = test_line
        lines.append(current_line)
        
        for i, line in enumerate(lines[-3:]): # show last 3 lines
            surf = font_large.render(line, True, (240, 240, 240))
            screen.blit(surf, (text_rect.x + 15, text_rect.y + 15 + i * 50))
            
        # Draw UI Buttons
        def draw_btn(r, label, tag):
            hover = r.collidepoint(cursor_x, cursor_y)
            is_clicked = ui_anim_timers[tag] > 0
            
            dwell = 0.0
            if hovered_key and hasattr(hovered_key, 'text') and hovered_key.text == tag:
                dwell = min(1.0, hovered_key.hover_time / dwell_limit)
                hover = True
                
            color = (0, 230, 118) if is_clicked else ((0, 210, 255) if hover else (50, 50, 55))
            draw_r = r.inflate(-4, -4) if is_clicked else r
            pygame.draw.rect(screen, color, draw_r, border_radius=8)
            surf = font_ui.render(label, True, (0, 0, 0) if (hover or is_clicked) else (200, 200, 200))
            screen.blit(surf, surf.get_rect(center=draw_r.center))
            
            if hover and not is_clicked and dwell > 0:
                prog_rect = pygame.Rect(draw_r.left + 5, draw_r.bottom - 6, (draw_r.width - 10) * dwell, 3)
                pygame.draw.rect(screen, (255, 50, 50), prog_rect, border_radius=2)
                
            return hover
            
        h_mode = draw_btn(btn_mode, f"Mode: {'Mouse' if mouse_mode else 'Gaze'}", 'MODE')
        h_dm = draw_btn(btn_dwell_minus, "-", 'DM')
        h_dp = draw_btn(btn_dwell_plus, "+", 'DP')
        # Dwell Label
        screen.blit(font_ui.render(f"Dwell: {dwell_limit:.1f}s", True, (200, 200, 200)), (250, 225))
        
        h_sm = draw_btn(btn_smooth_minus, "-", 'SM')
        h_sp = draw_btn(btn_smooth_plus, "+", 'SP')
        screen.blit(font_ui.render(f"Smooth: {smoothing_alpha:.2f}", True, (200, 200, 200)), (510, 225))
        
        # Draw Webcam PiP
        if pip_surf:
            pip_rect = pip_surf.get_rect(topright=(W - 20, 20))
            screen.blit(pip_surf, pip_rect)
            pygame.draw.rect(screen, (100, 100, 100), pip_rect, 2)
            if not calibrator.is_calibrated and not mouse_mode:
                warn = font_ui.render("NOT CALIBRATED", True, (255, 50, 50))
                screen.blit(warn, (pip_rect.left - 10, pip_rect.bottom + 10))

        # Draw Keyboard
        for k in keyboard.keys:
            is_hover = (k == hovered_key)
            ratio = min(1.0, k.hover_time / dwell_limit) if is_hover else 0.0
            k.draw(screen, font_keys, is_hover, ratio, dt)

        # Draw Calibration Screen Overlay
        if calibrator.calibrating:
            screen.fill((20, 20, 25))
            cx, cy = calibrator.points[calibrator.point_idx]
            
            # Draw all calibration points as dim dots
            for i, (px, py) in enumerate(calibrator.points):
                if i < calibrator.point_idx:
                    # Already recorded — green
                    pygame.draw.circle(screen, (0, 180, 80), (px, py), 10)
                    pygame.draw.circle(screen, (0, 255, 120), (px, py), 4)
                elif i == calibrator.point_idx:
                    # Current point — red with progress ring
                    pygame.draw.circle(screen, (255, 0, 0), (cx, cy), 18)
                    pygame.draw.circle(screen, (255, 255, 255), (cx, cy), 6)
                    
                    # Draw progress ring during collection
                    if calibrator.collecting and calib_progress > 0:
                        angle = calib_progress * 2 * math.pi
                        for a in range(int(angle * 30)):
                            theta = a / 30.0
                            rx = cx + int(25 * math.cos(theta - math.pi/2))
                            ry = cy + int(25 * math.sin(theta - math.pi/2))
                            pygame.draw.circle(screen, (0, 210, 255), (rx, ry), 2)
                else:
                    # Future points — dim
                    pygame.draw.circle(screen, (60, 60, 65), (px, py), 8)
            
            # Instructions
            is_tilt_point = calibrator.point_idx >= calibrator.n_spatial_points
            tilt_prompt = None
            if is_tilt_point:
                tilt_i = calibrator.point_idx - calibrator.n_spatial_points
                if 0 <= tilt_i < len(calibrator.tilt_prompts):
                    tilt_prompt = calibrator.tilt_prompts[tilt_i]

            if not calibrator.collecting:
                if tilt_prompt:
                    msg = font_large.render(f"Keep looking at the dot. {tilt_prompt}, then press SPACEBAR", True, (255, 255, 255))
                else:
                    msg = font_large.render("Look at the red dot and press SPACEBAR", True, (255, 255, 255))
            elif calibrator.needs_refixation:
                msg = font_large.render("Gaze wasn't steady — hold still, recollecting...", True, (255, 160, 0))
            elif tilt_prompt:
                msg = font_large.render(f"Keep your eyes on the dot — {tilt_prompt.lower()}...", True, (0, 210, 255))
            else:
                msg = font_large.render("Hold still... collecting samples", True, (0, 210, 255))
            screen.blit(msg, msg.get_rect(center=(W//2, H//2)))
            
            # Point counter
            label = "Head-pose step" if is_tilt_point else "Point"
            display_i = (calibrator.point_idx - calibrator.n_spatial_points + 1) if is_tilt_point else (calibrator.point_idx + 1)
            display_n = len(calibrator.tilt_prompts) if is_tilt_point else calibrator.n_spatial_points
            counter = font_ui.render(f"{label} {display_i} / {display_n}", True, (180, 180, 180))
            screen.blit(counter, counter.get_rect(center=(W//2, H//2 + 50)))
            
        elif calibrator.is_calibrated and calibrator.calib_rms_px is not None and calib_result_timer > 0:
            # Briefly show the estimated calibration accuracy right after finishing,
            # so a bad calibration is visible instead of silently trusted.
            rms = calibrator.calib_rms_px
            if rms < 40:
                quality, color = "Good", (0, 230, 118)
            elif rms < 80:
                quality, color = "Fair — consider recalibrating in better lighting", (255, 200, 0)
            else:
                quality, color = "Poor — please recalibrate (look steadily at each dot)", (255, 80, 80)
            result_msg = font_ui.render(f"Calibration accuracy: ~{rms:.0f}px ({quality})", True, color)
            screen.blit(result_msg, (20, H - 40))
            
        # Draw Gaze Cursor
        if not calibrator.calibrating:
            pygame.draw.circle(screen, (255, 255, 255), (int(cursor_x), int(cursor_y)), 10, 2)
            pygame.draw.circle(screen, (0, 210, 255), (int(cursor_x), int(cursor_y)), 4)
            
        pygame.display.flip()
        
    pygame.quit()
    tracker_thread.join()

if __name__ == "__main__":
    main()