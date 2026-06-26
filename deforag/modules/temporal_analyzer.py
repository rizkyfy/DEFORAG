"""
modules/temporal_analyzer.py
-----------------------------
Modul analisis temporal untuk DEFORAG.
Menganalisis konsistensi temporal pada video untuk mendeteksi deepfake.

Untuk gambar statis: mengembalikan skor netral dengan confidence rendah.
Untuk video (list frame): melakukan analisis:
  1. Eye blink detection (via mediapipe jika tersedia)
  2. Temporal consistency via optical flow
  3. Head pose consistency (via mediapipe jika tersedia)
  4. Face region texture consistency antar frame
"""

import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# OpenCV opsional (diperlukan untuk optical flow)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# MediaPipe opsional (untuk landmark wajah dan eye blink)
MEDIAPIPE_AVAILABLE = False
_mp = None
_mp_python = None
_mp_vision = None

try:
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

# Threshold Eye Aspect Ratio untuk deteksi mata tertutup
EAR_THRESHOLD = 0.20

# Blink rate normal: 15-20 blinks/menit
NORMAL_BLINK_RATE_MIN = 12.0   # blinks/menit
NORMAL_BLINK_RATE_MAX = 25.0   # blinks/menit

# Asumsi frame rate jika tidak diketahui
DEFAULT_FPS = 25.0

# Ambang batas optical flow untuk mendeteksi flicker
FLOW_FLICKER_THRESHOLD = 2.0   # pixel/frame

# Minimum jumlah frame untuk analisis yang bermakna
MIN_FRAMES_FOR_ANALYSIS = 10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _verdict_from_score(score: float) -> str:
    """Mengonversi skor numerik menjadi label verdict."""
    if score >= 0.70:
        return "FAKE"
    elif score >= 0.45:
        return "SUSPICIOUS"
    else:
        return "REAL"


def _normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Menormalkan nilai ke rentang [0, 1]."""
    if max_val <= min_val:
        return 0.0
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def _frame_to_gray(frame: np.ndarray) -> np.ndarray:
    """Mengonversi frame (RGB atau BGR) ke grayscale uint8."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        # Asumsikan RGB
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    elif frame.ndim == 2:
        gray = frame.astype(np.uint8)
    else:
        gray = frame[:, :, 0].astype(np.uint8)
    return gray


def _compute_ear(landmarks, left_eye_indices: list, right_eye_indices: list) -> float:
    """
    Menghitung Eye Aspect Ratio (EAR) dari landmark wajah.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Nilai EAR < 0.2 mengindikasikan mata tertutup (blink).

    Args:
        landmarks: Objek landmark dari MediaPipe.
        left_eye_indices: Indeks landmark untuk mata kiri.
        right_eye_indices: Indeks landmark untuk mata kanan.

    Returns:
        Rata-rata EAR kedua mata.
    """
    def _ear_single(indices: list) -> float:
        pts = []
        for i in indices:
            lm = landmarks[i]
            if hasattr(lm, 'x'):
                pts.append((lm.x, lm.y))
            else:
                pts.append((lm[0], lm[1]))
        if len(pts) < 6:
            return 0.3
        # Jarak vertikal
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        # Jarak horizontal
        h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return float(ear)

    ear_left = _ear_single(left_eye_indices)
    ear_right = _ear_single(right_eye_indices)
    return (ear_left + ear_right) / 2.0


# ---------------------------------------------------------------------------
# Sub-analisis 1: Eye blink detection (MediaPipe)
# ---------------------------------------------------------------------------

def _analyze_eye_blinks(all_landmarks: list, fps: float = DEFAULT_FPS) -> dict[str, Any]:
    """
    Mendeteksi kedipan mata untuk menilai naturalness gerakan wajah.

    Deepfake sering:
    - Tidak berkedip atau sangat jarang berkedip
    - Berkedip dengan pola yang tidak natural

    Returns:
        dict berisi 'score' dan 'details'
    """
    try:
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        ear_values: list[float] = []
        frames_with_face = 0

        for lm in all_landmarks:
            if lm is not None:
                frames_with_face += 1
                ear = _compute_ear(lm, LEFT_EYE, RIGHT_EYE)
                ear_values.append(ear)
            else:
                ear_values.append(0.3)  # Asumsikan mata terbuka

        if not ear_values or frames_with_face < 3:
            return {
                "score": 0.5,
                "details": {
                    "frames_with_face": frames_with_face,
                    "message": "Tidak cukup frame dengan wajah terdeteksi",
                },
            }

        ear_array = np.array(ear_values)

        # Deteksi blink: transisi dari EAR > threshold ke EAR < threshold
        below_threshold = ear_array < EAR_THRESHOLD
        blink_count = 0
        in_blink = False
        for is_closed in below_threshold:
            if is_closed and not in_blink:
                blink_count += 1
                in_blink = True
            elif not is_closed:
                in_blink = False

        # Hitung blink rate per menit
        duration_minutes = len(all_landmarks) / (fps * 60.0)
        blink_rate = blink_count / max(duration_minutes, 1e-6)

        # Evaluasi naturalness
        avg_ear = float(np.mean(ear_array))
        ear_std = float(np.std(ear_array))

        if blink_rate < NORMAL_BLINK_RATE_MIN * 0.5:
            # Sangat jarang berkedip = mencurigakan
            blink_score = 0.80
        elif blink_rate > NORMAL_BLINK_RATE_MAX * 2.0:
            # Berkedip terlalu sering = mencurigakan
            blink_score = 0.70
        elif NORMAL_BLINK_RATE_MIN <= blink_rate <= NORMAL_BLINK_RATE_MAX:
            # Normal
            blink_score = 0.15
        else:
            # Agak menyimpang
            blink_score = 0.40

        # EAR variability rendah = tidak natural
        if ear_std < 0.01:
            variability_score = 0.70
        elif ear_std < 0.03:
            variability_score = 0.45
        else:
            variability_score = 0.20

        final_score = float(np.clip(0.6 * blink_score + 0.4 * variability_score, 0.0, 1.0))

        return {
            "score": final_score,
            "details": {
                "blink_count": blink_count,
                "blink_rate_per_minute": round(blink_rate, 2),
                "avg_ear": round(avg_ear, 4),
                "ear_std": round(ear_std, 4),
                "frames_with_face": frames_with_face,
                "total_frames": len(all_landmarks),
                "normal_blink_range": f"{NORMAL_BLINK_RATE_MIN}-{NORMAL_BLINK_RATE_MAX}/menit",
            },
        }

    except Exception as exc:
        logger.warning("Gagal analisis eye blink: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Sub-analisis 2: Temporal consistency via optical flow
# ---------------------------------------------------------------------------

def _analyze_optical_flow(frames: list[np.ndarray]) -> dict[str, Any]:
    """
    Menganalisis konsistensi temporal menggunakan optical flow (Farneback).

    Deepfake sering menunjukkan:
    - Flicker di area boundary wajah
    - Gerakan yang tidak konsisten dengan gerakan background
    - Flow magnitude yang tiba-tiba berubah di area wajah

    Returns:
        dict berisi 'score' dan 'details'
    """
    if not CV2_AVAILABLE:
        return {
            "score": 0.5,
            "details": {
                "cv2_available": False,
                "message": "OpenCV tidak tersedia — analisis optical flow dilewati",
            },
        }

    try:
        if len(frames) < 2:
            return {"score": 0.5, "details": {"error": "Perlu minimal 2 frame"}}

        flow_magnitudes: list[float] = []
        flow_std_list: list[float] = []
        frame_diffs: list[float] = []

        prev_gray = _frame_to_gray(frames[0])

        for i in range(1, len(frames)):
            curr_gray = _frame_to_gray(frames[i])

            # Hitung optical flow Farneback
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    curr_gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                # Magnitude flow
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_magnitudes.append(float(np.mean(mag)))
                flow_std_list.append(float(np.std(mag)))
            except Exception:
                pass

            # Perbedaan frame langsung
            diff = float(np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float))))
            frame_diffs.append(diff)
            prev_gray = curr_gray

        if not flow_magnitudes:
            return {"score": 0.5, "details": {"error": "Gagal menghitung optical flow"}}

        mag_arr = np.array(flow_magnitudes)
        diff_arr = np.array(frame_diffs)

        # Deteksi flicker: perubahan flow magnitude yang tiba-tiba
        mag_diff = np.abs(np.diff(mag_arr))
        flicker_frames = int(np.sum(mag_diff > FLOW_FLICKER_THRESHOLD))
        flicker_ratio = flicker_frames / max(len(mag_diff), 1)

        # Variance flow magnitude (konsistensi gerakan)
        mag_cv = float(np.std(mag_arr) / (np.mean(mag_arr) + 1e-6))

        # Skor: flicker tinggi dan variance tinggi = mencurigakan
        flicker_score = _normalize_score(flicker_ratio, min_val=0.0, max_val=0.3)
        variance_score = _normalize_score(mag_cv, min_val=0.2, max_val=2.0)

        final_score = float(np.clip(0.6 * flicker_score + 0.4 * variance_score, 0.0, 1.0))

        return {
            "score": final_score,
            "details": {
                "mean_flow_magnitude": round(float(np.mean(mag_arr)), 4),
                "std_flow_magnitude": round(float(np.std(mag_arr)), 4),
                "coefficient_of_variation": round(mag_cv, 4),
                "flicker_frame_count": flicker_frames,
                "flicker_ratio": round(flicker_ratio, 4),
                "mean_frame_diff": round(float(np.mean(diff_arr)), 4),
            },
        }

    except Exception as exc:
        logger.warning("Gagal optical flow analysis: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Sub-analisis 3: Head pose consistency
# ---------------------------------------------------------------------------

def _analyze_head_pose(all_landmarks: list, fps: float = DEFAULT_FPS) -> dict[str, Any]:
    """
    Menganalisis konsistensi pose kepala antar frame menggunakan MediaPipe.

    Deepfake sering menunjukkan perubahan pose kepala yang tidak natural
    atau inkonsistensi dalam gerakan antara wajah dan leher/bahu.

    Returns:
        dict berisi 'score' dan 'details'
    """
    try:
        # Indeks landmark untuk estimasi pose (dahi, dagu, hidung, pipi)
        NOSE_TIP = 4
        CHIN = 152
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_OUTER = 263

        yaw_values: list[float] = []
        pitch_values: list[float] = []
        roll_values: list[float] = []

        for lm in all_landmarks:
            if lm is not None:
                def get_xy(idx):
                    pt = lm[idx]
                    if hasattr(pt, 'x'):
                        return pt.x, pt.y
                    return pt[0], pt[1]

                nose_x, nose_y = get_xy(NOSE_TIP)
                chin_x, chin_y = get_xy(CHIN)
                l_eye_x, l_eye_y = get_xy(LEFT_EYE_OUTER)
                r_eye_x, r_eye_y = get_xy(RIGHT_EYE_OUTER)

                # Estimasi pitch: y-pos hidung vs dagu
                pitch = nose_y - chin_y
                # Estimasi yaw: x-pos hidung vs tengah mata
                mid_eye_x = (l_eye_x + r_eye_x) / 2
                yaw = nose_x - mid_eye_x
                # Estimasi roll: kemiringan sumbu mata
                roll = l_eye_y - r_eye_y

                pitch_values.append(float(pitch))
                yaw_values.append(float(yaw))
                roll_values.append(float(roll))

        if len(pitch_values) < 3:
            return {
                "score": 0.5,
                "details": {"message": "Tidak cukup frame dengan pose terdeteksi"},
            }

        pitch_arr = np.array(pitch_values)
        yaw_arr = np.array(yaw_values)
        roll_arr = np.array(roll_values)

        # Hitung perubahan pose antar frame (harus halus/smooth)
        pitch_jumps = np.abs(np.diff(pitch_arr))
        yaw_jumps = np.abs(np.diff(yaw_arr))
        roll_jumps = np.abs(np.diff(roll_arr))

        # Frame dengan perubahan pose tiba-tiba
        threshold = 0.05  # dalam koordinat normalized (0-1)
        abrupt_changes = int(
            np.sum(pitch_jumps > threshold)
            + np.sum(yaw_jumps > threshold)
            + np.sum(roll_jumps > threshold)
        )
        total_checks = (len(pitch_jumps) + len(yaw_jumps) + len(roll_jumps))
        abrupt_ratio = abrupt_changes / max(total_checks, 1)

        final_score = float(np.clip(_normalize_score(abrupt_ratio, 0.0, 0.3), 0.0, 1.0))

        return {
            "score": final_score,
            "details": {
                "abrupt_pose_change_ratio": round(abrupt_ratio, 4),
                "pitch_std": round(float(np.std(pitch_arr)), 5),
                "yaw_std": round(float(np.std(yaw_arr)), 5),
                "roll_std": round(float(np.std(roll_arr)), 5),
                "frames_analyzed": len(pitch_values),
            },
        }

    except Exception as exc:
        logger.warning("Gagal head pose analysis: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Sub-analisis 4: Face region texture consistency
# ---------------------------------------------------------------------------

def _analyze_face_texture_consistency(frames: list[np.ndarray]) -> dict[str, Any]:
    """
    Menganalisis konsistensi tekstur area wajah antar frame.

    Deepfake sering menunjukkan fluktuasi tekstur wajah yang tidak natural
    karena proses blending atau rendering yang tidak sempurna.

    Pendekatan sederhana: bandingkan histogram warna pada region tengah gambar
    (diasumsikan sebagai area wajah).

    Returns:
        dict berisi 'score' dan 'details'
    """
    try:
        if len(frames) < 2:
            return {"score": 0.5, "details": {"error": "Perlu minimal 2 frame"}}

        # Ambil region tengah (perkiraan area wajah)
        h, w = frames[0].shape[:2]
        y0, y1 = h // 4, 3 * h // 4
        x0, x1 = w // 4, 3 * w // 4

        histograms: list[np.ndarray] = []
        for frame in frames:
            region = frame[y0:y1, x0:x1]
            if region.ndim == 3:
                gray_region = (0.299 * region[:,:,0] + 0.587 * region[:,:,1]
                               + 0.114 * region[:,:,2]).astype(np.float32)
            else:
                gray_region = region.astype(np.float32)

            hist, _ = np.histogram(gray_region, bins=64, range=(0, 255))
            hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-10)
            histograms.append(hist_norm)

        if len(histograms) < 2:
            return {"score": 0.5, "details": {"error": "Gagal membuat histogram"}}

        # Hitung korelasi histogram antar frame berurutan
        correlations = []
        for i in range(len(histograms) - 1):
            try:
                corr = float(np.corrcoef(histograms[i], histograms[i + 1])[0, 1])
                if not np.isnan(corr):
                    correlations.append(corr)
            except Exception:
                pass

        if not correlations:
            return {"score": 0.5, "details": {"error": "Gagal menghitung korelasi"}}

        mean_corr = float(np.mean(correlations))
        std_corr = float(np.std(correlations))

        # Frame dengan korelasi rendah (inkonsistensi tiba-tiba)
        low_corr_frames = int(np.sum(np.array(correlations) < 0.90))
        low_corr_ratio = low_corr_frames / max(len(correlations), 1)

        # Skor: korelasi rendah dan bervariasi = lebih mencurigakan
        consistency_score = max(0.0, 1.0 - mean_corr)
        variability_score = _normalize_score(std_corr, 0.0, 0.15)
        anomaly_score = _normalize_score(low_corr_ratio, 0.0, 0.3)

        final_score = float(np.clip(
            0.4 * consistency_score + 0.3 * variability_score + 0.3 * anomaly_score,
            0.0, 1.0
        ))

        return {
            "score": final_score,
            "details": {
                "mean_histogram_correlation": round(mean_corr, 4),
                "std_histogram_correlation": round(std_corr, 4),
                "low_correlation_frame_ratio": round(low_corr_ratio, 4),
                "frames_compared": len(correlations),
            },
        }

    except Exception as exc:
        logger.warning("Gagal face texture consistency analysis: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Kelas utama
# ---------------------------------------------------------------------------

class TemporalAnalyzer:
    """
    Menganalisis konsistensi temporal untuk mendeteksi deepfake pada video.

    Untuk gambar statis (is_video=False):
      Mengembalikan skor 0.5 dengan confidence rendah karena tidak ada
      informasi temporal yang dapat dianalisis.

    Untuk video (is_video=True, frames adalah list of numpy arrays):
      Menganalisis:
      1. Pola kedipan mata (via MediaPipe Tasks API)
      2. Konsistensi optical flow (via OpenCV)
      3. Konsistensi pose kepala (via MediaPipe Tasks API)
      4. Konsistensi tekstur wajah antar frame
    """

    SUB_WEIGHTS = {
        "blink": 0.30,
        "optical_flow": 0.35,
        "head_pose": 0.20,
        "face_texture": 0.15,
    }

    def __init__(self) -> None:
        self._face_mesh = None
        self._use_mediapipe = False

        if MEDIAPIPE_AVAILABLE:
            try:
                # Muat FaceLandmarker model task
                model_path = Path(__file__).parent / "face_landmarker.task"
                if model_path.exists():
                    options = _mp_vision.FaceLandmarkerOptions(
                        base_options=_mp_python.BaseOptions(model_asset_path=str(model_path)),
                        running_mode=_mp_vision.RunningMode.IMAGE,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_face_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                    self._face_mesh = _mp_vision.FaceLandmarker.create_from_options(options)
                    self._use_mediapipe = True
                    logger.info("TemporalAnalyzer: MediaPipe FaceLandmarker Tasks API diinisialisasi.")
            except Exception as e:
                logger.warning(f"TemporalAnalyzer: Gagal inisialisasi FaceLandmarker Tasks API: {e}")

    def analyze(
        self,
        image_path_or_frames: str | list[np.ndarray],
        is_video: bool = False,
        fps: float = DEFAULT_FPS,
    ) -> dict[str, Any]:
        """
        Melakukan analisis temporal.

        Args:
            image_path_or_frames:
                - Jika is_video=False: path ke file gambar (str).
                - Jika is_video=True: list of numpy arrays (frame-frame video).
            is_video: True jika input adalah video frames, False untuk gambar statis.
            fps: Frame rate video (digunakan untuk estimasi blink rate).

        Returns:
            dict dengan kunci:
              - score      : float [0, 1]
              - verdict    : str  – 'REAL' | 'SUSPICIOUS' | 'FAKE'
              - confidence : float [0, 1]
              - details    : dict
        """
        result_template: dict[str, Any] = {
            "score": 0.5,
            "verdict": "SUSPICIOUS",
            "confidence": 0.0,
            "details": {},
        }

        try:
            # --- Mode gambar statis ---
            if not is_video:
                image_path = str(image_path_or_frames)
                path = Path(image_path)
                if not path.exists():
                    raise FileNotFoundError(f"File tidak ditemukan: {image_path}")

                return {
                    "score": 0.50,
                    "verdict": "SUSPICIOUS",
                    "confidence": 0.15,
                    "details": {
                        "analysis_type": "static_image",
                        "message": (
                            "Analisis temporal tidak dapat dilakukan pada gambar statis. "
                            "Untuk analisis temporal yang bermakna, gunakan input video."
                        ),
                        "cv2_available": CV2_AVAILABLE,
                        "mediapipe_available": MEDIAPIPE_AVAILABLE,
                    },
                }

            # --- Mode video ---
            frames = image_path_or_frames
            if not isinstance(frames, list) or len(frames) == 0:
                raise ValueError("frames harus berupa list numpy arrays yang tidak kosong")

            n_frames = len(frames)

            if n_frames < MIN_FRAMES_FOR_ANALYSIS:
                return {
                    "score": 0.50,
                    "verdict": "SUSPICIOUS",
                    "confidence": 0.20,
                    "details": {
                        "analysis_type": "video",
                        "message": (
                            f"Jumlah frame ({n_frames}) terlalu sedikit untuk analisis "
                            f"yang bermakna (minimum {MIN_FRAMES_FOR_ANALYSIS} frame)"
                        ),
                        "total_frames": n_frames,
                    },
                }

            # Ekstrak landmark satu kali untuk semua frame (lebih cepat)
            all_landmarks = []
            if self._use_mediapipe and self._face_mesh is not None:
                for frame in frames:
                    try:
                        if frame.dtype != np.uint8:
                            frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
                        else:
                            frame_u8 = frame

                        if frame_u8.ndim == 2:
                            frame_rgb = np.stack([frame_u8] * 3, axis=-1)
                        else:
                            frame_rgb = frame_u8

                        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=frame_rgb)
                        detection_result = self._face_mesh.detect(mp_image)

                        if detection_result.face_landmarks:
                            all_landmarks.append(detection_result.face_landmarks[0])
                        else:
                            all_landmarks.append(None)
                    except Exception as exc:
                        logger.warning("TemporalAnalyzer: Error mengekstrak landmark frame: %s", exc)
                        all_landmarks.append(None)

            # Jalankan semua sub-analisis
            if self._use_mediapipe and all_landmarks:
                blink_result = _analyze_eye_blinks(all_landmarks, fps=fps)
                pose_result = _analyze_head_pose(all_landmarks, fps=fps)
            else:
                blink_result = {
                    "score": 0.5,
                    "details": {
                        "mediapipe_available": False,
                        "message": "MediaPipe tidak tersedia — analisis kedipan dilewati",
                    },
                }
                pose_result = {
                    "score": 0.5,
                    "details": {
                        "mediapipe_available": False,
                        "message": "MediaPipe tidak tersedia — analisis pose dilewati",
                    },
                }

            flow_result = _analyze_optical_flow(frames)
            texture_result = _analyze_face_texture_consistency(frames)

            sub_results = {
                "blink": blink_result,
                "optical_flow": flow_result,
                "head_pose": pose_result,
                "face_texture": texture_result,
            }

            # Hitung weighted average
            # Jika modul tertentu tidak tersedia, redistribute weight
            available_weights: dict[str, float] = {}
            for key, res in sub_results.items():
                details = res.get("details", {})
                is_available = (
                    not details.get("mediapipe_available") is False
                    and not details.get("cv2_available") is False
                    and "error" not in details
                )
                if is_available or key in ("face_texture",):
                    available_weights[key] = self.SUB_WEIGHTS[key]

            total_available_weight = sum(available_weights.values())

            if total_available_weight <= 0:
                available_weights = self.SUB_WEIGHTS.copy()
                total_available_weight = sum(available_weights.values())

            weighted_score = 0.0
            sub_scores: dict[str, float] = {}

            for key, weight in available_weights.items():
                normalized_weight = weight / total_available_weight
                sub_score = float(sub_results[key].get("score", 0.5))
                sub_scores[key] = sub_score
                weighted_score += normalized_weight * sub_score

            final_score = float(np.clip(weighted_score, 0.0, 1.0))

            # Confidence berdasarkan kesepakatan dan jumlah modul yang tersedia
            n_available = len(available_weights)
            score_std = float(np.std(list(sub_scores.values()))) if sub_scores else 0.5
            confidence = float(np.clip(
                (n_available / 4.0) * (1.0 - score_std * 1.5),
                0.10, 0.90
            ))

            verdict = _verdict_from_score(final_score)

            return {
                "score": round(final_score, 4),
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "details": {
                    "analysis_type": "video",
                    "total_frames": n_frames,
                    "fps": fps,
                    "sub_scores": sub_scores,
                    "available_modules": list(available_weights.keys()),
                    "cv2_available": CV2_AVAILABLE,
                    "mediapipe_available": self._use_mediapipe,
                    "sub_analyses": {k: v.get("details", {}) for k, v in sub_results.items()},
                },
            }

        except FileNotFoundError as fnf:
            logger.error("File tidak ditemukan: %s", fnf)
            result_template["details"] = {"error": str(fnf)}
            return result_template

        except Exception as exc:
            logger.error(
                "Kesalahan tidak terduga pada TemporalAnalyzer: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            result_template["details"] = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            return result_template
