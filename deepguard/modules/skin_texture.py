"""
modules/skin_texture.py
-----------------------
Modul analisis keaslian tekstur kulit (Skin Texture Authenticity).

AI-generated faces dan face-swap menerapkan smoothing artifisial pada kulit
yang menghilangkan noise frekuensi tinggi alami (pori, microdetail, sebaceous
glands). Modul ini mendeteksi over-smoothing ini melalui analisis frekuensi
di region kulit wajah yang terisolasi.

Metode:
  1. Isolasi region kulit wajah (pipi, dahi, hidung) via landmark + skin color
  2. Analisis power spectrum frekuensi tinggi (band 0.3–0.5 Nyquist)
  3. Hitung Natural Scene Statistics (NSS) via MSCN (Mean Subtracted
     Contrast Normalized) coefficients
  4. Bandingkan distribusi MSCN dengan prior foto asli (Gaussian fit)
  5. Hitung local variance consistency — over-smoothing menurunkan variance
     secara artifisial di seluruh region kulit

Sub-skor output:
  - high_freq_suppression  : seberapa tertekan frekuensi tinggi (tinggi = smoothing)
  - mscn_deviation         : deviasi MSCN dari distribusi Gaussian alami (tinggi = AI)
  - local_variance_flatness: kerataan variance lokal (tinggi = over-smoothed)
  - skin_texture_score     : skor gabungan 0.0–1.0

Dependensi: opencv-python, numpy, scipy, scikit-image
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.stats import kurtosis, skew, shapiro

# MediaPipe opsional
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

SKIN_LOWER_YCrCb = np.array([0,   133, 77],  dtype=np.uint8)
SKIN_UPPER_YCrCb = np.array([255, 173, 127], dtype=np.uint8)

PATCH_SIZE           = 32    # ukuran patch untuk analisis lokal
MIN_SKIN_PIXELS      = 500   # minimum piksel kulit untuk analisis valid
HIGH_FREQ_LOW_RATIO  = 0.30  # batas bawah band frekuensi tinggi (× Nyquist)
HIGH_FREQ_HIGH_RATIO = 0.50  # batas atas band frekuensi tinggi (× Nyquist)

# Prior MSCN untuk foto asli (dari literatur BRISQUE)
# MSCN foto asli mendekati Generalized Gaussian Distribution dengan:
# kurtosis ~3.0, skewness ~0.0
REAL_MSCN_KURTOSIS_RANGE = (2.0, 5.0)
REAL_MSCN_SKEW_RANGE     = (-0.5, 0.5)


# ---------------------------------------------------------------------------
# Kelas utama
# ---------------------------------------------------------------------------

class SkinTextureAnalyzer:
    """
    Mendeteksi over-smoothing artifisial pada tekstur kulit wajah.
    """

    def __init__(self, landmark_predictor=None) -> None:
        self.landmark_predictor = landmark_predictor
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
                    logger.info("SkinTextureAnalyzer: MediaPipe FaceLandmarker Tasks API diinisialisasi.")
            except Exception as e:
                logger.warning(f"SkinTextureAnalyzer: Gagal inisialisasi FaceLandmarker Tasks API: {e}")

    # ------------------------------------------------------------------
    def analyze(self, image: np.ndarray) -> dict[str, Any]:
        """
        Analisis utama. Menerima gambar BGR (OpenCV format).
        """
        try:
            if image is None or image.size == 0:
                return self._unavailable("Gambar kosong atau tidak valid")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # --- Isolasi region kulit ---
            skin_mask = self._get_skin_mask(image, h, w)
            if skin_mask.sum() < MIN_SKIN_PIXELS:
                return self._unavailable("Region kulit tidak cukup untuk analisis")

            # --- Sub-skor 1: High frequency suppression ---
            high_freq_suppression = self._compute_high_freq_suppression(
                gray, skin_mask
            )

            # --- Sub-skor 2: MSCN deviation ---
            mscn_deviation = self._compute_mscn_deviation(gray, skin_mask)

            # --- Sub-skor 3: Local variance flatness ---
            local_variance_flatness = self._compute_local_variance_flatness(
                gray, skin_mask, h, w
            )

            # --- Skor gabungan ---
            skin_texture_score = float(np.clip(
                0.40 * high_freq_suppression
                + 0.35 * mscn_deviation
                + 0.25 * local_variance_flatness,
                0.0, 1.0
            ))

            # --- Confidence ---
            skin_pixels = int(skin_mask.sum())
            confidence  = float(np.clip(skin_pixels / 10000, 0.3, 0.95))

            # --- Verdict ---
            if skin_texture_score >= 0.58:
                verdict = "FAKE"
            elif skin_texture_score >= 0.42:
                verdict = "SUSPICIOUS"
            else:
                verdict = "REAL"

            return {
                "score":      round(skin_texture_score, 4),
                "confidence": round(confidence, 4),
                "verdict":    verdict,
                "details": {
                    "sub_scores": {
                        "high_freq_suppression":   round(high_freq_suppression, 4),
                        "mscn_deviation":          round(mscn_deviation, 4),
                        "local_variance_flatness": round(local_variance_flatness, 4),
                    },
                    "skin_pixels": skin_pixels,
                },
            }

        except Exception as exc:
            logger.error("SkinTextureAnalyzer error: %s", exc, exc_info=True)
            return self._unavailable(f"Error: {exc}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_skin_mask(self, bgr: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Deteksi region kulit menggunakan YCrCb color space + MediaPipe jika tersedia.
        """
        if self._use_mediapipe:
            mask = self._mediapipe_skin_mask(bgr, h, w)
            if mask is not None and mask.sum() >= MIN_SKIN_PIXELS:
                return mask

        # Fallback to color-based middle ROI
        ycrcb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, SKIN_LOWER_YCrCb, SKIN_UPPER_YCrCb)

        # Morfologi untuk bersihkan noise
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,  kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        # Fokus ke area wajah (tengah-atas gambar, 60% height, 70% width)
        face_roi = np.zeros((h, w), dtype=np.uint8)
        y1, y2   = int(h * 0.05), int(h * 0.75)
        x1, x2   = int(w * 0.15), int(w * 0.85)
        face_roi[y1:y2, x1:x2] = 255

        return cv2.bitwise_and(skin_mask, face_roi)

    def _mediapipe_skin_mask(self, bgr, h, w):
        if not self._use_mediapipe or self._face_mesh is None:
            return None
            
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            detection_result = self._face_mesh.detect(mp_image)
            
            if not detection_result.face_landmarks:
                return None
                
            landmarks = detection_result.face_landmarks[0]
            
            # FACEOVAL
            FACEOVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            
            # Left Eye
            LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            
            # Right Eye
            RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Lips Outer
            LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 95, 88, 178]
            
            def get_pts(indices):
                pts = []
                for idx in indices:
                    lm = landmarks[idx]
                    pts.append([int(lm.x * w), int(lm.y * h)])
                return np.array(pts, dtype=np.int32)
                
            face_pts = get_pts(FACEOVAL)
            left_eye_pts = get_pts(LEFT_EYE)
            right_eye_pts = get_pts(RIGHT_EYE)
            lips_pts = get_pts(LIPS)
            
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_mask, [face_pts], 255)
            
            cv2.fillPoly(face_mask, [left_eye_pts], 0)
            cv2.fillPoly(face_mask, [right_eye_pts], 0)
            cv2.fillPoly(face_mask, [lips_pts], 0)
            
            # Erode slightly to avoid eyebrows/feature boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            face_mask = cv2.erode(face_mask, kernel)
            
            # Combine with color range mask
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            color_mask = cv2.inRange(ycrcb, SKIN_LOWER_YCrCb, SKIN_UPPER_YCrCb)
            
            skin_mask = cv2.bitwise_and(face_mask, color_mask)
            return skin_mask
        except Exception as e:
            logger.warning(f"SkinTextureAnalyzer: Gagal mask MediaPipe: {e}")
            return None

    def _compute_high_freq_suppression(
        self,
        gray: np.ndarray,
        skin_mask: np.ndarray,
    ) -> float:
        """
        Hitung seberapa tertekan komponen frekuensi tinggi di region kulit.

        AI smoothing menghapus detail frekuensi tinggi (0.3–0.5 Nyquist),
        sehingga rasio power HF/LF jauh lebih rendah dari foto asli.
        Normalisasi: rendah = asli, tinggi = over-smoothed.
        """
        # Isolasi piksel kulit dengan zero-padding
        skin_region = gray.astype(np.float32).copy()
        skin_region[skin_mask == 0] = 0

        # 2D FFT
        fft      = np.fft.fft2(skin_region)
        fft_shift = np.fft.fftshift(fft)
        power    = np.abs(fft_shift) ** 2

        h, w    = power.shape
        cy, cx  = h // 2, w // 2

        # Buat mask frekuensi (dalam koordinat normalized 0–1)
        Y, X    = np.ogrid[:h, :w]
        dist    = np.sqrt(((Y - cy) / cy) ** 2 + ((X - cx) / cx) ** 2)

        lf_mask = (dist < HIGH_FREQ_LOW_RATIO).astype(float)
        hf_mask = ((dist >= HIGH_FREQ_LOW_RATIO) & (dist <= HIGH_FREQ_HIGH_RATIO)).astype(float)

        lf_power = float(np.sum(power * lf_mask)) + 1e-10
        hf_power = float(np.sum(power * hf_mask)) + 1e-10

        hf_ratio = hf_power / lf_power

        # Kalibrasi: foto asli ~0.08–0.15, AI-smoothed ~0.01–0.04
        # Normalisasi: 0.15 → 0.0 (natural), 0.01 → 1.0 (heavy smoothing)
        normalized = float(np.clip(1.0 - (hf_ratio / 0.15), 0.0, 1.0))
        return normalized

    def _compute_mscn_deviation(
        self,
        gray: np.ndarray,
        skin_mask: np.ndarray,
    ) -> float:
        """
        Hitung deviasi MSCN coefficient dari distribusi Gaussian alami.

        MSCN (Mean Subtracted Contrast Normalized) dari foto asli mengikuti
        distribusi Gaussian umum (kurtosis ~3, skewness ~0).
        AI-generated dan over-smoothed images menyimpang dari distribusi ini.
        """
        # Hitung MSCN
        mu      = cv2.GaussianBlur(gray.astype(np.float64), (7, 7), 7.0 / 6.0)
        mu_sq   = cv2.GaussianBlur(gray.astype(np.float64) ** 2, (7, 7), 7.0 / 6.0)
        sigma   = np.sqrt(np.abs(mu_sq - mu ** 2)) + 1e-7
        mscn    = (gray.astype(np.float64) - mu) / sigma

        # Ambil hanya piksel kulit
        mscn_skin = mscn[skin_mask > 0]
        if len(mscn_skin) < 100:
            return 0.0

        # Hitung statistik
        kurt = float(kurtosis(mscn_skin, fisher=False))  # excess=False → normal=3
        skewness = float(skew(mscn_skin))

        # Deviasi dari prior foto asli
        kurt_dev  = abs(kurt - 3.0) / 3.0       # normalisasi: 0 = perfect Gaussian
        skew_dev  = abs(skewness) / 1.0          # normalisasi: 0 = symmetric

        deviation = float(np.clip(0.6 * kurt_dev + 0.4 * skew_dev, 0.0, 1.0))
        return deviation

    def _compute_local_variance_flatness(
        self,
        gray: np.ndarray,
        skin_mask: np.ndarray,
        h: int,
        w: int,
    ) -> float:
        """
        Hitung kerataan variance lokal di patch kulit.

        Foto asli: variance lokal bervariasi secara natural (ada pori, wrinkle,
        highlight, shadow) — distribusi variance memiliki spread yang lebar.
        AI over-smoothed: variance lokal sangat seragam dan rendah di semua
        patch kulit — distribusi variance sangat sempit dan terkonsentrasi.

        Metrik: coefficient of variation (CV) dari distribusi variance patch.
        Rendah CV = flat = AI smoothed. Normalisasi: tinggi = AI.
        """
        variances = []

        for y in range(0, h - PATCH_SIZE, PATCH_SIZE // 2):
            for x in range(0, w - PATCH_SIZE, PATCH_SIZE // 2):
                patch_skin = skin_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                # Hanya proses patch dengan cukup piksel kulit (>40%)
                if patch_skin.mean() < 0.4:
                    continue

                patch = gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE].astype(np.float64)
                variances.append(float(np.var(patch)))

        if len(variances) < 5:
            return 0.0

        variances = np.array(variances)
        mean_var  = float(np.mean(variances)) + 1e-6
        std_var   = float(np.std(variances))
        cv        = std_var / mean_var  # coefficient of variation

        # Foto asli: CV ~0.5–1.5 (bervariasi), AI: CV ~0.1–0.3 (flat)
        # Normalisasi: CV 0.5 → 0.0, CV 0.05 → 1.0
        flatness = float(np.clip(1.0 - (cv / 0.5), 0.0, 1.0))
        return flatness

    # ------------------------------------------------------------------
    def _unavailable(self, reason: str) -> dict[str, Any]:
        return {
            "score":      0.5,
            "confidence": 0.0,
            "verdict":    "N/A",
            "details":    {"reason": reason},
        }