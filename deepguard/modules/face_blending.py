"""
modules/face_blending.py
------------------------
Modul deteksi batas blending wajah (Face Blending Boundary).

Mendeteksi artefak inpainting/face-swap di sepanjang tepi wajah:
hairline, dagu, dan area telinga. Face-swap meninggalkan jejak
diskontinuitas frekuensi di batas region yang di-paste/blend,
bahkan ketika blending terlihat halus secara visual.

Metode:
  1. Deteksi landmark wajah (dlib/mediapipe) untuk isolasi batas wajah
  2. Ekstrak strip piksel di sepanjang kontur wajah (lebar ±8px)
  3. Hitung gradient magnitude di strip tersebut (Sobel)
  4. Bandingkan distribusi gradient di dalam vs luar kontur
  5. Hitung DCT coefficient ratio di sepanjang boundary strip
  6. Hitung Local Binary Pattern (LBP) entropy di boundary vs interior

Sub-skor output:
  - boundary_gradient_ratio : rasio gradient boundary vs interior (tinggi = anomali)
  - dct_discontinuity       : diskontinuitas koefisien DCT di batas (tinggi = anomali)
  - lbp_boundary_entropy    : selisih entropi LBP boundary vs interior (tinggi = anomali)
  - blend_score             : skor gabungan 0.0–1.0

Interpretasi:
  - Foto asli: gradients di boundary konsisten dengan interior
  - Face-swap: boundary_gradient_ratio tinggi, DCT discontinuity tinggi
  - Diffusion full: blend_score rendah (tidak ada boundary karena seluruh gambar generated)

Dependensi: opencv-python, numpy, scipy, dlib atau mediapipe
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.fft import dctn
from scipy.stats import entropy as scipy_entropy

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

BOUNDARY_STRIP_WIDTH = 8       # piksel di kiri/kanan kontur wajah
MIN_FACE_SIZE        = 80      # ukuran minimum wajah (px) untuk analisis valid
DCT_BLOCK_SIZE       = 8       # ukuran blok DCT (standar JPEG)
LBP_RADIUS           = 2
LBP_N_POINTS         = 16


# ---------------------------------------------------------------------------
# Kelas utama
# ---------------------------------------------------------------------------

class FaceBlendingAnalyzer:
    """
    Mendeteksi artefak blending di batas wajah untuk mengidentifikasi face-swap.
    """

    def __init__(self, face_detector=None, landmark_predictor=None) -> None:
        """
        Parameters
        ----------
        face_detector       : dlib.fhog_object_detector atau cv2.CascadeClassifier
        landmark_predictor  : dlib.shape_predictor (68-point model)
                              Jika None, fallback ke OpenCV Haar + approx boundary
        """
        self.face_detector       = face_detector
        self.landmark_predictor  = landmark_predictor
        self._use_dlib           = (face_detector is not None
                                    and landmark_predictor is not None)

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
                    logger.info("FaceBlendingAnalyzer: MediaPipe FaceLandmarker Tasks API diinisialisasi.")
            except Exception as e:
                logger.warning(f"FaceBlendingAnalyzer: Gagal inisialisasi FaceLandmarker Tasks API: {e}")

    # ------------------------------------------------------------------
    def analyze(self, image: np.ndarray) -> dict[str, Any]:
        """
        Analisis utama. Menerima gambar BGR (OpenCV format).

        Returns
        -------
        dict dengan keys: score, confidence, verdict, details
        """
        try:
            if image is None or image.size == 0:
                return self._unavailable("Gambar kosong atau tidak valid")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # --- Deteksi wajah & boundary mask ---
            boundary_mask, interior_mask, face_found = self._get_boundary_masks(
                gray, image, h, w
            )

            if not face_found:
                return self._unavailable("Wajah tidak terdeteksi dalam gambar")

            if boundary_mask.sum() < 100:
                return self._unavailable("Boundary region terlalu kecil untuk analisis")

            # --- Sub-skor 1: Gradient ratio ---
            boundary_gradient_ratio = self._compute_gradient_ratio(
                gray, boundary_mask, interior_mask
            )

            # --- Sub-skor 2: DCT discontinuity ---
            dct_discontinuity = self._compute_dct_discontinuity(
                gray, boundary_mask, interior_mask
            )

            # --- Sub-skor 3: LBP entropy diff ---
            lbp_boundary_entropy = self._compute_lbp_entropy_diff(
                gray, boundary_mask, interior_mask
            )

            # --- Blend score gabungan ---
            # Bobot: gradient ratio paling diskriminatif untuk face-swap
            blend_score = float(np.clip(
                0.45 * boundary_gradient_ratio
                + 0.35 * dct_discontinuity
                + 0.20 * lbp_boundary_entropy,
                0.0, 1.0
            ))

            # --- Confidence berdasarkan ukuran boundary ---
            boundary_pixels = int(boundary_mask.sum())
            confidence      = float(np.clip(boundary_pixels / 5000, 0.3, 0.95))

            # --- Verdict ---
            if blend_score >= 0.60:
                verdict = "FAKE"
            elif blend_score >= 0.42:
                verdict = "SUSPICIOUS"
            else:
                verdict = "REAL"

            return {
                "score":      round(blend_score, 4),
                "confidence": round(confidence, 4),
                "verdict":    verdict,
                "details": {
                    "sub_scores": {
                        "boundary_gradient_ratio": round(boundary_gradient_ratio, 4),
                        "dct_discontinuity":       round(dct_discontinuity, 4),
                        "lbp_boundary_entropy":    round(lbp_boundary_entropy, 4),
                    },
                    "boundary_pixels": boundary_pixels,
                    "method": getattr(self, "last_method", "haar_approx"),
                },
            }

        except Exception as exc:
            logger.error("FaceBlendingAnalyzer error: %s", exc, exc_info=True)
            return self._unavailable(f"Error: {exc}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_boundary_masks(
        self,
        gray: np.ndarray,
        bgr: np.ndarray,
        h: int,
        w: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Kembalikan (boundary_mask, interior_mask, face_found).

        Prioritas: MediaPipe -> dlib -> Haar cascade
        """
        boundary_mask = np.zeros((h, w), dtype=np.uint8)
        interior_mask = np.zeros((h, w), dtype=np.uint8)

        if self._use_mediapipe:
            b_mask, i_mask, found = self._mediapipe_boundary(gray, h, w, boundary_mask, interior_mask, bgr)
            if found:
                self.last_method = "mediapipe"
                return b_mask, i_mask, True

        if self._use_dlib:
            return self._dlib_boundary(gray, h, w, boundary_mask, interior_mask)
        else:
            return self._haar_boundary(gray, h, w, boundary_mask, interior_mask)

    def _mediapipe_boundary(self, gray, h, w, boundary_mask, interior_mask, bgr):
        if not self._use_mediapipe or self._face_mesh is None:
            return boundary_mask, interior_mask, False
            
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            detection_result = self._face_mesh.detect(mp_image)
            
            if not detection_result.face_landmarks:
                return boundary_mask, interior_mask, False
                
            landmarks = detection_result.face_landmarks[0]
            
            # FACEOVAL indices dari MediaPipe Face Mesh
            FACEOVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            
            pts = []
            for idx in FACEOVAL:
                lm = landmarks[idx]
                pt_x = int(lm.x * w)
                pt_y = int(lm.y * h)
                pts.append([pt_x, pt_y])
                
            pts = np.array(pts, dtype=np.int32)
            
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_mask, [pts], 255)
            
            # Erode untuk interior, dilate untuk outer boundary
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BOUNDARY_STRIP_WIDTH * 2, BOUNDARY_STRIP_WIDTH * 2))
            dilated = cv2.dilate(face_mask, kernel)
            eroded = cv2.erode(face_mask, kernel)
            
            boundary_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(eroded))
            interior_mask = eroded
            
            return boundary_mask, interior_mask, True
        except Exception as e:
            logger.warning(f"FaceBlendingAnalyzer: Gagal ekstraksi boundary MediaPipe: {e}")
            return boundary_mask, interior_mask, False

    def _dlib_boundary(self, gray, h, w, boundary_mask, interior_mask):
        self.last_method = "dlib_landmark"
        import dlib
        dets = self.face_detector(gray, 1)
        if len(dets) == 0:
            return boundary_mask, interior_mask, False

        det   = dets[0]
        shape = self.landmark_predictor(gray, det)
        pts   = np.array([[shape.part(i).x, shape.part(i).y]
                          for i in range(17, 68)], dtype=np.int32)

        # Gambar kontur wajah dari landmark
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [pts], 255)

        # Erode untuk interior, dilate untuk outer boundary
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (BOUNDARY_STRIP_WIDTH * 2,
                                              BOUNDARY_STRIP_WIDTH * 2))
        dilated  = cv2.dilate(face_mask, kernel)
        eroded   = cv2.erode(face_mask, kernel)

        boundary_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(eroded))
        interior_mask = eroded

        has_face = (face_mask.sum() > 0
                    and min(det.right() - det.left(),
                            det.bottom() - det.top()) >= MIN_FACE_SIZE)
        return boundary_mask, interior_mask, bool(has_face)

    def _haar_boundary(self, gray, h, w, boundary_mask, interior_mask):
        self.last_method = "haar_approx"
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )
        if len(faces) == 0:
            return boundary_mask, interior_mask, False

        # Ambil wajah terbesar
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])

        # Buat ellipse mask untuk approx kontur wajah
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy    = fx + fw // 2, fy + fh // 2
        cv2.ellipse(face_mask, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, 255, -1)

        kernel        = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BOUNDARY_STRIP_WIDTH * 2, BOUNDARY_STRIP_WIDTH * 2)
        )
        dilated       = cv2.dilate(face_mask, kernel)
        eroded        = cv2.erode(face_mask, kernel)
        boundary_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(eroded))
        interior_mask = eroded

        return boundary_mask, interior_mask, True

    def _compute_gradient_ratio(
        self,
        gray: np.ndarray,
        boundary_mask: np.ndarray,
        interior_mask: np.ndarray,
    ) -> float:
        """
        Hitung rasio rata-rata gradient magnitude di boundary vs interior.
        Face-swap: boundary gradient lebih tinggi dari interior (discontinuity).
        Foto asli: gradients relatif seragam.
        Normalisasi ke 0–1.
        """
        sobelx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag     = np.sqrt(sobelx ** 2 + sobely ** 2)

        b_vals = mag[boundary_mask > 0]
        i_vals = mag[interior_mask > 0]

        if len(b_vals) == 0 or len(i_vals) == 0:
            return 0.0

        b_mean = float(np.mean(b_vals))
        i_mean = float(np.mean(i_vals)) + 1e-6

        ratio = b_mean / i_mean
        # Foto asli: ratio ~1.0–1.5, face-swap: ratio >2.0
        # Normalisasi: ratio 1.0 → 0.0, ratio 3.0 → 1.0
        normalized = float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))
        return normalized

    def _compute_dct_discontinuity(
        self,
        gray: np.ndarray,
        boundary_mask: np.ndarray,
        interior_mask: np.ndarray,
    ) -> float:
        """
        Hitung diskontinuitas koefisien DCT antara blok boundary dan interior.
        Inpainting/face-swap meninggalkan perbedaan statistik koefisien DCT
        di blok yang melintasi batas manipulasi.
        """
        h, w        = gray.shape
        img_float   = gray.astype(np.float32)
        b_energies  = []
        i_energies  = []

        for y in range(0, h - DCT_BLOCK_SIZE, DCT_BLOCK_SIZE):
            for x in range(0, w - DCT_BLOCK_SIZE, DCT_BLOCK_SIZE):
                block    = img_float[y:y + DCT_BLOCK_SIZE, x:x + DCT_BLOCK_SIZE]
                b_region = boundary_mask[y:y + DCT_BLOCK_SIZE, x:x + DCT_BLOCK_SIZE]
                i_region = interior_mask[y:y + DCT_BLOCK_SIZE, x:x + DCT_BLOCK_SIZE]

                dct_block   = dctn(block, norm="ortho")
                ac_energy   = float(np.sum(dct_block[1:, 1:] ** 2))

                if b_region.mean() > 0.3:
                    b_energies.append(ac_energy)
                elif i_region.mean() > 0.3:
                    i_energies.append(ac_energy)

        if not b_energies or not i_energies:
            return 0.0

        b_mean = float(np.mean(b_energies))
        i_mean = float(np.mean(i_energies)) + 1e-6

        # Rasio AC energy — face-swap biasanya boundary lebih tinggi
        ratio      = abs(b_mean - i_mean) / max(b_mean, i_mean)
        normalized = float(np.clip(ratio * 2.0, 0.0, 1.0))
        return normalized

    def _compute_lbp_entropy_diff(
        self,
        gray: np.ndarray,
        boundary_mask: np.ndarray,
        interior_mask: np.ndarray,
    ) -> float:
        """
        Hitung selisih entropi LBP antara boundary dan interior.
        Face-swap: entropi LBP di boundary berbeda signifikan dari interior
        karena pola tekstur "dipotong" di batas blending.
        """
        from skimage.feature import local_binary_pattern

        lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")

        def lbp_entropy(mask: np.ndarray) -> float:
            vals = lbp[mask > 0]
            if len(vals) < 10:
                return 0.0
            hist, _ = np.histogram(vals, bins=LBP_N_POINTS + 2,
                                   range=(0, LBP_N_POINTS + 2), density=True)
            hist    = hist + 1e-10
            return float(scipy_entropy(hist))

        b_ent = lbp_entropy(boundary_mask)
        i_ent = lbp_entropy(interior_mask)

        if i_ent < 1e-6:
            return 0.0

        diff       = abs(b_ent - i_ent) / (i_ent + 1e-6)
        normalized = float(np.clip(diff, 0.0, 1.0))
        return normalized

    # ------------------------------------------------------------------
    def _unavailable(self, reason: str) -> dict[str, Any]:
        return {
            "score":      0.5,
            "confidence": 0.0,
            "verdict":    "N/A",
            "details":    {"reason": reason},
        }