"""
landmark_analyzer.py
====================
Modul analisis deepfake berbasis deteksi landmark wajah.

Menggunakan MediaPipe Face Landmarker (Tasks API, kompatibel 0.10.x+)
untuk menganalisis 478 landmark wajah:
  - Simetri wajah (perbandingan landmark kiri vs kanan)
  - Proporsi geometri (rasio mata-hidung-mulut)
  - Eye Aspect Ratio / naturalitas mata
  - Konsistensi blending area boundary wajah

Deepfake sering menampilkan asimetri yang tidak natural, proporsi yang
sedikit menyimpang, atau inkonsistensi geometri halus yang sulit dilihat
mata manusia namun dapat dideteksi secara matematis.

Skor mendekati 1.0 → lebih mencurigakan (FAKE)
Skor mendekati 0.0 → lebih natural (REAL)

Author  : DeepGuard Team
Version : 1.1.0  (MediaPipe Tasks API — kompatibel 0.10.x+)
"""

import logging
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lokasi & URL model Face Landmarker
# ---------------------------------------------------------------------------
_FACE_LANDMARKER_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_face_landmarker_model() -> bool:
    """
    Pastikan model face_landmarker.task tersedia.
    Download otomatis jika belum ada (~6 MB).

    Returns
    -------
    bool
        True jika model tersedia, False jika gagal.
    """
    if _FACE_LANDMARKER_MODEL_PATH.exists():
        return True

    logger.info(
        "Model Face Landmarker tidak ditemukan. Mengunduh dari Google (~6 MB)... "
        "Simpan ke: %s",
        _FACE_LANDMARKER_MODEL_PATH,
    )
    try:
        urllib.request.urlretrieve(
            _FACE_LANDMARKER_MODEL_URL,
            str(_FACE_LANDMARKER_MODEL_PATH),
        )
        logger.info("Model Face Landmarker berhasil diunduh.")
        return True
    except Exception as exc:
        logger.error(
            "Gagal mengunduh model Face Landmarker: %s\n"
            "Download manual dari:\n  %s\n"
            "Lalu simpan ke: %s",
            exc,
            _FACE_LANDMARKER_MODEL_URL,
            _FACE_LANDMARKER_MODEL_PATH,
        )
        return False


# ---------------------------------------------------------------------------
# Import MediaPipe Tasks API (0.10.x+)
# ---------------------------------------------------------------------------
MEDIAPIPE_AVAILABLE = False
_mp = None
_mp_python = None
_mp_vision = None

try:
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe %s (Tasks API) berhasil diimpor.", _mp.__version__)
except ImportError:
    logger.warning(
        "MediaPipe tidak tersedia. LandmarkAnalyzer akan mengembalikan "
        "skor netral 0.5. Install dengan: pip install mediapipe"
    )

# ---------------------------------------------------------------------------
# Import utilitas DeepGuard
# ---------------------------------------------------------------------------
try:
    from deepguard.utils.face_extractor import FaceExtractor
    from deepguard.utils.video_processor import VideoProcessor
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deepguard.utils.face_extractor import FaceExtractor
    from deepguard.utils.video_processor import VideoProcessor


# ===========================================================================
# Konstanta & Indeks Landmark MediaPipe Face Mesh
# ===========================================================================

# Indeks landmark untuk analisis simetri (pasangan kiri-kanan)
_SYMMETRY_PAIRS = [
    # Mata
    (33, 263),
    (133, 362),
    (159, 386),
    (145, 374),
    # Alis
    (70, 300),
    (63, 293),
    # Pipi & tulang pipi
    (234, 454),
    (93, 323),
    # Mulut
    (61, 291),
    (81, 311),
    (178, 402),
]

# Indeks landmark untuk Eye Aspect Ratio (EAR)
_EAR_LEFT  = [33, 160, 158, 133, 153, 144]
_EAR_RIGHT = [362, 385, 387, 263, 373, 380]

# Indeks landmark titik kunci wajah
_NOSE_TIP   = 1
_CHIN       = 152
_MOUTH_LEFT  = 61
_MOUTH_RIGHT = 291
_FOREHEAD    = 10

# Threshold kalibrasi
_ASYMMETRY_SUSPICIOUS_THRESHOLD = 0.05
_EAR_NATURAL_MIN = 0.15
_EAR_NATURAL_MAX = 0.50


class LandmarkAnalyzer:
    """
    Analyzer deepfake berbasis analisis geometri landmark wajah.

    Menggunakan MediaPipe Face Landmarker (Tasks API, 0.10.x+) dengan
    478 landmark untuk menghitung berbagai metrik geometri wajah.

    Jika MediaPipe tidak tersedia atau model gagal diunduh, analyzer
    akan mengembalikan skor netral 0.5 dengan keterangan yang sesuai.
    """

    MODULE_NAME = "Analisis Landmark Wajah (MediaPipe)"

    def __init__(self) -> None:
        """Inisialisasi LandmarkAnalyzer dengan Tasks API."""
        self._video_processor = VideoProcessor()
        self._face_mesh = None

        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe tidak tersedia — LandmarkAnalyzer nonaktif.")
            return

        model_ready = _ensure_face_landmarker_model()
        if not model_ready:
            logger.error(
                "Model Face Landmarker tidak tersedia — LandmarkAnalyzer nonaktif."
            )
            return

        try:
            options = _mp_vision.FaceLandmarkerOptions(
                base_options=_mp_python.BaseOptions(
                    model_asset_path=str(_FACE_LANDMARKER_MODEL_PATH)
                ),
                running_mode=_mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._face_mesh = _mp_vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker (Tasks API) diinisialisasi.")
        except Exception as exc:
            logger.error("Gagal menginisialisasi FaceLandmarker: %s", exc)
            self._face_mesh = None

    # ------------------------------------------------------------------
    # Deteksi landmark
    # ------------------------------------------------------------------

    def _detect_landmarks(
        self, image_rgb: np.ndarray
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Deteksi landmark wajah menggunakan MediaPipe Face Landmarker (Tasks API).

        Parameters
        ----------
        image_rgb : np.ndarray
            Gambar RGB uint8 shape (H, W, 3).

        Returns
        -------
        list of (x, y, z) or None
            Daftar 478 landmark (x, y, z); x/y dalam piksel, z normalized.
            None jika tidak ada wajah terdeteksi.
        """
        if self._face_mesh is None:
            return None

        try:
            h, w = image_rgb.shape[:2]

            # Pastikan array contiguous dan tipe uint8
            image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

            # Bungkus ke MediaPipe Image
            mp_image = _mp.Image(
                image_format=_mp.ImageFormat.SRGB,
                data=image_rgb,
            )

            detection_result = self._face_mesh.detect(mp_image)

            if not detection_result.face_landmarks:
                logger.debug("MediaPipe tidak mendeteksi wajah.")
                return None

            # Ambil wajah pertama — konversi ke koordinat piksel
            face_landmarks = detection_result.face_landmarks[0]
            landmarks = [
                (lm.x * w, lm.y * h, lm.z)
                for lm in face_landmarks
            ]
            return landmarks

        except Exception as exc:
            logger.error("Error saat deteksi landmark: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Analisis simetri
    # ------------------------------------------------------------------

    def _compute_symmetry_score(
        self,
        landmarks: List[Tuple[float, float, float]],
        img_w: float,
        img_h: float,
    ) -> Dict[str, float]:
        """
        Hitung skor asimetri wajah berdasarkan pasangan landmark kiri-kanan.
        """
        result = {"asymmetry_mean": 0.0, "asymmetry_std": 0.0, "asymmetry_score": 0.0}

        try:
            if len(landmarks) <= max(p for pair in _SYMMETRY_PAIRS for p in pair):
                return result

            nose_x = landmarks[_NOSE_TIP][0] if _NOSE_TIP < len(landmarks) else img_w / 2

            diffs: List[float] = []
            for left_idx, right_idx in _SYMMETRY_PAIRS:
                if left_idx >= len(landmarks) or right_idx >= len(landmarks):
                    continue

                left_lm  = landmarks[left_idx]
                right_lm = landmarks[right_idx]

                left_dist  = abs(left_lm[0]  - nose_x)
                right_dist = abs(right_lm[0] - nose_x)

                face_width_approx = img_w * 0.4
                if face_width_approx > 0:
                    diff = abs(left_dist - right_dist) / face_width_approx
                    diffs.append(diff)

            if not diffs:
                return result

            asym_mean = float(np.mean(diffs))
            asym_std  = float(np.std(diffs))

            if asym_mean <= _ASYMMETRY_SUSPICIOUS_THRESHOLD:
                asym_score = asym_mean / _ASYMMETRY_SUSPICIOUS_THRESHOLD * 0.3
            else:
                excess = asym_mean - _ASYMMETRY_SUSPICIOUS_THRESHOLD
                asym_score = 0.3 + min(0.7, excess / 0.15)

            result["asymmetry_mean"]  = round(asym_mean, 6)
            result["asymmetry_std"]   = round(asym_std, 6)
            result["asymmetry_score"] = round(float(np.clip(asym_score, 0.0, 1.0)), 4)

        except Exception as exc:
            logger.error("Error saat menghitung skor simetri: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Eye Aspect Ratio (EAR)
    # ------------------------------------------------------------------

    def _compute_ear(
        self,
        landmarks: List[Tuple[float, float, float]],
        indices: List[int],
    ) -> float:
        """Hitung Eye Aspect Ratio (EAR) untuk satu mata."""
        try:
            p = [np.array(landmarks[i][:2]) for i in indices]

            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h  = np.linalg.norm(p[0] - p[3])

            if h < 1e-6:
                return 0.0

            return float((v1 + v2) / (2.0 * h))
        except Exception:
            return 0.0

    def _compute_eye_score(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> Dict[str, float]:
        """Hitung skor anomali dari Eye Aspect Ratio."""
        result = {"ear_left": 0.0, "ear_right": 0.0, "ear_diff": 0.0, "eye_score": 0.0}

        try:
            if len(landmarks) < max(_EAR_LEFT + _EAR_RIGHT):
                return result

            ear_l = self._compute_ear(landmarks, _EAR_LEFT)
            ear_r = self._compute_ear(landmarks, _EAR_RIGHT)

            ear_diff = abs(ear_l - ear_r)
            ear_avg  = (ear_l + ear_r) / 2.0

            result["ear_left"]  = round(ear_l, 4)
            result["ear_right"] = round(ear_r, 4)
            result["ear_diff"]  = round(ear_diff, 4)

            eye_score = 0.0
            if ear_diff > 0.07:
                eye_score += min(0.5, (ear_diff - 0.07) / 0.15)
            if ear_avg < _EAR_NATURAL_MIN or ear_avg > _EAR_NATURAL_MAX:
                eye_score += 0.3

            result["eye_score"] = round(float(np.clip(eye_score, 0.0, 1.0)), 4)

        except Exception as exc:
            logger.error("Error saat menghitung EAR: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Analisis proporsi geometri
    # ------------------------------------------------------------------

    def _compute_proportion_score(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> Dict[str, float]:
        """Hitung skor anomali dari proporsi geometri wajah."""
        result = {
            "eye_nose_ratio": 0.0,
            "nose_mouth_ratio": 0.0,
            "proportion_score": 0.0,
        }

        try:
            n_lm = len(landmarks)
            required = [_NOSE_TIP, _CHIN, _MOUTH_LEFT, _MOUTH_RIGHT, _FOREHEAD, 159, 386]
            if any(idx >= n_lm for idx in required):
                return result

            nose       = np.array(landmarks[_NOSE_TIP][:2])
            chin       = np.array(landmarks[_CHIN][:2])
            mouth_l    = np.array(landmarks[_MOUTH_LEFT][:2])
            mouth_r    = np.array(landmarks[_MOUTH_RIGHT][:2])
            forehead   = np.array(landmarks[_FOREHEAD][:2])
            eye_l      = np.array(landmarks[159][:2])
            eye_r      = np.array(landmarks[386][:2])

            eye_center   = (eye_l + eye_r) / 2.0
            mouth_center = (mouth_l + mouth_r) / 2.0

            dist_forehead_eye  = float(np.linalg.norm(forehead - eye_center))
            dist_eye_nose      = float(np.linalg.norm(eye_center - nose))
            dist_nose_mouth    = float(np.linalg.norm(nose - mouth_center))
            dist_mouth_chin    = float(np.linalg.norm(mouth_center - chin))

            if dist_forehead_eye < 1e-6 or dist_nose_mouth < 1e-6:
                return result

            total_v = dist_forehead_eye + dist_eye_nose + dist_nose_mouth + dist_mouth_chin
            if total_v < 1e-6:
                return result

            ratio_eye_nose   = dist_eye_nose / total_v
            ratio_nose_mouth = dist_nose_mouth / total_v

            result["eye_nose_ratio"]   = round(ratio_eye_nose, 4)
            result["nose_mouth_ratio"] = round(ratio_nose_mouth, 4)

            deviation_eye_nose   = abs(ratio_eye_nose   - 0.25)
            deviation_nose_mouth = abs(ratio_nose_mouth - 0.25)
            avg_deviation = (deviation_eye_nose + deviation_nose_mouth) / 2.0

            proportion_score = min(1.0, avg_deviation / 0.12)
            result["proportion_score"] = round(float(proportion_score), 4)

        except Exception as exc:
            logger.error("Error saat menghitung proporsi: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Konsistensi area boundary wajah
    # ------------------------------------------------------------------

    def _compute_boundary_score(
        self,
        landmarks: List[Tuple[float, float, float]],
        image_rgb: np.ndarray,
    ) -> Dict[str, float]:
        """Analisis konsistensi area boundary (tepi) wajah."""
        result = {"boundary_gradient_mean": 0.0, "boundary_score": 0.0}

        try:
            h, w = image_rgb.shape[:2]

            face_oval_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                162, 21, 54, 103, 67, 109, 10
            ]

            valid_indices = [i for i in face_oval_indices if i < len(landmarks)]
            if len(valid_indices) < 5:
                return result

            contour_pts = np.array(
                [(int(landmarks[i][0]), int(landmarks[i][1]))
                 for i in valid_indices],
                dtype=np.int32
            )

            mask_inner = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_inner, [contour_pts], 255)

            kernel    = np.ones((11, 11), np.uint8)
            mask_outer = cv2.dilate(mask_inner, kernel, iterations=1)
            mask_erode = cv2.erode(mask_inner, kernel, iterations=1)
            boundary_mask = cv2.bitwise_and(
                cv2.bitwise_xor(mask_outer, mask_erode),
                np.ones((h, w), dtype=np.uint8) * 255
            )

            if np.sum(boundary_mask > 0) < 10:
                return result

            gray_float = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

            boundary_pixels = gradient_mag[boundary_mask > 0]
            if len(boundary_pixels) == 0:
                return result

            boundary_grad_mean = float(np.mean(boundary_pixels))
            result["boundary_gradient_mean"] = round(boundary_grad_mean, 4)

            if boundary_grad_mean < 5.0:
                boundary_score = (5.0 - boundary_grad_mean) / 5.0
            elif boundary_grad_mean > 50.0:
                boundary_score = min(0.5, (boundary_grad_mean - 50.0) / 100.0)
            else:
                boundary_score = 0.0

            result["boundary_score"] = round(float(np.clip(boundary_score, 0.0, 1.0)), 4)

        except Exception as exc:
            logger.debug("Error saat menghitung boundary score: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Analisis satu gambar
    # ------------------------------------------------------------------

    def _analyze_single_image(
        self, image_rgb: np.ndarray
    ) -> Dict[str, Any]:
        """Pipeline analisis landmark pada satu gambar."""
        h, w = image_rgb.shape[:2]

        landmarks = self._detect_landmarks(image_rgb)
        if landmarks is None:
            return {
                "score": 0.5,
                "face_detected": False,
                "note": "Wajah tidak terdeteksi oleh MediaPipe",
            }

        symmetry   = self._compute_symmetry_score(landmarks, float(w), float(h))
        eye        = self._compute_eye_score(landmarks)
        proportion = self._compute_proportion_score(landmarks)
        boundary   = self._compute_boundary_score(landmarks, image_rgb)

        weights = {
            "symmetry":   0.40,
            "eye":        0.25,
            "proportion": 0.20,
            "boundary":   0.15,
        }

        component_scores = {
            "symmetry":   symmetry.get("asymmetry_score", 0.0),
            "eye":        eye.get("eye_score", 0.0),
            "proportion": proportion.get("proportion_score", 0.0),
            "boundary":   boundary.get("boundary_score", 0.0),
        }

        combined_score = sum(
            component_scores[k] * weights[k]
            for k in weights
        )
        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        return {
            "score": combined_score,
            "face_detected": True,
            "symmetry": symmetry,
            "eye": eye,
            "proportion": proportion,
            "boundary": boundary,
            "component_scores": component_scores,
        }

    # ------------------------------------------------------------------
    # API Publik
    # ------------------------------------------------------------------

    def analyze(
        self, image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Analisis gambar atau video menggunakan geometri landmark wajah.

        Parameters
        ----------
        image_path : str | Path
            Path ke file gambar atau video.

        Returns
        -------
        dict
            - ``score``      : float [0–1], mendekati 1.0 = FAKE
            - ``verdict``    : str, 'FAKE' | 'REAL' | 'SUSPICIOUS'
            - ``confidence`` : float [0–1]
            - ``details``    : dict, detail analisis landmark
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        if not MEDIAPIPE_AVAILABLE:
            return self._build_result(
                score=0.5,
                face_detected=False,
                source_path=str(path),
                note="mediapipe tidak tersedia; install dengan: pip install mediapipe",
            )

        if self._face_mesh is None:
            return self._build_result(
                score=0.5,
                face_detected=False,
                source_path=str(path),
                note=(
                    "FaceLandmarker gagal diinisialisasi. "
                    "Pastikan face_landmarker.task tersedia di folder modules/ "
                    f"atau download dari: {_FACE_LANDMARKER_MODEL_URL}"
                ),
            )

        logger.info("LandmarkAnalyzer memproses: %s", path.name)

        try:
            is_video = self._video_processor.is_video(path)

            if is_video:
                frames = self._video_processor.extract_frames(path, n_frames=8)
                if not frames:
                    return self._build_result(0.5, False, str(path), error="Tidak ada frame")

                all_scores: List[float] = []
                all_face_detected = False

                for frame_pil in frames:
                    frame_np = np.array(frame_pil)
                    result_int = self._analyze_single_image(frame_np)
                    if result_int.get("face_detected"):
                        all_face_detected = True
                    all_scores.append(result_int.get("score", 0.5))

                avg_score = float(np.mean(all_scores))
                return self._build_result(
                    avg_score,
                    all_face_detected,
                    str(path),
                    n_frames=len(all_scores),
                    frame_scores=all_scores,
                )

            else:
                pil_img = Image.open(str(path)).convert("RGB")
                img_rgb = np.array(pil_img)

                result_int = self._analyze_single_image(img_rgb)
                return self._build_result(
                    score=result_int.get("score", 0.5),
                    face_detected=result_int.get("face_detected", False),
                    source_path=str(path),
                    symmetry_details=result_int.get("symmetry"),
                    eye_details=result_int.get("eye"),
                    proportion_details=result_int.get("proportion"),
                    boundary_details=result_int.get("boundary"),
                    component_scores=result_int.get("component_scores"),
                    note=result_int.get("note"),
                )

        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Error saat analisis landmark: %s", exc)
            return self._build_result(0.5, False, str(path), error=str(exc))

    def _build_result(
        self,
        score: float,
        face_detected: bool,
        source_path: str,
        symmetry_details: Optional[Dict] = None,
        eye_details: Optional[Dict] = None,
        proportion_details: Optional[Dict] = None,
        boundary_details: Optional[Dict] = None,
        component_scores: Optional[Dict] = None,
        n_frames: int = 1,
        frame_scores: Optional[List[float]] = None,
        note: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bangun dictionary hasil analisis terstandarisasi."""
        verdict    = "FAKE" if score >= 0.5 else "REAL"
        confidence = abs(score - 0.5) * 2.0

        details: Dict[str, Any] = {
            "module": self.MODULE_NAME,
            "source_path": source_path,
            "face_detected": face_detected,
            "mediapipe_available": MEDIAPIPE_AVAILABLE,
            "n_frames_analyzed": n_frames,
        }

        if symmetry_details:
            details["symmetry"] = symmetry_details
        if eye_details:
            details["eye_aspect_ratio"] = eye_details
        if proportion_details:
            details["facial_proportions"] = proportion_details
        if boundary_details:
            details["boundary_consistency"] = boundary_details
        if component_scores:
            details["component_scores"] = component_scores
        if frame_scores is not None:
            details["frame_scores"] = [round(s, 4) for s in frame_scores]
        if note is not None:
            details["note"] = note
        if error is not None:
            details["error"] = error

        return {
            "score": round(float(score), 6),
            "verdict": verdict,
            "confidence": round(float(confidence), 6),
            "details": details,
        }

    def __del__(self) -> None:
        """Tutup FaceLandmarker saat objek dihancurkan."""
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        status = "aktif" if self._face_mesh is not None else "nonaktif"
        return f"LandmarkAnalyzer(mediapipe={status})"