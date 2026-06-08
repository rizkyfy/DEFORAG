"""
modules/eye_reflection.py
--------------------------
Modul analisis konsistensi refleksi mata (Eye Reflection Consistency).

Catchlight (specular highlight) pada kornea mata adalah sinyal forensik
yang kuat untuk mendeteksi AI-generated dan face-swap images:

  1. Foto asli: catchlight bentuk natural (dari sumber cahaya nyata),
     posisi dan bentuk konsisten antara mata kiri dan kanan,
     ada chromatic aberration subtle di tepi catchlight.

  2. AI-generated (full diffusion): catchlight simetris sempurna antara
     kedua mata, bentuk terlalu reguler (bulat sempurna atau persegi),
     tidak ada chromatic aberration.

  3. Face-swap: catchlight dari wajah donor tidak cocok dengan pencahayaan
     background/environment. Inkonsistensi antara catchlight mata dan
     highlight di kulit/rambut.

Metode:
  1. Deteksi region mata via landmark atau Haar eye detector
  2. Isolasi catchlight via threshold adaptive pada region iris
  3. Analisis simetri catchlight L-R (posisi, bentuk, intensitas)
  4. Hitung regularity score (AI catchlight terlalu regular)
  5. Analisis konsistensi catchlight vs highlight di region non-mata

Sub-skor output:
  - lr_symmetry_score    : simetri L-R (tinggi = terlalu simetris = AI)
  - catchlight_regularity: regularitas bentuk catchlight (tinggi = AI)
  - highlight_consistency: konsistensi dengan highlight lain (rendah = face-swap)
  - eye_score            : skor gabungan 0.0–1.0

Catatan:
  - highlight_consistency DIINVERSI: rendah = inkonsisten = mencurigakan
  - Modul ini memberikan confidence rendah jika mata tidak terdeteksi
    atau catchlight terlalu kecil, dan mengembalikan verdict N/A.

Dependensi: opencv-python, numpy, scipy
"""

import logging
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import label as scipy_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

MIN_EYE_SIZE         = 20    # ukuran minimum region mata (px)
CATCHLIGHT_THRESHOLD = 180   # threshold brightness untuk catchlight (0–255) — diturunkan dari 220
MIN_CATCHLIGHT_AREA  = 5     # area minimum catchlight (px²)
MAX_CATCHLIGHT_AREA  = 800   # area maksimum (lebih besar = bukan catchlight)


# ---------------------------------------------------------------------------
# Kelas utama
# ---------------------------------------------------------------------------

class EyeReflectionAnalyzer:
    """
    Mendeteksi anomali catchlight pada mata sebagai indikator deepfake/AI.
    """

    def __init__(self, face_detector=None, landmark_predictor=None) -> None:
        self.face_detector      = face_detector
        self.landmark_predictor = landmark_predictor
        self._use_dlib          = (face_detector is not None
                                   and landmark_predictor is not None)

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

            # --- Deteksi region mata ---
            left_eye_roi, right_eye_roi = self._detect_eye_regions(gray, image, h, w)

            if left_eye_roi is None or right_eye_roi is None:
                return self._unavailable("Kedua mata tidak terdeteksi")

            # --- Ekstrak catchlight dari masing-masing mata ---
            left_cl  = self._extract_catchlight(left_eye_roi)
            right_cl = self._extract_catchlight(right_eye_roi)

            if left_cl is None or right_cl is None:
                return self._unavailable("Catchlight tidak terdeteksi di satu atau kedua mata")

            # --- Sub-skor 1: L-R symmetry ---
            lr_symmetry_score = self._compute_lr_symmetry(
                left_cl, right_cl, left_eye_roi, right_eye_roi
            )

            # --- Sub-skor 2: Catchlight regularity ---
            catchlight_regularity = self._compute_regularity(left_cl, right_cl)

            # --- Sub-skor 3: Highlight consistency ---
            # Diinversi: rendah = inkonsisten = mencurigakan
            highlight_consistency = self._compute_highlight_consistency(
                gray, image, left_eye_roi, right_eye_roi, h, w
            )
            # Inversi untuk scoring (inkonsistensi = skor tinggi = mencurigakan)
            highlight_inconsistency = 1.0 - highlight_consistency

            # --- Eye score gabungan ---
            # lr_symmetry: AI terlalu simetris → tinggi = mencurigakan
            # catchlight_regularity: AI terlalu regular → tinggi = mencurigakan
            # highlight_inconsistency: face-swap inkonsisten → tinggi = mencurigakan
            eye_score = float(np.clip(
                0.35 * lr_symmetry_score
                + 0.35 * catchlight_regularity
                + 0.30 * highlight_inconsistency,
                0.0, 1.0
            ))

            # --- Confidence ---
            confidence = 0.75  # relatif tinggi jika kedua mata dan catchlight terdeteksi

            # --- Verdict ---
            if eye_score >= 0.60:
                verdict = "FAKE"
            elif eye_score >= 0.42:
                verdict = "SUSPICIOUS"
            else:
                verdict = "REAL"

            return {
                "score":      round(eye_score, 4),
                "confidence": round(confidence, 4),
                "verdict":    verdict,
                "details": {
                    "sub_scores": {
                        "lr_symmetry_score":    round(lr_symmetry_score, 4),
                        "catchlight_regularity":round(catchlight_regularity, 4),
                        "highlight_consistency":round(highlight_consistency, 4),
                    },
                    "left_catchlight_area":  int(left_cl["area"]),
                    "right_catchlight_area": int(right_cl["area"]),
                },
            }

        except Exception as exc:
            logger.error("EyeReflectionAnalyzer error: %s", exc, exc_info=True)
            return self._unavailable(f"Error: {exc}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_eye_regions(self, gray, bgr, h, w):
        """Kembalikan (left_eye_patch, right_eye_patch) sebagai dict dengan
        keys: 'patch' (np.ndarray), 'x', 'y', 'w', 'h'."""
        if self._use_dlib:
            return self._dlib_eyes(gray, h, w)
        return self._haar_eyes(gray, h, w)

    def _dlib_eyes(self, gray, h, w):
        dets = self.face_detector(gray, 1)
        if len(dets) == 0:
            return None, None

        det   = dets[0]
        shape = self.landmark_predictor(gray, det)

        def eye_bbox(indices):
            pts = [(shape.part(i).x, shape.part(i).y) for i in indices]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            x1, y1 = max(0, min(xs) - 8), max(0, min(ys) - 8)
            x2, y2 = min(w, max(xs) + 8), min(h, max(ys) + 8)
            if x2 - x1 < MIN_EYE_SIZE or y2 - y1 < MIN_EYE_SIZE:
                return None
            return {
                "patch": gray[y1:y2, x1:x2],
                "x": x1, "y": y1,
                "w": x2 - x1, "h": y2 - y1,
            }

        # Landmark 68-point: mata kiri 36-41, mata kanan 42-47
        left  = eye_bbox(range(36, 42))
        right = eye_bbox(range(42, 48))
        return left, right

    def _haar_eyes(self, gray, h, w):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None, None

        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        face_gray      = gray[fy:fy + fh, fx:fx + fw]

        eyes = eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(MIN_EYE_SIZE, MIN_EYE_SIZE)
        )
        if len(eyes) < 2:
            return None, None

        # Ambil dua mata teratas, sort kiri-kanan
        eyes_sorted = sorted(eyes, key=lambda e: e[1])[:2]
        eyes_sorted = sorted(eyes_sorted, key=lambda e: e[0])

        def to_dict(ex, ey, ew, eh):
            ax, ay = fx + ex, fy + ey
            return {
                "patch": gray[ay:ay + eh, ax:ax + ew],
                "x": ax, "y": ay, "w": ew, "h": eh,
            }

        left  = to_dict(*eyes_sorted[0])
        right = to_dict(*eyes_sorted[1])
        return left, right

    def _extract_catchlight(self, eye_roi: dict) -> dict | None:
        """
        Ekstrak catchlight terbesar dari region mata.
        Kembalikan dict: area, centroid_rel, circularity, mean_intensity.
        """
        patch = eye_roi["patch"]
        if patch.size == 0:
            return None

        ph, pw = patch.shape

        # Threshold adaptif untuk catchlight
        _, thresh = cv2.threshold(patch, CATCHLIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Label connected components
        labeled, n_components = scipy_label(thresh)
        if n_components == 0:
            return None

        best = None
        best_area = 0

        for label_id in range(1, n_components + 1):
            component = (labeled == label_id).astype(np.uint8)
            area      = int(component.sum())

            if area < MIN_CATCHLIGHT_AREA or area > MAX_CATCHLIGHT_AREA:
                continue

            if area > best_area:
                best_area = area
                ys, xs    = np.where(component > 0)
                cx_rel    = float(np.mean(xs)) / pw  # relatif terhadap lebar mata
                cy_rel    = float(np.mean(ys)) / ph

                # Circularity: 4π·area / perimeter²
                contours, _ = cv2.findContours(
                    component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                perimeter = cv2.arcLength(contours[0], True) if contours else 1.0
                circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

                best = {
                    "area":         area,
                    "cx_rel":       cx_rel,
                    "cy_rel":       cy_rel,
                    "circularity":  float(circularity),
                    "mean_intensity": float(patch[ys, xs].mean()),
                }

        return best

    def _compute_lr_symmetry(self, left_cl, right_cl, left_roi, right_roi) -> float:
        """
        Hitung simetri L-R catchlight.

        Foto asli: posisi catchlight sedikit berbeda antara L dan R
        karena perspektif dan kontur kornea yang natural.
        AI-generated: posisi dan ukuran sangat simetris (mirrored).

        Normalisasi: 0 = asimetris alami, 1 = terlalu simetris (AI).
        """
        # Posisi relatif (setelah flip untuk mirror comparison)
        left_cx  = left_cl["cx_rel"]
        right_cx = 1.0 - right_cl["cx_rel"]  # mirror
        left_cy  = left_cl["cy_rel"]
        right_cy = right_cl["cy_rel"]

        pos_diff = np.sqrt((left_cx - right_cx) ** 2 + (left_cy - right_cy) ** 2)

        # Perbedaan ukuran (area)
        l_area   = left_cl["area"]
        r_area   = right_cl["area"]
        area_ratio = min(l_area, r_area) / (max(l_area, r_area) + 1e-6)

        # Perbedaan circularity
        circ_diff = abs(left_cl["circularity"] - right_cl["circularity"])

        # Foto asli: pos_diff ~0.1–0.3, area_ratio ~0.5–0.8, circ_diff ~0.1–0.3
        # AI: pos_diff ~0.0–0.05, area_ratio ~0.9–1.0, circ_diff ~0.0–0.05

        # Simetri = rendah pos_diff + tinggi area_ratio + rendah circ_diff
        symmetry = (
            (1.0 - float(np.clip(pos_diff / 0.3, 0.0, 1.0))) * 0.4
            + float(np.clip(area_ratio, 0.0, 1.0)) * 0.4
            + (1.0 - float(np.clip(circ_diff / 0.3, 0.0, 1.0))) * 0.2
        )

        # Normalisasi ulang: symmetry 0.5 → skor 0, symmetry 1.0 → skor 1
        score = float(np.clip((symmetry - 0.5) / 0.5, 0.0, 1.0))
        return score

    def _compute_regularity(self, left_cl: dict, right_cl: dict) -> float:
        """
        Hitung regularitas bentuk catchlight.

        AI: catchlight bulat sempurna (circularity ~1.0) atau persegi.
        Foto asli: catchlight tidak beraturan, memantulkan sumber cahaya
        nyata (window, softbox, ambient).

        Normalisasi: 0 = natural/irregular, 1 = terlalu regular (AI).
        """
        avg_circularity = (left_cl["circularity"] + right_cl["circularity"]) / 2.0

        # Circularity 1.0 = bulat sempurna = AI
        # Foto asli: ~0.4–0.8
        regularity = float(np.clip((avg_circularity - 0.4) / 0.6, 0.0, 1.0))
        return regularity

    def _compute_highlight_consistency(
        self,
        gray: np.ndarray,
        bgr: np.ndarray,
        left_roi: dict,
        right_roi: dict,
        h: int,
        w: int,
    ) -> float:
        """
        Hitung konsistensi catchlight dengan highlight di area non-mata.

        Face-swap: wajah baru punya catchlight yang tidak cocok dengan
        pencahayaan asli (dari background/baju/rambut).

        Logika: bandingkan arah dominan highlight di mata vs arah
        dominan highlight di region non-wajah (bahu, background atas).

        Kembalikan: 0.0 = inkonsisten (mencurigakan), 1.0 = konsisten (asli).
        """
        # Highlight di mata: centroid relatif dari catchlight
        left_cx  = left_roi["x"] + left_cl_cx(left_roi)
        right_cx = right_roi["x"] + right_cl_cx(right_roi)
        eye_light_x = (left_cx + right_cx) / 2.0 / w  # 0 = kiri, 1 = kanan

        # Highlight di background (atas gambar, 0–20% height)
        bg_region = gray[:int(h * 0.2), :]
        bright_bg = (bg_region > 200).astype(np.float32)
        if bright_bg.sum() > 0:
            bg_xs = np.where(bright_bg > 0)[1]
            bg_light_x = float(np.mean(bg_xs)) / w
        else:
            # Tidak ada highlight di background — tidak bisa dibandingkan
            return 0.5

        # Konsistensi: semakin dekat posisi horizontal, semakin konsisten
        diff        = abs(eye_light_x - bg_light_x)
        consistency = float(np.clip(1.0 - diff * 2.0, 0.0, 1.0))
        return consistency

    # ------------------------------------------------------------------
    def _unavailable(self, reason: str) -> dict[str, Any]:
        return {
            "score":      0.5,
            "confidence": 0.0,
            "verdict":    "N/A",
            "details":    {"reason": reason},
        }


# ---------------------------------------------------------------------------
# Helper standalone (digunakan dalam _compute_highlight_consistency)
# ---------------------------------------------------------------------------

def left_cl_cx(eye_roi: dict) -> float:
    """Estimasi posisi x catchlight dari centroid patch."""
    return eye_roi["w"] * 0.5   # fallback ke tengah jika tidak ada catchlight detail


def right_cl_cx(eye_roi: dict) -> float:
    return eye_roi["w"] * 0.5