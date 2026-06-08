"""
frequency_analyzer.py
=====================
Modul analisis deepfake berbasis frekuensi spatial (FFT & DCT).

Deepfake yang dihasilkan oleh GAN atau model generatif sering meninggalkan
jejak yang dapat dideteksi di domain frekuensi:
  - Energi frekuensi tinggi yang tidak natural
  - Artefak periodik pada magnitude spectrum
  - Distribusi statistik spectrum yang menyimpang dari gambar asli

Alur analisis:
  1. Konversi gambar ke grayscale
  2. FFT 2D → magnitude spectrum → log scale
  3. DCT pada blok 8×8 (mirip analisis JPEG)
  4. Ekstraksi fitur statistik spectrum
  5. Deteksi artefak periodik
  6. Kombinasikan skor menjadi skor kecurigaan tunggal

Author  : DeepGuard Team
Version : 1.0.0
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import opsional: scipy
# ---------------------------------------------------------------------------
try:
    from scipy import fft as scipy_fft
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
    logger.info("scipy berhasil diimpor.")
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy tidak tersedia. Beberapa fitur analisis frekuensi dinonaktifkan.")

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
# Konstanta & Parameter Analisis
# ===========================================================================

# Rasio energi frekuensi tinggi: frekuensi di atas persentil ini dianggap "tinggi"
_HIGH_FREQ_PERCENTILE = 70

# Ukuran blok DCT
_DCT_BLOCK_SIZE = 8

# Threshold untuk menganggap suatu peak di spectrum sebagai artefak periodik
_PERIODIC_NOISE_ZSCORE = 3.5

# Bobot untuk setiap komponen skor (jumlah harus 1.0)
_WEIGHT_HIGH_FREQ_ENERGY   = 0.30
_WEIGHT_SPECTRAL_FLATNESS  = 0.20
_WEIGHT_PERIODIC_ARTIFACTS = 0.25
_WEIGHT_DCT_STATISTICS     = 0.15
_WEIGHT_KURTOSIS           = 0.10


class FrequencyAnalyzer:
    """
    Analyzer deepfake berbasis analisis frekuensi spatial.

    Tidak memerlukan model terlatih; semua analisis berbasis
    properti matematika dari gambar.

    Skor mendekati 1.0 menunjukkan gambar lebih mencurigakan (FAKE).
    Skor mendekati 0.0 menunjukkan gambar lebih natural (REAL).
    """

    MODULE_NAME = "Analisis Frekuensi (FFT/DCT)"

    def __init__(self) -> None:
        """Inisialisasi FrequencyAnalyzer."""
        self._face_extractor = FaceExtractor()
        self._video_processor = VideoProcessor()
        logger.info("FrequencyAnalyzer diinisialisasi.")

    # ------------------------------------------------------------------
    # Utilitas preprocessing
    # ------------------------------------------------------------------

    def _load_to_grayscale(
        self, source: Union[str, Path, np.ndarray, Image.Image]
    ) -> Optional[np.ndarray]:
        """
        Muat gambar dan konversi ke grayscale float32.

        Parameters
        ----------
        source : str | Path | np.ndarray | PIL.Image.Image
            Sumber gambar.

        Returns
        -------
        np.ndarray or None
            Array float32 grayscale shape (H, W), atau None jika gagal.
        """
        try:
            if isinstance(source, (str, Path)):
                pil_img = Image.open(str(source)).convert("RGB")
                img_np = np.array(pil_img)
            elif isinstance(source, np.ndarray):
                img_np = source
            elif isinstance(source, Image.Image):
                img_np = np.array(source.convert("RGB"))
            else:
                return None

            if img_np.ndim == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np

            return gray.astype(np.float32)

        except Exception as exc:
            logger.error("Gagal memuat gambar ke grayscale: %s", exc)
            return None

    def _get_analysis_image(
        self, image_path: Path
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        Dapatkan gambar untuk dianalisis: wajah jika ada, gambar penuh jika tidak.

        Parameters
        ----------
        image_path : Path
            Path ke file gambar.

        Returns
        -------
        tuple[np.ndarray or None, bool]
            (gambar_rgb_uint8, apakah_wajah_ditemukan)
        """
        try:
            pil_img = Image.open(str(image_path)).convert("RGB")
            img_rgb = np.array(pil_img)
        except Exception as exc:
            logger.error("Gagal memuat gambar: %s", exc)
            return None, False

        face_np = self._face_extractor.extract(img_rgb)
        if face_np is not None:
            return face_np, True
        else:
            return img_rgb, False

    # ------------------------------------------------------------------
    # Analisis FFT
    # ------------------------------------------------------------------

    def _compute_fft_features(
        self, gray: np.ndarray
    ) -> Dict[str, float]:
        """
        Hitung fitur dari FFT 2D magnitude spectrum.

        Parameters
        ----------
        gray : np.ndarray
            Gambar grayscale float32 shape (H, W).

        Returns
        -------
        dict
            Dictionary berisi berbagai metrik frekuensi:
            - high_freq_energy_ratio : float [0–1]
            - spectral_flatness       : float [0–1]
            - mean_magnitude         : float
            - std_magnitude          : float
            - kurtosis               : float
            - skewness               : float
        """
        features: Dict[str, float] = {}

        try:
            # Pastikan ukuran gambar cukup
            h, w = gray.shape
            if h < 8 or w < 8:
                logger.warning("Gambar terlalu kecil untuk FFT: %dx%d", h, w)
                return {}

            # Terapkan window Hann untuk mengurangi spectral leakage
            window_h = np.hanning(h)
            window_w = np.hanning(w)
            window_2d = np.outer(window_h, window_w)
            gray_windowed = gray * window_2d

            # FFT 2D
            fft2d = np.fft.fft2(gray_windowed)
            fft_shifted = np.fft.fftshift(fft2d)
            magnitude = np.abs(fft_shifted)

            # Log magnitude (lebih stabil secara numerik)
            log_magnitude = np.log1p(magnitude)

            # Flatten untuk analisis statistik
            mag_flat = log_magnitude.flatten()

            # Statistik dasar
            features["mean_magnitude"] = float(np.mean(mag_flat))
            features["std_magnitude"] = float(np.std(mag_flat))

            # Kurtosis dan skewness (memerlukan scipy atau numpy)
            if SCIPY_AVAILABLE:
                features["kurtosis"] = float(scipy_stats.kurtosis(mag_flat))
                features["skewness"] = float(scipy_stats.skew(mag_flat))
            else:
                # Implementasi manual kurtosis dan skewness
                mean_val = np.mean(mag_flat)
                std_val = np.std(mag_flat) + 1e-8
                features["kurtosis"] = float(
                    np.mean(((mag_flat - mean_val) / std_val) ** 4) - 3.0
                )
                features["skewness"] = float(
                    np.mean(((mag_flat - mean_val) / std_val) ** 3)
                )

            # ---- Rasio energi frekuensi tinggi ----
            # Buat mask: luar lingkaran radius r dianggap frekuensi tinggi
            center_h, center_w = h // 2, w // 2
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = np.sqrt(
                (y_coords - center_h) ** 2 + (x_coords - center_w) ** 2
            )
            max_radius = min(center_h, center_w)
            threshold_radius = max_radius * (_HIGH_FREQ_PERCENTILE / 100.0)

            low_freq_mask  = dist_from_center <= threshold_radius
            high_freq_mask = ~low_freq_mask

            energy_total     = float(np.sum(magnitude ** 2))
            energy_high_freq = float(np.sum((magnitude * high_freq_mask) ** 2))

            if energy_total > 0:
                features["high_freq_energy_ratio"] = float(
                    energy_high_freq / energy_total
                )
            else:
                features["high_freq_energy_ratio"] = 0.0

            # ---- Spectral Flatness (Wiener Entropy) ----
            # Ukuran seberapa datar distribusi energi spectrum
            # Tinggi = lebih noise-like, Rendah = lebih tonal
            magnitude_nonzero = magnitude[magnitude > 0]
            if len(magnitude_nonzero) > 0:
                geom_mean = np.exp(np.mean(np.log(magnitude_nonzero)))
                arith_mean = np.mean(magnitude_nonzero)
                if arith_mean > 0:
                    features["spectral_flatness"] = float(geom_mean / arith_mean)
                else:
                    features["spectral_flatness"] = 0.0
            else:
                features["spectral_flatness"] = 0.0

        except Exception as exc:
            logger.error("Error saat menghitung fitur FFT: %s", exc)

        return features

    # ------------------------------------------------------------------
    # Deteksi artefak periodik
    # ------------------------------------------------------------------

    def _detect_periodic_artifacts(
        self, gray: np.ndarray
    ) -> Dict[str, float]:
        """
        Deteksi artefak periodik pada magnitude spectrum.

        GAN deepfake sering menghasilkan pola grid periodik yang tampak
        sebagai peak signifikan di FFT spectrum (di luar komponen DC).

        Parameters
        ----------
        gray : np.ndarray
            Gambar grayscale float32.

        Returns
        -------
        dict
            - n_periodic_peaks   : int, jumlah peak yang terdeteksi
            - max_peak_zscore    : float, z-score peak tertinggi
            - periodic_score     : float [0–1], skor artefak periodik
        """
        result: Dict[str, float] = {
            "n_periodic_peaks": 0.0,
            "max_peak_zscore": 0.0,
            "periodic_score": 0.0,
        }

        try:
            h, w = gray.shape
            if h < 16 or w < 16:
                return result

            fft2d = np.fft.fft2(gray)
            fft_shifted = np.fft.fftshift(fft2d)
            magnitude = np.abs(fft_shifted)

            # Hapus komponen DC (titik tengah)
            center_h, center_w = h // 2, w // 2
            dc_radius = max(3, min(h, w) // 20)
            y_coords, x_coords = np.ogrid[:h, :w]
            dc_mask = (y_coords - center_h) ** 2 + (x_coords - center_w) ** 2 <= dc_radius ** 2
            magnitude_no_dc = magnitude.copy()
            magnitude_no_dc[dc_mask] = 0.0

            mag_flat = magnitude_no_dc.flatten()
            mag_nonzero = mag_flat[mag_flat > 0]

            if len(mag_nonzero) < 10:
                return result

            mag_mean = np.mean(mag_nonzero)
            mag_std = np.std(mag_nonzero) + 1e-8

            # Hitung z-score setiap piksel
            z_scores = (magnitude_no_dc - mag_mean) / mag_std

            # Peak yang signifikan
            peak_mask = z_scores > _PERIODIC_NOISE_ZSCORE
            n_peaks = int(np.sum(peak_mask))

            if n_peaks > 0:
                max_zscore = float(np.max(z_scores[peak_mask]))
            else:
                max_zscore = 0.0

            # Normalisasi skor artefak periodik ke [0, 1]
            # Semakin banyak peak dan semakin tinggi z-score → skor lebih tinggi
            # Referensi: >50 peak sangat mencurigakan
            normalized_peaks = min(1.0, n_peaks / 20.0)
            normalized_zscore = min(1.0, max_zscore / 10.0)
            periodic_score = (normalized_peaks + normalized_zscore) / 2.0

            result["n_periodic_peaks"] = float(n_peaks)
            result["max_peak_zscore"] = round(max_zscore, 4)
            result["periodic_score"] = round(periodic_score, 4)

        except Exception as exc:
            logger.error("Error saat mendeteksi artefak periodik: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Analisis DCT pada blok 8×8
    # ------------------------------------------------------------------

    def _compute_dct_features(
        self, gray: np.ndarray
    ) -> Dict[str, float]:
        """
        Hitung fitur statistik dari DCT pada blok 8×8.

        DCT-8×8 adalah dasar kompresi JPEG. Anomali pada distribusi
        koefisien DCT dapat mengindikasikan gambar yang telah dimanipulasi.

        Parameters
        ----------
        gray : np.ndarray
            Gambar grayscale float32.

        Returns
        -------
        dict
            - dct_mean : float, rata-rata koefisien DCT (AC only)
            - dct_std  : float, standar deviasi koefisien DCT
            - dct_score : float [0–1], skor anomali DCT
        """
        features: Dict[str, float] = {
            "dct_mean": 0.0,
            "dct_std": 0.0,
            "dct_score": 0.0,
        }

        try:
            h, w = gray.shape

            # Pastikan dimensi habis dibagi block_size
            h_crop = (h // _DCT_BLOCK_SIZE) * _DCT_BLOCK_SIZE
            w_crop = (w // _DCT_BLOCK_SIZE) * _DCT_BLOCK_SIZE
            gray_crop = gray[:h_crop, :w_crop]

            n_blocks_h = h_crop // _DCT_BLOCK_SIZE
            n_blocks_w = w_crop // _DCT_BLOCK_SIZE

            if n_blocks_h == 0 or n_blocks_w == 0:
                return features

            # Kumpulkan koefisien DCT dari semua blok
            ac_coeffs: List[float] = []

            for i in range(n_blocks_h):
                for j in range(n_blocks_w):
                    block = gray_crop[
                        i * _DCT_BLOCK_SIZE:(i + 1) * _DCT_BLOCK_SIZE,
                        j * _DCT_BLOCK_SIZE:(j + 1) * _DCT_BLOCK_SIZE,
                    ]

                    if SCIPY_AVAILABLE:
                        dct_block = scipy_fft.dctn(block, norm="ortho")
                    else:
                        # Fallback: DCT manual menggunakan cosine transform
                        dct_block = cv2.dct(block.astype(np.float64))

                    # Ambil koefisien AC (semua kecuali DC di (0,0))
                    ac_flat = dct_block.flatten()[1:]
                    ac_coeffs.extend(ac_flat.tolist())

            if not ac_coeffs:
                return features

            ac_arr = np.array(ac_coeffs, dtype=np.float32)
            dct_mean = float(np.mean(np.abs(ac_arr)))
            dct_std  = float(np.std(ac_arr))

            features["dct_mean"] = round(dct_mean, 4)
            features["dct_std"]  = round(dct_std, 4)

            # Normalisasi skor DCT
            # Nilai referensi empiris: mean > 15 mulai mencurigakan
            dct_score = min(1.0, dct_mean / 15.0)
            features["dct_score"] = round(dct_score, 4)

        except Exception as exc:
            logger.error("Error saat menghitung fitur DCT: %s", exc)

        return features

    # ------------------------------------------------------------------
    # Kombinasi skor
    # ------------------------------------------------------------------

    def _combine_scores(
        self,
        fft_features: Dict[str, float],
        periodic_features: Dict[str, float],
        dct_features: Dict[str, float],
    ) -> float:
        """
        Kombinasikan berbagai metrik frekuensi menjadi satu skor kecurigaan.

        Parameters
        ----------
        fft_features : dict
            Fitur dari _compute_fft_features.
        periodic_features : dict
            Fitur dari _detect_periodic_artifacts.
        dct_features : dict
            Fitur dari _compute_dct_features.

        Returns
        -------
        float
            Skor gabungan dalam rentang [0.0, 1.0].
        """
        components: Dict[str, float] = {}

        # ---- Komponen 1: Rasio energi frekuensi tinggi ----
        # Deepfake sering punya energi frekuensi tinggi yang tidak natural
        # Normalisasi: > 0.35 sudah cukup mencurigakan
        high_freq = fft_features.get("high_freq_energy_ratio", 0.0)
        # Skala linear 0–0.5 → 0–1
        components["high_freq_energy"] = min(1.0, high_freq / 0.25)

        # ---- Komponen 2: Spectral Flatness ----
        # Gambar asli biasanya memiliki spectral flatness sedang
        # Deepfake bisa memiliki distribusi yang terlalu rata atau terlalu tajam
        flatness = fft_features.get("spectral_flatness", 0.5)
        # Smooth scoring curve across the full flatness range
        if flatness < 0.05:
            flatness_score = 0.7  # Too flat = suspicious
        elif flatness < 0.15:
            flatness_score = 0.5 - (flatness - 0.05) * 3.0  # Transition to normal
        elif flatness < 0.70:
            flatness_score = 0.1  # Normal range
        elif flatness < 0.80:
            flatness_score = 0.1 + (flatness - 0.70) * 5.0  # Transition
        else:
            flatness_score = 0.6  # Too not-flat = suspicious
        components["spectral_flatness"] = min(1.0, flatness_score)

        # ---- Komponen 3: Artefak periodik ----
        components["periodic_artifacts"] = periodic_features.get("periodic_score", 0.0)

        # ---- Komponen 4: DCT statistics ----
        components["dct_anomaly"] = dct_features.get("dct_score", 0.0)

        # ---- Komponen 5: Kurtosis excess ----
        # Kurtosis tinggi = distribusi ekor berat = lebih banyak extreme value
        # Deepfake cenderung memiliki kurtosis spectrum yang berbeda dari gambar asli
        kurtosis = fft_features.get("kurtosis", 0.0)
        # Referensi: kurtosis > 10 atau < -2 mulai mencurigakan
        if abs(kurtosis) > 1.0:
            kurtosis_score = min(1.0, (abs(kurtosis) - 1.0) / 8.0)
        else:
            kurtosis_score = 0.0
        components["kurtosis_anomaly"] = kurtosis_score

        # ---- Gabungkan dengan bobot ----
        weights = {
            "high_freq_energy":  _WEIGHT_HIGH_FREQ_ENERGY,
            "spectral_flatness": _WEIGHT_SPECTRAL_FLATNESS,
            "periodic_artifacts": _WEIGHT_PERIODIC_ARTIFACTS,
            "dct_anomaly":       _WEIGHT_DCT_STATISTICS,
            "kurtosis_anomaly":  _WEIGHT_KURTOSIS,
        }

        total_score = sum(
            components.get(k, 0.0) * w
            for k, w in weights.items()
        )

        return float(np.clip(total_score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Analisis satu gambar
    # ------------------------------------------------------------------

    def _analyze_single_image(
        self, image_rgb: np.ndarray
    ) -> Dict[str, Any]:
        """
        Jalankan seluruh pipeline analisis frekuensi pada satu gambar.

        Parameters
        ----------
        image_rgb : np.ndarray
            Gambar RGB uint8.

        Returns
        -------
        dict
            Hasil analisis intermediate (sebelum diformat sebagai output final).
        """
        # Konversi ke grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Jalankan semua analisis
        fft_features      = self._compute_fft_features(gray)
        periodic_features = self._detect_periodic_artifacts(gray)
        dct_features      = self._compute_dct_features(gray)

        # Gabungkan skor
        combined_score = self._combine_scores(fft_features, periodic_features, dct_features)

        return {
            "combined_score": combined_score,
            "fft_features": fft_features,
            "periodic_features": periodic_features,
            "dct_features": dct_features,
        }

    # ------------------------------------------------------------------
    # API Publik
    # ------------------------------------------------------------------

    def analyze(
        self, image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Analisis gambar atau video menggunakan analisis frekuensi untuk mendeteksi deepfake.

        Parameters
        ----------
        image_path : str | Path
            Path ke file gambar atau video.

        Returns
        -------
        dict
            Hasil analisis berisi:
            - ``score``      : float [0–1], mendekati 1.0 = FAKE
            - ``verdict``    : str, 'FAKE' atau 'REAL'
            - ``confidence`` : float [0–1], tingkat keyakinan
            - ``details``    : dict, detail fitur frekuensi

        Raises
        ------
        FileNotFoundError
            Jika file tidak ditemukan.
        """
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        logger.info("FrequencyAnalyzer memproses: %s", path.name)

        try:
            # Tentukan apakah video atau gambar
            is_video = self._video_processor.is_video(path)

            if is_video:
                # Ambil beberapa frame dan rata-rata hasilnya
                frames = self._video_processor.extract_frames(path, n_frames=8)
                if not frames:
                    return self._build_result(0.5, False, str(path), error="Tidak ada frame diekstrak")

                all_scores: List[float] = []
                all_fft_features: List[Dict] = []
                all_periodic_features: List[Dict] = []

                for frame_pil in frames:
                    frame_np = np.array(frame_pil)
                    face = self._face_extractor.extract(frame_np)
                    analysis_img = face if face is not None else frame_np
                    result_intermediate = self._analyze_single_image(analysis_img)
                    all_scores.append(result_intermediate["combined_score"])
                    all_fft_features.append(result_intermediate["fft_features"])
                    all_periodic_features.append(result_intermediate["periodic_features"])

                avg_score = float(np.mean(all_scores))
                # Gunakan feature dari frame pertama sebagai representatif
                details_fft = all_fft_features[0] if all_fft_features else {}
                details_periodic = all_periodic_features[0] if all_periodic_features else {}
                face_found = False  # Tidak dilacak per-video secara mudah
                return self._build_result(
                    avg_score, face_found, str(path),
                    fft_features=details_fft,
                    periodic_features=details_periodic,
                    n_frames=len(all_scores),
                    frame_scores=all_scores,
                )
            else:
                # Gambar statis
                analysis_img, face_found = self._get_analysis_image(path)
                if analysis_img is None:
                    return self._build_result(0.5, False, str(path), error="Gagal memuat gambar")

                result_intermediate = self._analyze_single_image(analysis_img)
                return self._build_result(
                    result_intermediate["combined_score"],
                    face_found,
                    str(path),
                    fft_features=result_intermediate["fft_features"],
                    periodic_features=result_intermediate["periodic_features"],
                    dct_features=result_intermediate["dct_features"],
                )

        except FileNotFoundError:
            raise
        except Exception as exc:
            logger.error("Error saat analisis frekuensi: %s", exc)
            return self._build_result(0.5, False, str(path), error=str(exc))

    def _build_result(
        self,
        score: float,
        face_found: bool,
        source_path: str,
        fft_features: Optional[Dict] = None,
        periodic_features: Optional[Dict] = None,
        dct_features: Optional[Dict] = None,
        n_frames: int = 1,
        frame_scores: Optional[List[float]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Bangun dictionary hasil analisis terstandarisasi.

        Parameters
        ----------
        score : float
            Skor kecurigaan 0–1.
        face_found : bool
            Apakah wajah terdeteksi.
        source_path : str
            Path file sumber.
        fft_features : dict, optional
            Fitur FFT yang dihitung.
        periodic_features : dict, optional
            Fitur artefak periodik.
        dct_features : dict, optional
            Fitur DCT.
        n_frames : int
            Jumlah frame yang dianalisis.
        frame_scores : list, optional
            Skor per frame (untuk video).
        error : str, optional
            Pesan error jika ada.

        Returns
        -------
        dict
            Hasil terstandarisasi.
        """
        verdict = "FAKE" if score >= 0.45 else "REAL"
        confidence = abs(score - 0.5) * 2.0  # Normalisasi ke [0, 1]

        details: Dict[str, Any] = {
            "module": self.MODULE_NAME,
            "source_path": source_path,
            "face_detected": face_found,
            "n_frames_analyzed": n_frames,
        }

        if fft_features:
            details["fft_features"] = fft_features
        if periodic_features:
            details["periodic_artifacts"] = periodic_features
        if dct_features:
            details["dct_features"] = dct_features
        if frame_scores is not None:
            details["frame_scores"] = [round(s, 4) for s in frame_scores]
        if error is not None:
            details["error"] = error

        return {
            "score": round(float(score), 6),
            "verdict": verdict,
            "confidence": round(float(confidence), 6),
            "details": details,
        }

    def __repr__(self) -> str:
        return "FrequencyAnalyzer()"
