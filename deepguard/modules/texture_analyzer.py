"""
modules/texture_analyzer.py
----------------------------
Modul analisis tekstur untuk DeepGuard.
Mengimplementasikan teknik Steganalysis Rich Model (SRM) yang disederhanakan,
analisis Local Binary Pattern (LBP), dan estimasi PRNU.

Teknik yang digunakan:
  1. SRM-like: residual noise via multiple high-pass filter
  2. Local Binary Pattern (LBP) analysis
  3. PRNU (Photo Response Non-Uniformity) estimation
"""

import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Scikit-image opsional (untuk LBP)
try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Scipy opsional
try:
    from scipy import ndimage
    from scipy.stats import skew, kurtosis as sp_kurtosis
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

# Klip truncated residual
T_CLIP = 4.0

# Jumlah titik LBP dan radius
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

# Jumlah grid untuk analisis PRNU
PRNU_GRID = 6


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_gray(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Memuat gambar dan mengonversinya menjadi array float grayscale dan RGB.

    Returns:
        (gray_float32, rgb_float32) dalam rentang [0, 255]
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)
    return gray, rgb


def _convolve2d_simple(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Konvolusi 2D sederhana menggunakan scipy jika tersedia,
    atau implementasi manual menggunakan numpy sebagai fallback.
    """
    if SCIPY_AVAILABLE:
        return ndimage.convolve(image, kernel, mode="reflect")

    # Fallback: konvolusi manual untuk kernel 3x3
    kh, kw = kernel.shape
    if kh == 1 and kw == 3:
        # Filter horizontal 1D
        out = np.zeros_like(image)
        for j, k in enumerate(kernel[0]):
            out += k * np.roll(image, j - kw // 2, axis=1)
        return out
    elif kh == 3 and kw == 1:
        out = np.zeros_like(image)
        for i, k in enumerate(kernel[:, 0]):
            out += k * np.roll(image, i - kh // 2, axis=0)
        return out
    else:
        # Konvolusi full dengan nested loop (lambat tapi benar untuk fallback)
        ph, pw = kh // 2, kw // 2
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
        out = np.zeros_like(image)
        for i in range(kh):
            for j in range(kw):
                out += kernel[i, j] * padded[i: i + image.shape[0], j: j + image.shape[1]]
        return out


def _compute_statistical_moments(arr: np.ndarray) -> dict[str, float]:
    """Menghitung momen statistik dari array: mean, var, skewness, kurtosis."""
    flat = arr.flatten()
    mean = float(np.mean(flat))
    var = float(np.var(flat))
    std = float(np.std(flat))

    if std > 1e-8:
        skewness = float(np.mean(((flat - mean) / std) ** 3))
        kurt = float(np.mean(((flat - mean) / std) ** 4)) - 3.0  # excess kurtosis
    else:
        skewness = 0.0
        kurt = 0.0

    return {
        "mean": round(mean, 6),
        "variance": round(var, 6),
        "std": round(std, 6),
        "skewness": round(skewness, 6),
        "excess_kurtosis": round(kurt, 6),
    }


def _verdict_from_score(score: float) -> str:
    """Mengonversi skor numerik menjadi label verdict."""
    if score >= 0.55:
        return "FAKE"
    elif score >= 0.40:
        return "SUSPICIOUS"
    else:
        return "REAL"


# ---------------------------------------------------------------------------
# Sub-analisis 1: SRM-like residual analysis
# ---------------------------------------------------------------------------

def _srm_analysis(gray: np.ndarray) -> dict[str, Any]:
    """
    Analisis residual noise terinspirasi SRM (Steganalysis Rich Model).

    Menerapkan tiga jenis high-pass filter untuk mengekstrak residual:
    - Horizontal: [-1, 2, -1] / 2
    - Vertikal: [-1, 2, -1]ᵀ / 2
    - Diagonal: [[-1,2,-1],[2,-4,2],[-1,2,-1]] / 4

    Residual yang tidak biasa mengindikasikan manipulasi.

    Returns:
        dict berisi 'score' dan 'details'
    """
    try:
        # Definisikan filter
        kernel_h = np.array([[[-1, 2, -1]]], dtype=np.float32).reshape(1, 3) / 2.0
        kernel_v = np.array([[-1], [2], [-1]], dtype=np.float32) / 2.0
        kernel_d = np.array(
            [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32
        ) / 4.0

        # Terapkan filter
        res_h = _convolve2d_simple(gray, kernel_h)
        res_v = _convolve2d_simple(gray, kernel_v)
        res_d = _convolve2d_simple(gray, kernel_d)

        # Truncated residuals
        res_h_trunc = np.clip(res_h, -T_CLIP * 10, T_CLIP * 10)
        res_v_trunc = np.clip(res_v, -T_CLIP * 10, T_CLIP * 10)
        res_d_trunc = np.clip(res_d, -T_CLIP * 10, T_CLIP * 10)

        # Statistik per residual
        stats_h = _compute_statistical_moments(res_h_trunc)
        stats_v = _compute_statistical_moments(res_v_trunc)
        stats_d = _compute_statistical_moments(res_d_trunc)

        # Co-occurrence matrix dari residual diagonal (simplified)
        # Kuantisasi residual ke range kecil untuk co-occurrence
        n_bins = 16
        res_d_norm = np.clip(res_d_trunc, -8, 8)
        res_d_quant = ((res_d_norm + 8) / 16 * (n_bins - 1)).astype(int)
        
        # Co-occurrence: P(i,j) = jumlah pasangan piksel bersebelahan
        co_matrix = np.zeros((n_bins, n_bins), dtype=np.float32)
        shifted = np.roll(res_d_quant, 1, axis=1)
        for val in range(n_bins * n_bins):
            i, j = val // n_bins, val % n_bins
            co_matrix[i, j] = float(np.sum((res_d_quant == i) & (shifted == j)))
        co_matrix /= co_matrix.sum() + 1e-10

        # Entropi co-occurrence (rendah = lebih terstruktur = mencurigakan untuk GAN)
        co_entropy = float(-np.sum(co_matrix * np.log2(co_matrix + 1e-10)))

        # Skor: kurtosis ekstrim dan entropi rendah = lebih mencurigakan
        # GAN: kurtosis biasanya lebih rendah (distribusi lebih merata)
        # Foto asli: kurtosis biasanya > 2 (heavy tail noise)
        kurt_avg = (
            abs(stats_h["excess_kurtosis"])
            + abs(stats_v["excess_kurtosis"])
            + abs(stats_d["excess_kurtosis"])
        ) / 3.0

        # Entropy rendah = lebih mencurigakan (noise lebih terstruktur)
        max_entropy = np.log2(n_bins * n_bins)
        entropy_score = 1.0 - (co_entropy / max_entropy)

        # Kurt rendah dan entropy rendah = GAN
        kurt_score = max(0.0, 1.0 - kurt_avg / 4.0)

        final_score = float(np.clip(0.5 * kurt_score + 0.5 * entropy_score, 0.0, 1.0))

        return {
            "score": final_score,
            "details": {
                "stats_horizontal": stats_h,
                "stats_vertical": stats_v,
                "stats_diagonal": stats_d,
                "co_occurrence_entropy": round(co_entropy, 4),
                "avg_abs_kurtosis": round(kurt_avg, 4),
            },
        }

    except Exception as exc:
        logger.warning("Gagal SRM analysis: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Sub-analisis 2: Local Binary Pattern (LBP)
# ---------------------------------------------------------------------------

def _lbp_analysis(gray: np.ndarray) -> dict[str, Any]:
    """
    Analisis Local Binary Pattern (LBP) untuk mendeteksi anomali tekstur.

    LBP mengodekan struktur lokal gambar. Gambar asli memiliki distribusi
    histogram LBP yang berbeda dari gambar yang dihasilkan AI.

    Returns:
        dict berisi 'score' dan 'details'
    """
    try:
        # Normalisasi ke uint8 untuk LBP
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

        if SKIMAGE_AVAILABLE:
            # LBP resmi dari scikit-image
            lbp = local_binary_pattern(
                gray_uint8,
                P=LBP_N_POINTS,
                R=LBP_RADIUS,
                method=LBP_METHOD,
            )
            n_bins = LBP_N_POINTS + 2  # untuk metode 'uniform'
        else:
            # Implementasi LBP sederhana sebagai fallback (radius=1, 8 titik)
            lbp = _simple_lbp(gray_uint8)
            n_bins = 256

        # Histogram LBP yang dinormalisasi
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
        hist = hist.astype(np.float64)
        hist_norm = hist / (hist.sum() + 1e-10)

        # Ukuran uniformitas: fraksi pola uniform dalam metode 'uniform'
        if SKIMAGE_AVAILABLE:
            # Bin terakhir adalah non-uniform
            uniformity = float(1.0 - hist_norm[-1])
        else:
            uniformity = 0.5  # tidak diketahui tanpa skimage

        # Entropi histogram LBP
        lbp_entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-10)))
        max_lbp_entropy = np.log2(n_bins)
        normalized_entropy = lbp_entropy / max_lbp_entropy

        # Puncak histogram: gambar sintetis sering punya distribusi LBP yang sangat
        # terkonsentrasi pada beberapa bin
        sorted_hist = np.sort(hist_norm)[::-1]
        top_3_fraction = float(np.sum(sorted_hist[:3]))

        # Skor: uniformitas sangat tinggi atau sangat rendah = mencurigakan
        if uniformity > 0.90:
            uniformity_score = 0.7  # terlalu uniform = AI-generated
        elif uniformity < 0.40:
            uniformity_score = 0.6  # terlalu non-uniform = manipulasi
        else:
            uniformity_score = 0.35  # rentang normal

        # Entropy sangat rendah atau peak terkonsentrasi = mencurigakan
        entropy_score = max(0.0, 1.0 - normalized_entropy) * 0.6
        concentration_score = min(1.0, top_3_fraction * 5) * 0.4

        final_score = float(np.clip(
            0.5 * uniformity_score + 0.3 * entropy_score + 0.2 * concentration_score,
            0.0, 1.0
        ))

        return {
            "score": final_score,
            "details": {
                "uniformity": round(uniformity, 4),
                "lbp_entropy": round(lbp_entropy, 4),
                "normalized_entropy": round(normalized_entropy, 4),
                "top_3_bins_fraction": round(top_3_fraction, 4),
                "n_bins": n_bins,
                "skimage_available": SKIMAGE_AVAILABLE,
            },
        }

    except Exception as exc:
        logger.warning("Gagal LBP analysis: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


def _simple_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Implementasi LBP sederhana (radius=1, 8 tetangga) sebagai fallback.
    Mengembalikan array LBP dengan nilai 0-255.
    """
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp |= (shifted >= gray).astype(np.uint8) << bit
    return lbp


# ---------------------------------------------------------------------------
# Sub-analisis 3: PRNU estimation
# ---------------------------------------------------------------------------

def _prnu_analysis(gray: np.ndarray, rgb: np.ndarray) -> dict[str, Any]:
    """
    Estimasi sederhana Photo Response Non-Uniformity (PRNU).

    PRNU adalah noise pattern unik dari sensor kamera. Foto asli menghasilkan
    PRNU yang konsisten di seluruh gambar. Deepfake/AI-generated image biasanya
    tidak memiliki PRNU yang koheren.

    Pendekatan:
    1. Estimasi noise dengan Wiener-like denoising sederhana
    2. Analisis konsistensi noise antar region
    3. Hitung auto-korelasi noise pattern

    Returns:
        dict berisi 'score' dan 'details'
    """
    try:
        h, w = gray.shape
        cell_h = max(1, h // PRNU_GRID)
        cell_w = max(1, w // PRNU_GRID)

        # Ekstrak noise residual: gambar asli - versi smoothed
        # Smoothing sederhana via averaging filter 5x5
        kernel_size = 5
        avg_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        smoothed = _convolve2d_simple(gray, avg_kernel)
        noise_residual = gray - smoothed

        # Analisis per region
        region_noise_patterns = []
        for i in range(PRNU_GRID):
            for j in range(PRNU_GRID):
                y0, y1 = i * cell_h, min((i + 1) * cell_h, h)
                x0, x1 = j * cell_w, min((j + 1) * cell_w, w)
                region_noise = noise_residual[y0:y1, x0:x1]
                if region_noise.size > 0:
                    region_noise_patterns.append(region_noise.flatten())

        if len(region_noise_patterns) < 2:
            return {"score": 0.5, "details": {"error": "Tidak cukup region"}}

        # Hitung korelasi antar noise region (min size untuk korelasi)
        min_size = min(len(p) for p in region_noise_patterns)
        # Potong semua ke ukuran minimum
        patterns_trimmed = [p[:min_size] for p in region_noise_patterns]

        # Matriks korelasi antar region
        n_regions = len(patterns_trimmed)
        correlations = []
        for i in range(n_regions):
            for j in range(i + 1, min(i + 4, n_regions)):  # batasi kombinasi
                try:
                    r = float(np.corrcoef(patterns_trimmed[i], patterns_trimmed[j])[0, 1])
                    if not np.isnan(r):
                        correlations.append(abs(r))
                except Exception:
                    pass

        if not correlations:
            return {"score": 0.5, "details": {"error": "Gagal hitung korelasi"}}

        mean_correlation = float(np.mean(correlations))
        std_correlation = float(np.std(correlations))

        # Foto asli: ada korelasi rendah-moderat tapi konsisten (PRNU coherent)
        # AI-generated: korelasi biasanya sangat rendah dan bervariasi
        # Face-swap: korelasi tidak konsisten (beberapa region punya PRNU berbeda)

        # Auto-korelasi noise residual keseluruhan
        flat_noise = noise_residual.flatten()
        noise_norm = flat_noise - flat_noise.mean()
        noise_norm_std = noise_norm.std()
        if noise_norm_std > 1e-8:
            autocorr_lag1 = float(np.corrcoef(noise_norm[:-1], noise_norm[1:])[0, 1])
        else:
            autocorr_lag1 = 0.0

        # Skor PRNU:
        # - Korelasi antar region rendah & tidak konsisten = AI-generated
        # - Auto-korelasi tinggi = noise terstruktur (GAN-like)
        consistency_score = max(0.0, 1.0 - mean_correlation * 3.0)
        autocorr_score = abs(autocorr_lag1)

        final_score = float(np.clip(0.6 * consistency_score + 0.4 * autocorr_score, 0.0, 1.0))

        return {
            "score": final_score,
            "details": {
                "mean_inter_region_correlation": round(mean_correlation, 4),
                "std_inter_region_correlation": round(std_correlation, 4),
                "autocorrelation_lag1": round(autocorr_lag1, 4),
                "num_region_pairs_compared": len(correlations),
                "interpretation": (
                    "PRNU tidak koheren — kemungkinan gambar sintetis atau face-swap"
                    if final_score > 0.6
                    else "PRNU konsisten — kemungkinan foto asli"
                ),
            },
        }

    except Exception as exc:
        logger.warning("Gagal PRNU analysis: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Kelas utama
# ---------------------------------------------------------------------------

class TextureAnalyzer:
    """
    Menganalisis tekstur gambar menggunakan pendekatan SRM-like, LBP, dan PRNU.

    Setiap sub-analisis menghasilkan skor [0,1]:
      0.0 = sangat meyakinkan ASLI
      1.0 = sangat meyakinkan PALSU/DIMANIPULASI

    Sub-skor digabung dengan bobot:
      - SRM   : 40%
      - LBP   : 35%
      - PRNU  : 25%
    """

    SUB_WEIGHTS = {
        "srm": 0.40,
        "lbp": 0.35,
        "prnu": 0.25,
    }

    def analyze(self, image_path: str) -> dict[str, Any]:
        """
        Melakukan analisis tekstur pada gambar.

        Args:
            image_path: Path ke file gambar.

        Returns:
            dict dengan kunci:
              - score      : float [0, 1]
              - verdict    : str  – 'REAL' | 'SUSPICIOUS' | 'FAKE'
              - confidence : float [0, 1]
              - details    : dict – hasil tiap sub-analisis
        """
        result_template: dict[str, Any] = {
            "score": 0.5,
            "verdict": "SUSPICIOUS",
            "confidence": 0.0,
            "details": {},
        }

        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"File tidak ditemukan: {image_path}")

            # Muat gambar
            gray, rgb = _load_gray(str(path))

            # Pastikan gambar cukup besar untuk analisis
            if gray.shape[0] < 32 or gray.shape[1] < 32:
                result_template["details"] = {
                    "error": "Gambar terlalu kecil untuk analisis tekstur"
                }
                result_template["confidence"] = 0.1
                return result_template

            # Jalankan sub-analisis
            srm_result = _srm_analysis(gray)
            lbp_result = _lbp_analysis(gray)
            prnu_result = _prnu_analysis(gray, rgb)

            sub_results = {
                "srm": srm_result,
                "lbp": lbp_result,
                "prnu": prnu_result,
            }

            # Hitung weighted average
            weighted_score = 0.0
            total_weight = 0.0
            sub_scores: dict[str, float] = {}

            for key, res in sub_results.items():
                weight = self.SUB_WEIGHTS.get(key, 0.33)
                sub_score = float(res.get("score", 0.5))
                sub_scores[key] = sub_score
                weighted_score += weight * sub_score
                total_weight += weight

            final_score = weighted_score / max(total_weight, 1e-10)

            # Confidence berdasarkan kesepakatan sub-skor
            scores_list = list(sub_scores.values())
            score_std = float(np.std(scores_list))
            confidence = float(np.clip(1.0 - score_std * 1.8, 0.10, 0.90))

            verdict = _verdict_from_score(final_score)

            return {
                "score": round(float(final_score), 4),
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "details": {
                    "sub_scores": sub_scores,
                    "sub_weights": self.SUB_WEIGHTS,
                    "image_shape": list(gray.shape),
                    "skimage_available": SKIMAGE_AVAILABLE,
                    "scipy_available": SCIPY_AVAILABLE,
                    "sub_analyses": {k: v.get("details", {}) for k, v in sub_results.items()},
                },
            }

        except FileNotFoundError as fnf:
            logger.error("File tidak ditemukan: %s", fnf)
            result_template["details"] = {"error": str(fnf)}
            return result_template

        except Exception as exc:
            logger.error(
                "Kesalahan tidak terduga pada TextureAnalyzer: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            result_template["details"] = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            return result_template
