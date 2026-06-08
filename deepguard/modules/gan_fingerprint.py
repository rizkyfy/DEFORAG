"""
modules/gan_fingerprint.py
--------------------------
Modul analisis sidik jari GAN (Generative Adversarial Network) untuk DeepGuard.
Mendeteksi artefak-artefak khas yang ditinggalkan oleh model GAN pada gambar.

Teknik yang digunakan:
  1. Deteksi artefak checkerboard via FFT
  2. Analisis inkonsistensi warna
  3. Analisis noise fingerprint (high-pass filter)
  4. Deteksi inkonsistensi ketajaman (sharpness) antar region
"""

import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Scipy opsional – digunakan untuk statistik tambahan
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konstanta global
# ---------------------------------------------------------------------------
# Ukuran grid untuk analisis regional
GRID_SIZE = 8

# Ambang batas frekuensi checkerboard (sebagai fraksi dari dimensi)
CHECKERBOARD_FREQ_THRESHOLD = 0.45

# Klip untuk residual noise
NOISE_CLIP_VALUE = 3.0

# Bobot sub-skor
SUB_WEIGHTS = {
    "checkerboard":            0.40,  # naik — sinyal terkuat GAN/Diffusion
    "noise_fingerprint":       0.35,  # naik — kurtosis & periodic peak kuat
    "sharpness_inconsistency": 0.15,
    "color_inconsistency":     0.10,  # turun — kurang diskriminatif
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_image_as_array(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Memuat gambar dari path dan mengembalikan array RGB dan grayscale.

    Returns:
        (rgb_array, gray_array) dalam float32 [0, 255]
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float32)
    gray = np.array(img.convert("L"), dtype=np.float32)
    return rgb, gray


def _normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Menormalkan nilai ke rentang [0, 1]."""
    if max_val == min_val:
        return 0.0
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def _verdict_from_score(score: float) -> str:
    """Mengonversi skor numerik menjadi label verdict."""
    if score >= 0.55:
        return "FAKE"
    elif score >= 0.40:
        return "SUSPICIOUS"
    else:
        return "REAL"


# ---------------------------------------------------------------------------
# Sub-analisis 1: Checkerboard artifact via FFT
# ---------------------------------------------------------------------------

def _analyze_checkerboard(gray: np.ndarray) -> dict[str, Any]:
    """
    Mendeteksi artefak checkerboard yang khas dari operasi transposed convolution GAN.

    GAN menggunakan transposed convolution yang menghasilkan pola periodik pada
    frekuensi N/2 di spektrum Fourier. Fungsi ini:
      1. Menghitung 2D FFT dari gambar grayscale
      2. Menganalisis magnitudo pada frekuensi kritis (mendekati Nyquist)
      3. Membandingkan energi frekuensi tinggi vs keseluruhan spektrum

    Returns:
        dict berisi 'score' [0,1] dan 'details'
    """
    try:
        h, w = gray.shape

        # FFT 2D dengan windowing untuk mengurangi spectral leakage
        window = np.outer(np.hanning(h), np.hanning(w))
        fft2d = np.fft.fft2(gray * window)
        magnitude = np.abs(np.fft.fftshift(fft2d))

        # Log-magnitude untuk visualisasi yang lebih stabil
        log_mag = np.log1p(magnitude)

        # Koordinat pusat spektrum
        cy, cx = h // 2, w // 2

        # --- Deteksi puncak pada frekuensi kritis ---
        # Region frekuensi kritis: sekitar 40-50% dari frekuensi Nyquist
        freq_lo = int(min(h, w) * CHECKERBOARD_FREQ_THRESHOLD * 0.8)
        freq_hi = int(min(h, w) * CHECKERBOARD_FREQ_THRESHOLD * 1.2)

        # Buat mask annular (cincin) untuk frekuensi kritis
        y_idx, x_idx = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

        mask_critical = (dist_from_center >= freq_lo) & (dist_from_center <= freq_hi)
        mask_low_freq = dist_from_center < freq_lo * 0.5

        energy_critical = float(np.mean(log_mag[mask_critical]))
        energy_low = float(np.mean(log_mag[mask_low_freq])) if mask_low_freq.any() else 1.0

        # Rasio energi frekuensi kritis vs frekuensi rendah
        ratio = energy_critical / (energy_low + 1e-6)

        # Deteksi puncak simetris (khas GAN checkerboard)
        # Kuadran kanan atas dan kiri bawah harus simetris
        half_h = min(cy, h - cy)
        half_w = min(cx, w - cx)
        quad1 = log_mag[cy - half_h : cy, cx : cx + half_w]  # atas kanan
        quad3 = log_mag[cy : cy + half_h, cx - half_w : cx]  # bawah kiri
        quad3_flipped = quad3[::-1, ::-1]

        if quad1.shape == quad3_flipped.shape:
            symmetry_score = float(np.corrcoef(quad1.flatten(), quad3_flipped.flatten())[0, 1])
            symmetry_score = max(0.0, symmetry_score)
        else:
            symmetry_score = 0.5

        # Skor akhir checkerboard: kombinasi rasio energi dan simetri
        # Rasio log normal: ~0.70 untuk gambar alami. Naikkan min_val agar tidak false positive.
        ratio_score = _normalize_score(ratio, min_val=0.68, max_val=0.85)
        # Gunakan perkalian agar symmetry_score bertindak sebagai gating/validasi pola
        final_score = float(np.clip(ratio_score * symmetry_score, 0.0, 1.0))

        return {
            "score": final_score,
            "details": {
                "energy_ratio_critical_vs_low": round(ratio, 4),
                "spectral_symmetry": round(symmetry_score, 4),
                "energy_critical_band": round(energy_critical, 4),
                "energy_low_band": round(energy_low, 4),
                "interpretation": (
                    "Pola frekuensi kritis tinggi — kemungkinan artefak checkerboard GAN"
                    if final_score > 0.6
                    else "Distribusi spektrum normal"
                ),
            },
        }

    except Exception as exc:
        logger.warning("Gagal analisis checkerboard: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Sub-analisis 2: Inkonsistensi warna
# ---------------------------------------------------------------------------

def _analyze_color_inconsistency(rgb: np.ndarray) -> dict[str, Any]:
    """
    Menganalisis inkonsistensi distribusi warna antar region gambar.

    GAN sering menghasilkan warna yang tidak konsisten di area boundary karena
    proses blending yang tidak sempurna. Analisis:
      1. Bagi gambar menjadi grid NxN
      2. Hitung statistik warna (mean, std) per region per channel
      3. Ukur variabilitas statistik antar region
      4. Region dengan variabilitas tinggi mengindikasikan manipulasi

    Returns:
        dict berisi 'score' [0,1] dan 'details'
    """
    try:
        h, w, _ = rgb.shape
        cell_h = max(1, h // GRID_SIZE)
        cell_w = max(1, w // GRID_SIZE)

        region_means = []   # [n_cells, 3]
        region_stds = []    # [n_cells, 3]

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                y0, y1 = i * cell_h, min((i + 1) * cell_h, h)
                x0, x1 = j * cell_w, min((j + 1) * cell_w, w)
                cell = rgb[y0:y1, x0:x1]
                if cell.size == 0:
                    continue
                region_means.append(cell.reshape(-1, 3).mean(axis=0))
                region_stds.append(cell.reshape(-1, 3).std(axis=0))

        if not region_means:
            return {"score": 0.5, "details": {"error": "Tidak ada region yang valid"}}

        means_arr = np.array(region_means)   # [N, 3]
        stds_arr = np.array(region_stds)     # [N, 3]

        # Variabilitas mean antar region (seharusnya moderat pada foto asli)
        inter_region_mean_std = float(np.mean(np.std(means_arr, axis=0)))
        inter_region_std_std = float(np.mean(np.std(stds_arr, axis=0)))

        # Foto asli: variabilitas mean biasanya 20-60, GAN bisa sangat rendah atau sangat tinggi
        # Khususnya std dari std sangat rendah pada GAN (terlalu uniform)
        low_variance_flag = inter_region_mean_std < 15.0
        high_variance_flag = inter_region_mean_std > 65.0

        # Cek inkonsistensi lokal: beberapa region punya std sangat berbeda dari tetangga
        std_outliers = 0
        global_std_mean = float(np.mean(stds_arr))
        global_std_std = float(np.std(stds_arr))
        for s in stds_arr:
            if np.any(np.abs(s - global_std_mean) > 2.5 * global_std_std):
                std_outliers += 1

        outlier_ratio = std_outliers / len(stds_arr)

        # Komposit skor
        score = 0.0
        if low_variance_flag:
            score += 0.4   # terlalu uniform → GAN
        if high_variance_flag:
            score += 0.3   # terlalu beragam → mungkin splicing
        score += 0.5 * outlier_ratio  # outlier regional

        final_score = float(np.clip(score, 0.0, 1.0))

        return {
            "score": final_score,
            "details": {
                "inter_region_mean_variability": round(inter_region_mean_std, 4),
                "inter_region_std_variability": round(inter_region_std_std, 4),
                "outlier_region_ratio": round(outlier_ratio, 4),
                "low_variance_flag": low_variance_flag,
                "high_variance_flag": high_variance_flag,
                "grid_size": GRID_SIZE,
                "num_regions_analyzed": len(region_means),
            },
        }

    except Exception as exc:
        logger.warning("Gagal analisis color inconsistency: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Sub-analisis 3: Noise fingerprint
# ---------------------------------------------------------------------------

def _analyze_noise_fingerprint(gray: np.ndarray) -> dict[str, Any]:
    """
    Menganalisis pola noise residual menggunakan high-pass filter (Laplacian).

    Kamera asli menghasilkan pola noise sensor yang spesifik (PRNU-like).
    GAN menghasilkan noise yang lebih terstruktur/berulang. Analisis:
      1. Terapkan Laplacian filter untuk mengekstrak residual noise
      2. Analisis distribusi statistik noise
      3. Cek periodisitas dalam residual (GAN sering punya periodisitas)

    Returns:
        dict berisi 'score' [0,1] dan 'details'
    """
    try:
        # Kernel Laplacian untuk high-pass filtering
        laplacian_kernel = np.array(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]],
            dtype=np.float32
        )

        # Konvolusi manual sederhana menggunakan numpy (tanpa scipy)
        if SCIPY_AVAILABLE:
            residual = ndimage.convolve(gray, laplacian_kernel)
        else:
            # Fallback: estimasi Laplacian via finite differences
            residual = (
                -np.roll(gray, -1, axis=0)
                - np.roll(gray, 1, axis=0)
                - np.roll(gray, -1, axis=1)
                - np.roll(gray, 1, axis=1)
                + 4 * gray
            )

        # Clip residual
        residual_clipped = np.clip(residual, -NOISE_CLIP_VALUE * 10, NOISE_CLIP_VALUE * 10)

        # Statistik residual
        noise_mean = float(np.mean(residual_clipped))
        noise_std = float(np.std(residual_clipped))
        noise_abs_mean = float(np.mean(np.abs(residual_clipped)))

        # Kurtosis manual (distribusi normal: kurtosis ≈ 3)
        if noise_std > 1e-6:
            kurtosis = float(np.mean(((residual_clipped - noise_mean) / noise_std) ** 4))
        else:
            kurtosis = 3.0

        # GAN sering punya kurtosis rendah (distribusi lebih merata)
        # Foto asli: kurtosis tinggi (heavy-tailed)
        # Hanya kurtosis rendah (< 6.0) yang dianggap mencurigakan (pola halus/rata/artificial)
        if kurtosis < 6.0:
            kurtosis_deviation = 6.0 - kurtosis
        else:
            kurtosis_deviation = 0.0

        # Analisis periodisitas residual via FFT
        fft_residual = np.abs(np.fft.fft2(residual_clipped))
        fft_residual_shifted = np.fft.fftshift(fft_residual)
        log_fft = np.log1p(fft_residual_shifted)

        # Cari puncak yang tidak biasa (periodic artifact)
        fft_mean = float(np.mean(log_fft))
        fft_std = float(np.std(log_fft))
        peak_mask = log_fft > (fft_mean + 3.5 * fft_std)
        peak_count = int(np.sum(peak_mask))

        # Foto asli: sangat sedikit peak ekstrim; GAN: lebih banyak
        peak_ratio = peak_count / log_fft.size

        # Skor gabungan
        kurtosis_score = _normalize_score(kurtosis_deviation, min_val=0, max_val=4.0)
        peak_score = _normalize_score(peak_ratio * 1000, min_val=0, max_val=2.5)
        final_score = float(np.clip(0.5 * kurtosis_score + 0.5 * peak_score, 0.0, 1.0))

        return {
            "score": final_score,
            "details": {
                "noise_std": round(noise_std, 4),
                "noise_abs_mean": round(noise_abs_mean, 4),
                "kurtosis": round(kurtosis, 4),
                "kurtosis_deviation_from_baseline": round(kurtosis_deviation, 4),
                "periodic_peak_count": peak_count,
                "periodic_peak_ratio": round(peak_ratio, 6),
                "interpretation": (
                    "Noise residual menunjukkan pola GAN"
                    if final_score > 0.6
                    else "Noise residual konsisten dengan foto asli"
                ),
            },
        }

    except Exception as exc:
        logger.warning("Gagal analisis noise fingerprint: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Sub-analisis 4: Inkonsistensi ketajaman (sharpness)
# ---------------------------------------------------------------------------

def _analyze_sharpness_inconsistency(gray: np.ndarray) -> dict[str, Any]:
    """
    Mendeteksi inkonsistensi ketajaman lokal antar region gambar.

    GAN dan face-swap sering menghasilkan area yang tidak konsisten dalam hal
    ketajaman: area wajah palsu bisa terlalu tajam atau terlalu blur dibanding
    background. Analisis:
      1. Hitung local sharpness via variance of Laplacian di setiap region
      2. Bandingkan sharpness antar region
      3. Region dengan sharpness sangat berbeda mengindikasikan manipulasi

    Returns:
        dict berisi 'score' [0,1] dan 'details'
    """
    try:
        h, w = gray.shape
        cell_h = max(1, h // GRID_SIZE)
        cell_w = max(1, w // GRID_SIZE)

        sharpness_map = []

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                y0, y1 = i * cell_h, min((i + 1) * cell_h, h)
                x0, x1 = j * cell_w, min((j + 1) * cell_w, w)
                cell = gray[y0:y1, x0:x1]

                if cell.size < 4:
                    continue

                # Variance of Laplacian sebagai ukuran sharpness
                if SCIPY_AVAILABLE:
                    lap = ndimage.laplace(cell)
                else:
                    lap = (
                        -np.roll(cell, -1, axis=0)
                        - np.roll(cell, 1, axis=0)
                        - np.roll(cell, -1, axis=1)
                        - np.roll(cell, 1, axis=1)
                        + 4 * cell
                    )
                sharpness = float(np.var(lap))
                sharpness_map.append(sharpness)

        if len(sharpness_map) < 2:
            return {"score": 0.5, "details": {"error": "Tidak cukup region"}}

        sharpness_arr = np.array(sharpness_map)
        global_mean = float(np.mean(sharpness_arr))
        global_std = float(np.std(sharpness_arr))

        # Koefisien variasi (CV) — lebih tinggi = lebih inconsistent
        cv = global_std / (global_mean + 1e-6)

        # Deteksi outlier region (> 2 sigma dari mean)
        outlier_count = int(np.sum(np.abs(sharpness_arr - global_mean) > 2.0 * global_std))
        outlier_ratio = outlier_count / len(sharpness_arr)

        # Kisaran min-max (GAN sering punya kisaran sangat lebar atau sangat sempit)
        sharpness_range = float(np.max(sharpness_arr) - np.min(sharpness_arr))
        normalized_range = sharpness_range / (global_mean + 1e-6)

        # Foto asli: CV biasanya 0.3–1.5 (variasi wajar karena depth of field)
        # GAN: bisa <0.1 (terlalu uniform) atau >3.0 (terlalu inconsistent)
        cv_score = 0.0
        if cv < 0.15:
            cv_score = 0.7  # terlalu uniform
        elif cv > 2.5:
            cv_score = 0.8  # terlalu inkonsisten
        else:
            cv_score = _normalize_score(cv, min_val=0.15, max_val=2.5) * 0.6

        outlier_score = min(1.0, outlier_ratio * 3)
        final_score = float(np.clip(0.6 * cv_score + 0.4 * outlier_score, 0.0, 1.0))

        return {
            "score": final_score,
            "details": {
                "sharpness_mean": round(global_mean, 4),
                "sharpness_std": round(global_std, 4),
                "coefficient_of_variation": round(cv, 4),
                "outlier_region_ratio": round(outlier_ratio, 4),
                "sharpness_normalized_range": round(normalized_range, 4),
                "interpretation": (
                    "Inkonsistensi ketajaman terdeteksi — kemungkinan manipulasi"
                    if final_score > 0.55
                    else "Ketajaman konsisten di seluruh gambar"
                ),
            },
        }

    except Exception as exc:
        logger.warning("Gagal analisis sharpness inconsistency: %s", exc)
        return {"score": 0.5, "details": {"error": str(exc)}}


# ---------------------------------------------------------------------------
# Kelas utama
# ---------------------------------------------------------------------------

class GANFingerprintAnalyzer:
    """
    Menganalisis gambar untuk mendeteksi artefak khas yang ditinggalkan oleh GAN.

    Mengombinasikan empat teknik analisis:
    1. Deteksi artefak checkerboard via spektrum FFT
    2. Analisis inkonsistensi warna antar region
    3. Analisis noise fingerprint menggunakan high-pass filter
    4. Deteksi inkonsistensi ketajaman lokal

    Setiap sub-analisis menghasilkan skor [0,1] di mana:
      0.0 = sangat meyakinkan ASLI
      1.0 = sangat meyakinkan PALSU

    Sub-skor digabung dengan bobot yang telah ditentukan (SUB_WEIGHTS).
    """

    def __init__(self) -> None:
        self.weights = SUB_WEIGHTS.copy()

    # ------------------------------------------------------------------
    def analyze(self, image_path: str) -> dict[str, Any]:
        """
        Melakukan analisis sidik jari GAN pada gambar.

        Args:
            image_path: Path absolut atau relatif ke file gambar.

        Returns:
            dict dengan kunci:
              - score      : float [0, 1] – skor kecurigaan
              - verdict    : str  – 'REAL' | 'SUSPICIOUS' | 'FAKE'
              - confidence : float [0, 1] – tingkat keyakinan analisis
              - details    : dict – hasil tiap sub-analisis
        """
        result_template = {
            "score": 0.5,
            "verdict": "SUSPICIOUS",
            "confidence": 0.0,
            "details": {},
        }

        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

            # --- Muat gambar ---
            rgb, gray = _load_image_as_array(str(path))

            # --- Jalankan semua sub-analisis ---
            checkerboard_result = _analyze_checkerboard(gray)
            color_result = _analyze_color_inconsistency(rgb)
            noise_result = _analyze_noise_fingerprint(gray)
            sharpness_result = _analyze_sharpness_inconsistency(gray)

            sub_results = {
                "checkerboard": checkerboard_result,
                "color_inconsistency": color_result,
                "noise_fingerprint": noise_result,
                "sharpness_inconsistency": sharpness_result,
            }

            # --- Hitung weighted average ---
            total_weight = 0.0
            weighted_score = 0.0
            sub_scores: dict[str, float] = {}

            for key, res in sub_results.items():
                weight = self.weights.get(key, 0.25)
                sub_score = res.get("score", 0.5)
                sub_scores[key] = sub_score
                weighted_score += weight * sub_score
                total_weight += weight

            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.5

            # --- Hitung confidence berdasarkan konsensus sub-skor ---
            scores_list = list(sub_scores.values())
            score_std = float(np.std(scores_list))
            # Confidence tinggi bila semua sub-skor searah; rendah bila divergen
            confidence = float(np.clip(1.0 - score_std * 2.0, 0.1, 0.95))

            verdict = _verdict_from_score(final_score)

            return {
                "score": round(float(final_score), 4),
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "details": {
                    "sub_scores": sub_scores,
                    "sub_weights": self.weights,
                    "image_shape": list(rgb.shape),
                    "sub_analyses": {k: v.get("details", {}) for k, v in sub_results.items()},
                },
            }

        except FileNotFoundError as fnf:
            logger.error("File tidak ditemukan: %s", fnf)
            result_template["details"] = {"error": f"File tidak ditemukan: {fnf}"}
            return result_template

        except Exception as exc:
            logger.error("Kesalahan tidak terduga pada GANFingerprintAnalyzer: %s\n%s",
                         exc, traceback.format_exc())
            result_template["details"] = {"error": str(exc), "traceback": traceback.format_exc()}
            return result_template
