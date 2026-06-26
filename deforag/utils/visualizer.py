"""
visualizer.py
=============
Modul utilitas untuk membuat visualisasi hasil analisis deepfake.

Menyediakan fungsi untuk:
  - Visualisasi FFT magnitude spectrum
  - Bar chart horizontal skor analisis
  - Overlay bounding box wajah pada gambar
  - Menyimpan ringkasan visualisasi ke direktori output

Author  : ANTENK TEAM
Version : 1.0.0
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif (aman untuk server/tanpa display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta warna (BGR untuk OpenCV, RGB untuk Matplotlib)
# ---------------------------------------------------------------------------
_COLOR_FAKE_BGR = (0, 0, 220)       # Merah
_COLOR_REAL_BGR = (34, 180, 34)     # Hijau
_COLOR_BOX_BGR  = (0, 200, 255)     # Kuning-oranye

_COLOR_FAKE_RGB = (220, 0, 0)
_COLOR_REAL_RGB = (34, 180, 34)


class Visualizer:
    """
    Kelas untuk membuat visualisasi analisis deepfake.

    Semua metode mengembalikan numpy array uint8 (BGR untuk kompatibilitas OpenCV)
    kecuali dinyatakan lain, atau menyimpan gambar ke disk.
    """

    # ------------------------------------------------------------------
    # Visualisasi FFT Spectrum
    # ------------------------------------------------------------------

    def create_frequency_spectrum(self, image_np: np.ndarray) -> np.ndarray:
        """
        Buat visualisasi FFT magnitude spectrum dari gambar.

        Spectrum yang tampak merata dan tidak memiliki artefak periodik
        biasanya menandakan gambar asli. Sebaliknya, pola simetris atau
        artefak berulang di spectrum dapat mengindikasikan gambar deepfake.

        Parameters
        ----------
        image_np : np.ndarray
            Gambar input sebagai array numpy (RGB atau BGR, uint8 atau float).
            Bisa berwarna atau grayscale.

        Returns
        -------
        np.ndarray
            Visualisasi spectrum sebagai array numpy uint8 RGB
            ukuran yang sama dengan input (atau 512×512 jika input kecil).
        """
        try:
            # Pastikan ukuran minimum untuk visualisasi yang baik
            target_size = max(512, max(image_np.shape[:2]))

            # Konversi ke grayscale
            if image_np.ndim == 3:
                if image_np.shape[2] == 4:
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np.copy()

            # Resize ke ukuran target (power-of-2 lebih baik untuk FFT,
            # namun numpy.fft.fft2 bekerja baik pada semua ukuran)
            gray_resized = cv2.resize(
                gray.astype(np.float32),
                (target_size, target_size),
                interpolation=cv2.INTER_LANCZOS4,
            )

            # Terapkan windowing Hann untuk mengurangi spectral leakage
            window = np.outer(
                np.hanning(target_size), np.hanning(target_size)
            )
            gray_windowed = gray_resized * window

            # Hitung FFT 2D
            fft2d = np.fft.fft2(gray_windowed)
            # Geser komponen DC ke tengah
            fft_shifted = np.fft.fftshift(fft2d)
            # Magnitude dalam skala logaritmik
            magnitude = np.log1p(np.abs(fft_shifted))

            # Normalisasi ke [0, 255]
            mag_norm = cv2.normalize(
                magnitude, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # Terapkan colormap JET untuk visualisasi yang menarik
            spectrum_colored = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
            # Konversi BGR → RGB
            spectrum_rgb = cv2.cvtColor(spectrum_colored, cv2.COLOR_BGR2RGB)

            # Tambahkan judul menggunakan matplotlib untuk output yang bersih
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#1a1a2e")
            fig.suptitle(
                "Analisis Frekuensi (FFT Magnitude Spectrum)",
                color="white", fontsize=13, fontweight="bold"
            )

            ax_orig = axes[0]
            ax_orig.imshow(
                cv2.resize(image_np if image_np.ndim == 3 else
                           np.stack([image_np]*3, axis=-1),
                           (target_size, target_size)),
            )
            ax_orig.set_title("Gambar Asli", color="white", fontsize=11)
            ax_orig.axis("off")

            ax_spec = axes[1]
            ax_spec.imshow(spectrum_rgb)
            ax_spec.set_title("Magnitude Spectrum (log)", color="white", fontsize=11)
            ax_spec.axis("off")

            plt.tight_layout()

            # Render figure ke numpy array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            return buf

        except Exception as exc:
            logger.error("Gagal membuat visualisasi spectrum: %s", exc)
            # Kembalikan gambar placeholder hitam
            placeholder = np.zeros((256, 512, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "Spectrum tidak tersedia",
                (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1
            )
            return placeholder

    # ------------------------------------------------------------------
    # Bar Chart Skor
    # ------------------------------------------------------------------

    def create_score_bar(
        self,
        scores_dict: Dict[str, float],
        title: str = "Skor Analisis",
    ) -> np.ndarray:
        """
        Buat bar chart horizontal dari skor analisis per modul.

        Parameters
        ----------
        scores_dict : dict
            Dictionary dengan nama modul sebagai key dan skor (0.0–1.0) sebagai value.
            Contoh: {'CNN': 0.87, 'Frekuensi': 0.62, 'Landmark': 0.45}
        title : str, optional
            Judul grafik. Default 'Skor Analisis'.

        Returns
        -------
        np.ndarray
            Bar chart sebagai array numpy uint8 RGB.
        """
        try:
            if not scores_dict:
                raise ValueError("scores_dict tidak boleh kosong.")

            labels = list(scores_dict.keys())
            values = [float(v) for v in scores_dict.values()]

            n = len(labels)
            fig_h = max(3.0, n * 1.0 + 1.5)
            fig, ax = plt.subplots(figsize=(8, fig_h), facecolor="#1a1a2e")
            ax.set_facecolor("#0f0f23")

            # Tentukan warna bar berdasarkan skor
            colors = []
            for v in values:
                if v >= 0.6:
                    colors.append("#e74c3c")   # Merah – FAKE
                elif v >= 0.4:
                    colors.append("#f39c12")   # Oranye – Tidak pasti
                else:
                    colors.append("#2ecc71")   # Hijau – REAL

            # Gambar bar horizontal
            bars = ax.barh(
                labels, values,
                color=colors,
                edgecolor="none",
                height=0.55,
            )

            # Tambahkan nilai di ujung bar
            for bar, val in zip(bars, values):
                xpos = min(val + 0.02, 0.98)
                ax.text(
                    xpos, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center", ha="left",
                    color="white", fontsize=10, fontweight="bold",
                )

            # Garis threshold 0.5
            ax.axvline(x=0.5, color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.6)
            ax.text(
                0.505, n - 0.5, "Threshold (0.5)",
                color="#aaaaaa", fontsize=8, va="top"
            )

            # Styling
            ax.set_xlim(0.0, 1.05)
            ax.set_xlabel("Skor Kecurigaan", color="#cccccc", fontsize=10)
            ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
            ax.tick_params(colors="#cccccc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

            # Legend
            legend_elements = [
                mpatches.Patch(color="#2ecc71", label="REAL (< 0.4)"),
                mpatches.Patch(color="#f39c12", label="Tidak Pasti (0.4–0.6)"),
                mpatches.Patch(color="#e74c3c", label="FAKE (> 0.6)"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="lower right",
                facecolor="#1a1a2e",
                labelcolor="white",
                fontsize=8,
                edgecolor="#444466",
            )

            plt.tight_layout()

            # Render ke numpy array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            return buf

        except Exception as exc:
            logger.error("Gagal membuat bar chart: %s", exc)
            placeholder = np.zeros((200, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "Chart tidak tersedia",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1
            )
            return placeholder

    # ------------------------------------------------------------------
    # Overlay Bounding Box Wajah
    # ------------------------------------------------------------------

    def create_face_overlay(
        self,
        image_np: np.ndarray,
        face_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Buat overlay bounding box wajah pada gambar.

        Jika face_box tidak disediakan, hanya gambar asli yang dikembalikan
        tanpa modifikasi.

        Parameters
        ----------
        image_np : np.ndarray
            Gambar input sebagai array numpy RGB uint8.
        face_box : tuple of int, optional
            Bounding box dalam format (x1, y1, x2, y2).

        Returns
        -------
        np.ndarray
            Gambar dengan overlay bounding box sebagai array numpy RGB uint8.
        """
        try:
            # Buat salinan agar tidak mengubah array asli
            overlay = image_np.copy()
            if overlay.dtype != np.uint8:
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            # Konversi ke BGR untuk operasi OpenCV
            if overlay.ndim == 3 and overlay.shape[2] == 3:
                bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            else:
                bgr = overlay.copy()

            if face_box is not None:
                x1, y1, x2, y2 = (int(v) for v in face_box)

                # Gambar bounding box dengan sudut membulat (simulasi)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), _COLOR_BOX_BGR, 2)

                # Sudut dekoratif (corner brackets)
                corner_len = max(10, (x2 - x1) // 8)
                corner_thickness = 3
                bracket_color = (0, 255, 220)  # Cyan-teal

                # Kiri-atas
                cv2.line(bgr, (x1, y1), (x1 + corner_len, y1), bracket_color, corner_thickness)
                cv2.line(bgr, (x1, y1), (x1, y1 + corner_len), bracket_color, corner_thickness)
                # Kanan-atas
                cv2.line(bgr, (x2, y1), (x2 - corner_len, y1), bracket_color, corner_thickness)
                cv2.line(bgr, (x2, y1), (x2, y1 + corner_len), bracket_color, corner_thickness)
                # Kiri-bawah
                cv2.line(bgr, (x1, y2), (x1 + corner_len, y2), bracket_color, corner_thickness)
                cv2.line(bgr, (x1, y2), (x1, y2 - corner_len), bracket_color, corner_thickness)
                # Kanan-bawah
                cv2.line(bgr, (x2, y2), (x2 - corner_len, y2), bracket_color, corner_thickness)
                cv2.line(bgr, (x2, y2), (x2, y2 - corner_len), bracket_color, corner_thickness)

                # Label "WAJAH TERDETEKSI"
                label = "WAJAH TERDETEKSI"
                (lw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                label_x = x1
                label_y = max(y1 - 8, lh + 4)
                # Background label
                cv2.rectangle(
                    bgr,
                    (label_x, label_y - lh - 4),
                    (label_x + lw + 8, label_y + 4),
                    (0, 0, 0), cv2.FILLED
                )
                cv2.putText(
                    bgr, label,
                    (label_x + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    bracket_color, 1, cv2.LINE_AA
                )

            # Konversi kembali ke RGB
            result_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return result_rgb

        except Exception as exc:
            logger.error("Gagal membuat face overlay: %s", exc)
            return image_np.copy()

    # ------------------------------------------------------------------
    # Simpan Visualisasi ke Disk
    # ------------------------------------------------------------------

    def save_visualization(
        self,
        output_dir: Union[str, Path],
        image_path: Union[str, Path],
        results: Dict[str, Any],
    ) -> str:
        """
        Buat dan simpan gambar ringkasan visualisasi hasil analisis.

        Menghasilkan satu gambar gabungan yang terdiri dari:
          - Gambar asli (atau frame pertama jika video)
          - Bar chart skor per modul
          - Teks ringkasan verdict keseluruhan

        Parameters
        ----------
        output_dir : str | Path
            Direktori tempat file visualisasi akan disimpan.
            Akan dibuat otomatis jika belum ada.
        image_path : str | Path
            Path file gambar/video yang dianalisis (untuk penamaan output).
        results : dict
            Hasil analisis dari DEFORAG. Harus memiliki key:
            - ``verdict``    : str, 'FAKE' atau 'REAL'
            - ``score``      : float, skor gabungan 0–1
            - ``confidence`` : float, keyakinan 0–1
            - ``modules``    : dict, skor per modul (opsional)

        Returns
        -------
        str
            Path absolut ke file visualisasi yang disimpan.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(image_path)

        # Tentukan nama file output berdasarkan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = image_path.stem[:30]  # Batasi panjang nama file
        out_filename = f"deforag_{stem}_{timestamp}.png"
        out_path = output_dir / out_filename

        try:
            # ---- Muat gambar asli ----
            try:
                orig_img = np.array(Image.open(str(image_path)).convert("RGB"))
            except Exception:
                orig_img = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.putText(
                    orig_img, "Gambar tidak\ndapat dimuat",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1
                )

            # Resize gambar asli ke 320×320 untuk tampilan
            display_size = (320, 320)
            orig_display = cv2.resize(orig_img, display_size, interpolation=cv2.INTER_AREA)

            # ---- Ambil skor per modul ----
            modules = results.get("modules", {})
            scores_dict: Dict[str, float] = {}
            for mod_name, mod_result in modules.items():
                if isinstance(mod_result, dict):
                    scores_dict[mod_name] = float(mod_result.get("score", 0.5))
                elif isinstance(mod_result, (int, float)):
                    scores_dict[mod_name] = float(mod_result)

            # Jika tidak ada data modul, tampilkan skor keseluruhan saja
            if not scores_dict:
                scores_dict = {"Skor Keseluruhan": float(results.get("score", 0.5))}

            # ---- Buat bar chart ----
            bar_chart = self.create_score_bar(scores_dict, title="Skor per Modul")
            bar_resized = cv2.resize(bar_chart, (640, display_size[1]), interpolation=cv2.INTER_AREA)

            # ---- Buat panel teks ringkasan ----
            verdict = results.get("verdict", "UNKNOWN")
            score_val = float(results.get("score", 0.5))
            confidence = float(results.get("confidence", 0.0))

            text_panel = np.zeros((display_size[1], 320, 3), dtype=np.uint8)
            text_panel[:] = (15, 15, 35)  # Background gelap

            is_fake = verdict.upper() == "FAKE"
            verdict_color = _COLOR_FAKE_BGR if is_fake else _COLOR_REAL_BGR

            # Teks Verdict
            cv2.putText(
                text_panel, "HASIL ANALISIS",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (180, 180, 200), 1, cv2.LINE_AA
            )
            cv2.putText(
                text_panel, verdict,
                (20, 110), cv2.FONT_HERSHEY_DUPLEX,
                2.5 if len(verdict) <= 4 else 1.8,
                verdict_color, 3, cv2.LINE_AA
            )

            # Teks Skor dan Confidence
            cv2.putText(
                text_panel, f"Skor     : {score_val:.4f}",
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (220, 220, 220), 1, cv2.LINE_AA
            )
            cv2.putText(
                text_panel, f"Keyakinan: {confidence:.2%}",
                (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (220, 220, 220), 1, cv2.LINE_AA
            )
            cv2.putText(
                text_panel, f"File: {image_path.name[:25]}",
                (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (160, 160, 160), 1, cv2.LINE_AA
            )
            cv2.putText(
                text_panel, f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                (20, 265), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (160, 160, 160), 1, cv2.LINE_AA
            )

            # Watermark
            cv2.putText(
                text_panel, "DEFORAG v1.0",
                (20, display_size[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (80, 80, 120), 1, cv2.LINE_AA
            )

            # ---- Gabungkan semua panel secara horizontal ----
            # Panel: [Gambar Asli | Teks Ringkasan | Bar Chart]
            top_row = np.hstack([orig_display, text_panel])
            # Bar chart lebih lebar, sehingga kita buat baris kedua
            separator = np.zeros((8, top_row.shape[1], 3), dtype=np.uint8)
            separator[:] = (50, 50, 80)

            # Resize bar chart agar lebarnya sama dengan top_row
            bar_final = cv2.resize(
                bar_chart,
                (top_row.shape[1], 200),
                interpolation=cv2.INTER_AREA
            )

            # Konversi bar_final dari RGB ke BGR (bar_chart dari matplotlib adalah RGB)
            bar_final_bgr = cv2.cvtColor(bar_final, cv2.COLOR_RGB2BGR)

            # Konversi orig_display & text_panel ke BGR (sudah BGR)
            full_image = np.vstack([top_row, separator, bar_final_bgr])

            # Simpan ke disk
            cv2.imwrite(str(out_path), full_image)
            logger.info("Visualisasi disimpan ke: %s", out_path)

            return str(out_path)

        except Exception as exc:
            logger.error("Gagal menyimpan visualisasi: %s", exc)
            # Simpan gambar kosong sebagai placeholder
            placeholder = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.imwrite(str(out_path), placeholder)
            return str(out_path)

    # ------------------------------------------------------------------
    # Representasi
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return "Visualizer()"
