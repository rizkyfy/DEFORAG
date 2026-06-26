"""
video_processor.py
==================
Modul utilitas untuk memproses file video dan mengekstrak frame.

Fitur utama:
  - Ekstraksi frame tersebar merata dari rentang 10%–90% durasi video
  - Pengambilan metadata video (fps, durasi, dimensi)
  - Deteksi apakah suatu path adalah file video atau gambar

Author  : ANTENK TEAM
Version : 1.0.0
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ekstensi file yang didukung
# ---------------------------------------------------------------------------
_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv",
    ".flv", ".webm", ".m4v", ".mpeg", ".mpg",
    ".3gp", ".ts", ".mts", ".m2ts",
}

_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff",
    ".tif", ".webp", ".gif", ".heic", ".heif",
}


class VideoProcessor:
    """
    Kelas untuk memproses file video: mengekstrak frame dan membaca metadata.

    Semua metode bersifat stateless (tidak ada state yang tersimpan antar panggilan),
    sehingga instance dapat digunakan secara thread-safe untuk berbagai file video.
    """

    # ------------------------------------------------------------------
    # API Publik
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        video_path: Union[str, Path],
        n_frames: int = 15,
    ) -> List[Image.Image]:
        """
        Ekstrak sejumlah frame dari video yang tersebar merata di rentang 10%–90%.

        Rentang 10%–90% digunakan untuk menghindari frame pembuka/penutup
        yang sering kali berisi adegan transisi, layar hitam, atau kredit.

        Parameters
        ----------
        video_path : str | Path
            Path ke file video.
        n_frames : int, optional
            Jumlah frame yang akan diekstrak. Default 15.

        Returns
        -------
        list of PIL.Image.Image
            Daftar frame sebagai PIL Image dalam mode RGB.
            Mengembalikan list kosong jika terjadi kesalahan.

        Raises
        ------
        FileNotFoundError
            Jika file video tidak ditemukan.
        ValueError
            Jika n_frames kurang dari 1.
        """
        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

        if n_frames < 1:
            raise ValueError(f"n_frames harus >= 1, didapat: {n_frames}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Gagal membuka video: %s", video_path)
            return []

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                logger.warning(
                    "Jumlah frame tidak dapat dibaca dari video: %s", video_path
                )
                return []

            # Hitung indeks frame yang akan diekstrak (10% – 90% durasi)
            start_frame = int(total_frames * 0.10)
            end_frame = int(total_frames * 0.90)

            # Pastikan rentang valid
            if start_frame >= end_frame:
                start_frame = 0
                end_frame = total_frames - 1

            # Distribusi merata dalam rentang
            if n_frames == 1:
                frame_indices = [int((start_frame + end_frame) / 2)]
            else:
                frame_indices = [
                    int(start_frame + i * (end_frame - start_frame) / (n_frames - 1))
                    for i in range(n_frames)
                ]

            frames: List[Image.Image] = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame_bgr = cap.read()

                if not ret or frame_bgr is None:
                    logger.debug("Gagal membaca frame ke-%d dari '%s'.", idx, video_path)
                    continue

                # Konversi BGR → RGB → PIL Image
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)

            logger.info(
                "Berhasil mengekstrak %d dari %d frame yang diminta dari '%s'.",
                len(frames),
                n_frames,
                video_path.name,
            )
            return frames

        except Exception as exc:
            logger.error("Error saat mengekstrak frame dari '%s': %s", video_path, exc)
            return []

        finally:
            cap.release()

    def get_video_info(
        self, video_path: Union[str, Path]
    ) -> Dict[str, Union[float, int]]:
        """
        Dapatkan informasi metadata dari file video.

        Parameters
        ----------
        video_path : str | Path
            Path ke file video.

        Returns
        -------
        dict
            Dictionary berisi:
            - ``fps``         : float, jumlah frame per detik
            - ``duration``    : float, durasi dalam detik
            - ``frame_count`` : int, total jumlah frame
            - ``width``       : int, lebar frame (piksel)
            - ``height``      : int, tinggi frame (piksel)

        Raises
        ------
        FileNotFoundError
            Jika file video tidak ditemukan.
        RuntimeError
            Jika video tidak dapat dibuka oleh OpenCV.
        """
        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Gagal membuka video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Durasi dihitung dari frame_count dan fps; hindari pembagian nol
            duration = frame_count / fps if fps > 0 else 0.0

            info = {
                "fps": round(fps, 4),
                "duration": round(duration, 4),
                "frame_count": frame_count,
                "width": width,
                "height": height,
            }
            logger.debug("Info video '%s': %s", video_path.name, info)
            return info

        finally:
            cap.release()

    def is_video(self, path: Union[str, Path]) -> bool:
        """
        Periksa apakah suatu path merujuk ke file video berdasarkan ekstensinya.

        Parameters
        ----------
        path : str | Path
            Path file yang akan diperiksa.

        Returns
        -------
        bool
            True jika ekstensi file termasuk video yang didukung.
        """
        ext = Path(path).suffix.lower()
        return ext in _VIDEO_EXTENSIONS

    def is_image(self, path: Union[str, Path]) -> bool:
        """
        Periksa apakah suatu path merujuk ke file gambar berdasarkan ekstensinya.

        Parameters
        ----------
        path : str | Path
            Path file yang akan diperiksa.

        Returns
        -------
        bool
            True jika ekstensi file termasuk gambar yang didukung.
        """
        ext = Path(path).suffix.lower()
        return ext in _IMAGE_EXTENSIONS

    # ------------------------------------------------------------------
    # Utilitas tambahan
    # ------------------------------------------------------------------

    def get_supported_video_extensions(self) -> List[str]:
        """
        Kembalikan daftar ekstensi video yang didukung.

        Returns
        -------
        list of str
            Daftar ekstensi dalam format '.ext' (huruf kecil).
        """
        return sorted(_VIDEO_EXTENSIONS)

    def get_supported_image_extensions(self) -> List[str]:
        """
        Kembalikan daftar ekstensi gambar yang didukung.

        Returns
        -------
        list of str
            Daftar ekstensi dalam format '.ext' (huruf kecil).
        """
        return sorted(_IMAGE_EXTENSIONS)

    def __repr__(self) -> str:
        return "VideoProcessor()"
