"""
face_extractor.py
=================
Modul utilitas untuk mendeteksi dan mengekstrak wajah dari gambar.

Mendukung dua backend:
  - facenet_pytorch MTCNN (lebih akurat, direkomendasikan)
  - OpenCV Haar Cascade (fallback jika facenet_pytorch tidak tersedia)

Author  : ANTENK TEAM
Version : 1.0.0
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coba import MTCNN — urutan prioritas:
#   1. facenet_pytorch (lebih akurat, butuh build tools)
#   2. mtcnn (pip install mtcnn, lebih mudah di Windows)
#   3. OpenCV Haar Cascade (selalu tersedia, paling ringan)
# ---------------------------------------------------------------------------
MTCNN_AVAILABLE   = False
MTCNN_SIMPLE      = False  # flag untuk paket 'mtcnn'

try:
    from facenet_pytorch import MTCNN
    import torch
    MTCNN_AVAILABLE = True
    logger.info("facenet_pytorch MTCNN berhasil diimpor.")
except ImportError:
    try:
        from mtcnn import MTCNN as _MTCNN_SIMPLE
        MTCNN_SIMPLE = True
        logger.info("mtcnn (pip package) berhasil diimpor sebagai fallback.")
    except ImportError:
        logger.warning(
            "Tidak ada MTCNN yang tersedia. Menggunakan OpenCV Haar Cascade."
        )


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
# Path ke file Haar Cascade bawaan OpenCV
_HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceExtractor:
    """
    Kelas untuk mendeteksi dan mengekstrak region wajah dari gambar.

    Menggunakan MTCNN dari facenet_pytorch sebagai detektor utama.
    Jika tidak tersedia, fallback ke OpenCV Haar Cascade.

    Parameters
    ----------
    device : str, optional
        Perangkat komputasi ('cpu' atau 'cuda'). Default 'cpu'.
    image_size : int, optional
        Ukuran output crop wajah (persegi). Default 224.
    margin : float, optional
        Margin tambahan di sekitar wajah yang terdeteksi (0.0–1.0).
        Nilai 0.3 berarti 30% dari ukuran wajah ditambahkan di setiap sisi.
        Default 0.3.
    """

    def __init__(
        self,
        device: str = "cpu",
        image_size: int = 224,
        margin: float = 0.3,
    ) -> None:
        self.device = device
        self.image_size = image_size
        self.margin = margin

        # Inisialisasi detektor berdasarkan ketersediaan library
        self._backend: str
        if MTCNN_AVAILABLE:
            self._init_mtcnn()
        elif MTCNN_SIMPLE:
            self._init_mtcnn_simple()
        else:
            self._init_opencv()

    # ------------------------------------------------------------------
    # Inisialisasi backend
    # ------------------------------------------------------------------

    def _init_mtcnn(self) -> None:
        """Inisialisasi MTCNN dari facenet_pytorch."""
        try:
            self._mtcnn = MTCNN(
                image_size=self.image_size,
                margin=int(self.image_size * self.margin),
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                keep_all=True,
                device=self.device,
            )
            self._backend = "mtcnn"
            logger.info("Backend MTCNN (facenet_pytorch) diinisialisasi.")
        except Exception as exc:
            logger.error("Gagal menginisialisasi MTCNN: %s. Beralih ke OpenCV.", exc)
            self._init_opencv()

    def _init_mtcnn_simple(self) -> None:
        """Inisialisasi MTCNN dari paket 'mtcnn' (Windows-friendly)."""
        try:
            self._mtcnn_simple = _MTCNN_SIMPLE()
            self._backend = "mtcnn_simple"
            logger.info("Backend MTCNN (mtcnn package) diinisialisasi.")
        except Exception as exc:
            logger.error("Gagal inisialisasi mtcnn: %s. Beralih ke OpenCV.", exc)
            self._init_opencv()

    def _init_opencv(self) -> None:
        """Inisialisasi OpenCV Haar Cascade sebagai fallback."""
        if not os.path.isfile(_HAAR_CASCADE_PATH):
            raise FileNotFoundError(
                f"File Haar Cascade tidak ditemukan: {_HAAR_CASCADE_PATH}"
            )
        self._haar = cv2.CascadeClassifier(_HAAR_CASCADE_PATH)
        self._backend = "opencv"
        logger.info("Backend OpenCV Haar Cascade diinisialisasi.")

    # ------------------------------------------------------------------
    # Utilitas internal
    # ------------------------------------------------------------------

    def _load_image(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Image.Image:
        """
        Muat gambar dari berbagai tipe input dan kembalikan sebagai PIL Image RGB.

        Parameters
        ----------
        image : str | Path | np.ndarray | PIL.Image.Image
            Sumber gambar. Bisa berupa path file, array numpy, atau PIL Image.

        Returns
        -------
        PIL.Image.Image
            Gambar dalam mode RGB.
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Asumsikan numpy array dalam format BGR (OpenCV) atau RGB
            if image.ndim == 2:
                # Grayscale → RGB
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 4:
                # BGRA → RGB
                pil_img = Image.fromarray(
                    cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                )
            else:
                # Coba deteksi apakah BGR atau RGB; konversi ke PIL (RGB)
                pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(
                f"Tipe input tidak didukung: {type(image)}. "
                "Gunakan str, Path, numpy.ndarray, atau PIL.Image.Image."
            )
        return pil_img

    def _add_margin(
        self, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
    ) -> tuple:
        """
        Tambahkan margin pada bounding box wajah dan pastikan tidak keluar batas gambar.

        Parameters
        ----------
        x1, y1 : int
            Koordinat kiri-atas bounding box.
        x2, y2 : int
            Koordinat kanan-bawah bounding box.
        img_w, img_h : int
            Lebar dan tinggi gambar asli.

        Returns
        -------
        tuple[int, int, int, int]
            Bounding box baru (x1, y1, x2, y2) setelah ditambahkan margin.
        """
        w = x2 - x1
        h = y2 - y1
        mx = int(w * self.margin)
        my = int(h * self.margin)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(img_w, x2 + mx)
        y2 = min(img_h, y2 + my)
        return x1, y1, x2, y2

    def _crop_and_resize(
        self, pil_img: Image.Image, box: tuple
    ) -> np.ndarray:
        """
        Crop region wajah dan resize ke image_size × image_size.

        Parameters
        ----------
        pil_img : PIL.Image.Image
            Gambar asli.
        box : tuple
            (x1, y1, x2, y2) koordinat region yang akan di-crop.

        Returns
        -------
        np.ndarray
            Array numpy uint8 RGB ukuran (image_size, image_size, 3).
        """
        x1, y1, x2, y2 = box
        crop = pil_img.crop((x1, y1, x2, y2))
        crop = crop.resize((self.image_size, self.image_size), Image.LANCZOS)
        return np.array(crop, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Deteksi menggunakan MTCNN
    # ------------------------------------------------------------------

    def _extract_mtcnn(
        self, pil_img: Image.Image
    ) -> List[np.ndarray]:
        """
        Deteksi semua wajah menggunakan MTCNN.

        Returns
        -------
        list of np.ndarray
            Daftar crop wajah sebagai array numpy RGB uint8.
        """
        img_w, img_h = pil_img.size

        try:
            # MTCNN detect mengembalikan (boxes, probs)
            boxes, probs = self._mtcnn.detect(pil_img)
        except Exception as exc:
            logger.debug("MTCNN detect error: %s", exc)
            return []

        if boxes is None:
            return []

        crops: List[np.ndarray] = []
        for box, prob in zip(boxes, probs):
            if prob is None or prob < 0.85:
                continue  # Abaikan deteksi dengan confidence rendah
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, y1, x2, y2 = self._add_margin(x1, y1, x2, y2, img_w, img_h)
            crop = self._crop_and_resize(pil_img, (x1, y1, x2, y2))
            crops.append(crop)

        return crops

    # ------------------------------------------------------------------
    # Deteksi menggunakan mtcnn package (fallback)
    # ------------------------------------------------------------------

    def _extract_mtcnn_simple(
        self, pil_img: Image.Image
    ) -> List[np.ndarray]:
        """
        Deteksi semua wajah menggunakan package 'mtcnn' (Windows fallback).

        Returns
        -------
        list of np.ndarray
            Daftar crop wajah sebagai array numpy RGB uint8.
        """
        img_np = np.array(pil_img)
        img_w, img_h = pil_img.size

        try:
            # detect_faces mengembalikan list of dict: [{'box': [x,y,w,h], 'confidence': c, ...}]
            faces = self._mtcnn_simple.detect_faces(img_np)
        except Exception as exc:
            logger.debug("mtcnn_simple detect error: %s", exc)
            return []

        if not faces:
            return []

        crops: List[np.ndarray] = []
        for face in faces:
            prob = face.get("confidence", 0.0)
            if prob < 0.85:
                continue  # Abaikan deteksi dengan confidence rendah
            x, y, w, h = face.get("box", [0, 0, 0, 0])
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1, x2, y2 = self._add_margin(x1, y1, x2, y2, img_w, img_h)
            crop = self._crop_and_resize(pil_img, (x1, y1, x2, y2))
            crops.append(crop)

        return crops

    # ------------------------------------------------------------------
    # Deteksi menggunakan OpenCV Haar Cascade
    # ------------------------------------------------------------------

    def _extract_opencv(
        self, pil_img: Image.Image
    ) -> List[np.ndarray]:
        """
        Deteksi semua wajah menggunakan OpenCV Haar Cascade.

        Returns
        -------
        list of np.ndarray
            Daftar crop wajah sebagai array numpy RGB uint8.
        """
        img_np = np.array(pil_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_h, img_w = gray.shape

        # Deteksi wajah
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return []

        crops: List[np.ndarray] = []
        for x, y, w, h in faces:
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1, x2, y2 = self._add_margin(x1, y1, x2, y2, img_w, img_h)
            crop = self._crop_and_resize(pil_img, (x1, y1, x2, y2))
            crops.append(crop)

        return crops

    # ------------------------------------------------------------------
    # API Publik
    # ------------------------------------------------------------------

    def extract(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Optional[np.ndarray]:
        """
        Ekstrak satu wajah terbesar dari gambar.

        Jika terdapat lebih dari satu wajah, wajah dengan area terbesar
        yang akan dikembalikan.

        Parameters
        ----------
        image : str | Path | np.ndarray | PIL.Image.Image
            Sumber gambar.

        Returns
        -------
        np.ndarray or None
            Array numpy RGB uint8 ukuran (image_size, image_size, 3),
            atau None jika tidak ada wajah terdeteksi.
        """
        try:
            pil_img = self._load_image(image)
        except Exception as exc:
            logger.error("Gagal memuat gambar: %s", exc)
            return None

        crops = self.extract_all(pil_img)

        if not crops:
            logger.debug("Tidak ada wajah yang terdeteksi.")
            return None

        if len(crops) == 1:
            return crops[0]

        # Pilih crop terbesar (semua sudah di-resize, jadi sama besar)
        # Gunakan variance tertinggi sebagai proxy wajah paling menonjol
        best_idx = int(
            np.argmax([np.var(c.astype(np.float32)) for c in crops])
        )
        return crops[best_idx]

    def extract_all(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> List[np.ndarray]:
        """
        Ekstrak semua wajah yang terdeteksi dari gambar.

        Parameters
        ----------
        image : str | Path | np.ndarray | PIL.Image.Image
            Sumber gambar.

        Returns
        -------
        list of np.ndarray
            Daftar crop wajah. Setiap elemen adalah array numpy RGB uint8
            ukuran (image_size, image_size, 3). List kosong jika tidak ada
            wajah terdeteksi.
        """
        try:
            pil_img = self._load_image(image)
        except Exception as exc:
            logger.error("Gagal memuat gambar: %s", exc)
            return []

        if self._backend == "mtcnn":
            crops = self._extract_mtcnn(pil_img)
        elif self._backend == "mtcnn_simple":
            crops = self._extract_mtcnn_simple(pil_img)
        else:
            crops = self._extract_opencv(pil_img)


        logger.debug(
            "Terdeteksi %d wajah menggunakan backend '%s'.",
            len(crops),
            self._backend,
        )
        return crops

    # ------------------------------------------------------------------
    # Representasi objek
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FaceExtractor(backend='{self._backend}', "
            f"device='{self.device}', "
            f"image_size={self.image_size}, "
            f"margin={self.margin})"
        )
