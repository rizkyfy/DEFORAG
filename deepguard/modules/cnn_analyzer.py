"""
cnn_analyzer.py
===============
Modul analisis deepfake berbasis Convolutional Neural Network (CNN).

Menggunakan arsitektur EfficientNet-B4 sebagai backbone dengan classifier
head khusus (DeepGuardModel). Model ini dilatih untuk membedakan gambar
asli (REAL) dari gambar hasil rekayasa deepfake (FAKE).

Alur analisis:
  1. Muat gambar atau video
  2. Ekstrak wajah menggunakan FaceExtractor
  3. Preprocessing dengan albumentations (resize + normalisasi ImageNet)
  4. Inferensi melalui DeepGuardModel
  5. Kembalikan skor, verdict, dan detail

Author  : DeepGuard Team
Version : 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import opsional: PyTorch + timm + albumentations
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info("PyTorch berhasil diimpor.")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch tidak tersedia. CNNAnalyzer tidak akan berfungsi.")

try:
    import timm
    TIMM_AVAILABLE = True
    logger.info("timm berhasil diimpor.")
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm tidak tersedia.")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
    logger.info("albumentations berhasil diimpor.")
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning("albumentations tidak tersedia. Akan menggunakan preprocessing manual.")

# ---------------------------------------------------------------------------
# Import utilitas DeepGuard
# ---------------------------------------------------------------------------
try:
    from deepguard.utils.face_extractor import FaceExtractor
    from deepguard.utils.video_processor import VideoProcessor
except ImportError:
    # Fallback untuk import relatif (saat dijalankan secara langsung)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deepguard.utils.face_extractor import FaceExtractor
    from deepguard.utils.video_processor import VideoProcessor


# ===========================================================================
# Arsitektur Model
# ===========================================================================

if TORCH_AVAILABLE and TIMM_AVAILABLE:

    class DeepGuardModel(nn.Module):
        """
        Model deepfake detection berbasis EfficientNet-B4.

        Backbone EfficientNet-B4 mengekstrak feature representasi gambar
        (1792-dim), kemudian classifier head mengkonversinya menjadi
        skor tunggal (logit).

        Output adalah logit mentah; terapkan sigmoid untuk mendapatkan
        probabilitas FAKE (0 = REAL, 1 = FAKE).
        """

        def __init__(self) -> None:
            super().__init__()
            # Backbone EfficientNet-B4 tanpa head klasifikasi bawaan
            self.backbone = timm.create_model(
                "efficientnet_b4",
                pretrained=False,
                num_classes=0,      # Hapus classifier head bawaan
                global_pool="avg",  # Global average pooling → (B, 1792)
            )
            # Classifier head khusus DeepGuard
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(1792, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Batch gambar shape (B, 3, H, W), ternormalisasi ImageNet.

            Returns
            -------
            torch.Tensor
                Logit shape (B,). Terapkan sigmoid untuk probabilitas FAKE.
            """
            features = self.backbone(x)          # (B, 1792)
            logits = self.classifier(features)   # (B, 1)
            return logits.squeeze(1)             # (B,)

else:
    # Stub jika PyTorch atau timm tidak tersedia
    class DeepGuardModel:  # type: ignore[no-redef]
        """Stub kelas jika PyTorch/timm tidak tersedia."""
        def __init__(self) -> None:
            raise ImportError("PyTorch dan timm diperlukan untuk DeepGuardModel.")


# ===========================================================================
# Preprocessing Pipeline
# ===========================================================================

def _build_transform(image_size: int = 224):
    """
    Buat pipeline preprocessing menggunakan albumentations.

    Jika albumentations tidak tersedia, kembalikan fungsi preprocessing manual.

    Parameters
    ----------
    image_size : int
        Ukuran target (persegi) untuk resize.

    Returns
    -------
    callable
        Transform yang dapat dipanggil pada numpy array (H, W, C) uint8.
    """
    if ALBUMENTATIONS_AVAILABLE:
        transform = A.Compose([
            A.Resize(image_size, image_size, interpolation=1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
        return transform
    else:
        # Preprocessing manual tanpa albumentations
        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        import cv2

        def manual_transform(image: np.ndarray) -> "torch.Tensor":
            img = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = (img - _mean) / _std
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (C, H, W)
            return img_tensor

        return manual_transform


# ===========================================================================
# CNNAnalyzer
# ===========================================================================

class CNNAnalyzer:
    """
    Analyzer deepfake berbasis CNN (EfficientNet-B4 + DeepGuardModel).

    Parameters
    ----------
    model_path : str | Path
        Path ke file bobot model (.pth atau .pt).
    config_path : str | Path
        Path ke file konfigurasi model dalam format JSON.
        Harus memiliki key ``threshold`` (float, default 0.4983).
    device : str, optional
        Perangkat komputasi. Jika None, akan dipilih secara otomatis
        (CUDA jika tersedia, selain itu CPU).

    Raises
    ------
    ImportError
        Jika PyTorch atau timm tidak tersedia.
    FileNotFoundError
        Jika model_path atau config_path tidak ditemukan.
    """

    # Nama modul untuk pelaporan
    MODULE_NAME = "CNN (EfficientNet-B4)"

    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch diperlukan untuk CNNAnalyzer. "
                "Install dengan: pip install torch torchvision"
            )
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm diperlukan untuk CNNAnalyzer. "
                "Install dengan: pip install timm"
            )

        self.model_path = Path(model_path)
        self.config_path = Path(config_path)

        # Tentukan device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Muat konfigurasi
        self.config = self._load_config()
        self.threshold: float = float(self.config.get("threshold", 0.4983))
        self.image_size: int = int(self.config.get("image_size", 224))

        # Inisialisasi utilitas
        self._face_extractor = FaceExtractor(
            device=self.device,
            image_size=self.image_size,
        )
        self._video_processor = VideoProcessor()
        self._transform = _build_transform(self.image_size)

        # Muat model
        self._model = self._load_model()

        logger.info(
            "CNNAnalyzer diinisialisasi: device=%s, threshold=%.4f",
            self.device, self.threshold,
        )

    # ------------------------------------------------------------------
    # Muat konfigurasi dan model
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        """
        Muat konfigurasi dari file JSON.

        Returns
        -------
        dict
            Konfigurasi model.
        """
        if not self.config_path.is_file():
            logger.warning(
                "File konfigurasi tidak ditemukan: %s. Menggunakan default.",
                self.config_path,
            )
            return {"threshold": 0.4983, "image_size": 224}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info("Konfigurasi dimuat dari: %s", self.config_path)
            return config
        except Exception as exc:
            logger.error("Gagal memuat konfigurasi: %s. Menggunakan default.", exc)
            return {"threshold": 0.4983, "image_size": 224}

    def _load_model(self) -> "DeepGuardModel":
        """
        Muat bobot model dari file .pth.

        Returns
        -------
        DeepGuardModel
            Model dalam mode evaluasi pada device yang ditentukan.

        Raises
        ------
        FileNotFoundError
            Jika file model tidak ditemukan.
        RuntimeError
            Jika gagal memuat bobot model.
        """
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"File model tidak ditemukan: {self.model_path}"
            )

        model = DeepGuardModel()

        try:
            state_dict = torch.load(
                str(self.model_path),
                map_location=self.device,
                weights_only=True,  # Lebih aman dari pickle injection
            )

            # Tangani kemungkinan dibungkus dalam dict
            if isinstance(state_dict, dict):
                if "model_state" in state_dict:
                    state_dict = state_dict["model_state"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

            model.load_state_dict(state_dict, strict=True)
            logger.info("Bobot model berhasil dimuat dari: %s", self.model_path)

        except Exception as exc:
            raise RuntimeError(
                f"Gagal memuat bobot model dari '{self.model_path}': {exc}"
            ) from exc

        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image_np: np.ndarray) -> "torch.Tensor":
        """
        Preprocessing gambar numpy menjadi tensor siap inferensi.

        Parameters
        ----------
        image_np : np.ndarray
            Gambar RGB uint8 shape (H, W, 3).

        Returns
        -------
        torch.Tensor
            Tensor shape (1, 3, image_size, image_size) pada device yang sesuai.
        """
        if ALBUMENTATIONS_AVAILABLE:
            transformed = self._transform(image=image_np)
            tensor = transformed["image"].unsqueeze(0)  # (1, C, H, W)
        else:
            tensor = self._transform(image_np).unsqueeze(0)  # (1, C, H, W)

        return tensor.float().to(self.device)

    # ------------------------------------------------------------------
    # Inferensi satu gambar
    # ------------------------------------------------------------------

    def _infer_single(self, image_np: np.ndarray) -> float:
        """
        Jalankan inferensi pada satu gambar dan kembalikan probabilitas FAKE.

        Parameters
        ----------
        image_np : np.ndarray
            Gambar RGB uint8.

        Returns
        -------
        float
            Probabilitas FAKE dalam rentang [0.0, 1.0].
        """
        tensor = self._preprocess(image_np)

        with torch.no_grad():
            logit = self._model(tensor)          # (1,)
            prob = torch.sigmoid(logit).item()   # Probabilitas FAKE

        return float(prob)

    # ------------------------------------------------------------------
    # Proses gambar tunggal
    # ------------------------------------------------------------------

    def _analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analisis satu file gambar.

        Parameters
        ----------
        image_path : Path
            Path ke file gambar.

        Returns
        -------
        dict
            Hasil analisis: score, verdict, confidence, details.
        """
        # Muat gambar
        pil_img = Image.open(str(image_path)).convert("RGB")
        img_np = np.array(pil_img)

        # Coba ekstrak wajah
        face_np = self._face_extractor.extract(img_np)
        face_found = face_np is not None

        if face_found:
            input_np = face_np
            logger.debug("Wajah terdeteksi; menganalisis crop wajah.")
        else:
            # Resize gambar penuh ke image_size
            import cv2
            input_np = cv2.resize(img_np, (self.image_size, self.image_size))
            logger.debug("Wajah tidak terdeteksi; menganalisis gambar penuh.")

        score = self._infer_single(input_np)

        return self._build_result(
            score=score,
            n_frames=1,
            face_found=face_found,
            source_type="image",
            source_path=str(image_path),
        )

    # ------------------------------------------------------------------
    # Proses video
    # ------------------------------------------------------------------

    def _analyze_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Analisis file video dengan mengambil beberapa frame.

        Score akhir adalah rata-rata dari semua frame yang berhasil dianalisis.

        Parameters
        ----------
        video_path : Path
            Path ke file video.

        Returns
        -------
        dict
            Hasil analisis: score, verdict, confidence, details.
        """
        n_frames_to_extract = int(self.config.get("video_frames", 15))

        frames = self._video_processor.extract_frames(
            video_path, n_frames=n_frames_to_extract
        )

        if not frames:
            logger.warning("Tidak ada frame berhasil diekstrak dari video.")
            return self._build_result(
                score=0.5,
                n_frames=0,
                face_found=False,
                source_type="video",
                source_path=str(video_path),
                error="Tidak ada frame yang dapat diekstrak",
            )

        scores: List[float] = []
        face_found_count = 0

        for i, pil_frame in enumerate(frames):
            try:
                frame_np = np.array(pil_frame)

                # Coba ekstrak wajah dari frame
                face_np = self._face_extractor.extract(frame_np)
                if face_np is not None:
                    input_np = face_np
                    face_found_count += 1
                else:
                    import cv2
                    input_np = cv2.resize(frame_np, (self.image_size, self.image_size))

                score = self._infer_single(input_np)
                scores.append(score)
                logger.debug("Frame %d/%d → skor: %.4f", i + 1, len(frames), score)

            except Exception as exc:
                logger.warning("Gagal memproses frame %d: %s", i, exc)
                continue

        if not scores:
            return self._build_result(
                score=0.5,
                n_frames=len(frames),
                face_found=False,
                source_type="video",
                source_path=str(video_path),
                error="Semua frame gagal diproses",
            )

        avg_score = float(np.mean(scores))
        face_found = face_found_count > 0

        return self._build_result(
            score=avg_score,
            n_frames=len(scores),
            face_found=face_found,
            source_type="video",
            source_path=str(video_path),
            frame_scores=scores,
        )

    # ------------------------------------------------------------------
    # Pembuatan hasil
    # ------------------------------------------------------------------

    def _build_result(
        self,
        score: float,
        n_frames: int,
        face_found: bool,
        source_type: str,
        source_path: str,
        frame_scores: Optional[List[float]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Bangun dictionary hasil analisis terstandarisasi.

        Parameters
        ----------
        score : float
            Skor kecurigaan 0–1 (mendekati 1 = FAKE).
        n_frames : int
            Jumlah frame yang diproses (1 untuk gambar).
        face_found : bool
            Apakah wajah berhasil terdeteksi.
        source_type : str
            'image' atau 'video'.
        source_path : str
            Path sumber file.
        frame_scores : list of float, optional
            Skor per frame (untuk video).
        error : str, optional
            Pesan error jika ada.

        Returns
        -------
        dict
            Hasil terstandarisasi dengan keys: score, verdict, confidence, details.
        """
        # Verdict berdasarkan threshold
        verdict = "FAKE" if score >= self.threshold else "REAL"

        # Confidence: seberapa jauh dari threshold (dinormalisasi ke 0–1)
        # Threshold bisa tidak tepat di 0.5, jadi kita normalisasi relatif terhadapnya
        if score >= self.threshold:
            confidence = min(1.0, (score - self.threshold) / (1.0 - self.threshold))
        else:
            confidence = min(1.0, (self.threshold - score) / self.threshold)

        result = {
            "score": round(score, 6),
            "verdict": verdict,
            "confidence": round(float(confidence), 6),
            "details": {
                "module": self.MODULE_NAME,
                "source_type": source_type,
                "source_path": source_path,
                "n_frames_processed": n_frames,
                "face_detected": face_found,
                "threshold_used": self.threshold,
                "device": self.device,
            },
        }

        if frame_scores is not None:
            result["details"]["frame_scores"] = [round(s, 4) for s in frame_scores]
            result["details"]["frame_score_std"] = round(float(np.std(frame_scores)), 4)

        if error is not None:
            result["details"]["error"] = error

        return result

    # ------------------------------------------------------------------
    # API Publik
    # ------------------------------------------------------------------

    def analyze(
        self, image_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Analisis gambar atau video untuk mendeteksi deepfake.

        Parameters
        ----------
        image_path : str | Path
            Path ke file gambar atau video yang akan dianalisis.

        Returns
        -------
        dict
            Hasil analisis berisi:
            - ``score``      : float [0–1], mendekati 1.0 = FAKE
            - ``verdict``    : str, 'FAKE' atau 'REAL'
            - ``confidence`` : float [0–1], tingkat keyakinan prediksi
            - ``details``    : dict, detail analisis tambahan

        Raises
        ------
        FileNotFoundError
            Jika file tidak ditemukan.
        ValueError
            Jika format file tidak didukung.
        """
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        logger.info("CNNAnalyzer memproses: %s", path.name)

        try:
            if self._video_processor.is_video(path):
                return self._analyze_video(path)
            elif self._video_processor.is_image(path):
                return self._analyze_image(path)
            else:
                raise ValueError(
                    f"Format file tidak didukung: '{path.suffix}'. "
                    "Gunakan gambar (jpg/png/...) atau video (mp4/avi/...)."
                )
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            logger.error("Error saat analisis CNN: %s", exc)
            return self._build_result(
                score=0.5,
                n_frames=0,
                face_found=False,
                source_type="unknown",
                source_path=str(path),
                error=str(exc),
            )

    def __repr__(self) -> str:
        return (
            f"CNNAnalyzer(model='{self.model_path.name}', "
            f"device='{self.device}', "
            f"threshold={self.threshold})"
        )

