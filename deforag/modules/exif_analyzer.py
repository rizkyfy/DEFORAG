"""
modules/exif_analyzer.py
------------------------
Modul analisis metadata EXIF untuk DEFORAG.
Memeriksa metadata gambar untuk mendeteksi tanda-tanda manipulasi atau pembuatan
menggunakan perangkat lunak AI/editing.

Prinsip scoring:
  - "Tidak ada EXIF" ≠ mencurigakan. Banyak gambar legitimate memang tidak
    punya EXIF: screenshot, gambar dari WhatsApp/social media (EXIF di-strip
    otomatis), PNG biasa, hasil export software desain, dll.
  - Skor dimulai dari NETRAL (0.5) untuk semua kasus.
  - Confidence diturunkan secara jujur ketika data tidak cukup untuk menyimpulkan.
  - "Tidak ada bukti manipulasi" ≠ "Ada bukti manipulasi".

Menggunakan pustaka PIL/Pillow untuk membaca EXIF.
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

# Daftar perangkat lunak yang mencurigakan
SUSPICIOUS_SOFTWARE = [
    "photoshop", "gimp", "stable diffusion", "midjourney", "dall-e",
    "dall·e", "adobe", "lightroom", "capture one", "affinity photo",
    "snapseed", "vsco", "faceapp", "facetune", "retouch", "meitu",
    "deepfake", "faceswap", "deepfacelab", "artbreeder", "runway",
    "novelai", "automatic1111", "comfyui", "invoke", "diffusion",
    "openart", "getimg", "nightcafe", "dreamstudio", "leonardo",
    "canva", "fotor", "pixlr", "remove.bg", "cleanup.pictures",
]

# Format yang secara umum TIDAK menyimpan EXIF — tidak perlu dicurigai
EXIF_EXEMPT_FORMATS = {"PNG", "WEBP", "BMP", "GIF", "TIFF", "SVG"}

# Daftar kamera yang dikenal dan terpercaya
KNOWN_CAMERA_MAKERS = [
    "canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
    "leica", "hasselblad", "pentax", "ricoh", "sigma", "apple",
    "samsung", "google", "huawei", "xiaomi", "motorola", "lg",
    "oneplus", "oppo", "vivo", "realme",
]

# Selisih waktu maksimum yang dianggap normal (dalam detik)
DATETIME_SUSPICIOUS_THRESHOLD_SECONDS = 86400  # 1 hari

# Format datetime EXIF standar
EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _decode_exif_value(value: Any) -> Any:
    """Mendekode nilai EXIF menjadi tipe Python yang dapat diserialisasi."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    if isinstance(value, tuple):
        return [_decode_exif_value(v) for v in value]
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            return float(value)
        except Exception:
            return str(value)
    return value


def _parse_exif_datetime(dt_string: str) -> datetime | None:
    """Mengurai string datetime format EXIF menjadi objek datetime."""
    try:
        return datetime.strptime(dt_string.strip(), EXIF_DATETIME_FORMAT)
    except (ValueError, AttributeError):
        return None


def _verdict_from_score(score: float) -> str:
    """Mengonversi skor numerik menjadi label verdict."""
    if score >= 0.70:
        return "FAKE"
    elif score >= 0.45:
        return "SUSPICIOUS"
    else:
        return "REAL"


def _extract_gps_info(gps_data: dict) -> dict[str, Any] | None:
    """Mengekstrak informasi GPS dari data EXIF GPS IFD."""
    try:
        gps_info: dict[str, Any] = {}
        for key, val in gps_data.items():
            tag_name = GPSTAGS.get(key, str(key))
            gps_info[tag_name] = _decode_exif_value(val)
        return gps_info
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Kelas utama
# ---------------------------------------------------------------------------

class EXIFAnalyzer:
    """
    Menganalisis metadata EXIF gambar untuk mendeteksi tanda-tanda manipulasi.

    Memeriksa:
    - Keberadaan metadata EXIF (dengan mempertimbangkan format file)
    - Software yang digunakan untuk membuat/mengedit gambar
    - Konsistensi tanggal/waktu
    - Informasi kamera
    - Konsistensi ukuran file vs resolusi
    - Perbedaan antara thumbnail dan gambar utama
    - Informasi GPS (jika tersedia)

    Output skor [0,1]:
      0.0 = metadata sangat bersih, kemungkinan besar foto asli
      1.0 = metadata sangat mencurigakan, kemungkinan besar palsu/diedit

    CATATAN PENTING:
      Tidak adanya EXIF BUKAN bukti manipulasi. Confidence diturunkan secara
      jujur ketika data tidak mencukupi untuk menyimpulkan sesuatu.
    """

    def analyze(self, image_path: str) -> dict[str, Any]:
        """
        Menganalisis metadata EXIF gambar.

        Args:
            image_path: Path ke file gambar.

        Returns:
            dict dengan kunci:
              - score      : float [0, 1]
              - verdict    : str  – 'REAL' | 'SUSPICIOUS' | 'FAKE'
              - confidence : float [0, 1]
              - details    : dict – metadata lengkap dan faktor risiko
        """
        result_template: dict[str, Any] = {
            "score": 0.5,
            "verdict": "SUSPICIOUS",
            "confidence": 0.2,  # default confidence sangat rendah bila error
            "details": {},
        }

        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"File tidak ditemukan: {image_path}")

            # --- Buka gambar ---
            img = Image.open(str(path))
            file_size_bytes = path.stat().st_size
            image_format = (img.format or "").upper()

            # Inisialisasi variabel pelacak
            risk_factors: list[str] = []
            green_flags: list[str] = []
            info_notes: list[str] = []   # catatan netral, tidak mempengaruhi skor
            all_exif: dict[str, Any] = {}
            score_adjustment = 0.0  # positif = lebih mencurigakan

            # ---------------------------------------------------------------
            # 1. Ekstrak data EXIF
            # ---------------------------------------------------------------
            raw_exif_dict: dict[int, Any] = {}

            try:
                raw_exif = img._getexif()  # type: ignore[attr-defined]
                if raw_exif:
                    raw_exif_dict = raw_exif
                else:
                    # Coba cara alternatif via getexif() (PIL >= 6.0)
                    exif_obj = img.getexif()
                    if exif_obj:
                        raw_exif_dict = dict(exif_obj)
            except (AttributeError, Exception):
                # Format seperti PNG tidak punya _getexif — ini normal
                try:
                    exif_obj = img.getexif()
                    if exif_obj:
                        raw_exif_dict = dict(exif_obj)
                except Exception:
                    pass

            # Decode semua tag EXIF
            for tag_id, value in raw_exif_dict.items():
                tag_name = TAGS.get(tag_id, f"Tag_{tag_id}")
                if tag_name == "GPSInfo" and isinstance(value, dict):
                    all_exif["GPSInfo"] = _extract_gps_info(value)
                else:
                    all_exif[tag_name] = _decode_exif_value(value)

            has_exif = bool(all_exif)
            is_exempt_format = image_format in EXIF_EXEMPT_FORMATS

            # ---------------------------------------------------------------
            # 2. Evaluasi kondisi EXIF
            # ---------------------------------------------------------------

            # --- KONDISI: Tidak ada EXIF ---
            if not has_exif:
                if is_exempt_format:
                    # PNG, WebP, BMP, GIF — memang jarang punya EXIF, NORMAL
                    green_flags.append(
                        f"Format {image_format} umumnya tidak menyimpan EXIF — "
                        "ini normal dan tidak mencurigakan."
                    )
                    # Tidak ada score_adjustment — benar-benar netral
                else:
                    # JPEG tanpa EXIF lebih jarang, tapi masih banyak penyebab legitimate:
                    # - Dibagikan via WhatsApp/Telegram/social media (strip EXIF otomatis)
                    # - Di-export dari software desain
                    # - Screenshot yang disave ulang sebagai JPEG
                    info_notes.append(
                        f"Format {image_format} tanpa EXIF — kemungkinan penyebab: "
                        "dibagikan via media sosial (strip EXIF otomatis), screenshot, "
                        "atau export dari software desain. Tidak cukup untuk menyimpulkan manipulasi."
                    )
                    score_adjustment += 0.08  # penalty kecil, jauh lebih rendah dari 0.35

            else:
                # --- Ada EXIF — evaluasi lebih lanjut ---

                # --- FLAG 1: Software mencurigakan ---
                software = str(all_exif.get("Software", "")).lower()
                software_display = all_exif.get("Software", "")
                if software:
                    matched_sus = next(
                        (sw for sw in SUSPICIOUS_SOFTWARE if sw in software), None
                    )
                    if matched_sus:
                        risk_factors.append(
                            f"Software mencurigakan terdeteksi di EXIF: '{software_display}' "
                            f"(cocok dengan pattern: '{matched_sus}')"
                        )
                        score_adjustment += 0.45
                    else:
                        info_notes.append(f"Software tercatat di EXIF: '{software_display}'")

                # --- FLAG 2: Inkonsistensi DateTime ---
                dt_original_str = all_exif.get("DateTimeOriginal", "")
                dt_digitized_str = all_exif.get("DateTimeDigitized", "")
                dt_modified_str = all_exif.get("DateTime", "")

                dt_original = _parse_exif_datetime(str(dt_original_str))
                dt_digitized = _parse_exif_datetime(str(dt_digitized_str))
                dt_modified = _parse_exif_datetime(str(dt_modified_str))

                if dt_original and dt_modified:
                    diff_seconds = abs((dt_modified - dt_original).total_seconds())
                    if diff_seconds > DATETIME_SUSPICIOUS_THRESHOLD_SECONDS:
                        risk_factors.append(
                            f"Selisih besar antara DateTimeOriginal dan DateTime: "
                            f"{diff_seconds / 3600:.1f} jam — kemungkinan diedit pasca-pengambilan."
                        )
                        score_adjustment += 0.20
                    else:
                        green_flags.append("Timestamp DateTimeOriginal dan DateTime konsisten.")

                if dt_original and dt_digitized:
                    diff2 = abs((dt_digitized - dt_original).total_seconds())
                    if diff2 > 60:
                        risk_factors.append(
                            f"DateTimeOriginal dan DateTimeDigitized tidak sinkron "
                            f"(selisih {diff2:.0f} detik)."
                        )
                        score_adjustment += 0.10

                # --- FLAG 3: Tidak ada kamera tapi ada software editing ---
                camera_make = str(all_exif.get("Make", "")).strip()
                camera_model = str(all_exif.get("Model", "")).strip()
                has_camera_info = bool(camera_make or camera_model)

                if not has_camera_info and software:
                    risk_factors.append(
                        "Tidak ada informasi kamera tetapi ada software tercatat — "
                        "kemungkinan gambar dibuat atau diedit sepenuhnya oleh komputer."
                    )
                    score_adjustment += 0.30

                # --- FLAG 4: Ukuran file vs resolusi ---
                try:
                    width, height = img.size
                    pixels = width * height
                    bits_per_pixel = (file_size_bytes * 8) / max(pixels, 1)
                    if bits_per_pixel < 0.3:
                        risk_factors.append(
                            f"Ukuran file sangat kecil untuk resolusinya "
                            f"({bits_per_pixel:.2f} bit/pixel) — mungkin di-recompress atau generated."
                        )
                        score_adjustment += 0.15
                    all_exif["_computed_bits_per_pixel"] = round(bits_per_pixel, 3)
                    all_exif["_image_resolution"] = f"{width}x{height}"
                    all_exif["_file_size_bytes"] = file_size_bytes
                except Exception:
                    pass

                # --- GREEN FLAG 1: Serial number kamera ---
                serial = (
                    all_exif.get("BodySerialNumber", "")
                    or all_exif.get("CameraSerialNumber", "")
                )
                if serial:
                    green_flags.append(
                        f"Serial number kamera ditemukan: {serial} — metadata tampak otentik."
                    )
                    score_adjustment -= 0.20

                # --- GREEN FLAG 2: Kamera yang dikenal ---
                if camera_make:
                    make_lower = camera_make.lower()
                    matched_maker = next(
                        (k for k in KNOWN_CAMERA_MAKERS if k in make_lower), None
                    )
                    if matched_maker:
                        green_flags.append(
                            f"Kamera dikenal terdeteksi: {camera_make} {camera_model}."
                        )
                        score_adjustment -= 0.15

                # --- GREEN FLAG 3: EXIF lengkap ---
                if len(all_exif) > 15:
                    green_flags.append(
                        f"Metadata EXIF lengkap ({len(all_exif)} field) — "
                        "menunjukkan pengambilan dari kamera nyata."
                    )
                    score_adjustment -= 0.10

                # --- GREEN FLAG 4: Semua timestamp konsisten ---
                if dt_original and dt_modified:
                    diff_ok = abs((dt_modified - dt_original).total_seconds())
                    if diff_ok < 10 and not any("Timestamp" in r for r in risk_factors):
                        green_flags.append("Semua timestamp EXIF konsisten.")
                        score_adjustment -= 0.05

            # ---------------------------------------------------------------
            # 3. Hitung skor final
            # ---------------------------------------------------------------

            # Selalu mulai dari NETRAL — tidak ada asumsi awal
            base_score = 0.5
            final_score = float(max(0.0, min(1.0, base_score + score_adjustment)))

            # ---------------------------------------------------------------
            # 4. Hitung confidence secara jujur
            # ---------------------------------------------------------------
            num_signals = len(risk_factors) + len(green_flags)

            if not has_exif:
                # Tanpa EXIF kita benar-benar tidak punya data yang cukup
                confidence = 0.20
            elif num_signals == 0:
                # Ada EXIF tapi tidak ada signal yang jelas
                confidence = 0.35
            elif num_signals <= 2:
                confidence = 0.50
            elif num_signals <= 5:
                confidence = 0.70
            else:
                confidence = 0.85

            verdict = _verdict_from_score(final_score)

            # Jika confidence sangat rendah dan skor mendekati netral, override verdict
            if confidence <= 0.25 and 0.40 <= final_score <= 0.60:
                verdict = "SUSPICIOUS"  # tidak cukup data untuk REAL atau FAKE

            return {
                "score": round(final_score, 4),
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "details": {
                    "has_exif": has_exif,
                    "image_format": image_format,
                    "exif_exempt_format": is_exempt_format,
                    "exif_field_count": len(all_exif),
                    "risk_factors": risk_factors,
                    "green_flags": green_flags,
                    "info_notes": info_notes,
                    "score_adjustment": round(score_adjustment, 4),
                    "all_exif": all_exif,
                    "file_info": {
                        "path": str(path),
                        "size_bytes": file_size_bytes,
                        "format": img.format,
                        "mode": img.mode,
                        "resolution": f"{img.width}x{img.height}",
                    },
                },
            }

        except FileNotFoundError as fnf:
            logger.error("File tidak ditemukan: %s", fnf)
            result_template["details"] = {"error": str(fnf)}
            return result_template

        except Exception as exc:
            logger.error(
                "Kesalahan tidak terduga pada EXIFAnalyzer: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            result_template["details"] = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            return result_template