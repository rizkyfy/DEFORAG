"""
report/pdf_generator.py
------------------------
Generator laporan PDF forensik untuk DeepGuard.
Menghasilkan laporan forensik digital profesional dalam Bahasa Indonesia
menggunakan pustaka ReportLab.

Struktur laporan:
  1. Header — judul dan identitas laporan
  2. Ringkasan Eksekutif — verdict, confidence, risk score
  3. Informasi Media — detail file dan EXIF
  4. Hasil Analisis Per Modul — tabel per modul
  5. Keputusan Ensemble — bobot dan skor gabungan
  6. Kesimpulan & Rekomendasi — paragraf formal
  7. Disclaimer — batasan analisis komputasional
  8. Footer — identitas sistem dan timestamp
"""

import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import ReportLab (wajib)
# ---------------------------------------------------------------------------
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        HRFlowable,
        Image as RLImage,
        PageBreak,
        PageTemplate,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.flowables import KeepTogether
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError as _rl_err:
    REPORTLAB_AVAILABLE = False
    logger.error(
        "ReportLab tidak terinstal. Instal dengan: pip install reportlab\nDetail: %s",
        _rl_err,
    )


# ---------------------------------------------------------------------------
# Konstanta warna
# ---------------------------------------------------------------------------
COLOR_FAKE       = "#DC2626"   # merah — FAKE / risiko tinggi
COLOR_SUSPICIOUS = "#F59E0B"   # oranye — SUSPICIOUS / risiko sedang
COLOR_REAL       = "#059669"   # hijau — REAL / risiko rendah
COLOR_HEADER_BG  = "#1E293B"   # biru gelap — header utama
COLOR_SECTION_BG = "#F1F5F9"   # abu-abu terang — background section
COLOR_TABLE_EVEN = "#F8FAFC"   # baris genap tabel
COLOR_TABLE_ODD  = "#FFFFFF"   # baris ganjil tabel
COLOR_BORDER     = "#CBD5E1"   # warna border tabel
COLOR_TEXT_DARK  = "#0F172A"   # teks utama
COLOR_TEXT_MUTED = "#64748B"   # teks sekunder

# Dimensi halaman A4
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 2.0 * cm


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str):
    """Mengonversi kode warna hex ke objek ReportLab Color."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return colors.Color(r, g, b)


def _verdict_color(verdict: str) -> str:
    """Mengembalikan kode warna hex berdasarkan verdict."""
    verdict_upper = str(verdict).upper()
    if verdict_upper in ("FAKE", "PALSU"):
        return COLOR_FAKE
    elif verdict_upper in ("SUSPICIOUS", "MENCURIGAKAN"):
        return COLOR_SUSPICIOUS
    else:
        return COLOR_REAL


def _score_color(score: float) -> str:
    """Mengembalikan kode warna hex berdasarkan skor numerik."""
    if score >= 0.70:
        return COLOR_FAKE
    elif score >= 0.45:
        return COLOR_SUSPICIOUS
    else:
        return COLOR_REAL


def _format_verdict_id(verdict: str) -> str:
    """Menerjemahkan verdict ke Bahasa Indonesia."""
    mapping = {
        "FAKE": "PALSU / DEEPFAKE",
        "SUSPICIOUS": "MENCURIGAKAN",
        "REAL": "ASLI",
    }
    return mapping.get(str(verdict).upper(), str(verdict))


def _format_module_name(module_key: str) -> str:
    """Menerjemahkan nama modul ke label Bahasa Indonesia."""
    mapping = {
        "cnn":             "Analisis CNN (Deep Learning)",
        "frequency":       "Analisis Domain Frekuensi",
        "landmark":        "Analisis Landmark Wajah",
        "gan_fingerprint": "Analisis Sidik Jari GAN",
        "exif":            "Analisis Metadata EXIF",
        "texture":         "Analisis Tekstur",
        "temporal":        "Analisis Temporal",
        "eye_reflection":  "Analisis Refleksi Mata",
        "face_blending":   "Analisis Blending Batas Wajah",
        "skin_texture":    "Analisis Tekstur Kulit Wajah",
    }
    return mapping.get(module_key, module_key.replace("_", " ").title())


def _format_score_percent(score: float | None) -> str:
    """Memformat skor sebagai persentase."""
    if score is None:
        return "N/A"
    return f"{float(score) * 100:.1f}%"


def _generate_report_number() -> str:
    """Menghasilkan nomor laporan unik berdasarkan timestamp."""
    now = datetime.now()
    return f"DG-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}"


def _format_file_size(size_bytes: int | None) -> str:
    """Memformat ukuran file ke format yang mudah dibaca."""
    if size_bytes is None:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Style factory
# ---------------------------------------------------------------------------

def _build_styles() -> dict[str, ParagraphStyle]:
    """Membangun kumpulan style ParagraphStyle yang digunakan dalam laporan."""
    base = getSampleStyleSheet()

    styles: dict[str, ParagraphStyle] = {}

    # Judul utama
    styles["main_title"] = ParagraphStyle(
        "main_title",
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.white,
        alignment=TA_CENTER,
        spaceAfter=4,
        leading=20,
    )

    # Subjudul header
    styles["header_subtitle"] = ParagraphStyle(
        "header_subtitle",
        fontName="Helvetica",
        fontSize=9,
        textColor=_hex_to_rgb("#94A3B8"),
        alignment=TA_CENTER,
        spaceAfter=2,
    )

    # Judul section
    styles["section_title"] = ParagraphStyle(
        "section_title",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=_hex_to_rgb(COLOR_TEXT_DARK),
        spaceBefore=14,
        spaceAfter=6,
        leading=16,
    )

    # Body normal
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9,
        textColor=_hex_to_rgb(COLOR_TEXT_DARK),
        alignment=TA_LEFT,
        spaceAfter=4,
        leading=13,
    )

    # Body justify
    styles["body_justify"] = ParagraphStyle(
        "body_justify",
        fontName="Helvetica",
        fontSize=9,
        textColor=_hex_to_rgb(COLOR_TEXT_DARK),
        alignment=TA_JUSTIFY,
        spaceAfter=4,
        leading=13,
    )

    # Teks muted/sekunder
    styles["muted"] = ParagraphStyle(
        "muted",
        fontName="Helvetica",
        fontSize=8,
        textColor=_hex_to_rgb(COLOR_TEXT_MUTED),
        spaceAfter=2,
        leading=11,
    )

    # Verdict besar
    styles["verdict_big"] = ParagraphStyle(
        "verdict_big",
        fontName="Helvetica-Bold",
        fontSize=22,
        alignment=TA_CENTER,
        spaceAfter=8,
        leading=28,
    )

    # Label key-value
    styles["kv_key"] = ParagraphStyle(
        "kv_key",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=_hex_to_rgb(COLOR_TEXT_DARK),
        spaceAfter=2,
    )

    styles["kv_value"] = ParagraphStyle(
        "kv_value",
        fontName="Helvetica",
        fontSize=9,
        textColor=_hex_to_rgb(COLOR_TEXT_MUTED),
        spaceAfter=2,
    )

    # Footer
    styles["footer"] = ParagraphStyle(
        "footer",
        fontName="Helvetica",
        fontSize=7,
        textColor=_hex_to_rgb(COLOR_TEXT_MUTED),
        alignment=TA_CENTER,
    )

    # Disclaimer
    styles["disclaimer"] = ParagraphStyle(
        "disclaimer",
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=_hex_to_rgb(COLOR_TEXT_MUTED),
        alignment=TA_JUSTIFY,
        spaceAfter=4,
        leading=12,
    )

    return styles


# ---------------------------------------------------------------------------
# Komponen visual khusus
# ---------------------------------------------------------------------------

def _build_score_bar_table(score: float, width: float = 12 * cm) -> Table:
    """
    Membangun tabel yang menampilkan progress bar skor risiko.

    Args:
        score: Skor [0, 1]
        width: Lebar total bar dalam points ReportLab

    Returns:
        Objek Table yang siap ditambahkan ke dokumen
    """
    bar_color = _hex_to_rgb(_score_color(score))
    bg_color = _hex_to_rgb("#E2E8F0")

    filled_width = max(2, score * width)
    empty_width = max(0, width - filled_width)

    # Label skor
    score_pct = f"{score * 100:.1f}%"

    # Gunakan tabel dua-sel sebagai bar
    data = [
        [
            Paragraph(
                f'<font color="{_score_color(score)}" size="11"><b>Skor Risiko: {score_pct}</b></font>',
                ParagraphStyle(
                    "score_label",
                    fontName="Helvetica-Bold",
                    fontSize=11,
                    alignment=TA_LEFT,
                )
            ),
        ]
    ]

    score_table = Table(data, colWidths=[width])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _hex_to_rgb(COLOR_SECTION_BG)),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_hex_to_rgb(COLOR_SECTION_BG)]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, _hex_to_rgb(COLOR_BORDER)),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return score_table


def _build_module_table(
    module_key: str,
    result: dict[str, Any] | None,
    styles: dict[str, ParagraphStyle],
    col_widths: list[float],
) -> Table:
    """
    Membangun tabel hasil analisis untuk satu modul.

    Args:
        module_key: Identifier modul (e.g., 'cnn', 'exif')
        result:     Dict hasil analisis modul, atau None jika tidak tersedia
        styles:     Kumpulan style paragraph
        col_widths: Lebar kolom tabel

    Returns:
        Objek Table
    """
    module_name = _format_module_name(module_key)

    if result is None:
        data = [
            [Paragraph(f"<b>{module_name}</b>", styles["body"]),
             Paragraph("Tidak Tersedia", styles["muted"]),
             Paragraph("—", styles["muted"]),
             Paragraph("—", styles["muted"])],
        ]
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _hex_to_rgb(COLOR_TABLE_EVEN)),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, -1), _hex_to_rgb(COLOR_TEXT_MUTED)),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, _hex_to_rgb(COLOR_BORDER)),
        ]))
        return table

    score = float(result.get("score", 0.5))
    verdict = str(result.get("verdict", "N/A"))
    confidence = float(result.get("confidence", 0.0))
    verdict_id = _format_verdict_id(verdict)
    verdict_color = _verdict_color(verdict)

    data = [
        [
            Paragraph(f"<b>{module_name}</b>", styles["body"]),
            Paragraph(
                f'<font color="{verdict_color}"><b>{verdict_id}</b></font>',
                styles["body"]
            ),
            Paragraph(f"<b>{_format_score_percent(score)}</b>", styles["body"]),
            Paragraph(_format_score_percent(confidence), styles["muted"]),
        ]
    ]

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _hex_to_rgb(COLOR_TABLE_ODD)),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, _hex_to_rgb(COLOR_BORDER)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


# ---------------------------------------------------------------------------
# Kelas utama
# ---------------------------------------------------------------------------

class ForensicReportGenerator:
    """
    Menghasilkan laporan forensik digital PDF yang profesional untuk DeepGuard.

    Laporan mencakup:
    - Header profesional dengan nomor laporan
    - Ringkasan eksekutif dengan verdict visual
    - Informasi detail media yang dianalisis
    - Tabel hasil per modul analisis
    - Keputusan ensemble dengan bobot kontribusi
    - Kesimpulan dan rekomendasi formal
    - Disclaimer dan footer

    Persyaratan:
    - Python 3.10+
    - reportlab >= 3.6.0
    """

    def __init__(self) -> None:
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab diperlukan untuk generate laporan PDF. "
                "Instal dengan: pip install reportlab"
            )
        self._styles = _build_styles()

    # ------------------------------------------------------------------
    def generate(
        self,
        analysis_results: dict[str, Any],
        media_path: str,
        output_dir: str,
    ) -> str:
        """
        Menghasilkan laporan forensik PDF.

        Args:
            analysis_results: Dict hasil analisis gabungan dengan struktur:
                {
                    "ensemble": {final_score, verdict, confidence, risk_level,
                                 module_contributions, explanation, ...},
                    "modules": {
                        "cnn": {score, verdict, confidence, details},
                        "frequency": {...},
                        "landmark": {...},
                        "gan_fingerprint": {...},
                        "exif": {...},
                        "texture": {...},
                        "temporal": {...},
                    }
                }
            media_path: Path ke file media yang dianalisis.
            output_dir: Direktori output untuk file PDF.

        Returns:
            Path absolut ke file PDF yang dihasilkan.

        Raises:
            ImportError: Jika ReportLab tidak terinstal.
            Exception:   Jika terjadi kesalahan saat generate PDF.
        """
        try:
            # --- Persiapan path output ---
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"deepguard_report_{timestamp}.pdf"
            output_path = output_dir_path / output_filename

            # --- Inisialisasi dokumen ---
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=MARGIN,
                leftMargin=MARGIN,
                topMargin=MARGIN,
                bottomMargin=MARGIN,
                title="Laporan Forensik Digital - Analisis Deepfake",
                author="DeepGuard v1.0",
                subject="Forensic Analysis Report",
            )

            # --- Kumpulkan semua element ---
            story: list[Any] = []

            # Ekstrak data
            ensemble = analysis_results.get("ensemble", {})
            modules = analysis_results.get("modules", {})
            report_number = _generate_report_number()
            now = datetime.now()

            # ---- SECTION 1: HEADER ----
            self._add_header(story, report_number, now)

            # ---- SECTION 2: RINGKASAN EKSEKUTIF ----
            self._add_executive_summary(story, ensemble, now)

            # ---- SECTION 3: INFORMASI MEDIA ----
            self._add_media_info(story, media_path, modules.get("exif"))

            # ---- SECTION 4: HASIL ANALISIS PER MODUL ----
            self._add_module_results(story, modules)

            # ---- SECTION 5: KEPUTUSAN ENSEMBLE ----
            self._add_ensemble_decision(story, ensemble)

            # ---- SECTION 6: KESIMPULAN & REKOMENDASI ----
            self._add_conclusion(story, ensemble, media_path)

            # ---- SECTION 7: DISCLAIMER ----
            self._add_disclaimer(story)

            # ---- SECTION 8: FOOTER ----
            self._add_footer(story, report_number, now)

            # --- Build dokumen ---
            doc.build(story)

            logger.info("Laporan PDF berhasil dibuat: %s", output_path)
            return str(output_path)

        except Exception as exc:
            logger.error(
                "Gagal generate laporan PDF: %s\n%s",
                exc,
                traceback.format_exc(),
            )
            raise

    # ------------------------------------------------------------------
    # --- Section builders ---
    # ------------------------------------------------------------------

    def _add_header(
        self,
        story: list,
        report_number: str,
        now: datetime,
    ) -> None:
        """Menambahkan header laporan."""
        styles = self._styles

        # Background header menggunakan tabel satu baris
        header_content = [
            [
                Paragraph(
                    "LAPORAN FORENSIK DIGITAL",
                    styles["main_title"],
                ),
            ],
            [
                Paragraph(
                    "ANALISIS DEEPFAKE &amp; MANIPULASI MEDIA DIGITAL",
                    styles["header_subtitle"],
                ),
            ],
            [
                Paragraph(
                    f"DeepGuard v1.0 &nbsp;|&nbsp; No. Laporan: {report_number} "
                    f"&nbsp;|&nbsp; {now.strftime('%d %B %Y, %H:%M WIB')}",
                    styles["header_subtitle"],
                ),
            ],
        ]

        header_table = Table(
            header_content,
            colWidths=[PAGE_WIDTH - 2 * MARGIN],
            rowHeights=[28, 18, 16],
        )
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _hex_to_rgb(COLOR_HEADER_BG)),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_hex_to_rgb(COLOR_HEADER_BG)] * 3),
        ]))

        story.append(header_table)
        story.append(Spacer(1, 0.5 * cm))

    def _add_executive_summary(
        self,
        story: list,
        ensemble: dict[str, Any],
        now: datetime,
    ) -> None:
        """Menambahkan ringkasan eksekutif."""
        styles = self._styles
        story.append(
            Paragraph("1. RINGKASAN EKSEKUTIF", styles["section_title"])
        )
        story.append(HRFlowable(width="100%", thickness=1, color=_hex_to_rgb(COLOR_BORDER)))
        story.append(Spacer(1, 0.3 * cm))

        final_score = float(ensemble.get("final_score", 0.5))
        verdict = str(ensemble.get("verdict", "SUSPICIOUS"))
        confidence = float(ensemble.get("confidence", 0.0))
        risk_level = str(ensemble.get("risk_level", "SEDANG"))
        verdict_id = _format_verdict_id(verdict)
        v_color = _verdict_color(verdict)

        # Tabel ringkasan: verdict besar + info samping
        verdict_para = Paragraph(
            f'<font color="{v_color}" size="20"><b>● {verdict_id}</b></font>',
            ParagraphStyle(
                "vp",
                fontName="Helvetica-Bold",
                fontSize=18,
                alignment=TA_CENTER,
                leading=26,
            ),
        )

        info_data = [
            [Paragraph("<b>Skor Risiko Gabungan:</b>", styles["kv_key"]),
             Paragraph(
                 f'<font color="{_score_color(final_score)}"><b>{_format_score_percent(final_score)}</b></font>',
                 styles["body"]
             )],
            [Paragraph("<b>Tingkat Kepercayaan:</b>", styles["kv_key"]),
             Paragraph(_format_score_percent(confidence), styles["body"])],
            [Paragraph("<b>Level Risiko:</b>", styles["kv_key"]),
             Paragraph(
                 f'<font color="{_score_color(final_score)}"><b>{risk_level}</b></font>',
                 styles["body"]
             )],
            [Paragraph("<b>Tanggal Analisis:</b>", styles["kv_key"]),
             Paragraph(now.strftime("%d %B %Y, %H:%M:%S"), styles["body"])],
        ]

        info_table = Table(info_data, colWidths=[5 * cm, 6 * cm])
        info_table.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))

        combined_data = [
            [verdict_para, info_table],
        ]
        combined_table = Table(
            combined_data,
            colWidths=[8 * cm, 12 * cm],
        )
        combined_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), _hex_to_rgb(COLOR_SECTION_BG)),
            ("BACKGROUND", (1, 0), (1, 0), colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 16),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
            ("LEFTPADDING", (0, 0), (0, 0), 10),
            ("RIGHTPADDING", (0, 0), (0, 0), 10),
            ("LEFTPADDING", (1, 0), (1, 0), 16),
            ("BOX", (0, 0), (-1, -1), 1, _hex_to_rgb(COLOR_BORDER)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, _hex_to_rgb(COLOR_BORDER)),
        ]))
        story.append(combined_table)

        # Score bar
        story.append(Spacer(1, 0.3 * cm))
        story.append(_build_score_bar_table(final_score, width=PAGE_WIDTH - 2 * MARGIN))
        story.append(Spacer(1, 0.2 * cm))

    def _add_media_info(
        self,
        story: list,
        media_path: str,
        exif_result: dict[str, Any] | None,
    ) -> None:
        """Menambahkan informasi media yang dianalisis."""
        styles = self._styles
        story.append(
            Paragraph("2. INFORMASI MEDIA YANG DIANALISIS", styles["section_title"])
        )
        story.append(HRFlowable(width="100%", thickness=1, color=_hex_to_rgb(COLOR_BORDER)))
        story.append(Spacer(1, 0.2 * cm))

        path = Path(media_path)

        # Info dasar file
        try:
            file_size = path.stat().st_size if path.exists() else None
        except Exception:
            file_size = None

        # Coba dapatkan dimensi dari exif_result
        resolution = "N/A"
        file_format = path.suffix.upper().lstrip(".")
        if exif_result:
            details = exif_result.get("details", {})
            file_info = details.get("file_info", {})
            resolution = file_info.get("resolution", "N/A")
            file_format = file_info.get("format", file_format) or file_format

        media_data = [
            ["Nama File", path.name],
            ["Path", str(path)],
            ["Format", file_format or "N/A"],
            ["Ukuran File", _format_file_size(file_size)],
            ["Resolusi", resolution],
            ["Ada EXIF", "Ya" if (exif_result and exif_result.get("details", {}).get("has_exif")) else "Tidak"],
        ]

        # Tambahkan beberapa field EXIF jika ada
        if exif_result:
            details = exif_result.get("details", {})
            all_exif = details.get("all_exif", {})
            for exif_key in ["Make", "Model", "DateTimeOriginal", "Software", "GPSInfo"]:
                val = all_exif.get(exif_key)
                if val:
                    val_str = str(val)[:80] + ("..." if len(str(val)) > 80 else "")
                    media_data.append([exif_key, val_str])

        # Render tabel
        table_data = []
        for i, (key, val) in enumerate(media_data):
            bg = _hex_to_rgb(COLOR_TABLE_EVEN if i % 2 == 0 else COLOR_TABLE_ODD)
            table_data.append([
                Paragraph(f"<b>{key}</b>", styles["body"]),
                Paragraph(str(val), styles["body"]),
            ])

        media_table = Table(table_data, colWidths=[5 * cm, PAGE_WIDTH - 2 * MARGIN - 5 * cm])
        media_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _hex_to_rgb(COLOR_TABLE_ODD)),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [
                _hex_to_rgb(COLOR_TABLE_EVEN), _hex_to_rgb(COLOR_TABLE_ODD)
            ]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, -1), 1, _hex_to_rgb(COLOR_BORDER)),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, _hex_to_rgb(COLOR_BORDER)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(media_table)
        story.append(Spacer(1, 0.2 * cm))

    def _add_module_results(
        self,
        story: list,
        modules: dict[str, Any],
    ) -> None:
        """Menambahkan tabel hasil analisis per modul."""
        styles = self._styles
        story.append(
            Paragraph("3. HASIL ANALISIS PER MODUL", styles["section_title"])
        )
        story.append(HRFlowable(width="100%", thickness=1, color=_hex_to_rgb(COLOR_BORDER)))
        story.append(Spacer(1, 0.2 * cm))

        # Header tabel
        col_widths = [6 * cm, 4 * cm, 3 * cm, 3 * cm]
        header_data = [[
            Paragraph("<b>Modul Analisis</b>", styles["kv_key"]),
            Paragraph("<b>Verdict</b>", styles["kv_key"]),
            Paragraph("<b>Skor Risiko</b>", styles["kv_key"]),
            Paragraph("<b>Kepercayaan</b>", styles["kv_key"]),
        ]]
        header_table = Table(header_data, colWidths=col_widths)
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _hex_to_rgb(COLOR_HEADER_BG)),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(header_table)

        # Baris per modul
        module_order = ["cnn", "frequency", "landmark", "gan_fingerprint",
                        "exif", "texture", "temporal", "eye_reflection",
                        "face_blending", "skin_texture"]
        
        temporal_res = modules.get("temporal")
        if temporal_res and temporal_res.get("verdict") == "N/A":
            if "temporal" in module_order:
                module_order.remove("temporal")
                
        for mod_key in module_order:
            result = modules.get(mod_key)
            mod_table = _build_module_table(mod_key, result, styles, col_widths)
            story.append(mod_table)

        # Detail per modul dengan informasi tambahan
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            Paragraph("<b>Detail Analisis Penting:</b>", styles["body"])
        )

        detail_modules_shown = 0
        for mod_key in module_order:
            result = modules.get(mod_key)
            if not result:
                continue
            details = result.get("details", {})
            if not details or "error" in details:
                continue

            mod_name = _format_module_name(mod_key)
            score = float(result.get("score", 0.5))

            # Tampilkan hanya detail dari modul dengan skor tinggi atau menarik
            risk_factors = details.get("risk_factors", [])
            green_flags = details.get("green_flags", [])
            sub_scores = details.get("sub_scores", {})

            lines: list[str] = []
            if risk_factors:
                lines.append(f"<b>⚠ Faktor Risiko {mod_name}:</b>")
                for rf in risk_factors[:3]:
                    lines.append(f"  • {rf}")
            if green_flags:
                lines.append(f"<b>✓ Indikasi Positif {mod_name}:</b>")
                for gf in green_flags[:2]:
                    lines.append(f"  • {gf}")
            if sub_scores and score > 0.4:
                sub_str = ", ".join(
                    f"{k}: {v*100:.1f}%" for k, v in list(sub_scores.items())[:4]
                )
                lines.append(f"<b>Sub-skor {mod_name}:</b> {sub_str}")

            if lines:
                detail_text = "<br/>".join(lines)
                story.append(
                    Paragraph(detail_text, styles["muted"])
                )
                detail_modules_shown += 1

        if detail_modules_shown == 0:
            story.append(
                Paragraph(
                    "Tidak ada detail tambahan yang signifikan dari modul-modul analisis.",
                    styles["muted"],
                )
            )

        story.append(Spacer(1, 0.2 * cm))

    def _add_ensemble_decision(
        self,
        story: list,
        ensemble: dict[str, Any],
    ) -> None:
        """Menambahkan section keputusan ensemble."""
        styles = self._styles
        story.append(
            Paragraph("4. KEPUTUSAN ENSEMBLE", styles["section_title"])
        )
        story.append(HRFlowable(width="100%", thickness=1, color=_hex_to_rgb(COLOR_BORDER)))
        story.append(Spacer(1, 0.2 * cm))

        contributions: list[dict] = ensemble.get("module_contributions", [])
        weight_used: dict = ensemble.get("weight_used", {})
        explanation = str(ensemble.get("explanation", ""))

        # Tabel kontribusi modul
        if contributions:
            contrib_header = [[
                Paragraph("<b>Modul</b>", styles["kv_key"]),
                Paragraph("<b>Bobot</b>", styles["kv_key"]),
                Paragraph("<b>Skor Mentah</b>", styles["kv_key"]),
                Paragraph("<b>Kontribusi</b>", styles["kv_key"]),
                Paragraph("<b>Verdict Modul</b>", styles["kv_key"]),
            ]]

            col_w = [5.5 * cm, 2.5 * cm, 3 * cm, 3 * cm, 3.5 * cm]
            contrib_table_data = contrib_header[:]

            for i, contrib in enumerate(contributions):
                bg = COLOR_TABLE_EVEN if i % 2 == 0 else COLOR_TABLE_ODD
                raw_score = float(contrib.get("raw_score", 0))
                mod_verdict = str(contrib.get("module_verdict", "N/A"))
                v_color = _verdict_color(mod_verdict)

                contrib_table_data.append([
                    Paragraph(_format_module_name(str(contrib.get("module", ""))), styles["body"]),
                    Paragraph(f"{float(contrib.get('weight', 0)) * 100:.1f}%", styles["body"]),
                    Paragraph(
                        f'<font color="{_score_color(raw_score)}">'
                        f'{_format_score_percent(raw_score)}</font>',
                        styles["body"]
                    ),
                    Paragraph(
                        f'<font color="{_score_color(raw_score)}">'
                        f'{_format_score_percent(float(contrib.get("weighted_contribution", 0)))}'
                        f'</font>',
                        styles["body"]
                    ),
                    Paragraph(
                        f'<font color="{v_color}"><b>{_format_verdict_id(mod_verdict)}</b></font>',
                        styles["body"]
                    ),
                ])

            contrib_table = Table(contrib_table_data, colWidths=col_w)
            contrib_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _hex_to_rgb(COLOR_HEADER_BG)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    _hex_to_rgb(COLOR_TABLE_EVEN), _hex_to_rgb(COLOR_TABLE_ODD)
                ]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("BOX", (0, 0), (-1, -1), 1, _hex_to_rgb(COLOR_BORDER)),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, _hex_to_rgb(COLOR_BORDER)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(contrib_table)
        else:
            story.append(
                Paragraph(
                    "Data kontribusi modul tidak tersedia.",
                    styles["muted"],
                )
            )

        # Penjelasan ensemble
        if explanation:
            story.append(Spacer(1, 0.25 * cm))
            story.append(Paragraph("<b>Penjelasan Keputusan:</b>", styles["body"]))
            story.append(Paragraph(explanation, styles["body_justify"]))

        story.append(Spacer(1, 0.2 * cm))

    def _add_conclusion(
        self,
        story: list,
        ensemble: dict[str, Any],
        media_path: str,
    ) -> None:
        """Menambahkan section kesimpulan dan rekomendasi."""
        styles = self._styles
        story.append(
            Paragraph("5. KESIMPULAN DAN REKOMENDASI", styles["section_title"])
        )
        story.append(HRFlowable(width="100%", thickness=1, color=_hex_to_rgb(COLOR_BORDER)))
        story.append(Spacer(1, 0.2 * cm))

        final_score = float(ensemble.get("final_score", 0.5))
        verdict = str(ensemble.get("verdict", "SUSPICIOUS"))
        confidence = float(ensemble.get("confidence", 0.0))
        risk_level = str(ensemble.get("risk_level", "SEDANG"))
        verdict_id = _format_verdict_id(verdict)
        file_name = Path(media_path).name

        # Kesimpulan berdasarkan verdict
        if verdict.upper() == "FAKE":
            conclusion_text = (
                f"Berdasarkan analisis komputasional yang dilakukan oleh sistem DeepGuard v1.0 "
                f"terhadap berkas media <i>{file_name}</i>, diperoleh kesimpulan bahwa media "
                f"tersebut mengandung indikasi kuat sebagai konten palsu atau hasil manipulasi "
                f"digital (deepfake). Skor risiko gabungan sebesar {_format_score_percent(final_score)} "
                f"dengan tingkat kepercayaan {_format_score_percent(confidence)} menempatkan media "
                f"ini pada kategori risiko <b>{risk_level}</b>. "
                f"Sistem mendeteksi adanya anomali pada satu atau lebih aspek teknis yang "
                f"mengindikasikan penggunaan teknologi generatif AI atau face-swap."
            )
            recommendations = [
                "Lakukan investigasi forensik manual oleh pakar forensik digital bersertifikat.",
                "Jangan gunakan media ini sebagai bukti tanpa verifikasi lebih lanjut.",
                "Laporkan ke pihak berwenang jika media ini digunakan untuk penipuan atau disinformasi.",
                "Simpan media asli dan laporan ini sebagai bagian dari chain of custody.",
                "Pertimbangkan analisis dengan tool forensik tambahan (e.g., FotoForensics, Amped).",
            ]
        elif verdict.upper() == "SUSPICIOUS":
            conclusion_text = (
                f"Analisis komputasional DeepGuard v1.0 terhadap berkas <i>{file_name}</i> "
                f"menghasilkan skor risiko {_format_score_percent(final_score)} dengan tingkat "
                f"kepercayaan {_format_score_percent(confidence)}. Media ini dikategorikan sebagai "
                f"<b>MENCURIGAKAN</b> dengan level risiko <b>{risk_level}</b>. "
                f"Ditemukan beberapa anomali teknis yang tidak dapat diabaikan, namun tidak cukup "
                f"kuat untuk memastikan status PALSU secara definitif. Media ini mungkin merupakan "
                f"foto asli yang telah melalui proses editing ringan, atau merupakan deepfake dengan "
                f"kualitas tinggi yang sulit terdeteksi."
            )
            recommendations = [
                "Verifikasi sumber asli media melalui pencarian gambar terbalik (reverse image search).",
                "Analisis metadata tambahan menggunakan tool forensik profesional.",
                "Jika digunakan dalam konteks hukum, wajib mendapatkan second opinion dari ahli.",
                "Perhatikan konteks di mana media ini disebarkan.",
                "Pertimbangkan analisis lebih lanjut dengan model deepfake detection yang lebih spesifik.",
            ]
        else:
            conclusion_text = (
                f"Hasil analisis komputasional DeepGuard v1.0 terhadap berkas <i>{file_name}</i> "
                f"menunjukkan bahwa media ini kemungkinan besar merupakan konten <b>ASLI</b>. "
                f"Skor risiko {_format_score_percent(final_score)} dengan tingkat kepercayaan "
                f"{_format_score_percent(confidence)} menempatkan media pada kategori risiko "
                f"<b>{risk_level}</b>. "
                f"Tidak ditemukan artefak teknis yang signifikan mengindikasikan manipulasi digital."
            )
            recommendations = [
                "Media dapat digunakan dengan keyakinan moderat sebagai konten yang tidak dimanipulasi.",
                "Tetap lakukan verifikasi konteks dan sumber media sebelum menggunakannya.",
                "Perhatikan bahwa tidak ada sistem deteksi yang 100% akurat.",
                "Untuk kepentingan hukum/resmi, tetap diperlukan analisis forensik profesional.",
            ]

        story.append(Paragraph(conclusion_text, styles["body_justify"]))
        story.append(Spacer(1, 0.25 * cm))
        story.append(Paragraph("<b>Rekomendasi Tindak Lanjut:</b>", styles["body"]))

        for i, rec in enumerate(recommendations, 1):
            story.append(
                Paragraph(f"{i}. {rec}", styles["body"])
            )

        story.append(Spacer(1, 0.2 * cm))

    def _add_disclaimer(self, story: list) -> None:
        """Menambahkan section disclaimer."""
        styles = self._styles
        story.append(
            Paragraph("6. DISCLAIMER", styles["section_title"])
        )
        story.append(HRFlowable(width="100%", thickness=1, color=_hex_to_rgb(COLOR_BORDER)))
        story.append(Spacer(1, 0.2 * cm))

        disclaimer_text = (
            "Laporan ini dihasilkan secara otomatis oleh sistem DeepGuard v1.0 menggunakan "
            "metode analisis komputasional berbasis kecerdasan buatan dan teknik forensik digital. "
            "Hasil analisis ini TIDAK menggantikan investigasi forensik digital yang dilakukan oleh "
            "tenaga ahli bersertifikat. Tingkat akurasi sistem bergantung pada kualitas dan jenis "
            "media yang dianalisis, serta ketersediaan modul analisis. Tidak ada sistem deteksi "
            "deepfake yang dapat memberikan jaminan akurasi 100%. "
            "Penggunaan hasil laporan ini untuk kepentingan hukum, investigasi resmi, atau "
            "pengambilan keputusan kritis harus selalu disertai dengan verifikasi independen oleh "
            "pakar forensik digital berlisensi. Penyedia sistem DeepGuard tidak bertanggung jawab "
            "atas kerugian yang timbul akibat penggunaan laporan ini tanpa verifikasi lebih lanjut. "
            "Semua data yang diproses oleh sistem ini bersifat rahasia dan tidak disimpan di server "
            "eksternal manapun."
        )
        story.append(Paragraph(disclaimer_text, styles["disclaimer"]))
        story.append(Spacer(1, 0.2 * cm))

    def _add_footer(
        self,
        story: list,
        report_number: str,
        now: datetime,
    ) -> None:
        """Menambahkan footer laporan."""
        styles = self._styles
        story.append(HRFlowable(width="100%", thickness=1, color=_hex_to_rgb(COLOR_BORDER)))
        story.append(Spacer(1, 0.15 * cm))

        footer_data = [[
            Paragraph(
                f"DeepGuard v1.0 | Laporan Forensik Digital | No: {report_number}",
                styles["footer"],
            ),
            Paragraph(
                f"Dibuat: {now.strftime('%d/%m/%Y %H:%M:%S')} | "
                "Bersifat Rahasia — Hanya untuk Penggunaan Internal",
                styles["footer"],
            ),
        ]]

        footer_table = Table(
            footer_data,
            colWidths=[(PAGE_WIDTH - 2 * MARGIN) / 2, (PAGE_WIDTH - 2 * MARGIN) / 2],
        )
        footer_table.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("ALIGN", (0, 0), (0, 0), "LEFT"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ]))
        story.append(footer_table)
