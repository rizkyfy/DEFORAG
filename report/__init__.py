"""
report/__init__.py
------------------
Package initializer untuk modul report DEFORAG.
Mengekspor kelas ForensicReportGenerator untuk kemudahan import.
"""

from report.pdf_generator import ForensicReportGenerator

__all__ = ["ForensicReportGenerator"]
