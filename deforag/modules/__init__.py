"""
modules/__init__.py
===================
Package initializer untuk modul-modul analisis DEFORAG.

Mengimpor semua kelas analyzer utama sehingga dapat diakses langsung
dari namespace package:

    from deforag.modules import CNNAnalyzer, FrequencyAnalyzer, LandmarkAnalyzer

Author  : ANTENK TEAM
Version : 1.0.0
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import analyzer utama
# ---------------------------------------------------------------------------

try:
    from deforag.modules.cnn_analyzer import CNNAnalyzer
    __all__ = ["CNNAnalyzer"]
    logger.debug("CNNAnalyzer berhasil diimpor.")
except ImportError as e:
    logger.warning("Gagal mengimpor CNNAnalyzer: %s", e)
    CNNAnalyzer = None  # type: ignore[assignment,misc]

try:
    from deforag.modules.frequency_analyzer import FrequencyAnalyzer
    __all__ = getattr(__all__, "__iadd__", lambda x: x)(["FrequencyAnalyzer"]) or [
        *(__all__ if __all__ else []), "FrequencyAnalyzer"
    ]
    logger.debug("FrequencyAnalyzer berhasil diimpor.")
except ImportError as e:
    logger.warning("Gagal mengimpor FrequencyAnalyzer: %s", e)
    FrequencyAnalyzer = None  # type: ignore[assignment,misc]

try:
    from deforag.modules.landmark_analyzer import LandmarkAnalyzer
    logger.debug("LandmarkAnalyzer berhasil diimpor.")
except ImportError as e:
    logger.warning("Gagal mengimpor LandmarkAnalyzer: %s", e)
    LandmarkAnalyzer = None  # type: ignore[assignment,misc]

# Definisikan __all__ secara eksplisit
__all__ = ["CNNAnalyzer", "FrequencyAnalyzer", "LandmarkAnalyzer"]

__version__ = "1.0.0"
__author__  = "ANTENK TEAM"
