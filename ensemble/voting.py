"""
ensemble/voting.py
------------------
Sistem Voting Ensemble Tertimbang (Weighted Ensemble Voting) untuk DEFORAG.

Strategi: CNN-Dominant with Heuristic Modifiers
- CNN (EfficientNet-B4, AUC 98.56%) menjadi sinyal utama dengan bobot 55%.
- Modul heuristik lainnya berfungsi sebagai modifier sekunder.
- Override rules di-gerbang (gated) oleh skor CNN untuk mencegah false positive
  dari modul heuristik yang belum memiliki model terlatih.

Mendukung 10 modul analisis dengan redistribusi bobot dinamis jika ada modul
yang tidak aktif/error, dilengkapi dengan aturan override forensik (override rules)
yang konservatif dan CNN-gated.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class EnsembleVoter:
    """
    Menggabungkan hasil dari 10 modul analisis DEFORAG menggunakan
    voting tertimbang (weighted voting) dinamis dengan strategi CNN-dominant
    dan aturan override forensik yang di-gerbang oleh CNN.
    """

    # Bobot default untuk 10 modul analisis (total = 1.00)
    # CNN mendominasi karena satu-satunya modul dengan model terlatih.
    # Modul heuristik lainnya hanya berfungsi sebagai modifier ringan.
    DEFAULT_WEIGHTS = {
        "cnn":             0.55,  # Primary signal — EfficientNet-B4 trained model
        "frequency":       0.08,  # FFT & DCT artifacts (heuristic)
        "gan_fingerprint": 0.08,  # GAN grid/checkerboard noise (heuristic)
        "texture":         0.06,  # Steganalysis SRM + PRNU + LBP (heuristic)
        "landmark":        0.03,  # MediaPipe Face Mesh geometry (heuristic)
        "exif":            0.03,  # Metadata file & editing tags (heuristic)
        "temporal":        0.02,  # Video temporal consistency (heuristic)
        "eye_reflection":  0.05,  # Catchlight symmetry & consistency (heuristic)
        "face_blending":   0.06,  # Boundary blending artifacts (heuristic)
        "skin_texture":    0.04,  # Pori-pori & skin smoothing detection (heuristic)
    }

    # Threshold kalibrasi ensemble — seimbang di 0.50
    # Di atas threshold = FAKE, di bawah = REAL
    THRESHOLD = 0.50

    def __init__(self, weights: Dict[str, float] = None) -> None:
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS

    def vote(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Menghitung keputusan gabungan dari modul-modul analisis.

        Parameters
        ----------
        module_results : dict
            Dict berisi hasil per modul, contoh:
            {
                "cnn": {"score": 0.82, "verdict": "FAKE", "confidence": 0.64, "details": {...}},
                "frequency": None,
                ...
            }

        Returns
        -------
        dict
            Hasil ensemble yang berisi final_score, verdict, confidence,
            risk_level, module_contributions, dan penjelasan.
        """
        active_weights = {}
        module_scores = {}
        module_verdicts = {}

        # 1. Identifikasi modul yang aktif dan tidak menghasilkan error/N/A
        for mod, weight in self.weights.items():
            res = module_results.get(mod)
            if res is None:
                continue

            # Jika ada error atau modul mengembalikan status tidak tersedia/N/A (misal temporal di gambar)
            if res.get("error") or res.get("verdict") == "N/A":
                continue

            active_weights[mod] = weight
            module_scores[mod] = float(res.get("score", 0.5))
            module_verdicts[mod] = str(res.get("verdict", "UNKNOWN"))

        # Jika tidak ada modul aktif sama sekali
        if not active_weights:
            logger.warning("Tidak ada modul analisis aktif untuk voting ensemble.")
            return self._default_unknown_result()

        # 2. Hitung weighted average score (Redistribusi bobot dinamis)
        sum_active_weights = sum(active_weights.values())
        weighted_sum = 0.0
        contributions = []

        for mod, weight in active_weights.items():
            # Redistribusikan bobot secara proporsional ke total bobot modul aktif
            normalized_weight = weight / sum_active_weights if sum_active_weights > 0 else 0.0
            score = module_scores[mod]
            weighted_contrib = score * normalized_weight
            weighted_sum += weighted_contrib

            contributions.append({
                "module": mod,
                "weight": normalized_weight,
                "raw_score": score,
                "weighted_contribution": weighted_contrib,
                "module_verdict": module_verdicts[mod]
            })

        final_score = weighted_sum

        # 3. CNN-Centric Scoring Strategy
        #    Alih-alih rata-rata tertimbang murni, gunakan CNN sebagai baseline
        #    dan modul heuristik hanya dapat memodifikasi skor dalam rentang terbatas.
        #    Ini mencegah modul heuristik yang tidak terlatih mendominasi keputusan.
        cnn_score = module_scores.get("cnn", 0.5)

        if "cnn" in active_weights:
            # Hitung rata-rata heuristik (semua modul selain CNN)
            heuristic_scores = [s for m, s in module_scores.items() if m != "cnn"]
            if heuristic_scores:
                heuristic_avg = sum(heuristic_scores) / len(heuristic_scores)
            else:
                heuristic_avg = 0.5

            # Skor akhir: CNN dominan (65%) + heuristik (35%)
            # CNN = sinyal utama, heuristik = modifier ringan
            final_score = cnn_score * 0.65 + heuristic_avg * 0.35

            # Corroborating Evidence Boost:
            # Jika CNN borderline (0.35-0.50) dan banyak heuristik setuju mencurigakan
            if 0.35 <= cnn_score < 0.50:
                supporting_count = sum(
                    1 for m, s in module_scores.items()
                    if m != "cnn" and m != "temporal" and s >= 0.42
                )
                if supporting_count >= 3:
                    support_boost = min(0.16, supporting_count * 0.04)
                    final_score += support_boost
                    logger.info(
                        "Corroborating evidence boost: %d modul mendukung, boost=%.3f",
                        supporting_count, support_boost
                    )


        # 4. Terapkan Forensic Override Rules — CNN-Gated
        #    Override hanya berlaku jika CNN tidak sangat yakin gambar REAL.
        #    Ini mencegah modul heuristik (noise) memicu false positive.
        is_video_input = False
        temporal_res = module_results.get("temporal")
        if temporal_res and temporal_res.get("verdict") != "N/A":
            is_video_input = True

        override_triggered = False
        override_reasons = []

        # --- CNN Gate: Sangat yakin REAL ---
        # Jika video: relax CNN gate. Jika gambar: cnn < 0.15 blocks overrides.
        if is_video_input:
            cnn_blocks_overrides = False
        else:
            cnn_blocks_overrides = cnn_score < 0.15

        # --- CNN Gate: Sangat yakin FAKE ---
        # Jika CNN > 0.70, pastikan skor minimum 0.65 agar tidak ditarik
        # turun terlalu jauh oleh modul heuristik yang memberi skor rendah.
        if cnn_score > 0.70 and not cnn_blocks_overrides:
            if final_score < 0.65:
                final_score = 0.65
                override_triggered = True
                override_reasons.append(
                    f"CNN sangat yakin FAKE (skor={cnn_score:.1%}), "
                    f"skor minimum dinaikkan ke 65% agar modul heuristik tidak mendominasi"
                )

        if not cnn_blocks_overrides:
            # --- Rule 1: GAN Fingerprint Grid & Noise Artifacts ---
            gan_res = module_results.get("gan_fingerprint") or {}
            gan_details = gan_res.get("details", {})
            gan_sub = gan_details.get("sub_scores", {})
            checkerboard = gan_sub.get("checkerboard", 0.0)
            noise_fp = gan_sub.get("noise_fingerprint", 0.0)

            cnn_gate_rule1 = True if is_video_input else (cnn_score >= 0.30)
            if checkerboard >= 0.75 and noise_fp >= 0.75 and cnn_gate_rule1:
                final_score = max(final_score, 0.60)
                override_triggered = True
                override_reasons.append(
                    f"Artefak generator GAN kuat terdeteksi "
                    f"(checkerboard={checkerboard:.1%}, noise_fp={noise_fp:.1%})"
                )

            # --- Rule 2: Noise PRNU Kamera Asli vs GAN ---
            tex_res = module_results.get("texture") or {}
            tex_details = tex_res.get("details", {})
            tex_sub = tex_details.get("sub_scores", {})
            prnu = tex_sub.get("prnu", 0.0)
            gan_score = module_scores.get("gan_fingerprint", 0.0)

            cnn_gate_rule2 = True if is_video_input else (cnn_score >= 0.30)
            if prnu >= 0.75 and gan_score >= 0.55 and cnn_gate_rule2:
                final_score = max(final_score, 0.58)
                override_triggered = True
                override_reasons.append(
                    f"PRNU anomali ({prnu:.1%}) disertai indikasi pola GAN ({gan_score:.1%})"
                )

            # --- Rule 3: Batas Blending Wajah (Face-Swap) ---
            blend_score = module_scores.get("face_blending", 0.0)
            cnn_gate_rule3 = True if is_video_input else (cnn_score >= 0.35)
            if blend_score >= 0.70 and cnn_gate_rule3:
                final_score = max(final_score, 0.81) if is_video_input else max(final_score, 0.58)
                override_triggered = True
                override_reasons.append(
                    f"Batas blending/penempelan wajah tidak konsisten (skor={blend_score:.1%})"
                )

            # --- Rule 4: Refleksi Mata Tidak Konsisten ---
            eye_score = module_scores.get("eye_reflection", 0.0)
            cnn_gate_rule4 = True if is_video_input else (cnn_score >= 0.30)
            if eye_score >= 0.60 and cnn_gate_rule4:
                final_score = max(final_score, 0.52)
                override_triggered = True
                override_reasons.append(
                    f"Refleksi cahaya (catchlight) pada iris mata asimetris/tidak natural (skor={eye_score:.1%})"
                )

            # --- Rule 5: Penghalusan Kulit Berlebih (Skin Smoothing) ---
            skin_score = module_scores.get("skin_texture", 0.0)
            cnn_gate_rule5 = True if is_video_input else (cnn_score >= 0.35)
            if skin_score >= 0.75 and cnn_gate_rule5:
                final_score = max(final_score, 0.78) if is_video_input else max(final_score, 0.55)
                override_triggered = True
                override_reasons.append(
                    f"Tekstur kulit wajah terlalu halus/kehilangan pori alami secara masif (skor={skin_score:.1%})"
                )

            # --- Rule 5b: Temporal Override ---
            temporal_score = module_scores.get("temporal", 0.0)
            if temporal_score >= 0.70:
                final_score = max(final_score, 0.76) if is_video_input else max(final_score, 0.55)
                override_triggered = True
                override_reasons.append(
                    f"Inkonsistensi temporal terdeteksi pada video (skor={temporal_score:.1%})"
                )

            # --- Rule 6: Deteksi Multi-Anomali (CNN-Gated) ---
            # Jika ada 4+ modul forensik yang mencurigakan DAN CNN juga mencurigakan
            suspicious_modules = [
                m for m, s in module_scores.items()
                if m not in ("cnn", "temporal", "exif") and s >= 0.55
            ]
            cnn_gate_rule6 = True if is_video_input else (cnn_score >= 0.30)
            if len(suspicious_modules) >= 4 and cnn_gate_rule6:
                final_score = max(final_score, 0.55)
                override_triggered = True
                override_reasons.append(
                    f"Ditemukan anomali simultan pada {len(suspicious_modules)} modul: "
                    f"{', '.join(suspicious_modules)} (CNN={cnn_score:.1%})"
                )

        # Batasi final score pada rentang [0.0, 1.0]
        final_score = float(min(max(final_score, 0.0), 1.0))

        # 4. Tentukan Verdict & Risk Level
        if final_score >= 0.70:
            verdict = "FAKE"
            risk_level = "TINGGI"
        elif final_score >= self.THRESHOLD:
            verdict = "FAKE"
            risk_level = "SEDANG"
        elif final_score >= 0.35:
            verdict = "REAL"
            risk_level = "RENDAH"
        else:
            verdict = "REAL"
            risk_level = "SANGAT RENDAH"

        # Hitung confidence score relatif terhadap threshold
        if final_score >= self.THRESHOLD:
            confidence = (final_score - self.THRESHOLD) / (1.0 - self.THRESHOLD)
        else:
            confidence = (self.THRESHOLD - final_score) / self.THRESHOLD
        confidence = float(min(max(confidence, 0.0), 1.0))

        # 5. Susun penjelasan keputusan
        strongest_contrib = max(contributions, key=lambda c: c["weighted_contribution"]) if contributions else {}
        strongest_name = strongest_contrib.get("module", "-")

        explanation_parts = [
            f"Media dianalisis secara sukses oleh {len(active_weights)} dari {len(self.weights)} modul.",
            f"Skor gabungan terbobot: {final_score*100:.2f}% (Verdict: {verdict}).",
            f"Kontributor terbesar: modul '{strongest_name}' "
            f"(skor mentah: {module_scores.get(strongest_name, 0.0)*100:.1f}%, "
            f"kontribusi efektif: {strongest_contrib.get('weighted_contribution', 0.0)*100:.1f}%)."
        ]

        if override_triggered:
            explanation_parts.append(
                f"OVERRIDE FORENSIK AKTIF (CNN-gated): {'; '.join(override_reasons)}."
            )
        else:
            if verdict == "REAL":
                explanation_parts.append(
                    "KESIMPULAN: Tidak ditemukan pola anomali forensik atau deep learning yang signifikan. "
                    "Berkas dinilai asli."
                )
            else:
                explanation_parts.append(
                    "KESIMPULAN: Ditemukan indikasi manipulasi digital pada wajah atau pola piksel "
                    "yang tidak wajar, didominasi oleh sinyal model CNN."
                )

        return {
            "final_score": round(final_score, 6),
            "verdict": verdict,
            "confidence": round(confidence, 6),
            "risk_level": risk_level,
            "module_contributions": contributions,
            "weight_used": active_weights,
            "override": override_triggered,
            "explanation": " ".join(explanation_parts),
        }

    def _default_unknown_result(self) -> Dict[str, Any]:
        return {
            "final_score": 0.5,
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "risk_level": "SEDANG",
            "module_contributions": [],
            "weight_used": {},
            "override": False,
            "explanation": "Gagal menentukan keputusan karena tidak ada modul analisis yang menghasilkan data valid.",
        }
