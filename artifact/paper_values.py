"""Constants taken from paper/paper-174.pdf that the renderers need.

These are structural: they say how a table is built, not what it should come
out as. render.py does not compare anything against the paper - it renders the
figures and tables from whatever CSVs it is given.

The paper's own printed numbers are kept at the bottom, under REFERENCE, purely
as a record of what the submission claims. Nothing reads them.
"""

# ── Table 4: display scaling ────────────────────────────────────────────────
TABLE4_SCALE = dict(swaps=1e3, depth=1e3, latency=1e3, error=1.0)

# ── Table 5: surface-code circuit shape and scoring ─────────────────────────
#: Circuit shape per code distance; independent of the mapper.
TABLE5_SHAPE = {3: dict(qubits=17, cx_per_round=24),
                5: dict(qubits=49, cx_per_round=120),
                7: dict(qubits=97, cx_per_round=168)}

#: Loop-unroll factor used when scoring each backend's traces. Brisbane sweeps
#: r in {3,5,10,15,20} and is scored at the global default of 10; MECH is r=3
#: only and is scored at 3. Scoring MECH at 10 inflates every improvement by
#: roughly 10 points, so this is not a free parameter.
TABLE5_LOOP_ITERATIONS = {"ibm_brisbane": 10, "mech_heavy_hex": 3}

# ── Table 7 / Figure 10: ablation configurations ────────────────────────────
# Cumulative configurations:
#   (i)   Recon                  reconciliation passes only - the baseline. No
#                                error term, no depth-rate term C_rate, no
#                                steady-state loop-entry remapping.
#   (ii)  Recon+Err              adds hardware-error awareness.
#   (iii) Recon+Err+Depth        adds the depth-rate term C_rate.
#   (iv)  Recon+Err+Depth+Remap  adds SelectSteadyState remapping = full DynamiQ.
#
# The ablation CSVs carry all four side by side under historical column suffixes
# from the development sweep, not the paper's labels.
ABLATION_COLUMNS = {
    "(i) Recon":                  "no_remap_no_error",   # baseline
    "(ii) Recon+Err":             "no_remap",
    "(iii) Recon+Err+Depth":      "new_line",
    "(iv) Recon+Err+Depth+Remap": "default",             # full DynamiQ
}
ABLATION_BASELINE = "no_remap_no_error"
ABLATION_BACKEND = "ibm_brisbane_old"
ABLATION_STUDY = "ablation_study5"
#: Figure 10's panel legend labels, in plotting order.
ABLATION_PANEL_LABELS = {"no_remap_no_error": "(i)", "no_remap": "(ii)",
                         "new_line": "(iii)", "default": "(iv)"}

# ── Experiment settings stated in Sec. 7.1 ──────────────────────────────────
QISKIT_VERSION = "1.3.2"      # LightSABRE baseline used in the paper
CALIBRATION_DATE = "2025-10-17"
SABRE_SETTINGS = dict(routing_method="sabre", layout_method="identity",
                      optimization_level=1)


# ════════════════════════════════════════════════════════════════════════════
# REFERENCE - the paper's printed numbers. Nothing below is read by the
# artifact; it is kept so the submission's claims stay recorded next to the
# code that regenerates them.
# ════════════════════════════════════════════════════════════════════════════

# Table 3: average relative improvement (%) over Qiskit Sabre.
TABLE3 = {
    ("ibm_brisbane_old", 54):  dict(swaps=52.04, depth=13.92, latency=13.06, error=14.62),
    ("ibm_brisbane_old", 81):  dict(swaps=40.97, depth=14.66, latency=13.99, error=21.08),
    ("ibm_brisbane_old", 121): dict(swaps=51.73, depth=17.75, latency=18.59, error=16.87),
    ("ibm_kingston", 54):      dict(swaps=39.46, depth=11.48, latency=11.16, error=40.53),
    ("ibm_kingston", 81):      dict(swaps=48.71, depth=18.48, latency=18.35, error=38.99),
    ("ibm_kingston", 121):     dict(swaps=34.54, depth=8.70,  latency=8.80,  error=33.62),
}

# Table 4: chiplet QPUs. Absolute cells in the paper's display units.
TABLE4_ABS = {
    ("heavy_hexagon", 81):  dict(swaps=(2.97, 4.51), depth=(30.1, 34.8),
                                 latency=(45.8, 54.6), error=(11.38, 12.88)),
    ("heavy_hexagon", 121): dict(swaps=(4.57, 7.41), depth=(33.0, 34.4),
                                 latency=(50.4, 58.3), error=(12.31, 13.65)),
    ("ibm_flamingo", 256):  dict(swaps=(15.9, 12.1), depth=(40.9, 34.2),
                                 latency=(5122.0, 5270.4), error=(154.40, 181.72)),
}
TABLE4_OVERALL = {
    "heavy_hexagon": dict(swaps=36.2, depth=8.7, latency=14.9, error=10.7),
    "ibm_flamingo":  dict(swaps=-31.8, depth=-19.6, latency=2.8, error=15.0),
}

# Table 5: surface-code circuits. Latency in microseconds.
TABLE5_ABS = {
    ("ibm_brisbane", 3):   dict(swaps=(848, 1360),   depth=(395, 562),
                                latency=(229, 355),  error=(1.92, 2.84)),
    ("ibm_brisbane", 5):   dict(swaps=(4320, 4887),  depth=(806, 1132),
                                latency=(494, 711),  error=(3.80, 4.47)),
    ("ibm_brisbane", 7):   dict(swaps=(8271, 13387), depth=(909, 2192),
                                latency=(564, 1415), error=(5.05, 7.71)),
    ("mech_heavy_hex", 3): dict(swaps=(420, 480),    depth=(200, 185),
                                latency=(124, 112),  error=(0.09, 0.10)),
    ("mech_heavy_hex", 5): dict(swaps=(1462, 2380),  depth=(316, 446),
                                latency=(472, 737),  error=(0.17, 0.27)),
    ("mech_heavy_hex", 7): dict(swaps=(4552, 6284),  depth=(460, 706),
                                latency=(526, 1091), error=(0.25, 0.33)),
}
TABLE5_OVERALL = {
    "ibm_brisbane":   dict(swaps=31.5, depth=45.7, latency=48.1, error=28.4),
    "mech_heavy_hex": dict(swaps=29.6, depth=27.0, latency=42.1, error=28.6),
}

# Table 6: average DynamiQ mapping time (s), Intel i7-10750H at 2.60 GHz.
TABLE6 = {
    ("ibm_kingston", 54):      {"Small (10-30)": 0.22, "Med. (40-60)": 0.48, "Large (70-90)": 0.73},
    ("ibm_kingston", 81):      {"Small (10-30)": 0.38, "Med. (40-60)": 0.69, "Large (70-90)": 1.01},
    ("ibm_kingston", 121):     {"Small (10-30)": 0.86, "Med. (40-60)": 1.48, "Large (70-90)": 2.03},
    ("ibm_brisbane_old", 54):  {"Small (10-30)": 0.19, "Med. (40-60)": 0.37, "Large (70-90)": 0.56},
    ("ibm_brisbane_old", 81):  {"Small (10-30)": 0.34, "Med. (40-60)": 0.64, "Large (70-90)": 0.97},
    ("ibm_brisbane_old", 121): {"Small (10-30)": 0.67, "Med. (40-60)": 1.10, "Large (70-90)": 1.52},
}

# Table 7: ablation, improvement (%) over the reconciliation-only baseline.
# Cells are truncated toward zero to one decimal, not rounded.
TABLE7 = {
    81:  {"(ii) Recon+Err":             dict(swaps=-6.9, depth=-2.3, latency=-2.2, error=15.1),
          "(iii) Recon+Err+Depth":      dict(swaps=17.7, depth=6.5,  latency=6.1,  error=9.7),
          "(iv) Recon+Err+Depth+Remap": dict(swaps=48.2, depth=17.7, latency=17.1, error=20.5)},
    121: {"(ii) Recon+Err":             dict(swaps=-0.3, depth=2.1,  latency=3.5,  error=13.5),
          "(iii) Recon+Err+Depth":      dict(swaps=17.5, depth=6.6,  latency=6.9,  error=10.0),
          "(iv) Recon+Err+Depth+Remap": dict(swaps=55.5, depth=19.4, latency=20.1, error=18.0)},
}

# Table 8: sensitivity to <alpha, beta> on heavy-hexagon 8x8-2x2, 54 qbt.
# Improvement vs the spatial-only baseline <1, 0>; positive is better.
TABLE8 = [
    dict(alpha=0.50, beta=0.50, swaps=-42.00, depth=8.00,  latency=-23.83, error=-23.25),
    dict(alpha=0.60, beta=0.40, swaps=-21.98, depth=9.82,  latency=-9.73,  error=-10.63),
    dict(alpha=0.80, beta=0.20, swaps=-11.47, depth=11.35, latency=-1.82,  error=-5.35),
    dict(alpha=1.00, beta=0.40, swaps=-11.19, depth=16.99, latency=2.11,   error=-4.06),
    dict(alpha=1.00, beta=0.20, swaps=-2.61,  depth=12.40, latency=6.04,   error=0.35),
]
TABLE8_BASELINE = dict(alpha=1.0, beta=0.0)
#: The pair the paper states every other experiment uses.
EVALUATION_ALPHA_BETA = (1.0, 0.2)

# Sec. 7.6.1 (Figure 9) prose claims.
NESTED_CLAIM_TEXT = ("~80% reduction in SWAP count and 50-60% improvement in "
                     "latency, error and circuit depth (Sec. 7.6.1)")
NESTED_CLAIMS = dict(swaps=80.0, depth=None, latency=None, error=None)
NESTED_RANGE_METRICS = ("depth", "latency", "error")
NESTED_RANGE = (50.0, 60.0)
