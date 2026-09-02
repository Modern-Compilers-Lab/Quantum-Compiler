"""Constants from the paper that the renderers need: display scaling, circuit
shape, loop-unroll factors and the ablation column names."""

# ── Table 4: display scaling ────────────────────────────────────────────────
TABLE4_SCALE = dict(swaps=1e3, depth=1e3, latency=1e3, error=1.0)

# ── Table 5: surface-code circuit shape and scoring ─────────────────────────
#: Circuit shape per code distance; independent of the mapper.
TABLE5_SHAPE = {3: dict(qubits=17, cx_per_round=24),
                5: dict(qubits=49, cx_per_round=120),
                7: dict(qubits=97, cx_per_round=168)}

#: Loop-unroll factor per backend: Brisbane sweeps r in {3,5,10,15,20} and is
#: scored at 10; MECH is r=3 only and is scored at 3.
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
# The ablation CSVs carry all four side by side, under these column suffixes.
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
