"""Check every number the manuscript reports for the fourth pass against the
result files, so that no figure in the text rests on a transcription.

Run from anywhere; exits non-zero and prints each mismatch.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")

with open(os.path.join(RESULTS, "effect_size_calibration.json")) as f:
    cal = json.load(f)
with open(os.path.join(RESULTS, "effect_size_altnorm.json")) as f:
    alt = json.load(f)
tex = open(os.path.join(ROOT, "manuscript.tex"), encoding="utf-8").read()

problems = []


def want(label, value, pattern, tol=5e-4):
    """Assert the manuscript states `value` where `pattern` (a regex with one
    capturing group) matches."""
    m = re.search(pattern, tex)
    if not m:
        problems.append(f"{label}: no encuentro el patron en el manuscrito")
        return
    stated = float(m.group(1))
    if abs(stated - value) > tol:
        problems.append(f"{label}: el texto dice {stated}, los datos dan {value}")


pooled = cal["pooled_detection"]
want("thr gaussian pooled", round(pooled["gaussian_linear"]["threshold_rule"] * 100, 1),
     r"22\.4\\% vs (\d+\.\d)\\%")
want("thr nonlinear pooled", round(pooled["nonlinear_misspecified"]["threshold_rule"] * 100, 1),
     r"less often than Gaussian drift \((\d+\.\d)\\%")
want("thr clustered pooled", round(pooled["clustered"]["threshold_rule"] * 100, 1),
     r"clustered confounding 4\.2 points less often \((\d+\.\d)\\%")
want("thr contaminated pooled", round(pooled["outlier_contaminated"]["threshold_rule"] * 100, 1),
     r"it detects (\d+\.\d)\\%, a 10\.4-point advantage")
want("perm nonlinear pooled", round(pooled["nonlinear_misspecified"]["permutation_rule"] * 100, 1),
     r"weakest case too \((\d+\.\d)\\% pooled")
want("perm gaussian pooled", round(pooled["gaussian_linear"]["permutation_rule"] * 100, 1),
     r"pooled against (\d+\.\d)\\%\)")

alt_thr = alt["pooled_detection_threshold_rule"]
want("altnorm contaminated", round(alt_thr["normalised_by_base_sigma (this check)"] * 100, 1),
     r"it detects (\d+\.\d)\\% against Gaussian drift's")
want("altnorm gaussian ref", round(alt_thr["gaussian_linear_for_reference"] * 100, 1),
     r"against Gaussian drift's (\d+\.\d)\\%")

paired = cal["paired_vs_gaussian_linear"]
want("dif nonlinear (pts)",
     abs(round(paired["nonlinear_misspecified"]["difference_in_detection_rate"] * 100, 1)),
     r"\\emph\{(\d+\.\d) points\} less often than Gaussian drift")
want("dif clustered (pts)",
     abs(round(paired["clustered"]["difference_in_detection_rate"] * 100, 1)),
     r"clustered confounding (\d+\.\d) points less often")
want("dif contaminated (pts)",
     round(paired["outlier_contaminated"]["difference_in_detection_rate"] * 100, 1),
     r"a (\d+\.\d)-point advantage over Gaussian drift")
want("discordantes nonlinear a favor",
     paired["nonlinear_misspecified"]["gaussian_detects_other_misses"],
     r"(\d+) discordant pairs against 2 the other way")
want("discordantes clustered",
     paired["clustered"]["gaussian_detects_other_misses"],
     r"(\d+) against 52,")
want("discordantes clustered otro lado",
     paired["clustered"]["other_detects_gaussian_misses"],
     r"178 against (\d+),")

# tabla de la cuarta pasada, celda por celda
order = ["gaussian_linear", "outlier_contaminated", "nonlinear_misspecified",
         "clustered"]
tex_rows = {
    "<=0.1": r"\$\\leq\$0\.1\s*&\s*(\d+)\s*&(.+?)\\\\",
    "0.1-0.25": r"0\.1--0\.25\s*&\s*(\d+)\s*&(.+?)\\\\",
    "0.25-0.5": r"0\.25--0\.5\s*&\s*(\d+)\s*&(.+?)\\\\",
    "0.5-1": r"0\.5--1\s*&\s*(\d+)\s*&(.+?)\\\\",
    "1-2": r"\n1--2\s*&\s*(\d+)\s*&(.+?)\\\\",
    "2-4": r"\n2--4\s*&\s*(\d+)\s*&(.+?)\\\\",
    "4-8": r"\n4--8\s*&\s*(\d+)\s*&(.+?)\\\\",
    "8-16": r"\n8--16\s*&\s*(\d+)\s*&(.+?)\\\\",
    ">16": r"\$>\$16\s*&\s*(\d+)\s*&(.+?)\\\\",
}
for label, pat in tex_rows.items():
    m = re.search(pat, tex)
    if not m:
        problems.append(f"tabla, fila {label}: no encuentro la fila")
        continue
    n_stated = int(m.group(1))
    n_real = cal["detection_by_delta_bin"][label]["gaussian_linear"]["n"]
    if n_stated != n_real:
        problems.append(f"tabla {label}: n={n_stated} en el texto, {n_real} en los datos")
    cells = [c.strip() for c in m.group(2).split("&")]
    if len(cells) != 4:
        problems.append(f"tabla {label}: {len(cells)} celdas, esperaba 4")
        continue
    for ctype, cell in zip(order, cells):
        thr_s, perm_s = [float(v) for v in cell.split("/")]
        d = cal["detection_by_delta_bin"][label][ctype]
        if abs(thr_s - d["threshold_rule"]) > 5e-4:
            problems.append(f"tabla {label}/{ctype}: thr {thr_s} vs {d['threshold_rule']:.3f}")
        if abs(perm_s - d["permutation_rule"]) > 5e-4:
            problems.append(f"tabla {label}/{ctype}: perm {perm_s} vs {d['permutation_rule']:.3f}")

# totales declarados
n_total = 4000 + 8000 + 6000 + cal["n_scenarios"]
if f"{n_total:,}" not in tex.replace("30,000", "30,000"):
    problems.append(f"el total de escenarios deberia ser {n_total:,}")
if cal["n_scenarios"] != 12000:
    problems.append(f"la cuarta pasada tiene {cal['n_scenarios']} escenarios, no 12.000")

# el 90.7% del mecanismo no lineal en el bin alto
want("nonlinear en delta>16",
     round(cal["detection_by_delta_bin"][">16"]["nonlinear_misspecified"]["threshold_rule"] * 100, 1),
     r"detects it (\d+\.\d)\\% of the time")

if problems:
    print("PROBLEMAS:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print(f"OK: todas las cifras de la cuarta pasada cuadran con los ficheros de resultados "
      f"({cal['n_scenarios']} escenarios, total declarado {n_total:,})")
