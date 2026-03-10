"""
generate_latex.py
-----------------
Legge i CSV da metrics_tables/ e produce tabelle LaTeX in stile booktabs.
Gestisce SEQ, OMP, MPI e i nuovi dati CUDA (Algorithm, Precision, BlockDim, TileSize).

Output: plot_scripts/latex_tables/
"""

import os
import pandas as pd
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(SCRIPT_DIR, "metrics_tables")
OUT_DIR     = os.path.join(SCRIPT_DIR, "latex_tables")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Column configs ────────────────────────────────────────────────────────────
# (csv_column, latex_header, alignment_char)

SEQ_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("Align",          "Align",          "c"),
    ("Restrict",       "Restrict",       "c"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup",        "Speedup",        "r"),
]

SEQ_TILE_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("TileSize",       "Tile",           "r"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup",        "Speedup",        "r"),
]

# Nuova configurazione specifica per CUDA
CUDA_COLS = [
    ("N",              r"$N$",           "r"),
    ("Algorithm",      "Kernel",         "l"),
    ("Precision",      "Prec.",          "c"),
    ("BlockDim",       "Block",          "c"),
    ("TileSize",       "Tile",           "c"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup",        "Speedup",        "r"),
]

OMP_MPI_COLS = [
    ("N",              r"$N$",           "r"),
    ("Threads/Ranks",  "P",              "r"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup",        "Speedup",        "r"),
    ("Efficiency_%",   r"Eff. (\%)",     "r"),
]

# ── Table metadata ────────────────────────────────────────────────────────────
# (csv_filename, col_config, caption, label, sort_by)
TABLES = [
    # --- SEQ ---
    ("seq_ikj_metrics.csv",    SEQ_COLS,      "Sequential IKJ matrix multiplication", "seq_ikj", None),
    ("seq_tile_metrics.csv",   SEQ_TILE_COLS, "Sequential cache-tiled matrix multiplication", "seq_tile", None),
    ("mkl_seq_metrics.csv",    SEQ_COLS,      "MKL single-threaded DGEMM", "mkl_seq", None),
    
    # --- CUDA ---
    ("cuda_all_metrics.csv",   CUDA_COLS,     "CUDA kernels performance comparison -- NVIDIA Tesla T4", "cuda_all", ["N", "Algorithm", "Precision"]),
    ("cuda_naive_metrics.csv", CUDA_COLS,     "CUDA Naive kernel performance", "cuda_naive", ["N", "Precision"]),
    ("cuda_shared_metrics.csv", CUDA_COLS,    "CUDA Shared Memory tiled kernel", "cuda_shared", ["N", "Precision", "TileSize"]),
    ("cuda_cublas_metrics.csv", CUDA_COLS,    "CUDA cuBLAS Library performance", "cuda_cublas", ["N", "Precision"]),
    
    # --- OMP ---
    ("omp_ikj_metrics.csv",    OMP_MPI_COLS,  "OpenMP IKJ parallel performance", "omp_ikj", None),
    ("omp_tile_metrics.csv",   OMP_MPI_COLS,  "OpenMP cache-tiled parallel performance", "omp_tile", None),
    ("mkl_omp_metrics.csv",    OMP_MPI_COLS,  "OpenMP MKL parallel DGEMM", "mkl_omp", None),
    
    # --- MPI ---
    ("mpi_ikj_metrics.csv",    OMP_MPI_COLS,  "MPI row-slab IKJ performance", "mpi_ikj", None),
    ("mpi_cannon_metrics.csv", OMP_MPI_COLS,  "MPI Cannon's algorithm performance", "mpi_cannon", None),
    ("mpi_summa_metrics.csv",  OMP_MPI_COLS,  "MPI SUMMA algorithm performance", "mpi_summa", None),
]

# ── LaTeX builder ─────────────────────────────────────────────────────────────

def fmt_cell(val):
    """Format a single cell value for LaTeX."""
    if pd.isna(val) or str(val).strip() in ("", "-", "nan", "None"):
        return "--"
    if isinstance(val, (float, np.float64)):
        # Se è un intero (es. TileSize 16.0), rimuovi il decimale
        if val.is_integer():
            return str(int(val))
        return f"{val:.2f}"
    return str(val).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def make_table(df: pd.DataFrame, col_cfg: list, caption: str, label: str) -> str:
    """Return a complete LaTeX table string (booktabs style)."""
    # Keep only cols that exist in df
    actual_col_cfg = [(c, h, a) for c, h, a in col_cfg if c in df.columns]

    cols    = [c for c, _, _ in actual_col_cfg]
    headers = [h for _, h, _ in actual_col_cfg]
    aligns  = "".join(a for _, _, a in actual_col_cfg)

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{tab:{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{aligns}}}")
    lines.append(r"    \toprule")

    # Header row
    hdr = " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\"
    lines.append(f"    {hdr}")
    lines.append(r"    \midrule")

    # Data rows — insert \midrule between different N values
    prev_n = None
    group_col = "N" if "N" in cols else None

    for _, row in df.iterrows():
        if group_col is not None:
            curr_n = row[group_col]
            if prev_n is not None and curr_n != prev_n:
                lines.append(r"    \midrule")
            prev_n = curr_n

        cells = [fmt_cell(row[c]) for c in cols]
        lines.append("    " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Generate all tables ───────────────────────────────────────────────────────

master_inputs = []

for csv_name, col_cfg, caption, label, sort_by in TABLES:
    csv_path = os.path.join(METRICS_DIR, csv_name)
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_name} not found")
        continue

    df = pd.read_csv(csv_path)

    # Optional re-sort before rendering
    if sort_by:
        existing = [c for c in sort_by if c in df.columns]
        if existing:
            df = df.sort_values(existing).reset_index(drop=True)

    tex = make_table(df, col_cfg, caption, label)

    out_name = csv_name.replace("_metrics.csv", "_table.txt")
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w") as f:
        f.write(tex + "\n")

    print(f"  Saved {out_name:35s}  ({len(df)} rows)")
    master_inputs.append(out_name)


# ── Master file ───────────────────────────────────────────────────────────────
master_lines = [
    r"\documentclass{article}",
    r"\usepackage[a4paper, margin=2cm]{geometry}",
    r"\usepackage{booktabs}",
    r"\usepackage{caption}",
    r"\usepackage{float}",
    r"",
    r"\begin{document}",
    r"",
    r"\section*{Sequential (SEQ)}",
]

# Filtri per categorie
seq_files  = [f for f in master_inputs if f.startswith("seq_") or f.startswith("mkl_seq")]
cuda_files = [f for f in master_inputs if f.startswith("cuda_")]
omp_files  = [f for f in master_inputs if f.startswith("omp_") or f.startswith("mkl_omp")]
mpi_files  = [f for f in master_inputs if f.startswith("mpi_")]

for f in seq_files:
    master_lines.append(f"\\input{{{f}}}")
    master_lines.append(r"\medskip")

master_lines += [r"", r"\clearpage", r"\section*{CUDA}", r""]
for f in cuda_files:
    master_lines.append(f"\\input{{{f}}}")
    master_lines.append(r"\medskip")

master_lines += [r"", r"\clearpage", r"\section*{OpenMP (OMP)}", r""]
for f in omp_files:
    master_lines.append(f"\\input{{{f}}}")
    master_lines.append(r"\medskip")

master_lines += [r"", r"\clearpage", r"\section*{MPI}", r""]
for f in mpi_files:
    master_lines.append(f"\\input{{{f}}}")
    master_lines.append(r"\medskip")

master_lines += [r"\end{document}"]

master_path = os.path.join(OUT_DIR, "master.tex")
with open(master_path, "w") as f:
    f.write("\n".join(master_lines) + "\n")

print(f"\nGenerazione completata. Master file: {master_path}")