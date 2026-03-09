r"""
generate_latex.py
-----------------
Reads every metrics CSV from metrics_tables/ and produces a booktabs-style
LaTeX table for each one.

Output: plot_scripts/latex_tables/
    <name>_table.txt   -- one file per variant
    master.tex         -- file that \input{}s all tables (ready to compile)
"""

import os
import pandas as pd

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(SCRIPT_DIR, "metrics_tables")
OUT_DIR     = os.path.join(SCRIPT_DIR, "latex_tables")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Column configs ────────────────────────────────────────────────────────────
# (csv_column, latex_header, alignment_char)
# alignment: l=left  r=right  c=center

SEQ_IKJ_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("Align",          "Align",          "c"),
    ("Restrict",       "Restrict",       "c"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup_vs_seq", "Speedup",        "r"),
]

SEQ_PAD_COLS = SEQ_IKJ_COLS   # same shape

SEQ_TILE_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("TileSize",       "Tile",           "r"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup_vs_seq", "Speedup",        "r"),
]

MKL_SEQ_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup_vs_seq", "Speedup",        "r"),
]

CUDA_COLS = [
    ("N",              r"$N$",           "r"),
    ("Test",           "Kernel",         "l"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup_vs_seq", "Speedup",        "r"),
]

OMP_IKJ_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("Threads/Ranks",  "Threads",        "r"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup_vs_seq", "Speedup",        "r"),
    ("Efficiency_%",   r"Efficiency (\%)", "r"),
]

OMP_TILE_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("Threads/Ranks",  "Threads",        "r"),
    ("TileSize",       "Tile",           "r"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup_vs_seq", "Speedup",        "r"),
    ("Efficiency_%",   r"Efficiency (\%)", "r"),
]

MKL_OMP_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("Threads/Ranks",  "Threads",        "r"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup_vs_seq", "Speedup",        "r"),
    ("Efficiency_%",   r"Efficiency (\%)", "r"),
]

MPI_IKJ_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("Threads/Ranks",  "Ranks",          "r"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup_vs_seq", "Speedup",        "r"),
    ("Efficiency_%",   r"Efficiency (\%)", "r"),
]

MPI_CANNON_COLS  = MPI_IKJ_COLS
MPI_SUMMA_COLS   = MPI_IKJ_COLS

SCALAPACK_COLS = [
    ("N",              r"$N$",           "r"),
    ("Compiler",       "Compiler",       "l"),
    ("Threads/Ranks",  "Ranks",          "r"),
    ("TileSize",       "Tile",           "r"),
    ("Time_s",         "Time (s)",       "r"),
    ("GFLOPS",         "GFLOPS",         "r"),
    ("Speedup_vs_seq", "Speedup",        "r"),
    ("Efficiency_%",   r"Efficiency (\%)", "r"),
]

# ── Table metadata ────────────────────────────────────────────────────────────
# (csv_filename, col_config, caption, label, sort_by)
# sort_by: list of columns to sort by before rendering (None = keep CSV order)
TABLES = [
    # SEQ
    ("seq_ikj_metrics.csv",    SEQ_IKJ_COLS,
     "Sequential IKJ matrix multiplication -- Intel i9-12900K",
     "seq_ikj", None),
    ("seq_pad_metrics.csv",    SEQ_PAD_COLS,
     "Sequential IKJ with padding -- Intel i9-12900K",
     "seq_pad", None),
    ("seq_tile_metrics.csv",   SEQ_TILE_COLS,
     "Sequential cache-tiled matrix multiplication -- Intel i9-12900K",
     "seq_tile", None),
    ("mkl_seq_metrics.csv",    MKL_SEQ_COLS,
     "MKL single-threaded DGEMM -- Intel i9-12900K",
     "mkl_seq", None),
    # CUDA  – sort by N first so kernels for the same N are adjacent
    ("cuda_all_metrics.csv",   CUDA_COLS,
     "CUDA matrix multiplication kernels -- NVIDIA Tesla T4",
     "cuda_all", ["N", "Test"]),
    ("cuda_cublas_metrics.csv",CUDA_COLS,
     "CUDA cuBLAS DGEMM -- NVIDIA Tesla T4",
     "cuda_cublas", ["N"]),
    ("cuda_tile16_metrics.csv",CUDA_COLS,
     r"CUDA shared-memory tiled kernel (tile $=16$) -- NVIDIA Tesla T4",
     "cuda_tile16", ["N"]),
    ("cuda_tile32_metrics.csv",CUDA_COLS,
     r"CUDA shared-memory tiled kernel (tile $=32$) -- NVIDIA Tesla T4",
     "cuda_tile32", ["N"]),
    # OMP
    ("omp_ikj_metrics.csv",    OMP_IKJ_COLS,
     "OpenMP IKJ parallel matrix multiplication -- Intel i9-12900K",
     "omp_ikj", None),
    ("omp_tile_metrics.csv",   OMP_TILE_COLS,
     "OpenMP cache-tiled parallel matrix multiplication -- Intel i9-12900K",
     "omp_tile", None),
    ("mkl_omp_metrics.csv",    MKL_OMP_COLS,
     "OpenMP MKL parallel DGEMM -- Intel i9-12900K",
     "mkl_omp", None),
    # MPI
    ("mpi_ikj_metrics.csv",    MPI_IKJ_COLS,
     "MPI row-slab IKJ matrix multiplication -- Intel i9-12900K",
     "mpi_ikj", None),
    ("mpi_cannon_metrics.csv", MPI_CANNON_COLS,
     "MPI Cannon's algorithm -- Intel i9-12900K",
     "mpi_cannon", None),
    ("mpi_summa_metrics.csv",  MPI_SUMMA_COLS,
     "MPI SUMMA algorithm -- Intel i9-12900K",
     "mpi_summa", None),
    ("scalapack_metrics.csv",  SCALAPACK_COLS,
     "ScaLAPACK pdgemm -- Intel i9-12900K",
     "scalapack", None),
]


# ── LaTeX builder ─────────────────────────────────────────────────────────────

def fmt_cell(val):
    """Format a single cell value for LaTeX."""
    if pd.isna(val) or str(val).strip() in ("", "-", "nan"):
        return "--"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def make_table(df: pd.DataFrame, col_cfg: list, caption: str, label: str) -> str:
    """Return a complete LaTeX table string (booktabs style)."""
    # Keep only cols that exist in df
    col_cfg = [(c, h, a) for c, h, a in col_cfg if c in df.columns]

    cols    = [c for c, _, _ in col_cfg]
    headers = [h for _, h, _ in col_cfg]
    aligns  = "".join(a for _, _, a in col_cfg)

    sub = df[cols].copy()

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

    for _, row in sub.iterrows():
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

    # Optional re-sort before rendering (e.g. CUDA: group by N first)
    if sort_by:
        existing = [c for c in sort_by if c in df.columns]
        if existing:
            df = df.sort_values(existing).reset_index(drop=True)

    tex = make_table(df, col_cfg, caption, label)

    out_name = csv_name.replace("_metrics.csv", "_table.txt")
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w") as f:
        f.write(tex + "\n")

    print(f"  Saved {out_name:35s}  ({len(df)} rows,  {len(col_cfg)} cols)")
    master_inputs.append(out_name)


# ── Master file ───────────────────────────────────────────────────────────────
master_lines = [
    r"% master.tex  –  compile with:  pdflatex master.tex",
    r"% Requires: \usepackage{booktabs} in your preamble",
    r"",
    r"\documentclass{article}",
    r"\usepackage[a4paper, margin=2cm]{geometry}",
    r"\usepackage{booktabs}",
    r"\usepackage{caption}",
    r"\usepackage{float}",
    r"",
    r"\begin{document}",
    r"",
    r"\section*{SEQ}",
]
seq_files  = [f for f in master_inputs if f.startswith("seq_") or f.startswith("mkl_seq")]
omp_files  = [f for f in master_inputs if f.startswith("omp_") or f.startswith("mkl_omp")]
mpi_files  = [f for f in master_inputs if f.startswith("mpi_") or f.startswith("scalapack")]
cuda_files = [f for f in master_inputs if f.startswith("cuda_")]

for f in seq_files:
    master_lines.append(f"\\input{{{f}}}")
    master_lines.append(r"\medskip")

master_lines += [r"", r"\clearpage", r"\section*{CUDA}", r""]
for f in cuda_files:
    master_lines.append(f"\\input{{{f}}}")
    master_lines.append(r"\medskip")

master_lines += [r"", r"\clearpage", r"\section*{OMP}", r""]
for f in omp_files:
    master_lines.append(f"\\input{{{f}}}")
    master_lines.append(r"\medskip")

master_lines += [r"", r"\clearpage", r"\section*{MPI}", r""]
for f in mpi_files:
    master_lines.append(f"\\input{{{f}}}")
    master_lines.append(r"\medskip")

master_lines += [r"", r"\end{document}"]

master_path = os.path.join(OUT_DIR, "master.tex")
with open(master_path, "w") as f:
    f.write("\n".join(master_lines) + "\n")

print(f"\n  Saved master.tex  ({len(master_inputs)} tables included)")
print(f"\nAll LaTeX tables saved to: {OUT_DIR}")
