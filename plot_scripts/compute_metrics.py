"""
compute_metrics.py
------------------
For each benchmark part compute:

    GFLOPS           = 2 * N^3 / (Time_s * 1e9)
    Speedup_vs_seq   = T_baseline / Time_s
    Efficiency_%     = (Speedup_vs_seq / p) * 100   (OMP & MPI only, p = threads/ranks)

All float values are rounded to 2 decimal places.

Baseline = gcc seq-ikj, Align=0, Restrict=0, same N
           (plain sequential loop, no alignment hint, no restrict)

Output: one CSV per variant type under plot_scripts/metrics_tables/
"""

import os
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_CSV  = os.path.join(SCRIPT_DIR, "seq",  "benchmark_results_seq.csv")
OMP_CSV  = os.path.join(SCRIPT_DIR, "omp",  "benchmark_results_omp.csv")
MPI_CSV  = os.path.join(SCRIPT_DIR, "mpi",  "benchmark_results_mpi.csv")
CUDA_CSV = os.path.join(SCRIPT_DIR, "cuda", "cuda_matrix_benchmark_times.csv")

OUT_DIR = os.path.join(SCRIPT_DIR, "metrics_tables")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load data ────────────────────────────────────────────────────────────────
seq_df  = pd.read_csv(SEQ_CSV)
omp_df  = pd.read_csv(OMP_CSV)
mpi_df  = pd.read_csv(MPI_CSV)
cuda_df = pd.read_csv(CUDA_CSV)


# ── Baseline: gcc seq-ikj, Align=0, Restrict=0 for each N ───────────────────
baseline_rows = seq_df[
    (seq_df["Test"]     == "seq-ikj") &
    (seq_df["Compiler"] == "gcc")     &
    (seq_df["Align"]    == 0)         &
    (seq_df["Restrict"] == 0)
]
baseline = baseline_rows.groupby("N")["Time_s"].min().to_dict()

print("Baseline (T_seq  =  gcc seq-ikj  no-align  no-restrict):")
for n, t in sorted(baseline.items()):
    gf = 2 * n**3 / (t * 1e9)
    print(f"  N={n:6d}  T={t:10.3f} s   GFLOPS={gf:.3f}")
print()


# ── Helper functions ─────────────────────────────────────────────────────────

def gflops(N, T):
    return round(2.0 * float(N)**3 / (T * 1e9), 2)

def speedup(N, T):
    tb = baseline.get(int(N))
    return round(tb / T, 2) if tb is not None else None

def efficiency_pct(N, T, p):
    """Parallel efficiency as a percentage: (Speedup / p) * 100."""
    s = speedup(N, T)
    return round((s / p) * 100.0, 2) if (s is not None and p > 0) else None


def round_floats(df):
    """Round every float column to 2 decimal places."""
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(2)
    return df


def add_seq_metrics(df):
    df = df.copy()
    df["GFLOPS"]         = df.apply(lambda r: gflops(r["N"], r["Time_s"]),  axis=1)
    df["Speedup_vs_seq"] = df.apply(lambda r: speedup(r["N"], r["Time_s"]), axis=1)
    return round_floats(df)


def add_parallel_metrics(df):
    df = df.copy()
    df["GFLOPS"]         = df.apply(lambda r: gflops(r["N"], r["Time_s"]),                         axis=1)
    df["Speedup_vs_seq"] = df.apply(lambda r: speedup(r["N"], r["Time_s"]),                         axis=1)
    df["Efficiency_%"]   = df.apply(lambda r: efficiency_pct(r["N"], r["Time_s"], r["Threads/Ranks"]), axis=1)
    return round_floats(df)


def save_type(df, test_name, out_dir, label=""):
    sub = df[df["Test"] == test_name].copy()
    if sub.empty:
        print(f"  [SKIP] No data for '{test_name}'")
        return
    fname = test_name.replace("-", "_") + "_metrics.csv"
    sub.to_csv(os.path.join(out_dir, fname), index=False)
    rows = len(sub)
    print(f"  Saved {fname:38s}  ({rows} rows)  {label}")


# ══════════════════════════════════════════════════════════════════════════════
# SEQ
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("SEQ  —  seq-ikj | seq-pad | seq-tile | mkl-seq")
print("=" * 62)
seq_m = add_seq_metrics(seq_df)
for t in ["seq-ikj", "seq-pad", "seq-tile", "mkl-seq"]:
    save_type(seq_m, t, OUT_DIR)
seq_m.to_csv(os.path.join(OUT_DIR, "seq_all_metrics.csv"), index=False)
print(f"  Saved {'seq_all_metrics.csv':38s}  ({len(seq_m)} rows)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# OMP  (+Efficiency)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("OMP  —  omp-ikj | omp-tile | mkl-omp   [+Efficiency_% = (Speedup/p)*100]")
print("=" * 62)
omp_m = add_parallel_metrics(omp_df)
for t in ["omp-ikj", "omp-tile", "mkl-omp"]:
    save_type(omp_m, t, OUT_DIR)
omp_m.to_csv(os.path.join(OUT_DIR, "omp_all_metrics.csv"), index=False)
print(f"  Saved {'omp_all_metrics.csv':38s}  ({len(omp_m)} rows)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# MPI  (+Efficiency)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("MPI  —  mpi-ikj | mpi-cannon | mpi-summa | scalapack   [+Efficiency_%]")
print("=" * 62)
mpi_m = add_parallel_metrics(mpi_df)
for t in ["mpi-ikj", "mpi-cannon", "mpi-summa", "scalapack"]:
    save_type(mpi_m, t, OUT_DIR)
mpi_m.to_csv(os.path.join(OUT_DIR, "mpi_all_metrics.csv"), index=False)
print(f"  Saved {'mpi_all_metrics.csv':38s}  ({len(mpi_m)} rows)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# CUDA  — wide format  →  long format,  then GFLOPS + Speedup_vs_seq
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("CUDA  —  cublas | tile-16 | tile-32   [+GFLOPS +Speedup_vs_seq]")
print("=" * 62)

# Parse N from "4100*4100"  →  4100
cuda_df["N"] = cuda_df["matrix_size"].str.split("*").str[0].astype(int)

# Rename columns to friendly test names
cuda_df = cuda_df.rename(columns={
    "cublas_time":           "cublas",
    "shared_tiling_16_time": "tile-16",
    "shared_tiling_32_time": "tile-32",
})

# Melt to long format
cuda_long = cuda_df[["N", "cublas", "tile-16", "tile-32"]].melt(
    id_vars="N", var_name="Test", value_name="Time_s"
)

# Add metrics
cuda_long["GFLOPS"]         = cuda_long.apply(lambda r: gflops(r["N"], r["Time_s"]),  axis=1)
cuda_long["Speedup_vs_seq"] = cuda_long.apply(lambda r: speedup(r["N"], r["Time_s"]), axis=1)
cuda_long = round_floats(cuda_long)

# Sort nicely
cuda_long = cuda_long.sort_values(["Test", "N"]).reset_index(drop=True)

# Save per type
for t in ["cublas", "tile-16", "tile-32"]:
    sub = cuda_long[cuda_long["Test"] == t].copy()
    fname = "cuda_" + t.replace("-", "") + "_metrics.csv"
    sub.to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f"  Saved {fname:38s}  ({len(sub)} rows)")

# Save combined
cuda_long.to_csv(os.path.join(OUT_DIR, "cuda_all_metrics.csv"), index=False)
print(f"  Saved {'cuda_all_metrics.csv':38s}  ({len(cuda_long)} rows)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("SUMMARY  –  Best result per (Part, Type, N)")
print("=" * 72)

# SEQ
print("\n[SEQ]")
print(f"  {'Test':<12} {'N':>6}  {'Time_s':>8}  {'GFLOPS':>8}  {'Speedup':>8}  Compiler")
print(f"  {'-'*12} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
best_seq = seq_m.sort_values("Time_s").groupby(["Test","N"]).first().reset_index()
for _, r in best_seq.sort_values(["Test","N"]).iterrows():
    sp = r["Speedup_vs_seq"]
    print(f"  {r['Test']:<12} {int(r['N']):>6}  {r['Time_s']:>8.2f}  {r['GFLOPS']:>8.2f}  {sp:>8.2f}  {r['Compiler']}")

# OMP
print("\n[OMP]  (p = threads used for best run)")
print(f"  {'Test':<12} {'N':>6}  {'p':>3}  {'Time_s':>8}  {'GFLOPS':>8}  {'Speedup':>8}  {'Effic.%':>8}  Compiler")
print(f"  {'-'*12} {'-'*6}  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
best_omp = omp_m.sort_values("Time_s").groupby(["Test","N"]).first().reset_index()
for _, r in best_omp.sort_values(["Test","N"]).iterrows():
    sp = r["Speedup_vs_seq"]; ef = r["Efficiency_%"]
    print(f"  {r['Test']:<12} {int(r['N']):>6}  {int(r['Threads/Ranks']):>3}  "
          f"{r['Time_s']:>8.2f}  {r['GFLOPS']:>8.2f}  {sp:>8.2f}  {ef:>8.2f}  {r['Compiler']}")

# MPI
print("\n[MPI]  (p = ranks used for best run)")
print(f"  {'Test':<12} {'N':>6}  {'p':>3}  {'Time_s':>8}  {'GFLOPS':>8}  {'Speedup':>8}  {'Effic.%':>8}  Compiler")
print(f"  {'-'*12} {'-'*6}  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
best_mpi = mpi_m.sort_values("Time_s").groupby(["Test","N"]).first().reset_index()
for _, r in best_mpi.sort_values(["Test","N"]).iterrows():
    sp = r["Speedup_vs_seq"]; ef = r["Efficiency_%"]
    print(f"  {r['Test']:<12} {int(r['N']):>6}  {int(r['Threads/Ranks']):>3}  "
          f"{r['Time_s']:>8.2f}  {r['GFLOPS']:>8.2f}  {sp:>8.2f}  {ef:>8.2f}  {r['Compiler']}")

# CUDA
print("\n[CUDA]  (baseline = same gcc seq-ikj for same N)")
print(f"  {'Test':<10} {'N':>6}  {'Time_s':>8}  {'GFLOPS':>8}  {'Speedup':>8}")
print(f"  {'-'*10} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
for _, r in cuda_long.sort_values(["Test","N"]).iterrows():
    sp = r["Speedup_vs_seq"]
    print(f"  {r['Test']:<10} {int(r['N']):>6}  {r['Time_s']:>8.2f}  {r['GFLOPS']:>8.2f}  {sp:>8.2f}")

print()
print(f"All CSV tables saved to: {OUT_DIR}")
