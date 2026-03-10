"""
compute_metrics.py
------------------
For each benchmark part compute:

    GFLOPS           = 2 * N^3 / (Time_s * 1e9)
    Speedup          = T_baseline / Time_s
    Efficiency_%     = (Speedup / p) * 100   (OMP & MPI only, p = threads/ranks)

BASELINES UTILIZZATE:
    - Per SEQ: gcc seq-ikj (no align, no restrict) -> Calcola "Speedup" vs codice sequenziale base.
    - Per OMP, MPI, CUDA: mkl-seq (miglior compilatore) -> Calcola "Speedup" vs miglior algoritmo sequenziale noto.

Output: one CSV per variant type under plot_scripts/metrics_tables/
"""

import os
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_CSV  = os.path.join(SCRIPT_DIR, "seq",  "benchmark_results_seq.csv")
OMP_CSV  = os.path.join(SCRIPT_DIR, "omp",  "benchmark_results_omp_updated.csv")
MPI_CSV  = os.path.join(SCRIPT_DIR, "mpi",  "benchmark_results_mpi_updated.csv")

CUDA_SN_CSV = os.path.join(SCRIPT_DIR, "cuda", "cuda_shared_naive_benchmark.csv")
CUDA_CB_CSV = os.path.join(SCRIPT_DIR, "cuda", "cuda_cuBLAS_benchmark.csv")

OUT_DIR = os.path.join(SCRIPT_DIR, "metrics_tables")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load data ────────────────────────────────────────────────────────────────
# Se i file non esistono, creiamo dei DataFrame vuoti per non far crashare lo script
seq_df  = pd.read_csv(SEQ_CSV) if os.path.exists(SEQ_CSV) else pd.DataFrame()
omp_df  = pd.read_csv(OMP_CSV) if os.path.exists(OMP_CSV) else pd.DataFrame()
mpi_df  = pd.read_csv(MPI_CSV) if os.path.exists(MPI_CSV) else pd.DataFrame()

cuda_sn_df = pd.read_csv(CUDA_SN_CSV) if os.path.exists(CUDA_SN_CSV) else pd.DataFrame()
cuda_cb_df = pd.read_csv(CUDA_CB_CSV) if os.path.exists(CUDA_CB_CSV) else pd.DataFrame()


# ── BASELINE 1: gcc seq-ikj (Per calcolare lo speedup dei test sequenziali) ──
baseline_seq = {}
if not seq_df.empty:
    baseline_seq_rows = seq_df[
        (seq_df["Test"]     == "seq-ikj") &
        (seq_df["Compiler"] == "gcc")     &
        (seq_df["Align"]    == 0)         &
        (seq_df["Restrict"] == 0)
    ]
    baseline_seq = baseline_seq_rows.groupby("N")["Time_s"].min().to_dict()

# --- FORZATURA / OVERRIDE VALORI MANUALI GCC ---
expected_gcc_seq = {
    4100: 27.069,
    8200: 226.549,
    12300: 764.60,
    16400: 1812.39
}
baseline_seq.update(expected_gcc_seq) # Sovrascrive eventuali valori del CSV con quelli corretti

print("Baseline 1 - SEQ (T_base = gcc seq-ikj puro):")
for n, t in sorted(baseline_seq.items()):
    print(f"  N={n:6d}  T={t:10.3f} s")
print()


# ── BASELINE 2: mkl-seq (Per calcolare speedup ed efficienza di OMP/MPI/CUDA) ─
baseline_mkl = {}
if not seq_df.empty:
    baseline_mkl_rows = seq_df[seq_df["Test"] == "mkl-seq"]
    baseline_mkl = baseline_mkl_rows.groupby("N")["Time_s"].min().to_dict()

# --- FORZATURA / OVERRIDE VALORI MANUALI BEST SEQ (MKL) ---
expected_best_seq = {
    4100: 2.21,
    8200: 17.65,
    12300: 59.59,
    16400: 141.20
}
baseline_mkl.update(expected_best_seq) # Sovrascrive eventuali valori del CSV con quelli corretti

print("Baseline 2 - PARALLEL/CUDA (T_base = best seq / mkl-seq):")
for n, t in sorted(baseline_mkl.items()):
    print(f"  N={n:6d}  T={t:10.3f} s")
print()


# ── Helper functions (Sicure contro divisioni per zero, precisione mantenuta) ─

def gflops(N, T):
    if pd.isna(T) or T <= 1e-9:
        return 0.0
    return round(2.0 * float(N)**3 / (T * 1e9), 2)

def _exact_speedup(N, T, base_dict):
    """Calcola lo speedup non arrotondato (usato internamente per l'efficienza)."""
    tb = base_dict.get(int(N))
    if tb is None or pd.isna(T) or T <= 1e-9:
        return None
    return tb / T

def speedup(N, T, base_dict):
    """Calcola lo speedup e lo arrotonda a 2 decimali per le tabelle."""
    s = _exact_speedup(N, T, base_dict)
    return round(s, 2) if s is not None else None

def efficiency_pct(N, T, p, base_dict):
    """Calcola l'efficienza rispetto alla baseline passata, senza perdere precisione."""
    s_exact = _exact_speedup(N, T, base_dict)
    if s_exact is None or pd.isna(p) or p <= 0:
        return None
    return round((s_exact / p) * 100.0, 2)

def round_floats(df):
    """Arrotonda le colonne float a 2 decimali."""
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(2)
    return df


# ── Funzioni di aggiunta metriche ────────────────────────────────────────────

def add_seq_metrics(df):
    if df.empty: return df
    df = df.copy()
    df["GFLOPS"]  = df.apply(lambda r: gflops(r["N"], r["Time_s"]), axis=1)
    # Per i sequenziali usiamo Baseline 1 (gcc seq)
    df["Speedup"] = df.apply(lambda r: speedup(r["N"], r["Time_s"], baseline_seq), axis=1)
    return round_floats(df)

def add_parallel_metrics(df):
    if df.empty: return df
    df = df.copy()
    df["GFLOPS"]       = df.apply(lambda r: gflops(r["N"], r["Time_s"]), axis=1)
    # Per paralleli usiamo Baseline 2 (mkl)
    df["Speedup"]      = df.apply(lambda r: speedup(r["N"], r["Time_s"], baseline_mkl), axis=1)
    df["Efficiency_%"] = df.apply(lambda r: efficiency_pct(r["N"], r["Time_s"], r["Threads/Ranks"], baseline_mkl), axis=1)
    return round_floats(df)


def save_type(df, test_name, out_dir, label=""):
    if df.empty: return
    sub = df[df["Test"] == test_name].copy()
    if sub.empty:
        return
    fname = test_name.replace("-", "_") + "_metrics.csv"
    sub.to_csv(os.path.join(out_dir, fname), index=False)
    print(f"  Saved {fname:38s}  ({len(sub)} rows)  {label}")


# ══════════════════════════════════════════════════════════════════════════════
# SEQ (Usa gcc_seq come baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 66)
print("SEQ  —  seq-ikj | seq-pad | seq-tile | mkl-seq  [Baseline: gcc seq]")
print("=" * 66)
if not seq_df.empty:
    seq_m = add_seq_metrics(seq_df)
    for t in ["seq-ikj", "seq-pad", "seq-tile", "mkl-seq"]:
        save_type(seq_m, t, OUT_DIR)
    seq_m.to_csv(os.path.join(OUT_DIR, "seq_all_metrics.csv"), index=False)
    print(f"  Saved {'seq_all_metrics.csv':38s}  ({len(seq_m)} rows)\n")
else:
    print("  Nessun dato SEQ trovato.\n")


# ══════════════════════════════════════════════════════════════════════════════
# OMP (Usa mkl-seq come baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 66)
print("OMP  —  omp-ikj | omp-tile | mkl-omp   [Baseline: best seq]")
print("=" * 66)
if not omp_df.empty:
    omp_m = add_parallel_metrics(omp_df)
    for t in ["omp-ikj", "omp-tile", "mkl-omp"]:
        save_type(omp_m, t, OUT_DIR)
    omp_m.to_csv(os.path.join(OUT_DIR, "omp_all_metrics.csv"), index=False)
    print(f"  Saved {'omp_all_metrics.csv':38s}  ({len(omp_m)} rows)\n")
else:
    print("  Nessun dato OMP trovato.\n")


# ══════════════════════════════════════════════════════════════════════════════
# MPI (Usa mkl-seq come baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 66)
print("MPI  —  mpi-ikj | mpi-cannon | mpi-summa | scalapack   [Baseline: best seq]")
print("=" * 66)
if not mpi_df.empty:
    mpi_m = add_parallel_metrics(mpi_df)
    for t in ["mpi-ikj", "mpi-cannon", "mpi-summa", "scalapack"]:
        save_type(mpi_m, t, OUT_DIR)
    mpi_m.to_csv(os.path.join(OUT_DIR, "mpi_all_metrics.csv"), index=False)
    print(f"  Saved {'mpi_all_metrics.csv':38s}  ({len(mpi_m)} rows)\n")
else:
    print("  Nessun dato MPI trovato.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CUDA (Usa mkl-seq come baseline)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 75)
print("CUDA — Analisi Metriche (Naive, Shared, cuBLAS) [Baseline: best seq (MKL)]")
print("=" * 75)

# Percorsi dei nuovi file
CUDA_SN_CSV = os.path.join(SCRIPT_DIR, "cuda", "cuda_shared_naive_benchmark.csv")
CUDA_CB_CSV = os.path.join(SCRIPT_DIR, "cuda", "cuda_cuBLAS_benchmark.csv")

# Caricamento
cuda_sn_df = pd.read_csv(CUDA_SN_CSV) if os.path.exists(CUDA_SN_CSV) else pd.DataFrame()
cuda_cb_df = pd.read_csv(CUDA_CB_CSV) if os.path.exists(CUDA_CB_CSV) else pd.DataFrame()

if not cuda_sn_df.empty or not cuda_cb_df.empty:
    # 1. Standardizzazione cuBLAS (N, Algorithm, Precision, Time_s)
    if not cuda_cb_df.empty:
        cuda_cb_df = cuda_cb_df.rename(columns={'MatrixSize': 'N'})
        cuda_cb_df['BlockDim'] = ""
        cuda_cb_df['TileSize'] = ""
    
    # 2. Standardizzazione Naive/Shared (N, Algorithm, Precision, BlockDim, TileSize, Time_s)
    if not cuda_sn_df.empty:
        cuda_sn_df = cuda_sn_df.rename(columns={'MatrixSize': 'N'})
    
    # 3. Unione dei dataset
    cuda_combined = pd.concat([cuda_sn_df, cuda_cb_df], ignore_index=True)
    
    # 4. Formattazione Colonna Configurazione (BlockDim + Tile + Precision)
    def format_cuda_cfg(row):
        prec = row['Precision']
        algo = row['Algorithm']
        b_dim = str(row['BlockDim'])
        t_size = str(row['TileSize'])
        
        if algo == 'cuBLAS':
            return f"Library, {prec}"
        
        # Gestione Naive (senza Tile) e Shared (con Tile)
        if t_size == "" or pd.isna(row['TileSize']) or t_size == "nan":
            return f"dim.blocks: {b_dim}, {prec}"
        else:
            # Rimuove l'eventuale .0 se il tile è letto come float
            t_str = str(int(float(t_size))) 
            return f"dim.blocks: {b_dim}, tile {t_str}, {prec}"

    cuda_combined["Configuration"] = cuda_combined.apply(format_cuda_cfg, axis=1)
    
    # 5. Calcolo Metriche (GFLOPS e Speedup vs MKL)
    cuda_combined["GFLOPS"]  = cuda_combined.apply(lambda r: gflops(r["N"], r["Time_s"]), axis=1)
    cuda_combined["Speedup"] = cuda_combined.apply(lambda r: speedup(r["N"], r["Time_s"], baseline_mkl), axis=1)
    
    cuda_combined = round_floats(cuda_combined)
    
    # 6. Salvataggio CSV globale
    cuda_combined.to_csv(os.path.join(OUT_DIR, "cuda_all_metrics.csv"), index=False)
    
    # 7. Salvataggio file separati per Algoritmo (Best per ogni N)
    for algo in ["Naive", "Shared", "cuBLAS"]:
        sub = cuda_combined[cuda_combined["Algorithm"] == algo].copy()
        if not sub.empty:
            # Salviamo tutto il set per l'algoritmo
            fname = f"cuda_{algo.lower()}_metrics.csv"
            sub.to_csv(os.path.join(OUT_DIR, fname), index=False)
            print(f"  Saved {fname:38s} ({len(sub)} rows)")

    # 8. Visualizzazione SUMMARY a video
    print("\n[CUDA SUMMARY] Migliori performance per Algoritmo/Precisione:")
    print(f"  {'N':>6} | {'Algo':<8} | {'Prec':<7} | {'Time_s':>8} | {'GFLOPS':>9} | {'Speedup':>7}")
    print("-" * 75)
    
    # Raggruppiamo per trovare il tempo minimo per ogni N, Algoritmo e Precisione
    summary_cuda = cuda_combined.sort_values("Time_s").groupby(["N", "Algorithm", "Precision"]).first().reset_index()
    
    for _, r in summary_cuda.sort_values(["N", "Algorithm", "Precision"]).iterrows():
        sp = r["Speedup"]
        sp_str = f"{sp:>7.2f}" if pd.notna(sp) else f"{'N/A':>7}"
        print(f"  {int(r['N']):>6} | {r['Algorithm']:<8} | {r['Precision']:<7} | {r['Time_s']:>8.3f} | {r['GFLOPS']:>9.2f} | {sp_str}")

else:
    print("  Nessun dato CUDA trovato nei file specificati.")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("SUMMARY  –  Miglior risultato per (Categoria, Test, N)")
print("=" * 72)

# SEQ
if not seq_df.empty:
    print("\n[SEQ] Baseline = gcc seq")
    print(f"  {'Test':<12} {'N':>6}  {'Time_s':>8}  {'GFLOPS':>8}  {'Speedup':>8}  Compiler")
    print(f"  {'-'*12} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    best_seq = seq_m.sort_values("Time_s").groupby(["Test","N"]).first().reset_index()
    for _, r in best_seq.sort_values(["Test","N"]).iterrows():
        sp = r["Speedup"]
        sp_str = f"{sp:>8.2f}" if pd.notna(sp) else f"{'N/A':>8}"
        comp = r.get("Compiler", "N/A")
        print(f"  {r['Test']:<12} {int(r['N']):>6}  {r['Time_s']:>8.2f}  {r['GFLOPS']:>8.2f}  {sp_str}  {comp}")

# OMP
if not omp_df.empty:
    print("\n[OMP] Baseline = best seq (mkl)")
    print(f"  {'Test':<12} {'N':>6}  {'p':>3}  {'Time_s':>8}  {'GFLOPS':>8}  {'Speedup':>8}  {'Effic.%':>8}  Compiler")
    print(f"  {'-'*12} {'-'*6}  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    best_omp = omp_m.sort_values("Time_s").groupby(["Test","N"]).first().reset_index()
    for _, r in best_omp.sort_values(["Test","N"]).iterrows():
        sp = r["Speedup"]
        ef = r["Efficiency_%"]
        sp_str = f"{sp:>8.2f}" if pd.notna(sp) else f"{'N/A':>8}"
        ef_str = f"{ef:>8.2f}" if pd.notna(ef) else f"{'N/A':>8}"
        comp = r.get("Compiler", "N/A")
        print(f"  {r['Test']:<12} {int(r['N']):>6}  {int(r['Threads/Ranks']):>3}  "
              f"{r['Time_s']:>8.2f}  {r['GFLOPS']:>8.2f}  {sp_str}  {ef_str}  {comp}")

# MPI
if not mpi_df.empty:
    print("\n[MPI] Baseline = best seq (mkl)")
    print(f"  {'Test':<12} {'N':>6}  {'p':>3}  {'Time_s':>8}  {'GFLOPS':>8}  {'Speedup':>8}  {'Effic.%':>8}  Compiler")
    print(f"  {'-'*12} {'-'*6}  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    best_mpi = mpi_m.sort_values("Time_s").groupby(["Test","N"]).first().reset_index()
    for _, r in best_mpi.sort_values(["Test","N"]).iterrows():
        sp = r["Speedup"]
        ef = r["Efficiency_%"]
        sp_str = f"{sp:>8.2f}" if pd.notna(sp) else f"{'N/A':>8}"
        ef_str = f"{ef:>8.2f}" if pd.notna(ef) else f"{'N/A':>8}"
        comp = r.get("Compiler", "N/A")
        print(f"  {r['Test']:<12} {int(r['N']):>6}  {int(r['Threads/Ranks']):>3}  "
              f"{r['Time_s']:>8.2f}  {r['GFLOPS']:>8.2f}  {sp_str}  {ef_str}  {comp}")

# CUDA
print("\n" + "=" * 85)
print(f"{'SUMMARY TABLE (CUDA)':^85}")
print("=" * 85)

if not cuda_combined.empty:
    print(f"  {'N':>6} | {'Algorithm':<8} | {'Time_s':>8} | {'GFLOPS':>9} | {'Speedup':>7} | {'Configuration'}")
    print("-" * 85)
    
    # Seleziona il migliore per ogni N/Algoritmo/Precisione
    summary_cuda = cuda_combined.sort_values("Time_s").groupby(["N", "Algorithm", "Precision"]).first().reset_index()
    
    for _, r in summary_cuda.sort_values(["N", "Algorithm", "Precision"]).iterrows():
        print(f"  {int(r['N']):>6} | {r['Algorithm']:<8} | {r['Time_s']:>8.3f} | {r['GFLOPS']:>9.2f} | {r['Speedup']:>7.2f} | {r['Configuration']}")

print(f"\nFiles salvati in: {OUT_DIR}")

print()
print(f"Tutti i file CSV salvati nella cartella: {OUT_DIR}")