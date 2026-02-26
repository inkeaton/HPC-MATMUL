#!/usr/bin/env bash

# ==========================================
# 1. VISUAL SETUP & HARDWARE DETECTION
# ==========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[1;35m'
NC='\033[0m' # No Color

# Automatically extract CPU model from Debian (xargs trims whitespace)
CPU_MODEL=$(grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   Matrix Multiplication Benchmarking Suite           ${NC}"
echo -e "${CYAN}   CPU:  ${YELLOW}${CPU_MODEL}${NC}"
echo -e "${CYAN}   Time: ${YELLOW}${TIMESTAMP}${NC}"
echo -e "${CYAN}======================================================${NC}"

# ==========================================
# 2. CONFIGURATION ARRAYS
# ==========================================
COMPILERS=("gcc" "icx") 
SIZES=(1000 2000 5000)
THREAD_COUNTS=(1 2 4 8 16)
TILE_SIZES=(16 32 64 128) # Test different cache block boundaries
ITERATIONS=3              # Run each test 3 times and keep the fastest
CSV_FILE="benchmark_results.csv"

# Overwrite CSV entirely for a fresh run and set new headers
echo "CPU,Compiler,Test,N,Align,Restrict,Threads/Ranks,TileSize,Time_s" > "$CSV_FILE"

# ==========================================
# 3. CORE EXECUTOR FUNCTION
# ==========================================
# Params: <comp> <src> <name> <N> <ALIGN> <RESTRICT> <IS_OMP> <IS_MPI> <THREADS> <TILE>
run_test() {
    local comp=$1
    local src=$2
    local name=$3
    local n_val=$4
    local align=$5
    local restrict=$6
    local is_omp=$7
    local is_mpi=$8
    local num_threads=${9:-1}
    local tile_size=${10:-0} # 0 means no tile size
    
    local bin_out="/tmp/matmul_bench_bin"
    
    if [ ! -f "$src" ]; then
        echo -e "${RED}[SKIP] Source file $src not found.${NC}"
        return
    fi

    # Determine compiler wrapper and flags
    local cmd_comp="$comp"
    local flags="-O3 -DTIME -DN=$n_val"
    
    if [[ "$comp" == "gcc" ]]; then
        flags="$flags -march=native"
        [[ "$is_omp" == "1" ]] && flags="$flags -fopenmp"
    elif [[ "$comp" == "icx" || "$comp" == "icc" ]]; then
        flags="$flags -xHost"
        [[ "$is_omp" == "1" ]] && flags="$flags -qopenmp"
    fi

    # Handle MPI wrappers
    if [[ "$is_mpi" == "1" ]]; then
        if [[ "$comp" == "gcc" ]]; then cmd_comp="mpicc"; fi
        if [[ "$comp" == "icx" ]]; then cmd_comp="mpiicx"; fi
        if [[ "$comp" == "icc" ]]; then cmd_comp="mpiicc"; fi
    fi

    if ! command -v "$cmd_comp" &> /dev/null; then
        echo -e "${RED}[SKIP] Compiler '$cmd_comp' not found.${NC}"
        return
    fi

    # Append Macro Flags
    [[ "$align" == "1" ]] && flags="$flags -DALIGN"
    [[ "$restrict" == "1" ]] && flags="$flags -DRESTRICT"
    [[ "$tile_size" != "0" ]] && flags="$flags -DTILE_SIZE=$tile_size"

    # 1. COMPILE ONCE
    $cmd_comp $flags "$src" -o "$bin_out" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}[FAIL] Compilation failed: $cmd_comp $src${NC}"
        return
    fi

    # 2. SET UP EXECUTION
    local run_cmd="$bin_out"
    if [[ "$is_omp" == "1" ]]; then
        run_cmd="env OMP_NUM_THREADS=$num_threads $bin_out"
    elif [[ "$is_mpi" == "1" ]]; then
        run_cmd="mpirun -np $num_threads $bin_out"
    fi

    # 3. BEST-OF-N ITERATIONS
    local min_time="999999" 
    
    for ((i=1; i<=ITERATIONS; i++)); do
        local output
        output=$(eval "$run_cmd" 2>&1)
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}[FAIL] Execution crashed: $name${NC}"
            return
        fi

        local elapsed=$(echo "$output" | grep -oP 'elapsed=\K[0-9.]+' | head -n 1)

        if [[ -z "$elapsed" ]]; then
             echo -e "${RED}[FAIL] Could not parse Time.${NC}"
             return
        fi

        # AWK to find the smallest execution time
        min_time=$(awk -v e="$elapsed" -v m="$min_time" 'BEGIN { printf "%.3f", (e < m) ? e : m }')
    done

    # Formatting helper for terminal table
    local t_disp="-"
    [[ "$tile_size" != "0" ]] && t_disp="$tile_size"

    # 4. LOG RESULTS
    printf "  ${GREEN}✔${NC} %-8s | N=%-5s | A=%d,R=%d | Thr=%-2s | Tile=%-3s | ${YELLOW}%.3fs${NC}\n" "$name" "$n_val" "$align" "$restrict" "$num_threads" "$t_disp" "$min_time"
    
    # Append to CSV
    echo "\"$CPU_MODEL\",$comp,$name,$n_val,$align,$restrict,$num_threads,$t_disp,$min_time" >> "$CSV_FILE"
}

# ==========================================
# 4. TEST SUITES (Clean Output Organization)
# ==========================================

run_sequential() {
    echo -e "\n${YELLOW}======================================================${NC}"
    echo -e "${YELLOW}                 SEQUENTIAL TESTS                     ${NC}"
    echo -e "${YELLOW}======================================================${NC}"
    
    for comp in "${COMPILERS[@]}"; do
        echo -e "\n${CYAN}>>> COMPILER: ${comp^^} <<<${NC}"

        echo -e "\n${MAGENTA}--- Baseline & Optimized (seq-ikj) ---${NC}"
        for n in "${SIZES[@]}"; do
            run_test "$comp" "src/matmul_seq_ikj.c" "seq-ikj" "$n" 0 0 0 0 1 0
            run_test "$comp" "src/matmul_seq_ikj.c" "seq-ikj" "$n" 1 1 0 0 1 0
        done

        echo -e "\n${MAGENTA}--- Matrix Padding (seq-pad) ---${NC}"
        for n in "${SIZES[@]}"; do
            run_test "$comp" "src/matmul_seq_pad.c" "seq-pad" "$n" 1 1 0 0 1 0
        done

        echo -e "\n${MAGENTA}--- Cache Tiling (seq-tile) ---${NC}"
        for n in "${SIZES[@]}"; do
            for ts in "${TILE_SIZES[@]}"; do
                run_test "$comp" "src/matmul_seq_tile.c" "seq-tile" "$n" 1 1 0 0 1 "$ts"
            done
        done
    done
}

run_openmp() {
    echo -e "\n${YELLOW}======================================================${NC}"
    echo -e "${YELLOW}                   OPENMP TESTS                       ${NC}"
    echo -e "${YELLOW}======================================================${NC}"
    
    for comp in "${COMPILERS[@]}"; do
        echo -e "\n${CYAN}>>> COMPILER: ${comp^^} <<<${NC}"

        echo -e "\n${MAGENTA}--- Thread Scaling (omp-ikj) ---${NC}"
        for n in "${SIZES[@]}"; do
            for t in "${THREAD_COUNTS[@]}"; do
                run_test "$comp" "src/matmul_omp_ikj.c" "omp-ikj" "$n" 1 1 1 0 "$t" 0
            done
        done

        echo -e "\n${MAGENTA}--- Thread & Tile Scaling (omp-tile) ---${NC}"
        for n in "${SIZES[@]}"; do
            for t in "${THREAD_COUNTS[@]}"; do
                for ts in "${TILE_SIZES[@]}"; do
                    run_test "$comp" "src/matmul_omp_tile.c" "omp-tile" "$n" 1 1 1 0 "$t" "$ts"
                done
            done
        done
    done
}

run_mpi() {
    echo -e "\n${YELLOW}======================================================${NC}"
    echo -e "${YELLOW}                     MPI TESTS                        ${NC}"
    echo -e "${YELLOW}======================================================${NC}"
    
    for comp in "${COMPILERS[@]}"; do
        echo -e "\n${CYAN}>>> COMPILER: ${comp^^} <<<${NC}"

        echo -e "\n${MAGENTA}--- Distributed Process Scaling (mpi-ikj) ---${NC}"
        for n in "${SIZES[@]}"; do
            for t in "${THREAD_COUNTS[@]}"; do
                run_test "$comp" "src/matmul_mpi.c" "mpi-ikj" "$n" 1 1 0 1 "$t" 0
            done
        done
    done
}

# ==========================================
# 5. INPUT PARAMETER ROUTING
# ==========================================
TARGET=${1:-"all"}

case "$TARGET" in
    "seq") run_sequential ;;
    "omp") run_openmp ;;
    "mpi") run_mpi ;;
    "all")
        run_sequential
        run_openmp
        run_mpi
        ;;
    *)
        echo -e "${RED}Invalid target: $TARGET${NC}"
        echo "Usage: ./benchmark.sh [seq | omp | mpi | all]"
        exit 1
        ;;
esac

echo -e "\n${CYAN}Benchmarking complete. Best times saved to $CSV_FILE.${NC}"