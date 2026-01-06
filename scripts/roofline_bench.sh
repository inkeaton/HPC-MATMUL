#!/usr/bin/env bash
# ======================================================================================
# ROOFLINE MODEL BENCHMARKING SCRIPT FOR MATRIX MULTIPLICATION
# ======================================================================================
#
# DESCRIPTION:
#   Comprehensive benchmarking script that:
#   1. Characterizes system performance using likwid (peak GFLOPS, memory bandwidth)
#   2. Tests multiple compilers (gcc, icx) with various optimization flags
#   3. Generates assembly and vectorization reports for each configuration
#   4. Profiles each binary using likwid-perfctr
#   5. Plots a roofline model with all execution points
#
# USAGE:
#   ./roofline_bench.sh [OPTIONS]
#
# OPTIONS:
#   -s, --source FILE   Source file to benchmark (default: ../src/matmul_seq.c)
#   -n, --runs N        Number of benchmark runs per configuration (default: 3)
#   -o, --output DIR    Output directory (default: ./roofline_results)
#   -c, --core ID       CPU core to pin execution to (default: 0)
#   -h, --help          Show this help message
#
# REQUIREMENTS:
#   - likwid (likwid-bench, likwid-perfctr) for roofline data
#   - gcc and/or icx compilers
#   - Python 3 with matplotlib and numpy for plotting
#
# ======================================================================================

set -o pipefail

# ======================================================================================
# CONFIGURATION
# ======================================================================================

# Default values
SOURCE_FILE="../src/matmul_seq.c"
NUM_RUNS=3
OUTPUT_DIR="./roofline_results"
CORE_ID=0
MATRIX_SIZE=5000

# Compiler flag configurations
declare -A FLAGS_GCC=(
    ["O0"]="-O0 -fopenmp-simd"
    ["O2"]="-O2"
    ["O3"]="-O3"
    ["O3_native"]="-O3 -march=native"
    ["O3_native_unroll"]="-O3 -march=native -funroll-loops"
    ["Ofast_native"]="-Ofast -march=native -funroll-loops"
)

declare -A FLAGS_ICX=(
    ["O2"]="-O2"
    ["O3"]="-O3"
    ["O3_xHost"]="-O3 -xHost"
    ["O3_xHost_zmm"]="-O3 -xHost -qopt-zmm-usage=high"
    ["O3_xHost_fast"]="-O3 -xHost -fp-model=fast -funroll-loops"
)

# Vectorization report flags (compiler-specific)
declare -A VEC_REPORT_FLAGS=(
    ["gcc"]="-fopt-info-vec-optimized -fopt-info-vec-missed"
    ["icx"]="-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize"
)

# Common compilation flags
# Note: _POSIX_C_SOURCE=199309L enables clock_gettime and CLOCK_MONOTONIC
COMMON_FLAGS="-std=c11 -D_POSIX_C_SOURCE=199309L -Wall -DENABLE_TIMING"

# ======================================================================================
# COLORS AND OUTPUT FORMATTING
# ======================================================================================

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# Status symbols
SYMBOL_OK="✓"
SYMBOL_WARN="⚠"
SYMBOL_ERROR="✗"
SYMBOL_INFO="ℹ"
SYMBOL_ARROW="→"
SYMBOL_BULLET="•"

# Logging functions
log_info()    { echo -e "${BLUE}${SYMBOL_INFO}${RESET} $*"; }
log_success() { echo -e "${GREEN}${SYMBOL_OK}${RESET} $*"; }
log_warn()    { echo -e "${YELLOW}${SYMBOL_WARN}${RESET} ${YELLOW}$*${RESET}"; }
log_error()   { echo -e "${RED}${SYMBOL_ERROR}${RESET} ${RED}$*${RESET}"; }
log_step()    { echo -e "  ${DIM}${SYMBOL_ARROW}${RESET} $*"; }
log_detail()  { echo -e "    ${DIM}${SYMBOL_BULLET}${RESET} $*"; }

print_header() {
    local title="$1"
    local width=70
    local padding=$(( (width - ${#title} - 2) / 2 ))
    echo ""
    echo -e "${BOLD}${CYAN}$(printf '═%.0s' $(seq 1 $width))${RESET}"
    echo -e "${BOLD}${CYAN}$(printf ' %.0s' $(seq 1 $padding)) $title $(printf ' %.0s' $(seq 1 $padding))${RESET}"
    echo -e "${BOLD}${CYAN}$(printf '═%.0s' $(seq 1 $width))${RESET}"
    echo ""
}

print_section() {
    local title="$1"
    echo ""
    echo -e "${BOLD}${MAGENTA}── $title ──${RESET}"
    echo ""
}

print_subsection() {
    local title="$1"
    echo -e "  ${BOLD}${BLUE}$title${RESET}"
}

# ======================================================================================
# CLEANUP AND ERROR HANDLING
# ======================================================================================

CLEANUP_NEEDED=false

cleanup() {
    if [[ "$CLEANUP_NEEDED" == "true" ]]; then
        log_warn "Cleaning up temporary files..."
        # Add cleanup actions here if needed
    fi
}

trap cleanup EXIT

error_exit() {
    log_error "$1"
    exit 1
}

# ======================================================================================
# ARGUMENT PARSING
# ======================================================================================

show_help() {
    cat << EOF
${BOLD}Roofline Model Benchmarking Script${RESET}

${BOLD}USAGE:${RESET}
    $0 [OPTIONS]

${BOLD}OPTIONS:${RESET}
    -s, --source FILE   Source file to benchmark (default: $SOURCE_FILE)
    -n, --runs N        Number of benchmark runs per configuration (default: $NUM_RUNS)
    -o, --output DIR    Output directory (default: $OUTPUT_DIR)
    -c, --core ID       CPU core to pin execution to (default: $CORE_ID)
    -h, --help          Show this help message

${BOLD}EXAMPLE:${RESET}
    $0 -s src/matmul_seq.c -n 5 -o results/

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--source)
                SOURCE_FILE="$2"
                shift 2
                ;;
            -n|--runs)
                NUM_RUNS="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -c|--core)
                CORE_ID="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# ======================================================================================
# SYSTEM DETECTION
# ======================================================================================

# Availability flags
HAS_LIKWID=false
HAS_LIKWID_PERFCTR=false
HAS_GCC=false
HAS_ICX=false
HAS_AVX512=false
HAS_PYTHON=false
LIKWID_PERMS_OK=false

detect_tools() {
    print_section "Phase 1: System Detection"
    
    # Check for likwid-bench
    if command -v likwid-bench &> /dev/null; then
        HAS_LIKWID=true
        log_success "likwid-bench found: $(command -v likwid-bench)"
    else
        log_warn "likwid-bench not found - system characterization will be skipped"
    fi
    
    # Check for likwid-perfctr
    if command -v likwid-perfctr &> /dev/null; then
        HAS_LIKWID_PERFCTR=true
        log_success "likwid-perfctr found: $(command -v likwid-perfctr)"
    else
        log_warn "likwid-perfctr not found - profiling will be limited to timing only"
    fi
    
    # Check likwid permissions (MSR access)
    if [[ "$HAS_LIKWID" == "true" || "$HAS_LIKWID_PERFCTR" == "true" ]]; then
        check_likwid_permissions
    fi
    
    # Check for GCC
    if command -v gcc &> /dev/null; then
        HAS_GCC=true
        local gcc_version=$(gcc --version | head -n1)
        log_success "GCC found: $gcc_version"
    else
        log_warn "GCC not found - GCC configurations will be skipped"
    fi
    
    # Check for Intel ICX
    if command -v icx &> /dev/null; then
        HAS_ICX=true
        local icx_version=$(icx --version 2>&1 | head -n1)
        log_success "Intel ICX found: $icx_version"
    else
        log_warn "Intel ICX not found - ICX configurations will be skipped"
    fi
    
    # Verify at least one compiler is available
    if [[ "$HAS_GCC" == "false" && "$HAS_ICX" == "false" ]]; then
        error_exit "No compilers found! Please install gcc or icx."
    fi
    
    # Check for AVX-512 support
    detect_avx512
    
    # Check for Python with required packages
    check_python
    
    # Verify source file exists
    if [[ ! -f "$SOURCE_FILE" ]]; then
        error_exit "Source file not found: $SOURCE_FILE"
    else
        log_success "Source file found: $SOURCE_FILE"
    fi
}

check_likwid_permissions() {
    print_subsection "Checking likwid permissions..."
    
    # Check if MSR module is loaded
    if [[ ! -e /dev/cpu/0/msr ]]; then
        log_warn "MSR device not found (/dev/cpu/0/msr)"
        log_detail "Try: sudo modprobe msr"
        LIKWID_PERMS_OK=false
        return
    fi
    
    # Check if we can read MSR
    if [[ -r /dev/cpu/0/msr ]]; then
        log_success "MSR read access available"
        LIKWID_PERMS_OK=true
    else
        log_warn "Cannot read MSR device - likwid may require elevated permissions"
        log_detail "Options:"
        log_detail "  1. Run with sudo"
        log_detail "  2. Add user to 'msr' group"
        log_detail "  3. Use likwid accessDaemon (likwid-accessD)"
        LIKWID_PERMS_OK=false
    fi
    
    # Check for frequency scaling warnings
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
        local governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
        if [[ "$governor" != "performance" ]]; then
            log_warn "CPU governor is '$governor' (recommended: 'performance')"
            log_detail "For consistent results: sudo cpupower frequency-set -g performance"
        fi
    fi
}

detect_avx512() {
    print_subsection "Detecting SIMD capabilities..."
    
    if grep -q "avx512" /proc/cpuinfo 2>/dev/null; then
        HAS_AVX512=true
        log_success "AVX-512 support detected"
    else
        HAS_AVX512=false
        log_info "AVX-512 not available (will use AVX2/AVX)"
    fi
    
    # Show available vector extensions
    local simd_flags=""
    grep -q "sse4_2" /proc/cpuinfo && simd_flags+="SSE4.2 "
    grep -q "avx2" /proc/cpuinfo && simd_flags+="AVX2 "
    grep -q "avx512f" /proc/cpuinfo && simd_flags+="AVX-512F "
    grep -q "avx512vl" /proc/cpuinfo && simd_flags+="AVX-512VL "
    grep -q "fma" /proc/cpuinfo && simd_flags+="FMA "
    
    log_detail "Available SIMD: $simd_flags"
}

check_python() {
    print_subsection "Checking Python environment..."
    
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 --version 2>&1)
        log_success "Python3 found: $py_version"
        
        # Check for matplotlib and numpy
        local missing_pkgs=""
        python3 -c "import matplotlib" 2>/dev/null || missing_pkgs+="matplotlib "
        python3 -c "import numpy" 2>/dev/null || missing_pkgs+="numpy "
        
        if [[ -n "$missing_pkgs" ]]; then
            log_warn "Missing Python packages: $missing_pkgs"
            log_detail "Install with: pip install $missing_pkgs"
            HAS_PYTHON=false
        else
            log_success "Required Python packages available (matplotlib, numpy)"
            HAS_PYTHON=true
        fi
    else
        log_warn "Python3 not found - roofline plot will not be generated"
        HAS_PYTHON=false
    fi
}

# ======================================================================================
# DIRECTORY SETUP
# ======================================================================================

setup_directories() {
    print_section "Phase 2: Directory Setup"
    
    # Create main output directory
    mkdir -p "$OUTPUT_DIR"
    log_success "Created output directory: $OUTPUT_DIR"
    
    # Create subdirectories for each compiler
    for compiler in gcc icx; do
        mkdir -p "$OUTPUT_DIR/bin/$compiler"
        mkdir -p "$OUTPUT_DIR/asm/$compiler"
        mkdir -p "$OUTPUT_DIR/logs/$compiler"
        mkdir -p "$OUTPUT_DIR/reports/$compiler"
    done
    
    # Create data directory
    mkdir -p "$OUTPUT_DIR/data"
    
    log_step "Directory structure created:"
    log_detail "bin/      - Compiled binaries"
    log_detail "asm/      - Assembly output"
    log_detail "logs/     - Execution logs"
    log_detail "reports/  - Vectorization reports"
    log_detail "data/     - Benchmark data and plots"
    
    CLEANUP_NEEDED=true
}

# ======================================================================================
# SYSTEM CHARACTERIZATION (LIKWID)
# ======================================================================================

PEAK_GFLOPS=0
PEAK_BANDWIDTH=0

characterize_system() {
    print_section "Phase 3: System Characterization"
    
    if [[ "$HAS_LIKWID" == "false" ]]; then
        log_warn "Skipping system characterization (likwid-bench not available)"
        log_info "Using default roofline values or specify manually"
        return
    fi
    
    if [[ "$LIKWID_PERMS_OK" == "false" ]]; then
        log_warn "Attempting likwid-bench despite permission warnings..."
    fi
    
    local sysinfo_file="$OUTPUT_DIR/data/system_info.txt"
    
    # Save system information
    {
        echo "===== SYSTEM INFORMATION ====="
        echo "Date: $(date)"
        echo "Hostname: $(hostname 2>/dev/null || cat /etc/hostname 2>/dev/null || echo 'unknown')"
        echo ""
        echo "===== CPU INFO ====="
        lscpu 2>/dev/null || cat /proc/cpuinfo | head -50
        echo ""
    } > "$sysinfo_file"
    
    # Measure peak FLOPS
    measure_peak_flops
    
    # Measure peak bandwidth
    measure_peak_bandwidth
    
    # Save roofline parameters
    {
        echo ""
        echo "===== ROOFLINE PARAMETERS ====="
        echo "Peak GFLOPS: $PEAK_GFLOPS"
        echo "Peak Bandwidth (GB/s): $PEAK_BANDWIDTH"
        echo "Ridge Point (FLOP/Byte): $(echo "scale=2; $PEAK_GFLOPS / $PEAK_BANDWIDTH" | bc 2>/dev/null || echo "N/A")"
    } >> "$sysinfo_file"
    
    log_success "System info saved to: $sysinfo_file"
}

measure_peak_flops() {
    print_subsection "Measuring peak floating-point performance..."
    
    local bench_type
    if [[ "$HAS_AVX512" == "true" ]]; then
        bench_type="peakflops_avx512_fma"
    else
        bench_type="peakflops_avx_fma"
    fi
    
    log_step "Running likwid-bench -t $bench_type..."
    
    local output
    output=$(likwid-bench -t "$bench_type" -W N:2GB:1 2>&1) || {
        log_warn "likwid-bench peakflops failed, trying alternative..."
        # Try without specific workgroup
        output=$(likwid-bench -t "$bench_type" 2>&1) || {
            log_error "Could not measure peak FLOPS"
            PEAK_GFLOPS=100  # Default fallback
            return
        }
    }
    
    # Parse GFLOPS from output
    # Example line: "MFlops/s:            217980.45"
    PEAK_GFLOPS=$(echo "$output" | grep -i "MFlops/s" | tail -1 | awk '{print $NF/1000}')
    
    if [[ -z "$PEAK_GFLOPS" || "$PEAK_GFLOPS" == "0" ]]; then
        log_warn "Could not parse peak GFLOPS, using default value"
        PEAK_GFLOPS=100
    else
        log_success "Peak performance: ${BOLD}${PEAK_GFLOPS} GFLOPS${RESET}"
    fi
    
    # Save raw output
    echo "$output" > "$OUTPUT_DIR/data/likwid_peakflops.txt"
}

measure_peak_bandwidth() {
    print_subsection "Measuring peak memory bandwidth..."
    
    local bench_type
    if [[ "$HAS_AVX512" == "true" ]]; then
        bench_type="load_avx512"
    else
        bench_type="load_avx"
    fi
    
    log_step "Running likwid-bench -t $bench_type..."
    
    local output
    output=$(likwid-bench -t "$bench_type" -W N:2GB:1 2>&1) || {
        log_warn "likwid-bench load failed, trying alternative..."
        output=$(likwid-bench -t load 2>&1) || {
            log_error "Could not measure bandwidth"
            PEAK_BANDWIDTH=50  # Default fallback
            return
        }
    }
    
    # Parse bandwidth from output
    # Example line: "MByte/s:             58680.12"
    PEAK_BANDWIDTH=$(echo "$output" | grep -i "MByte/s" | tail -1 | awk '{print $NF/1000}')
    
    if [[ -z "$PEAK_BANDWIDTH" || "$PEAK_BANDWIDTH" == "0" ]]; then
        log_warn "Could not parse peak bandwidth, using default value"
        PEAK_BANDWIDTH=50
    else
        log_success "Peak bandwidth: ${BOLD}${PEAK_BANDWIDTH} GB/s${RESET}"
    fi
    
    # Save raw output
    echo "$output" > "$OUTPUT_DIR/data/likwid_bandwidth.txt"
}

# ======================================================================================
# COMPILATION PHASE
# ======================================================================================

# Results storage
declare -a RESULTS=()

compile_all() {
    print_section "Phase 4: Compilation"
    
    local total_configs=0
    local compiled=0
    
    # Count total configurations
    [[ "$HAS_GCC" == "true" ]] && total_configs=$((total_configs + ${#FLAGS_GCC[@]}))
    [[ "$HAS_ICX" == "true" ]] && total_configs=$((total_configs + ${#FLAGS_ICX[@]}))
    
    log_info "Compiling $total_configs configurations..."
    echo ""
    
    # Compile GCC configurations
    if [[ "$HAS_GCC" == "true" ]]; then
        print_subsection "GCC Configurations"
        for config_name in "${!FLAGS_GCC[@]}"; do
            compiled=$((compiled + 1))
            log_step "[$compiled/$total_configs] gcc_$config_name"
            compile_single "gcc" "$config_name" "${FLAGS_GCC[$config_name]}"
        done
    fi
    
    # Compile ICX configurations
    if [[ "$HAS_ICX" == "true" ]]; then
        echo ""
        print_subsection "Intel ICX Configurations"
        for config_name in "${!FLAGS_ICX[@]}"; do
            compiled=$((compiled + 1))
            log_step "[$compiled/$total_configs] icx_$config_name"
            compile_single "icx" "$config_name" "${FLAGS_ICX[$config_name]}"
        done
    fi
    
    echo ""
    log_success "Compilation complete: $compiled configurations"
}

compile_single() {
    local compiler="$1"
    local config_name="$2"
    local flags="$3"
    
    local bin_name="${compiler}_${config_name}"
    local bin_path="$OUTPUT_DIR/bin/$compiler/$bin_name"
    local asm_path="$OUTPUT_DIR/asm/$compiler/${bin_name}.s"
    local report_path="$OUTPUT_DIR/reports/$compiler/${bin_name}_vec_report.txt"
    local log_path="$OUTPUT_DIR/logs/$compiler/${bin_name}.log"
    
    local full_flags="$COMMON_FLAGS $flags"
    
    # Initialize log
    {
        echo "===== COMPILATION LOG ====="
        echo "Binary: $bin_name"
        echo "Compiler: $compiler"
        echo "Flags: $full_flags"
        echo "Source: $SOURCE_FILE"
        echo "Date: $(date)"
        echo ""
    } > "$log_path"
    
    # Compile binary
    if ! $compiler $full_flags "$SOURCE_FILE" -o "$bin_path" -lm 2>> "$log_path"; then
        log_error "    Compilation failed for $bin_name"
        return 1
    fi
    
    local bin_size=$(du -h "$bin_path" | cut -f1)
    log_detail "Binary: $bin_path ($bin_size)"
    
    # Generate assembly
    if $compiler $full_flags -S "$SOURCE_FILE" -o "$asm_path" 2>> "$log_path"; then
        log_detail "Assembly: $asm_path"
        analyze_assembly "$asm_path" "$log_path"
    fi
    
    # Generate vectorization report
    generate_vec_report "$compiler" "$config_name" "$flags" "$report_path" "$log_path"
    
    echo "" >> "$log_path"
}

analyze_assembly() {
    local asm_file="$1"
    local log_path="$2"
    
    # Count SIMD instructions (use tr to ensure clean output)
    local sse_count=$(grep -cE '\b(movaps|movups|addps|mulps|movapd|movupd|addpd|mulpd)\b' "$asm_file" 2>/dev/null | tr -d '\n' || echo -n 0)
    local avx_count=$(grep -cE '\bv(movaps|movups|addps|mulps|movapd|movupd|addpd|mulpd|fmadd)\b' "$asm_file" 2>/dev/null | tr -d '\n' || echo -n 0)
    local avx512_count=$(grep -cE '\b(zmm|k[0-7])\b' "$asm_file" 2>/dev/null | tr -d '\n' || echo -n 0)
    local fma_count=$(grep -cE '\bvfmadd|vfmsub|vfnmadd|vfnmsub\b' "$asm_file" 2>/dev/null | tr -d '\n' || echo -n 0)
    
    # Ensure we have numeric values
    [[ -z "$sse_count" ]] && sse_count=0
    [[ -z "$avx_count" ]] && avx_count=0
    [[ -z "$avx512_count" ]] && avx512_count=0
    [[ -z "$fma_count" ]] && fma_count=0
    
    {
        echo "===== ASSEMBLY ANALYSIS ====="
        echo "SSE instructions: $sse_count"
        echo "AVX instructions: $avx_count"
        echo "AVX-512 instructions: $avx512_count"
        echo "FMA instructions: $fma_count"
        echo ""
    } >> "$log_path"
    
    log_detail "SIMD: SSE=$sse_count AVX=$avx_count AVX512=$avx512_count FMA=$fma_count"
}

generate_vec_report() {
    local compiler="$1"
    local config_name="$2"
    local flags="$3"
    local report_path="$4"
    local log_path="$5"
    
    local vec_flags="${VEC_REPORT_FLAGS[$compiler]}"
    local full_flags="$COMMON_FLAGS $flags $vec_flags"
    
    # Capture vectorization report (redirect stderr to file for gcc/icx)
    {
        echo "===== VECTORIZATION REPORT ====="
        echo "Compiler: $compiler"
        echo "Flags: $full_flags"
        echo ""
    } > "$report_path"
    
    # Compile again with vectorization reports enabled
    $compiler $full_flags "$SOURCE_FILE" -o /dev/null 2>> "$report_path" || true
    
    # Count vectorized loops
    local vec_success=$(grep -ciE '(vectorized|LOOP WAS VECTORIZED|loop vectorized)' "$report_path" 2>/dev/null || echo 0)
    local vec_missed=$(grep -ciE '(not vectorized|missed|failed)' "$report_path" 2>/dev/null || echo 0)
    
    {
        echo ""
        echo "===== VECTORIZATION REPORT ====="
        echo "Successfully vectorized loops: $vec_success"
        echo "Failed/Missed vectorizations: $vec_missed"
        echo ""
    } >> "$log_path"
    
    log_detail "Vectorization: success=$vec_success missed=$vec_missed"
}

# ======================================================================================
# PROFILING PHASE
# ======================================================================================

profile_all() {
    print_section "Phase 5: Profiling"
    
    # Initialize CSV results file
    local csv_file="$OUTPUT_DIR/data/results.csv"
    echo "Compiler,Config,Flags,Binary_Size_KB,Mean_Time_s,Min_Time_s,Max_Time_s,GFLOPS,Operational_Intensity,SSE,AVX,AVX512,FMA" > "$csv_file"
    
    local total=0
    local current=0
    
    # Count binaries
    shopt -s nullglob
    for bin in "$OUTPUT_DIR"/bin/*/*; do
        [[ -f "$bin" ]] && total=$((total + 1))
    done
    shopt -u nullglob
    
    log_info "Profiling $total binaries ($NUM_RUNS runs each)..."
    echo ""
    
    # Profile each binary
    for compiler_dir in "$OUTPUT_DIR"/bin/*/; do
        local compiler=$(basename "$compiler_dir")
        print_subsection "Profiling $compiler binaries"
        
        for bin_path in "$compiler_dir"/*; do
            [[ -f "$bin_path" ]] || continue
            current=$((current + 1))
            
            local bin_name=$(basename "$bin_path")
            local config_name="${bin_name#${compiler}_}"
            
            log_step "[$current/$total] $bin_name"
            profile_single "$compiler" "$config_name" "$bin_path"
        done
        echo ""
    done
    
    log_success "Profiling complete. Results: $csv_file"
}

profile_single() {
    local compiler="$1"
    local config_name="$2"
    local bin_path="$3"
    
    local bin_name=$(basename "$bin_path")
    local log_path="$OUTPUT_DIR/logs/$compiler/${bin_name}.log"
    
    local times=()
    local gflops_values=()
    
    {
        echo ""
        echo "===== EXECUTION PROFILING ====="
        echo "Runs: $NUM_RUNS"
        echo "Core: $CORE_ID"
        echo ""
    } >> "$log_path"
    
    # Run benchmark multiple times
    for ((run=1; run<=NUM_RUNS; run++)); do
        local output
        local elapsed
        
        # Run with taskset to pin to core
        output=$(taskset -c "$CORE_ID" "$bin_path" 2>&1)
        
        # Extract timing from stderr output: [seq] N=5000, block=64, elapsed=X.XXX s
        elapsed=$(echo "$output" | grep -oP 'elapsed=\K[0-9.]+' || echo "")
        
        if [[ -z "$elapsed" ]]; then
            log_warn "    Run $run: Could not extract timing"
            continue
        fi
        
        times+=("$elapsed")
        
        # Calculate GFLOPS: 2 * N^3 / time / 10^9
        local gflops=$(echo "scale=2; 2 * $MATRIX_SIZE * $MATRIX_SIZE * $MATRIX_SIZE / $elapsed / 1000000000" | bc)
        gflops_values+=("$gflops")
        
        echo "Run $run: ${elapsed}s (${gflops} GFLOPS)" >> "$log_path"
    done
    
    # Calculate statistics
    local mean_time=$(calc_mean "${times[@]}")
    local min_time=$(calc_min "${times[@]}")
    local max_time=$(calc_max "${times[@]}")
    local mean_gflops=$(calc_mean "${gflops_values[@]}")
    
    log_detail "Time: ${mean_time}s (min=${min_time}s, max=${max_time}s)"
    log_detail "Performance: ${mean_gflops} GFLOPS"
    
    # Profile with likwid if available
    local op_intensity="N/A"
    if [[ "$HAS_LIKWID_PERFCTR" == "true" && "$LIKWID_PERMS_OK" == "true" ]]; then
        op_intensity=$(profile_with_likwid "$bin_path" "$log_path")
    fi
    
    # Get SIMD counts from log
    local sse=$(grep "SSE instructions:" "$log_path" | awk '{print $NF}' | tail -1)
    local avx=$(grep "AVX instructions:" "$log_path" | awk '{print $NF}' | tail -1)
    local avx512=$(grep "AVX-512 instructions:" "$log_path" | awk '{print $NF}' | tail -1)
    local fma=$(grep "FMA instructions:" "$log_path" | awk '{print $NF}' | tail -1)
    
    # Get flags
    local flags=""
    case "$compiler" in
        gcc) flags="${FLAGS_GCC[$config_name]}" ;;
        icx) flags="${FLAGS_ICX[$config_name]}" ;;
    esac
    
    # Get binary size
    local bin_size_kb=$(du -k "$bin_path" | cut -f1)
    
    # Append to CSV
    echo "$compiler,$config_name,\"$flags\",$bin_size_kb,$mean_time,$min_time,$max_time,$mean_gflops,$op_intensity,$sse,$avx,$avx512,$fma" >> "$OUTPUT_DIR/data/results.csv"
    
    # Store for roofline
    RESULTS+=("$compiler|$config_name|$mean_gflops|$op_intensity")
    
    {
        echo ""
        echo "===== SUMMARY ====="
        echo "Mean Time: ${mean_time}s"
        echo "Min Time: ${min_time}s"
        echo "Max Time: ${max_time}s"
        echo "Mean GFLOPS: $mean_gflops"
        echo "Operational Intensity: $op_intensity"
    } >> "$log_path"
}

profile_with_likwid() {
    local bin_path="$1"
    local log_path="$2"

    # Wrapper mode (no -m). We collect FLOPS and MEM separately and derive AI = GFLOPS / BW(GB/s).

    # 1) FLOPS measurement
    local flops_out
    flops_out=$(likwid-perfctr -C "$CORE_ID" -g FLOPS_DP "$bin_path" 2>&1) || {
        echo "N/A"
        return
    }

    # 2) Memory bandwidth measurement
    local mem_out
    mem_out=$(likwid-perfctr -C "$CORE_ID" -g MEM "$bin_path" 2>&1) || {
        # Log FLOPS output at least
        echo "$flops_out" >> "$log_path"
        echo "N/A"
        return
    }

    # Save full outputs
    echo "$flops_out" >> "$log_path"
    echo "$mem_out"   >> "$log_path"

    # Parse DP MFLOP/s (take first numeric field at end of line containing MFLOP/s)
    local mflops=$(echo "$flops_out" | grep -i "MFLOP/s" | awk '{print $NF}' | head -1)
    # Parse memory bandwidth in MBytes/s
    local mbytes=$(echo "$mem_out" | grep -i "MBytes/s" | awk '{print $NF}' | head -1)

    if [[ -z "$mflops" || -z "$mbytes" ]]; then
        echo "N/A"
        return
    fi

    # Convert to GFLOPS and GB/s
    local gflops
    local gbs
    gflops=$(echo "scale=4; $mflops / 1000" | bc 2>/dev/null)
    gbs=$(echo "scale=4; $mbytes / 1000" | bc 2>/dev/null)

    # Guard against zero/invalid
    if [[ -z "$gflops" || -z "$gbs" || "$gbs" == "0" ]]; then
        echo "N/A"
        return
    fi

    # Operational intensity = GFLOPS / GB/s (FLOP/Byte)
    local op_intensity
    op_intensity=$(echo "scale=4; $gflops / $gbs" | bc 2>/dev/null)

    if [[ -z "$op_intensity" ]]; then
        echo "N/A"
    else
        echo "$op_intensity"
    fi
}

# ======================================================================================
# STATISTICS FUNCTIONS
# ======================================================================================

calc_mean() {
    local sum=0
    local count=0
    for val in "$@"; do
        sum=$(echo "$sum + $val" | bc)
        count=$((count + 1))
    done
    [[ $count -eq 0 ]] && echo "0" && return
    echo "scale=3; $sum / $count" | bc
}

calc_min() {
    local min=""
    for val in "$@"; do
        if [[ -z "$min" ]] || (( $(echo "$val < $min" | bc -l) )); then
            min=$val
        fi
    done
    echo "${min:-0}"
}

calc_max() {
    local max=""
    for val in "$@"; do
        if [[ -z "$max" ]] || (( $(echo "$val > $max" | bc -l) )); then
            max=$val
        fi
    done
    echo "${max:-0}"
}

# ======================================================================================
# ROOFLINE PLOT GENERATION
# ======================================================================================

generate_roofline_plot() {
    print_section "Phase 6: Roofline Plot Generation"
    
    if [[ "$HAS_PYTHON" == "false" ]]; then
        log_warn "Skipping plot generation (Python/matplotlib not available)"
        return
    fi
    
    local csv_file="$OUTPUT_DIR/data/results.csv"
    local plot_script="$OUTPUT_DIR/data/plot_roofline.py"
    local plot_output="$OUTPUT_DIR/data/roofline.png"
    
    log_step "Generating Python plotting script..."
    
    cat > "$plot_script" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Roofline Model Plot Generator
Reads benchmark results and plots them against theoretical roofline limits.
"""

import sys
import csv
import numpy as np
import matplotlib

# Force non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Configuration
PEAK_GFLOPS = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
PEAK_BANDWIDTH = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
CSV_FILE = sys.argv[3] if len(sys.argv) > 3 else "results.csv"
OUTPUT_FILE = sys.argv[4] if len(sys.argv) > 4 else "roofline.png"

# Ridge point: where memory-bound transitions to compute-bound
RIDGE_POINT = PEAK_GFLOPS / PEAK_BANDWIDTH

# Compiler colors
COLORS = {
    'gcc': '#E74C3C',      # Red
    'icx': '#3498DB',      # Blue
    'icc': '#9B59B6',      # Purple
    'clang': '#2ECC71',    # Green
}

# Markers for different optimization levels
MARKERS = {
    'O0': 'o',
    'O2': 's',
    'O3': '^',
    'O3_native': 'D',
    'O3_native_unroll': 'p',
    'Ofast_native': '*',
    'O3_xHost': 'v',
    'O3_xHost_zmm': 'h',
    'O3_xHost_fast': 'P',
}

def get_marker(config):
    """Get marker for configuration name."""
    for key, marker in MARKERS.items():
        if key in config:
            return marker
    return 'o'

def roofline(x, peak_gflops, peak_bw):
    """Calculate roofline performance."""
    return np.minimum(peak_gflops, peak_bw * x)

def main():
    print(f"Generating roofline plot...")
    print(f"  Peak Performance: {PEAK_GFLOPS:.2f} GFLOPS")
    print(f"  Peak Bandwidth: {PEAK_BANDWIDTH:.2f} GB/s")
    print(f"  Ridge Point: {RIDGE_POINT:.2f} FLOP/Byte")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot roofline
    x = np.logspace(-1, 2, 500)  # 0.1 to 100 FLOP/Byte
    y = roofline(x, PEAK_GFLOPS, PEAK_BANDWIDTH)
    ax.loglog(x, y, 'k-', linewidth=2, label='Roofline')
    
    # Fill regions
    ax.fill_between(x, 0.1, y, alpha=0.1, color='gray')
    
    # Add ridge point marker
    ax.axvline(x=RIDGE_POINT, color='gray', linestyle='--', alpha=0.5)
    ax.text(RIDGE_POINT * 1.1, PEAK_GFLOPS * 0.5, 
            f'Ridge Point\n({RIDGE_POINT:.1f} FLOP/B)', 
            fontsize=9, alpha=0.7)
    
    # Read and plot benchmark results
    try:
        with open(CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                compiler = row['Compiler']
                config = row['Config']
                try:
                    gflops = float(row['GFLOPS']) if row['GFLOPS'] else 0
                except Exception:
                    continue

                # Try to get operational intensity, estimate if missing or malformed
                op_int_str = (row.get('Operational_Intensity', '') or '').strip()
                op_int = None
                if op_int_str in ['', 'N/A', '|']:
                    op_int = None
                else:
                    try:
                        op_int = float(op_int_str)
                    except Exception:
                        op_int = None

                if gflops <= 0:
                    continue

                # Fallback estimate for GEMM operational intensity when missing
                if op_int is None:
                    op_int = 5.0  # Rough AI for large GEMM (FLOP/Byte)
                
                color = COLORS.get(compiler, '#7F8C8D')
                marker = get_marker(config)
                
                ax.scatter(op_int, gflops, 
                          c=color, marker=marker, s=100, 
                          edgecolors='black', linewidths=0.5,
                          label=f'{compiler}_{config}', zorder=5)
                
                # Add label
                ax.annotate(f'{config}', (op_int, gflops),
                           textcoords="offset points", 
                           xytext=(5, 5), fontsize=7, alpha=0.8)
    
    except FileNotFoundError:
        print(f"Warning: CSV file '{CSV_FILE}' not found")
    except Exception as e:
        print(f"Warning: Error reading CSV: {e}")
    
    # Labels and formatting
    ax.set_xlabel('Operational Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title('Roofline Model - Matrix Multiplication Benchmark', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.1, PEAK_GFLOPS * 1.5)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Create legend for compilers
    legend_elements = [Patch(facecolor=color, label=comp) 
                       for comp, color in COLORS.items() if comp in ['gcc', 'icx']]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add annotations
    ax.annotate(f'Peak: {PEAK_GFLOPS:.1f} GFLOPS', 
                xy=(50, PEAK_GFLOPS), fontsize=10, alpha=0.7)
    ax.annotate('Memory Bound', xy=(0.2, 1), fontsize=10, 
                style='italic', alpha=0.6)
    ax.annotate('Compute Bound', xy=(20, PEAK_GFLOPS * 0.8), fontsize=10, 
                style='italic', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {OUTPUT_FILE}")
    
    # Also save as PDF
    pdf_output = OUTPUT_FILE.replace('.png', '.pdf')
    plt.savefig(pdf_output, bbox_inches='tight')
    print(f"  PDF saved: {pdf_output}")

if __name__ == '__main__':
    main()
PYTHON_SCRIPT

    log_step "Executing plot script..."
    
    if python3 "$plot_script" "$PEAK_GFLOPS" "$PEAK_BANDWIDTH" "$csv_file" "$plot_output" 2>&1; then
        log_success "Roofline plot generated: $plot_output"
    else
        log_error "Failed to generate roofline plot"
    fi
}

# ======================================================================================
# SUMMARY GENERATION
# ======================================================================================

generate_summary() {
    print_section "Phase 7: Summary"
    
    local summary_file="$OUTPUT_DIR/data/summary.txt"
    
    {
        echo "═══════════════════════════════════════════════════════════════════════"
        echo "              ROOFLINE BENCHMARK SUMMARY"
        echo "═══════════════════════════════════════════════════════════════════════"
        echo ""
        echo "Date: $(date)"
        echo "Source: $SOURCE_FILE"
        echo "Matrix Size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
        echo "Runs per configuration: $NUM_RUNS"
        echo ""
        echo "───────────────────────────────────────────────────────────────────────"
        echo "SYSTEM CHARACTERISTICS"
        echo "───────────────────────────────────────────────────────────────────────"
        echo "Peak Performance: ${PEAK_GFLOPS} GFLOPS"
        echo "Peak Bandwidth: ${PEAK_BANDWIDTH} GB/s"
        echo "Ridge Point: $(echo "scale=2; $PEAK_GFLOPS / $PEAK_BANDWIDTH" | bc 2>/dev/null || echo "N/A") FLOP/Byte"
        echo ""
        echo "───────────────────────────────────────────────────────────────────────"
        echo "RESULTS"
        echo "───────────────────────────────────────────────────────────────────────"
        
        if [[ -f "$OUTPUT_DIR/data/results.csv" ]]; then
            # Print CSV as formatted table
            column -t -s',' "$OUTPUT_DIR/data/results.csv" 2>/dev/null || cat "$OUTPUT_DIR/data/results.csv"
        fi
        
        echo ""
        echo "───────────────────────────────────────────────────────────────────────"
        echo "OUTPUT FILES"
        echo "───────────────────────────────────────────────────────────────────────"
        echo "Results CSV: $OUTPUT_DIR/data/results.csv"
        echo "Roofline Plot: $OUTPUT_DIR/data/roofline.png"
        echo "System Info: $OUTPUT_DIR/data/system_info.txt"
        echo ""
    } > "$summary_file"
    
    # Print summary to console
    echo ""
    log_success "Benchmark complete!"
    echo ""
    echo -e "${BOLD}Results Summary:${RESET}"
    echo ""
    
    # Print top performers
    if [[ -f "$OUTPUT_DIR/data/results.csv" ]]; then
        echo -e "  ${CYAN}Configuration${RESET}            ${CYAN}GFLOPS${RESET}"
        echo "  ───────────────────────────────────"
        tail -n +2 "$OUTPUT_DIR/data/results.csv" | sort -t',' -k8 -rn | head -5 | while IFS=',' read -r compiler config flags size mean min max gflops rest; do
            printf "  %-24s %s\n" "${compiler}_${config}" "$gflops"
        done
    fi
    
    echo ""
    echo -e "${BOLD}Output Directory:${RESET} $OUTPUT_DIR"
    echo ""
    echo -e "  ${DIM}├── data/${RESET}"
    echo -e "  ${DIM}│   ├── results.csv${RESET}        - All benchmark results"
    echo -e "  ${DIM}│   ├── roofline.png${RESET}       - Roofline visualization"
    echo -e "  ${DIM}│   └── system_info.txt${RESET}    - System characterization"
    echo -e "  ${DIM}├── bin/${RESET}                   - Compiled binaries"
    echo -e "  ${DIM}├── asm/${RESET}                   - Assembly files"
    echo -e "  ${DIM}├── logs/${RESET}                  - Per-binary detailed logs"
    echo -e "  ${DIM}└── reports/${RESET}               - Vectorization reports"
    echo ""
}

# ======================================================================================
# MAIN
# ======================================================================================

main() {
    print_header "ROOFLINE MODEL BENCHMARKING"
    
    log_info "Starting benchmark suite..."
    log_detail "Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
    log_detail "Operations: $((2 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE)) FLOPs per multiply"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Phase 1: Detect tools and capabilities
    detect_tools
    
    # Phase 2: Setup directories
    setup_directories
    
    # Phase 3: Characterize system performance
    characterize_system
    
    # Phase 4: Compile all configurations
    compile_all
    
    # Phase 5: Profile all binaries
    profile_all
    
    # Phase 6: Generate roofline plot
    generate_roofline_plot
    
    # Phase 7: Generate summary
    generate_summary
    
    CLEANUP_NEEDED=false
    log_success "All phases completed successfully!"
}

# Run main
main "$@"
