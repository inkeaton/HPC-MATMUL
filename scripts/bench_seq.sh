#!/bin/bash
#===============================================================================
# bench_seq.sh - Sequential Matrix Multiplication Benchmark Script
#
# This script compiles matmul.c with various compilers (gcc, icc, icx) and
# optimization flags, then benchmarks each variant multiple times to measure
# performance. Results are saved to individual log files and a summary CSV.
#
# Usage: ./scripts/bench_seq.sh [OPTIONS]
#   -n RUNS    Number of benchmark runs per executable (default: 5)
#   -w WARMUP  Number of warmup runs to discard (default: 1)
#   -h         Show this help message
#
# Output:
#   bin/seq_bench/           - Compiled executables
#   bin/seq_bench/logs/      - Individual result files per variant
#   bin/seq_bench/summary.csv - CSV summary for all variants
#===============================================================================

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# Cleanup function for graceful exit
cleanup() {
    rm -f mat-res.txt 2>/dev/null
    rm -f "$tmp_file" 2>/dev/null
    # Remove any stray optimization report files
    rm -f *.optrpt *.opt.yaml 2>/dev/null
}
trap cleanup EXIT

#-------------------------------------------------------------------------------
# CONFIGURATION
#-------------------------------------------------------------------------------

# Source file to compile
SRC_FILE="src/matmul.c"

# Output directories
BIN_DIR="bin/seq_bench"
LOG_DIR="${BIN_DIR}/logs"

# Benchmark parameters (can be overridden via CLI)
NUM_RUNS=5
NUM_WARMUP=1

# Compilers to test (will check availability)
COMPILERS=("gcc" "icc" "icx")

# Optimization flag sets per compiler
declare -A FLAGS_GCC=(
    ["O0"]="-O0"
    ["O2"]="-O2"
    ["O3"]="-O3"
    ["O3_native"]="-O3 -march=native"
    ["O3_native_fast"]="-O3 -march=native -ffast-math"
    ["O3_native_unroll"]="-O3 -march=native -funroll-loops"
)

declare -A FLAGS_ICC=(
    ["O0"]="-O0"
    ["O2"]="-O2"
    ["O3"]="-O3"
    ["O3_xHost"]="-O3 -xHost"
    ["O3_xHost_fast"]="-O3 -xHost -fp-model fast=2"
    ["O3_xHost_unroll"]="-O3 -xHost -funroll-loops"
)

declare -A FLAGS_ICX=(
    ["O0"]="-O0"
    ["O2"]="-O2"
    ["O3"]="-O3"
    ["O3_xHost"]="-O3 -xHost"
    ["O3_xHost_fast"]="-O3 -xHost -fp-model=fast"
    ["O3_xHost_unroll"]="-O3 -xHost -funroll-loops"
)

# Common flags for all compilations
COMMON_FLAGS="-std=c11 -Wall -DENABLE_TIMING=1"

# Vectorization report flags per compiler (used during analysis compilation)
# Note: ICX generates remarks to stderr, ICC generates .optrpt files
declare -A VEC_REPORT_FLAGS=(
    ["gcc"]="-fopt-info-vec-optimized -fopt-info-vec-missed"
    ["icc"]="-qopt-report=5 -qopt-report-phase=vec -diag-disable=10441"
    ["icx"]="-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize"
)

# Output directory for vectorization reports
VEC_DIR=""

# Reference output file for correctness checking
REFERENCE_OUTPUT=""
REFERENCE_CHECKSUM=""

#-------------------------------------------------------------------------------
# COLOR DEFINITIONS (ANSI escape codes)
#-------------------------------------------------------------------------------

# Reset
NC='\033[0m'

# Regular colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'

# Bold colors
BOLD='\033[1m'
BOLD_RED='\033[1;31m'
BOLD_GREEN='\033[1;32m'
BOLD_YELLOW='\033[1;33m'
BOLD_BLUE='\033[1;34m'
BOLD_MAGENTA='\033[1;35m'
BOLD_CYAN='\033[1;36m'

#-------------------------------------------------------------------------------
# UTILITY FUNCTIONS
#-------------------------------------------------------------------------------

# Print a decorated header
print_header() {
    local msg="$1"
    local width=70
    local padding=$(( (width - ${#msg} - 2) / 2 ))
    echo ""
    echo -e "${BOLD_CYAN}$(printf '═%.0s' $(seq 1 $width))${NC}"
    echo -e "${BOLD_CYAN}║${NC}$(printf ' %.0s' $(seq 1 $padding))${BOLD}${msg}${NC}$(printf ' %.0s' $(seq 1 $((width - padding - ${#msg} - 2))))${BOLD_CYAN}║${NC}"
    echo -e "${BOLD_CYAN}$(printf '═%.0s' $(seq 1 $width))${NC}"
}

# Print a section divider
print_section() {
    local msg="$1"
    echo ""
    echo -e "${BOLD_YELLOW}▶ ${msg}${NC}"
    echo -e "${YELLOW}$(printf '─%.0s' $(seq 1 50))${NC}"
}

# Log info message
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Log success message
log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

# Log warning message
log_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Log error message
log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Log progress with spinner
log_progress() {
    # Print a single, clean progress line (no carriage return artifacts)
    echo -e "${CYAN}[...]${NC} $1"
}

# Calculate mean of values
calc_mean() {
    local values=("$@")
    local sum=0
    local count=${#values[@]}
    for v in "${values[@]}"; do
        sum=$(echo "$sum + $v" | bc -l)
    done
    echo "scale=6; $sum / $count" | bc -l
}

# Calculate standard deviation
calc_stddev() {
    local mean=$1
    shift
    local values=("$@")
    local count=${#values[@]}
    
    # Handle edge case: single value or empty
    if [ $count -le 1 ]; then
        echo "0.000000"
        return
    fi
    
    local sum_sq=0
    for v in "${values[@]}"; do
        local diff=$(echo "$v - $mean" | bc -l)
        sum_sq=$(echo "$sum_sq + ($diff * $diff)" | bc -l)
    done
    # Use sample standard deviation (n-1) for better estimate
    local variance=$(echo "scale=10; $sum_sq / ($count - 1)" | bc -l)
    echo "scale=6; sqrt($variance)" | bc -l
}

# Calculate min of values
calc_min() {
    local values=("$@")
    local min=${values[0]}
    for v in "${values[@]}"; do
        if (( $(echo "$v < $min" | bc -l) )); then
            min=$v
        fi
    done
    echo $min
}

# Calculate max of values
calc_max() {
    local values=("$@")
    local max=${values[0]}
    for v in "${values[@]}"; do
        if (( $(echo "$v > $max" | bc -l) )); then
            max=$v
        fi
    done
    echo $max
}

# Calculate GFLOP/s for matrix multiplication
# For n×n matrices: 2*n^3 floating point operations (n^3 multiplies + n^3 adds)
calc_gflops() {
    local n=$1
    local time_sec=$2
    local flops=$(echo "2 * $n * $n * $n" | bc -l)
    local gflops=$(echo "scale=3; $flops / $time_sec / 1000000000" | bc -l)
    echo $gflops
}

# Compute checksum of output file for correctness verification
# Uses md5sum for fast hashing of the result matrix
compute_checksum() {
    local file=$1
    if [ -f "$file" ]; then
        md5sum "$file" 2>/dev/null | awk '{print $1}' || shasum "$file" 2>/dev/null | awk '{print $1}' || echo "checksum_failed"
    else
        echo "file_not_found"
    fi
}

# Verify correctness by comparing output checksum with reference
# Returns: "PASS" if match, "FAIL" if mismatch, "REFERENCE" if first run
# Note: This function outputs status:checksum and the caller must update REFERENCE_CHECKSUM
verify_correctness() {
    local exe_path=$1
    local output_file="mat-res.txt"
    
    # Run the executable to generate output
    "$exe_path" 2>/dev/null 1>/dev/null
    
    if [ ! -f "$output_file" ]; then
        echo "NO_OUTPUT:"
        return
    fi
    
    local checksum=$(compute_checksum "$output_file")
    
    # If no reference yet, this becomes the reference
    if [ -z "$REFERENCE_CHECKSUM" ]; then
        echo "REFERENCE:$checksum"
        return
    fi
    
    # Compare with reference
    if [ "$checksum" = "$REFERENCE_CHECKSUM" ]; then
        echo "PASS:$checksum"
    else
        echo "FAIL:$checksum"
    fi
}

# Extract matrix size from source file
get_matrix_size() {
    grep -E "^#define n [0-9]+" "$SRC_FILE" | awk '{print $3}'
}

# Check if a compiler is available
check_compiler() {
    command -v "$1" &> /dev/null
}

# Get compiler version
get_compiler_version() {
    local compiler=$1
    case $compiler in
        gcc)
            $compiler --version | head -n1
            ;;
        icc|icx)
            # Fix: Allow SIGPIPE failure by appending '|| true'
            $compiler --version 2>&1 | head -n1 || true
            ;;
        *)
            echo "Unknown"
            ;;
    esac
}

# Analyze assembly for SIMD instruction counts
# Returns counts of various vector instruction types
analyze_assembly() {
    local exe_path=$1
    local asm_file=$2
    
    # Disassemble the binary
    if ! objdump -d "$exe_path" > "$asm_file" 2>/dev/null; then
        echo "objdump failed"
        return 1
    fi
    
    if [ ! -s "$asm_file" ]; then
        echo "objdump produced empty output"
        return 1
    fi
    
    # Count SIMD instructions by category (using grep -E for extended regex)
    # SSE: Streaming SIMD Extensions (128-bit, xmm registers)
    local sse_count=$(grep -cE '(movaps|movups|addps|mulps|subps|divps|addss|mulss|movss|addpd|mulpd|subpd|divpd|addsd|mulsd|movsd|movapd|movupd|xorps|xorpd)' "$asm_file" 2>/dev/null || echo 0)
    # AVX: Advanced Vector Extensions (256-bit, ymm registers, v-prefix)
    local avx_count=$(grep -cE 'v(movaps|movups|addps|mulps|subps|divps|addss|vmulss|addpd|mulpd|subpd|divpd|addsd|mulsd|movapd|movupd|xorps|xorpd|fmadd|fmsub|fnmadd|fnmsub)' "$asm_file" 2>/dev/null || echo 0)
    # AVX-512: 512-bit vectors (zmm registers)
    local avx512_count=$(grep -cE 'zmm[0-9]' "$asm_file" 2>/dev/null || echo 0)
    # FMA: Fused Multiply-Add instructions
    local fma_count=$(grep -cE 'vfn?m(add|sub)' "$asm_file" 2>/dev/null || echo 0)
    # Scalar FP: Single scalar floating-point operations
    local scalar_fp=$(grep -cE '(addsd|mulsd|subsd|divsd|addss|mulss|subss|divss)[^p]' "$asm_file" 2>/dev/null || echo 0)
    # Total lines (approximation of total instructions)
    local total_instr=$(wc -l < "$asm_file" 2>/dev/null || echo 0)
    
    echo "SSE:${sse_count} AVX:${avx_count} AVX512:${avx512_count} FMA:${fma_count} ScalarFP:${scalar_fp} Total:${total_instr}"
}

# Generate vectorization report during compilation
generate_vec_report() {
    local compiler=$1
    local flags=$2
    local report_file=$3
    
    local vec_flags="${VEC_REPORT_FLAGS[$compiler]}"
    
    if [ -z "$vec_flags" ]; then
        echo "No vectorization report flags for $compiler" > "$report_file"
        return
    fi
    
    # Create a temporary object file
    local tmp_obj=$(mktemp --suffix=.o 2>/dev/null || mktemp -t obj.XXXXXX)
    
    # Compile with vectorization reporting
    case $compiler in
        gcc)
            # GCC outputs vectorization info to stderr
            ${compiler} ${COMMON_FLAGS} ${flags} ${vec_flags} -c "${SRC_FILE}" -o "$tmp_obj" 2> "$report_file" || true
            ;;
        icc)
            # ICC creates .optrpt files in current directory
            local base_name="$(basename ${SRC_FILE} .c)"
            local optrpt_file="${base_name}.optrpt"
            # Clean up any existing report
            rm -f "$optrpt_file" 2>/dev/null
            ${compiler} ${COMMON_FLAGS} ${flags} ${vec_flags} -c "${SRC_FILE}" -o "$tmp_obj" 2>&1 || true
            if [ -f "$optrpt_file" ]; then
                mv "$optrpt_file" "$report_file"
            else
                echo "No optimization report generated by ICC" > "$report_file"
            fi
            ;;
        icx)
            # ICX uses clang-style remarks (-Rpass) which go to stderr
            ${compiler} ${COMMON_FLAGS} ${flags} ${vec_flags} -c "${SRC_FILE}" -o "$tmp_obj" 2> "$report_file" || true
            # Check for any YAML or optrpt files
            local base_name="$(basename ${SRC_FILE} .c)"
            for ext in opt.yaml optrpt; do
                if [ -f "${base_name}.${ext}" ]; then
                    echo "" >> "$report_file"
                    echo "=== Additional Report (${ext}) ===" >> "$report_file"
                    cat "${base_name}.${ext}" >> "$report_file"
                    rm -f "${base_name}.${ext}"
                fi
            done
            ;;
    esac
    
    # Cleanup temporary object file
    rm -f "$tmp_obj" 2>/dev/null
}

# Parse vectorization report and extract summary
summarize_vec_report() {
    local report_file=$1
    local compiler=$2
    
    if [ ! -f "$report_file" ] || [ ! -s "$report_file" ]; then
        echo "Report not available"
        return
    fi
    
    case $compiler in
        gcc)
            # GCC uses "optimized:" for success and "missed:" for failures
            local vectorized=$(grep -cE "(optimized|VECTORIZED)" "$report_file" 2>/dev/null)
            vectorized=${vectorized:-0}
            local not_vectorized=$(grep -cE "(missed|not vectorized)" "$report_file" 2>/dev/null)
            not_vectorized=${not_vectorized:-0}
            echo "Vectorized: ${vectorized}, Missed: ${not_vectorized}"
            ;;
        icc)
            # ICC optimization report format
            local vectorized=$(grep -cE "(LOOP WAS VECTORIZED|VECTORIZED)" "$report_file" 2>/dev/null)
            vectorized=${vectorized:-0}
            local not_vectorized=$(grep -cE "(was not vectorized|NOT VECTORIZED)" "$report_file" 2>/dev/null)
            not_vectorized=${not_vectorized:-0}
            echo "Vectorized: ${vectorized}, Not vectorized: ${not_vectorized}"
            ;;
        icx)
            # ICX uses clang-style -Rpass remarks
            local vectorized=$(grep -cE "remark:.*vectorized" "$report_file" 2>/dev/null)
            vectorized=${vectorized:-0}
            local not_vectorized=$(grep -cE "remark:.*(not vectorized|failed)" "$report_file" 2>/dev/null)
            not_vectorized=${not_vectorized:-0}
            echo "Vectorized: ${vectorized}, Missed: ${not_vectorized}"
            ;;
        *)
            echo "Unknown compiler"
            ;;
    esac
}

#-------------------------------------------------------------------------------
# PARSE COMMAND LINE ARGUMENTS
#-------------------------------------------------------------------------------

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n RUNS    Number of benchmark runs per executable (default: $NUM_RUNS)"
    echo "  -w WARMUP  Number of warmup runs to discard (default: $NUM_WARMUP)"
    echo "  -h         Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -n 10 -w 2"
}

while getopts "n:w:h" opt; do
    case $opt in
        n)
            NUM_RUNS=$OPTARG
            ;;
        w)
            NUM_WARMUP=$OPTARG
            ;;
        h)
            show_help
            exit 0
            ;;
        \?)
            log_error "Invalid option: -$OPTARG"
            show_help
            exit 1
            ;;
    esac
done

#-------------------------------------------------------------------------------
# MAIN SCRIPT
#-------------------------------------------------------------------------------

print_header "Sequential Matrix Multiplication Benchmark"

# Check for required tools
for tool in bc objdump; do
    if ! command -v "$tool" &> /dev/null; then
        log_error "Required tool '$tool' not found. Please install it."
        exit 1
    fi
done

# Display configuration
print_section "Configuration"
log_info "Source file: ${SRC_FILE}"
log_info "Benchmark runs: ${NUM_RUNS} (+ ${NUM_WARMUP} warmup)"
log_info "Output directory: ${BIN_DIR}"

# Get matrix size
MATRIX_SIZE=$(get_matrix_size)
log_info "Matrix size: ${MATRIX_SIZE} x ${MATRIX_SIZE}"

# Create base output directories (per-compiler subdirs created later)
mkdir -p "$BIN_DIR" "$LOG_DIR"
log_success "Created output directories"

# Collect CPU information
print_section "System Information"
CPU_INFO=$(lscpu 2>/dev/null || echo "lscpu not available")
CPU_MODEL=$(echo "$CPU_INFO" | grep "Model name:" | sed 's/Model name:\s*//')
CPU_CORES=$(echo "$CPU_INFO" | grep "^CPU(s):" | awk '{print $2}')
CPU_CACHE_L1D=$(echo "$CPU_INFO" | grep "L1d cache:" | sed 's/L1d cache:\s*//')
CPU_CACHE_L2=$(echo "$CPU_INFO" | grep "L2 cache:" | sed 's/L2 cache:\s*//')
CPU_CACHE_L3=$(echo "$CPU_INFO" | grep "L3 cache:" | sed 's/L3 cache:\s*//')
CPU_FLAGS=$(echo "$CPU_INFO" | grep "Flags:" | sed 's/Flags:\s*//')

log_info "CPU: ${CPU_MODEL}"
log_info "Cores: ${CPU_CORES}"
log_info "Cache: L1d=${CPU_CACHE_L1D}, L2=${CPU_CACHE_L2}, L3=${CPU_CACHE_L3}"

# Save system info to file
SYSINFO_FILE="${LOG_DIR}/system_info.txt"
{
    echo "========================================"
    echo "SYSTEM INFORMATION"
    echo "========================================"
    echo "Date: $(date)"
    echo "Hostname: $(hostname 2>/dev/null || cat /etc/hostname 2>/dev/null || echo "unknown")"
    echo ""
    echo "CPU Information:"
    echo "$CPU_INFO"
    echo ""
    echo "Memory Information:"
    free -h 2>/dev/null || echo "free command not available"
} > "$SYSINFO_FILE"
log_success "Saved system info to ${SYSINFO_FILE}"

# Check available compilers
print_section "Checking Compilers"
declare -a AVAILABLE_COMPILERS=()

for compiler in "${COMPILERS[@]}"; do
    if check_compiler "$compiler"; then
        version=$(get_compiler_version "$compiler")
        log_success "${compiler}: ${version}"
        AVAILABLE_COMPILERS+=("$compiler")
    else
        log_warn "${compiler}: Not found, skipping"
    fi
done

if [ ${#AVAILABLE_COMPILERS[@]} -eq 0 ]; then
    log_error "No compilers available. Exiting."
    exit 1
fi

# Initialize CSV summary file (in base log directory)
CSV_FILE="${LOG_DIR}/summary.csv"
echo "Compiler,Flags,Binary_Size_KB,Mean_Time_s,Min_Time_s,Max_Time_s,StdDev_s,GFLOPS,SSE_Instr,AVX_Instr,AVX512_Instr,FMA_Instr,Correctness,Checksum" > "$CSV_FILE"

# Arrays to store results for final comparison
declare -a ALL_NAMES=()
declare -a ALL_TIMES=()
declare -a ALL_GFLOPS=()
declare -a ALL_CORRECT=()

#-------------------------------------------------------------------------------
# COMPILATION AND BENCHMARKING
#-------------------------------------------------------------------------------

print_section "Compilation and Benchmarking"

for compiler in "${AVAILABLE_COMPILERS[@]}"; do
    echo ""
    echo -e "${BOLD_MAGENTA}━━━ Compiler: ${compiler} ━━━${NC}"
    
    # Create per-compiler subdirectories
    COMPILER_BIN_DIR="${BIN_DIR}/${compiler}"
    COMPILER_LOG_DIR="${LOG_DIR}/${compiler}"
    VEC_DIR="${COMPILER_LOG_DIR}/vec_reports"
    ASM_DIR="${COMPILER_LOG_DIR}/asm"
    mkdir -p "$COMPILER_BIN_DIR" "$COMPILER_LOG_DIR" "$VEC_DIR" "$ASM_DIR"
    
    # Get flag set for this compiler
    declare -n FLAGS="FLAGS_${compiler^^}"
    
    for flag_name in "${!FLAGS[@]}"; do
        flags="${FLAGS[$flag_name]}"
        exe_name="matmul_${flag_name}"
        display_name="${compiler}/${exe_name}"  # Full name for display/ranking
        exe_path="${COMPILER_BIN_DIR}/${exe_name}"
        log_file="${COMPILER_LOG_DIR}/${exe_name}.txt"
        
        log_progress "Compiling ${exe_name}..."
        
        # Compile
        compile_cmd="${compiler} ${COMMON_FLAGS} ${flags} ${SRC_FILE} -o ${exe_path} -lm"
        compile_start=$(date +%s.%N)
        
        if ! compile_output=$($compile_cmd 2>&1); then
            echo ""
            log_error "Compilation failed for ${exe_name}"
            echo "$compile_output" >> "${log_file}"
            continue
        fi
        
        compile_end=$(date +%s.%N)
        compile_time=$(echo "$compile_end - $compile_start" | bc -l)
        
        # Get binary size
        binary_size=$(stat --printf="%s" "$exe_path" 2>/dev/null || stat -f%z "$exe_path" 2>/dev/null)
        binary_size_kb=$(echo "scale=2; $binary_size / 1024" | bc -l)
        
        log_success "Compiled ${exe_name} (${binary_size_kb} KB, ${compile_time}s)"
        
        # Generate vectorization report
        log_progress "Generating vectorization report for ${exe_name}..."
        vec_report_file="${VEC_DIR}/${exe_name}_vec.txt"
        generate_vec_report "$compiler" "$flags" "$vec_report_file"
        vec_summary=$(summarize_vec_report "$vec_report_file" "$compiler")
        
        # Analyze assembly
        log_progress "Analyzing assembly for ${exe_name}..."
        asm_file="${ASM_DIR}/${exe_name}.asm"
        asm_analysis=$(analyze_assembly "$exe_path" "$asm_file")
        
        # Parse assembly analysis results (portable, no grep -P dependency)
        sse_count=$(echo "$asm_analysis" | sed -n 's/.*SSE:\([0-9]*\).*/\1/p')
        avx_count=$(echo "$asm_analysis" | sed -n 's/.*AVX:\([0-9]*\).*/\1/p')
        avx512_count=$(echo "$asm_analysis" | sed -n 's/.*AVX512:\([0-9]*\).*/\1/p')
        fma_count=$(echo "$asm_analysis" | sed -n 's/.*FMA:\([0-9]*\).*/\1/p')
        scalar_fp=$(echo "$asm_analysis" | sed -n 's/.*ScalarFP:\([0-9]*\).*/\1/p')
        
        # Start log file
        {
            echo "========================================"
            echo "BENCHMARK RESULTS: ${exe_name}"
            echo "========================================"
            echo ""
            echo "Compiler: ${compiler}"
            echo "Compiler Version: $(get_compiler_version $compiler)"
            echo "Flags: ${flags}"
            echo "Common Flags: ${COMMON_FLAGS}"
            echo "Full Command: ${compile_cmd}"
            echo ""
            echo "Binary Size: ${binary_size_kb} KB"
            echo "Compile Time: ${compile_time} s"
            echo ""
            echo "Matrix Size: ${MATRIX_SIZE} x ${MATRIX_SIZE}"
            echo "Benchmark Runs: ${NUM_RUNS} (+ ${NUM_WARMUP} warmup)"
            echo ""
            echo "----------------------------------------"
            echo "EXECUTION TIMES"
            echo "----------------------------------------"
        } > "$log_file"
        
        # Warmup runs
        log_progress "Running warmup (${NUM_WARMUP} runs)..."
        for ((w=1; w<=NUM_WARMUP; w++)); do
            "$exe_path" 2>/dev/null 1>/dev/null
        done
        
        # Correctness check
        log_progress "Verifying correctness for ${exe_name}..."
        correctness_result=$(verify_correctness "$exe_path")
        correctness_status=$(echo "$correctness_result" | cut -d: -f1)
        correctness_checksum=$(echo "$correctness_result" | cut -d: -f2)
        
        case $correctness_status in
            REFERENCE)
                # Set the reference checksum in parent shell (subshell can't modify parent vars)
                REFERENCE_CHECKSUM="$correctness_checksum"
                log_info "Set as reference (checksum: ${correctness_checksum:0:8}...)"
                correctness_display="${GREEN}REFERENCE${NC}"
                ;;
            PASS)
                log_success "Correctness check PASSED (checksum: ${correctness_checksum:0:8}...)"
                correctness_display="${GREEN}PASS${NC}"
                ;;
            FAIL)
                log_error "Correctness check FAILED (expected: ${REFERENCE_CHECKSUM:0:8}..., got: ${correctness_checksum:0:8}...)"
                correctness_display="${RED}FAIL${NC}"
                ;;
            *)
                log_warn "Correctness check skipped (no output)"
                correctness_display="${YELLOW}SKIP${NC}"
                ;;
        esac
        
        # Clean up output file from correctness check
        rm -f mat-res.txt 2>/dev/null
        
        # Benchmark runs
        declare -a times=()
        log_progress "Benchmarking ${exe_name}..."
        
        for ((r=1; r<=NUM_RUNS; r++)); do
            # Run and capture timing from stderr
            run_output=$("$exe_path" 2>&1 1>/dev/null)
            
            # Extract time from output like "[seq] n=2000 elapsed=1.234 s"
            # Using portable sed instead of grep -P
            elapsed=$(echo "$run_output" | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p' | head -1)
            
            if [ -z "$elapsed" ] || [ "$elapsed" = "0" ]; then
                # Fallback: use shell timing
                start_time=$(date +%s.%N)
                "$exe_path" 2>/dev/null 1>/dev/null
                end_time=$(date +%s.%N)
                elapsed=$(echo "$end_time - $start_time" | bc -l)
            fi
            
            times+=("$elapsed")
            echo "Run $r: ${elapsed} s" >> "$log_file"
        done
        
        # Calculate statistics
        mean_time=$(calc_mean "${times[@]}")
        min_time=$(calc_min "${times[@]}")
        max_time=$(calc_max "${times[@]}")
        stddev=$(calc_stddev "$mean_time" "${times[@]}")
        gflops=$(calc_gflops "$MATRIX_SIZE" "$mean_time")
        
        # Append statistics to log file
        {
            echo ""
            echo "----------------------------------------"
            echo "STATISTICS"
            echo "----------------------------------------"
            echo "Mean Time:   ${mean_time} s"
            echo "Min Time:    ${min_time} s"
            echo "Max Time:    ${max_time} s"
            echo "Std Dev:     ${stddev} s"
            echo "GFLOP/s:     ${gflops}"
            echo ""
            echo "----------------------------------------"
            echo "ADDITIONAL INFO"
            echo "----------------------------------------"
            echo ""
            echo "Binary file info:"
            file "$exe_path"
            echo ""
            echo "Binary sections (size):"
            size "$exe_path" 2>/dev/null || echo "size command not available"
            echo ""
            echo "----------------------------------------"
            echo "VECTORIZATION ANALYSIS"
            echo "----------------------------------------"
            echo "Vectorization Summary: ${vec_summary}"
            echo "Full report: ${vec_report_file}"
            echo ""
            echo "----------------------------------------"
            echo "ASSEMBLY ANALYSIS"
            echo "----------------------------------------"
            echo "SIMD Instruction Counts:"
            echo "  SSE instructions:    ${sse_count:-0}"
            echo "  AVX instructions:    ${avx_count:-0}"
            echo "  AVX-512 instructions: ${avx512_count:-0}"
            echo "  FMA instructions:    ${fma_count:-0}"
            echo "  Scalar FP ops:       ${scalar_fp:-0}"
            echo ""
            echo "Full disassembly: ${asm_file}"
            echo ""
            echo "----------------------------------------"
            echo "CORRECTNESS CHECK"
            echo "----------------------------------------"
            echo "Status: ${correctness_status}"
            echo "Checksum: ${correctness_checksum}"
            echo "Reference Checksum: ${REFERENCE_CHECKSUM}"
        } >> "$log_file"
        
        # Add to CSV
        echo "${compiler},\"${flags}\",${binary_size_kb},${mean_time},${min_time},${max_time},${stddev},${gflops},${sse_count:-0},${avx_count:-0},${avx512_count:-0},${fma_count:-0},${correctness_status},${correctness_checksum}" >> "$CSV_FILE"
        
        # Store for comparison (use display_name for ranking table)
        ALL_NAMES+=("$display_name")
        ALL_TIMES+=("$mean_time")
        ALL_GFLOPS+=("$gflops")
        ALL_CORRECT+=("$correctness_status")
        
        # Print result
        echo -e "     ${CYAN}→${NC} Mean: ${BOLD}${mean_time}s${NC} | GFLOP/s: ${BOLD_GREEN}${gflops}${NC} | StdDev: ${stddev}s"
        echo -e "     ${CYAN}→${NC} SIMD: SSE=${sse_count:-0} AVX=${avx_count:-0} AVX512=${avx512_count:-0} FMA=${fma_count:-0}"
        echo -e "     ${CYAN}→${NC} ${vec_summary}"
        echo -e "     ${CYAN}→${NC} Correctness: ${correctness_display}"
        
        # Cleanup temp files
        rm -f mat-res.txt 2>/dev/null
    done
done

#-------------------------------------------------------------------------------
# FINAL SUMMARY
#-------------------------------------------------------------------------------

print_header "Benchmark Summary"

# Check if we have any results
if [ ${#ALL_NAMES[@]} -eq 0 ]; then
    log_error "No successful benchmarks to report."
    exit 1
fi

# Find best performer
best_idx=0
best_time=${ALL_TIMES[0]}
for i in "${!ALL_TIMES[@]}"; do
    if (( $(echo "${ALL_TIMES[$i]} < $best_time" | bc -l) )); then
        best_time=${ALL_TIMES[$i]}
        best_idx=$i
    fi
done

echo ""
echo -e "${BOLD}Performance Ranking (by execution time):${NC}"
echo ""

# Sort and display results
# Create temporary file for sorting
tmp_file=$(mktemp)
for i in "${!ALL_NAMES[@]}"; do
    echo "${ALL_TIMES[$i]} ${ALL_NAMES[$i]} ${ALL_GFLOPS[$i]} ${ALL_CORRECT[$i]}" >> "$tmp_file"
done

rank=1
while read -r time name gflops correct; do
    # Format correctness indicator
    case $correct in
        PASS|REFERENCE) correct_icon="${GREEN}✓${NC}" ;;
        FAIL) correct_icon="${RED}✗${NC}" ;;
        *) correct_icon="${YELLOW}?${NC}" ;;
    esac
    
    if [ "$rank" -eq 1 ]; then
        echo -e "${BOLD_GREEN}  #${rank}${NC} ${BOLD}${name}${NC} [${correct_icon}]"
        echo -e "      Time: ${GREEN}${time}s${NC} | GFLOP/s: ${GREEN}${gflops}${NC} ${BOLD_YELLOW}⭐ BEST${NC}"
    else
        speedup=$(echo "scale=2; $time / $best_time" | bc -l)
        echo -e "  ${CYAN}#${rank}${NC} ${name} [${correct_icon}]"
        echo -e "      Time: ${time}s | GFLOP/s: ${gflops} | Slowdown: ${speedup}x"
    fi
    ((rank++))
done < <(sort -n "$tmp_file")

rm -f "$tmp_file"

echo ""
echo -e "${BOLD}Output Files:${NC}"
echo -e "  ${CYAN}•${NC} Base directory: ${BIN_DIR}/"
echo -e "  ${CYAN}•${NC} CSV summary:    ${CSV_FILE}"
echo -e "  ${CYAN}•${NC} System info:    ${SYSINFO_FILE}"
echo -e "  ${CYAN}•${NC} Per-compiler subdirectories:"
for comp in "${AVAILABLE_COMPILERS[@]}"; do
    echo -e "      ${CYAN}└─${NC} ${comp}/  (binaries, logs, vec_reports/, asm/)"
done

print_header "Benchmark Complete"

echo ""
log_success "All benchmarks completed successfully!"
echo ""
