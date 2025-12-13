
# Compilers and base flags
CC            = gcc        # options: gcc, icc, icx
MPICC         = mpiicc     # or mpicc for gcc
BASE_CFLAGS   = -O2 -march=native
OMP_FLAG      = -fopenmp
LDFLAGS       =

# Derived flags (with / without timing)
CFLAGS_TIMED   = $(BASE_CFLAGS) -DENABLE_TIMING
CFLAGS_NOTIME  = $(BASE_CFLAGS)

# Source and output folders
SRC_DIR  := src
BIN_DIR  := bin

# Source files
SEQ_SRC         := $(SRC_DIR)/matmul_seq.c
OMP_SRC         := $(SRC_DIR)/matmul_omp.c
MPI_SRC         := $(SRC_DIR)/matmul_mpi.c
MPI_SHARED_SRC  := $(SRC_DIR)/matmul_mpi_shared.c

# Binaries (timed and no-timing variants)
SEQ_BIN        := $(BIN_DIR)/matmul_seq
SEQ_BIN_NT     := $(BIN_DIR)/matmul_seq_notime
OMP_BIN        := $(BIN_DIR)/matmul_omp
OMP_BIN_NT     := $(BIN_DIR)/matmul_omp_notime
MPI_BIN        := $(BIN_DIR)/matmul_mpi
MPI_BIN_NT     := $(BIN_DIR)/matmul_mpi_notime
MPI_SH_BIN     := $(BIN_DIR)/matmul_mpi_shared
MPI_SH_BIN_NT  := $(BIN_DIR)/matmul_mpi_shared_notime

# Targets
.PHONY: all timed notimed seq seq-notime omp omp-notime mpi mpi-notime mpi-shared mpi-shared-notime clean dirs

all: timed notimed

timed: dirs seq omp mpi mpi-shared
notimed: dirs seq-notime omp-notime mpi-notime mpi-shared-notime

# Create bin directory
dirs:
	@mkdir -p $(BIN_DIR)

# Sequential
seq: $(SEQ_SRC)
	$(CC) $(CFLAGS_TIMED) $< -o $(SEQ_BIN) $(LDFLAGS)

seq-notime: $(SEQ_SRC)
	$(CC) $(CFLAGS_NOTIME) $< -o $(SEQ_BIN_NT) $(LDFLAGS)

# OpenMP
omp: $(OMP_SRC)
	$(CC) $(CFLAGS_TIMED) $(OMP_FLAG) $< -o $(OMP_BIN) $(LDFLAGS)

omp-notime: $(OMP_SRC)
	$(CC) $(CFLAGS_NOTIME) $(OMP_FLAG) $< -o $(OMP_BIN_NT) $(LDFLAGS)

# MPI (regular)
mpi: $(MPI_SRC)
	$(MPICC) $(CFLAGS_TIMED) $< -o $(MPI_BIN) $(LDFLAGS)

mpi-notime: $(MPI_SRC)
	$(MPICC) $(CFLAGS_NOTIME) $< -o $(MPI_BIN_NT) $(LDFLAGS)

# MPI (shared-memory variant)
mpi-shared: $(MPI_SHARED_SRC)
	$(MPICC) $(CFLAGS_TIMED) $< -o $(MPI_SH_BIN) $(LDFLAGS)

mpi-shared-notime: $(MPI_SHARED_SRC)
	$(MPICC) $(CFLAGS_NOTIME) $< -o $(MPI_SH_BIN_NT) $(LDFLAGS)

# remove binaries
clean:
	rm -f $(SEQ_BIN) $(SEQ_BIN_NT) $(OMP_BIN) $(OMP_BIN_NT) \
	      $(MPI_BIN) $(MPI_BIN_NT) $(MPI_SH_BIN) $(MPI_SH_BIN_NT)
