
# Toggle timing (1=enabled, 0=disabled) across all builds
TIMING ?= 0
TIMING_FLAG = -D ENABLE_TIMING=$(TIMING)

# compilers and flags
CC      ?= gcc
MPICC   ?= mpicc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra $(TIMING_FLAG)
LDFLAGS ?=

# source folder and binary output folder
SRC_DIR := src
BIN_DIR := bin

# source files
SEQ_SRC := $(SRC_DIR)/matmul.c
OMP_SRC := $(SRC_DIR)/matmul_omp.c
MPI_SRC := $(SRC_DIR)/matmul_mpi.c

# binary files
SEQ_BIN := $(BIN_DIR)/matmul_seq
OMP_BIN := $(BIN_DIR)/matmul_omp
MPI_BIN := $(BIN_DIR)/matmul_mpi

# targets
.PHONY: all clean dirs

all: dirs seq omp mpi

# compile binaries
dirs:
	@mkdir -p $(BIN_DIR)

seq: $(SEQ_SRC)
	$(CC) $(CFLAGS) $(TIMING_FLAG) $< -o $(SEQ_BIN) $(LDFLAGS)

omp: $(OMP_SRC)
	$(CC) $(CFLAGS) $(TIMING_FLAG) -fopenmp $< -o $(OMP_BIN) $(LDFLAGS)

mpi: $(MPI_SRC)
	$(MPICC) $(filter-out -fopenmp,$(CFLAGS)) $(TIMING_FLAG) $< -o $(MPI_BIN) $(LDFLAGS)

# remove binaries
clean:
	rm -f $(SEQ_BIN) $(OMP_BIN) $(MPI_BIN)
