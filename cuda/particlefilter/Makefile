#makefile

# include ../../common/make.config

CUDA_DIR ?= /usr/local/cuda
ifneq ($(strip $(CUDA_HOME)),)
CUDA_DIR := $(CUDA_HOME)
endif
NVCC ?= nvcc

INCLUDE ?= $(CUDA_DIR)/include
CUDA_LIB_DIR ?= $(CUDA_DIR)/lib64

all: naive float

CUDA_ARCH=-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80

naive: ex_particle_CUDA_naive_seq.cu
	$(NVCC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math $(CUDA_ARCH) ex_particle_CUDA_naive_seq.cu -o particlefilter_naive
	
float: ex_particle_CUDA_float_seq.cu
	$(NVCC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math $(CUDA_ARCH) ex_particle_CUDA_float_seq.cu -o particlefilter_float

clean:
	rm particlefilter_naive particlefilter_float
