# CUDA Makefile

TARGET = particle_filter
SRC    = ex_particle_CUDA_float_seq.cu

NVCC   = nvcc
CFLAGS = -O2
LDFLAGS = -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) -x 128 -y 128 -z 10 -np 1000

clean:
	rm -f $(TARGET)