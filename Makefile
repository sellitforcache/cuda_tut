NVCC = nvcc -ccbin=/usr/local/Cellar/gcc46/4.6.4/bin
ARCH = -arch sm_30 -m64

all: matmul

matmul:
	$(NVCC) $(ARCH) -use_fast_math  matmul.cu -o matmul

clean:
	rm matmul
