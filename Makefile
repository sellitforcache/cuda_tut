NVCC = nvcc -ccbin=/usr/local/Cellar/gcc46/4.6.4/bin
ARCH = -arch sm_30 -m64
CBLAS_LOC = /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A

all: matmul

matmul:
	$(NVCC) $(ARCH) -use_fast_math -I$(CBLAS_LOC)/Headers -L$(CBLAS_LOC) -lblas matmul.cu -o matmul

clean:
	rm matmul
