compile:
	nvcc -Wno-deprecated-gpu-targets -std=c++11 -gencode=arch=compute_35,code=sm_35 main.cu -o cuda
clean:
	rm cuda
