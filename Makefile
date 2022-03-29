compile:
	nvcc -Wno-deprecated-gpu-targets -gencode=arch=compute_35,code=sm_35 cuda_main.cu -o cuda
run:
	./cuda data_image.txt query_image.txt 10 10
clean:
	rm cuda