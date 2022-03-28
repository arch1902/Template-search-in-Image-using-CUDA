compile:
	nvcc cuda_main.cu -o cuda
run:
	./cuda data_image.txt query_image.txt 10 10
clean:
	rm cuda