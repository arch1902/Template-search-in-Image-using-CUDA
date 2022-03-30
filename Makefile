compile:
	nvcc -std=c++11 -Wno-deprecated-gpu-targets -gencode=arch=compute_35,code=sm_35 main.cu -o cuda
run:
	./cuda data_image.txt query_image.txt 10 10 1
clean:
	rm cuda

main:
	rm -f main
	g++ -o main main.cpp
	./main data_image.txt query_image.txt 10 10