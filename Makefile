compile:
	g++ -std=c++11 main.cpp -o main
run:
	./main data_image.txt query_image.txt 10 10
clean:
	rm main