all: scan_1 scan_2

%: %.cu
	nvcc -o $@ $<
	# nvcc -Xcompiler '-std=c++98 -Wall' -g -O3 -o $@ $<

%: %.cc
	$(CXX) -std=c++98 -Wall -g -O3 -march=native -mtune=native -o $@ $<


clean:
	-rm all