all: scan_0 scan_0_cpu scan_1 scan_1_cpu scan_2 scan_2_cpu

%: %.cu
	nvcc -o $@ $<
	#nvcc -Xcompiler '-std=c++98 -Wall' -g -O3 -o $@ $<

%: %.cc
	$(CXX) -std=c++98 -Wall -g -O3 -march=native -mtune=native -o $@ $<

clean:
	-rm all
