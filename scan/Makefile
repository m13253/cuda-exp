all: scan_0 scan_1 scan_2

%: %.cu
	nvcc -Xcompiler '-std=c++98 -Wall' -g -o $@ $<
