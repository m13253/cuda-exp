all: matmul_0_cpu matmul_0_gpu matmul_1_cpu matmul_1_gpu

%_gpu: %.cu
	nvcc -O3 -g -o $@ $<

%_cpu: %.cc
	$(CXX) -O3 -g -o $@ $<

clean:
	rm -f matmul_0_cpu matmul_0_gpu matmul_1_cpu matmul_1_gpu
