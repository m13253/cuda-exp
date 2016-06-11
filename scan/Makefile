all: scan_0_cpu scan_0_gpu scan_1_cpu scan_1_gpu scan_2_cpu scan_2_gpu

%_gpu: %.cu
	nvcc -O3 -g -o $@ $<

%_cpu: %.cc
	$(CXX) -O3 -g -o $@ $<

clean:
	rm -f scan_0_cpu scan_0_gpu scan_1_cpu scan_1_gpu scan_2_cpu scan_2_gpu
