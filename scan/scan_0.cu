#include <algorithm>
#include <cstdlib>
#include <cstdio>

__global__ static void fill(int a[]) {
    unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
    a[n] = n;
}

__global__ static void scan(int a[]) {
    unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for(unsigned i = 0; i <= n; ++i) {
        sum += a[i];
    }
    __syncthreads();
    a[n] = sum;
}

static cudaError_t report_error(void) {
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
    return err;
}

int main(int argc, char *argv[]) {
    unsigned blk_size = 64;
    if(argc >= 2) {
        std::sscanf(argv[1], "%u", &blk_size);
        blk_size = ((blk_size-1)/4+1)*4;
    }
    unsigned blk_cnt = 4;
    if(argc >= 3) {
        std::sscanf(argv[2], "%u", &blk_cnt);
    }
    unsigned times = 1;
    if(argc >= 4) {
        std::sscanf(argv[3], "%u", &times);
    } else {
        std::fprintf(stderr, "Block configuration: %u x %u (%u)\n", blk_size, blk_cnt, blk_size * blk_cnt);
    }

    int *a;
    cudaMalloc(&a, blk_cnt * blk_size * sizeof (int)); report_error();
    fill<<<blk_cnt, blk_size>>>(a); report_error();

    cudaEvent_t start, end;
    cudaEventCreate(&start); report_error();
    cudaEventCreate(&end); report_error();
    cudaEventRecord(start); report_error();
    cudaEventSynchronize(start); report_error();
    for(unsigned i = 0; i < times; ++i) {
        scan<<<blk_cnt, blk_size>>>(a); report_error();
    }
    cudaEventRecord(end); report_error();
    cudaEventSynchronize(end); report_error();
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end); report_error();
    cudaEventDestroy(end); report_error();
    cudaEventDestroy(start); report_error();

    if(argc < 4) {
        size_t n = std::min<size_t>(256, blk_size * blk_cnt);
        int *h_a = new int[n];
        cudaMemcpy(h_a, a, n * sizeof (int), cudaMemcpyDeviceToHost); report_error();
        std::printf("[%d", h_a[0]);
        for(size_t i = 1; i < n; ++i) {
            std::printf(", %u", h_a[i]);
        }
        std::puts("]");
        delete[] h_a;
    } else {
        std::printf("%.9g", elapsed * 0.001f / times);
    }

    cudaFree(a); report_error();

    return 0;
}
