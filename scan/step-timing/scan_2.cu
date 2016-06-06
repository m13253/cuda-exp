#include <algorithm>
#include <cstdlib>
#include <cstdio>

__global__ static void fill(int a[]) {
    unsigned n = blockIdx.x * blockDim.x + threadIdx.x;
    a[n] = n;
}

__global__ static void scan(int a[], int blk_sum[]) {
    extern __shared__ int slice[];
    unsigned n = threadIdx.x;
    slice[n] = a[blockIdx.x*blockDim.x*2 + n];
    slice[blockDim.x + n] = a[blockIdx.x*blockDim.x*2 + blockDim.x + n];
    __syncthreads();
    unsigned i;
    for(i = 1; i < blockDim.x*2; i *= 2) {
        unsigned from = (n * 2 + 1) * i - 1;
        unsigned to = from + i;
        if(to < blockDim.x*2) {
            slice[to] += slice[from];
        }
        __syncthreads();
    }
    for(i /= 2; i != 0; i /= 2) {
        unsigned from = (n + 1) * i * 2 - 1;
        unsigned to = from + i;
        if(to < blockDim.x*2) {
            slice[to] += slice[from];
        }
        __syncthreads();
    }
    a[blockIdx.x*blockDim.x*2 + n] = slice[n];
    a[blockIdx.x*blockDim.x*2 + blockDim.x + n] = slice[blockDim.x + n];
    if(blk_sum && n == 0) {
        blk_sum[blockIdx.x+1] = slice[blockDim.x*2-1];
    }
}

__global__ static void fix(int a[], int blk_sum[]) {
    a[blockIdx.x*blockDim.x + threadIdx.x] += blk_sum[blockIdx.x/2];
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
    unsigned device_id = 0;
    if(argc >= 2 && std::sscanf(argv[1], "-d%u", &device_id) == 1) {
        --argc; ++argv;
    }
    cudaSetDevice(device_id); report_error();

    unsigned blk_size = 128;
    if(argc >= 2) {
        std::sscanf(argv[1], "%u", &blk_size);
        blk_size = ((blk_size-1)/4+1)*8;
    }
    unsigned blk_cnt = 2;
    if(argc >= 3) {
        std::sscanf(argv[2], "%u", &blk_cnt);
        blk_cnt = ((blk_cnt-1)/4+1)*4/2;
    }
    unsigned times = 10;
    if(argc >= 4) {
        std::sscanf(argv[3], "%u", &times);
    }

    int *a;
    cudaMalloc(&a, blk_cnt * blk_size * sizeof (int)); report_error();
    fill<<<blk_cnt*2, blk_size/2>>>(a); report_error();

    int *blk_sum;
    cudaMalloc(&blk_sum, blk_cnt * sizeof (int)); report_error();
    cudaMemset(blk_sum, 0, sizeof (int)); report_error();

    cudaEvent_t start, step1, step2, end;
    cudaEventCreate(&start); report_error();
    cudaEventCreate(&step1); report_error();
    cudaEventCreate(&step2); report_error();
    cudaEventCreate(&end); report_error();

    cudaEventRecord(start); report_error();
    cudaEventSynchronize(start); report_error();

    for(unsigned i = 0; i < times; ++i) {
        scan<<<blk_cnt, blk_size/2, blk_size * sizeof (int)>>>(a, blk_sum); report_error();
    }

    cudaEventRecord(step1); report_error();
    cudaEventSynchronize(step1); report_error();

    for(unsigned i = 0; i < times; ++i) {
        scan<<<1, blk_cnt/2, blk_cnt * sizeof (int)>>>(blk_sum, NULL); report_error();
    }

    cudaEventRecord(step2); report_error();
    cudaEventSynchronize(step2); report_error();

    for(unsigned i = 0; i < times; ++i) {
        fix<<<blk_cnt*2, blk_size/2>>>(a, blk_sum); report_error();
    }

    cudaEventRecord(end); report_error();
    cudaEventSynchronize(end); report_error();

    float elapsed01, elapsed12, elapsed23;
    cudaEventElapsedTime(&elapsed01, start, step1); report_error();
    cudaEventElapsedTime(&elapsed12, step1, step2); report_error();
    cudaEventElapsedTime(&elapsed23, step2, end); report_error();

    cudaEventDestroy(end); report_error();
    cudaEventDestroy(step2); report_error();
    cudaEventDestroy(step1); report_error();
    cudaEventDestroy(start); report_error();

    cudaFree(blk_sum); report_error();
    cudaFree(a); report_error();

    std::printf("step 1 scan: %.9g s\n", elapsed01 * 0.001f / times);
    std::printf("step 2 scan: %.9g s\n", elapsed12 * 0.001f / times);
    std::printf("step 3 fix:  %.9g s\n", elapsed23 * 0.001f / times);

    return 0;
}
