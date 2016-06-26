#include <algorithm>
#include <cstdio>

static void print_matrix(const char *name, const float *matrix, int h, int w) {
    int eff_h = std::min(h, 8);
    int eff_w = std::min(w, 8);
    std::printf("%s = [\n", name);
    for(int i = 0; i < eff_h; ++i) {
        for(int j = 0; j < eff_w; ++j) {
            std::printf(" %5g", matrix[i*w+j]);
        }
        if(eff_w == w) {
            std::puts("");
        } else {
            std::puts(" ...");
        }
    }
    if(eff_h == h) {
        std::puts("]\n");
    } else {
        std::puts("... ]\n");
    }
}

__global__ static void matrix_mul(float q[], const float a[], const float b[], int size, int blocks) {
    extern __shared__ float shared[];
    float *tile_a = shared;
    float *tile_b = shared + size;
    int i = threadIdx.x, j = threadIdx.y;
    tile_a[i*size+j] = a[calc_index(blockIdx.x, blockIdx.z, i, j, size, blocks)];
    tile_b[i*size+j] = b[calc_index(blockIdx.z, blockIdx.y, i, j, size, blocks)];
    __syncthreads();
    // STUB
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
    int size = 8;
    if(argc >= 2) {
        std::sscanf(argv[1], "%d", &size);
    }
    int block_size = 8;
    if(argc >= 3) {
        std::sscanf(argv[2], "%d", &block_size);
    }
    size = ((size-1)/block_size+1) * block_size;

    float *a = new float[size*size];
    float *b = new float[size*size];
    for(int i = 0; i < size*size; ++i) {
        a[i] = i+1;
    }
    for(int i = 0; i < size*size; ++i) {
        b[i] = i+1;
    }
    print_matrix("a", a, size, size);
    print_matrix("b", b, size, size);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size*size*sizeof *d_a); report_error();
    cudaMalloc(&d_b, size*size*sizeof *d_b); report_error();
    cudaMalloc(&d_c, size*size*sizeof *d_c); report_error();
    cudaMemcpy(d_a, a, size*size*sizeof *a, cudaMemcpyHostToDevice); report_error();
    cudaMemcpy(d_b, b, size*size*sizeof *b, cudaMemcpyHostToDevice); report_error();
    cudaMemset(d_c, 0, size*size*sizeof *d_c); report_error();

    matrix_mul<<<dim3(size/block_size, size/block_size, size/block_size), dim3(block_size, block_size), size*sizeof (float)*2>>>(d_c, d_a, d_b, block_size, size/block_size); report_error();

    float *c = new float[size*size];
    cudaMemcpy(c, d_c, size*size*sizeof *c, cudaMemcpyDeviceToHost); report_error();
    print_matrix("a * b", c, size, size);
    delete[] c;

    cudaFree(d_c); report_error();
    cudaFree(d_b); report_error();
    cudaFree(d_a); report_error();
    delete[] b;
    delete[] a;

    return 0;
}
