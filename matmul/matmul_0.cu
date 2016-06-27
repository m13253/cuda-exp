#include <algorithm>
#include <cstdio>
#include <cstring>

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

__global__ static void matrix_mul(float q[], const float a[], const float b[], int size) {
    int i = threadIdx.x, j = threadIdx.y;
    float s = 0;
    for(int k = 0; k < size; ++k) {
        s += a[i*size+k] * b[k*size+j];
    }
    q[i*size+j] = s;
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
    bool bench = argc >= 4 && std::strcmp(argv[3], "bench") == 0;

    float *a = new float[size*size];
    float *b = new float[size*size];
    for(int i = 0; i < size*size; ++i) {
        a[i] = i+1;
    }
    for(int i = 0; i < size*size; ++i) {
        b[i] = i+1;
    }
    if(!bench) {
        print_matrix("a", a, size, size);
        print_matrix("b", b, size, size);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size*size*sizeof *d_a); report_error();
    cudaMalloc(&d_b, size*size*sizeof *d_b); report_error();
    cudaMalloc(&d_c, size*size*sizeof *d_c); report_error();
    cudaMemcpy(d_a, a, size*size*sizeof *a, cudaMemcpyHostToDevice); report_error();
    cudaMemcpy(d_b, b, size*size*sizeof *b, cudaMemcpyHostToDevice); report_error();

    cudaEvent_t start, end;
    cudaEventCreate(&start); report_error();
    cudaEventCreate(&end); report_error();
    cudaEventRecord(start); report_error();
    cudaEventSynchronize(start); report_error();

    matrix_mul<<<1, dim3(size, size)>>>(d_c, d_a, d_b, size); report_error();

    cudaEventRecord(end); report_error();
    cudaEventSynchronize(end); report_error();
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end); report_error();
    cudaEventDestroy(end); report_error();
    cudaEventDestroy(start); report_error();

    if(!bench) {
        float *c = new float[size*size];
        cudaMemcpy(c, d_c, size*size*sizeof *c, cudaMemcpyDeviceToHost); report_error();
        print_matrix("a * b", c, size, size);
        delete[] c;
        std::printf("%.9g s elapsed\n", elapsed * 0.001f);
    } else {
        std::printf("%.9g", elapsed * 0.001f);
    }

    cudaFree(d_c); report_error();
    cudaFree(d_b); report_error();
    cudaFree(d_a); report_error();
    delete[] b;
    delete[] a;

    return 0;
}
