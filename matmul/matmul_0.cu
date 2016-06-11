#include <cstdio>

static void print_matrix(const char *name, const float *matrix, int h, int w) {
    std::printf("%s = [\n", name);
    for(int i = 0; i < h; ++i) {
        for(int j = 0; j < w; ++j) {
            std::printf(" %3.0f.", matrix[i*w+j]);
        }
        std::puts("");
    }
    std::puts("]\n");
}

__global__ static void matrix_mul(float q[], const float a[], const float b[], int ah, int aw, int bw) {
    int i = threadIdx.x, j = threadIdx.y;
    q[i*bw+j] = 0;
    for(int k = 0; k < aw; ++k) {
        q[i*bw+j] += a[i*aw+k] * b[k*bw+j];
    }
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
    int ah = 4, aw = 3, bw = 5;
    switch(argc) {
    default:
        std::sscanf(argv[3], "%d", &bw);
        /* fall-through */
    case 3:
        std::sscanf(argv[2], "%d", &aw);
        /* fall-through */
    case 2:
        std::sscanf(argv[1], "%d", &ah);
        /* fall-through */
    case 1:
    case 0:
        break;
    }

    float *a = new float[ah*aw];
    float *b = new float[aw*bw];
    for(int i = 0; i < ah*aw; ++i) {
        a[i] = i+1;
    }
    for(int i = 0; i < aw*bw; ++i) {
        b[i] = i+1;
    }
    print_matrix("a", a, ah, aw);
    print_matrix("b", b, aw, bw);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, ah*aw*sizeof *d_a); report_error();
    cudaMalloc(&d_b, aw*bw*sizeof *d_b); report_error();
    cudaMalloc(&d_c, ah*bw*sizeof *d_c); report_error();
    cudaMemcpy(d_a, a, sizeof a, cudaMemcpyHostToDevice); report_error();
    cudaMemcpy(d_b, b, sizeof b, cudaMemcpyHostToDevice); report_error();

    matrix_mul<<<1, dim3(ah, bw)>>>(d_c, d_a, d_b, ah, aw, bw); report_error();

    float *c = new float[ah*bw];
    cudaMemcpy(c, d_c, ah*bw*sizeof *c, cudaMemcpyDeviceToHost); report_error();
    print_matrix("a * b", c, ah, bw);
    delete[] c;

    cudaFree(d_c); report_error();
    cudaFree(d_b); report_error();
    cudaFree(d_a); report_error();
    delete[] b;
    delete[] a;

    return 0;
}
