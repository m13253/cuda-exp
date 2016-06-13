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

static void matrix_mul(float q[], const float a[], const float b[], int size) {
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            float s = 0;
            for(int k = 0; k < size; ++k) {
                s += a[i*size+k] * b[k*size+j];
            }
            q[i*size+j] = s;
        }
    }
}

int main(int argc, char *argv[]) {
    int size = 8;
    if(argc >= 2) {
        std::sscanf(argv[1], "%d", &size);
    }

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

    float *c = new float[size*size];

    matrix_mul(c, a, b, size);

    print_matrix("a * b", c, size, size);

    delete[] c;
    delete[] b;
    delete[] a;

    return 0;
}
