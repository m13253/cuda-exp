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

static void matrix_mul(float q[], const float a[], const float b[], int ah, int aw, int bw) {
    for(int i = 0; i < ah; ++i) {
        for(int j = 0; j < bw; ++j) {
            float s = 0;
            for(int k = 0; k < aw; ++k) {
                s += a[i*aw+k] * b[k*bw+j];
            }
            q[i*bw+j] = s;
        }
    }
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

    float *c = new float[ah*bw];

    matrix_mul(c, a, b, ah, aw, bw);

    print_matrix("a * b", c, ah, bw);

    delete[] c;
    delete[] b;
    delete[] a;

    return 0;
}
