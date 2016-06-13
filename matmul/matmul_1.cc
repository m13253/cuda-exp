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

static int calc_index(int bi, int bj, int i, int j, int size, int blocks) {
    return (bi*size + i)*size*blocks + bj*size+j;
}

static void matrix_mul(float q[], const float a[], const float b[], int size, int blocks) {
    for(int bi = 0; bi < blocks; ++bi) {
        for(int bj = 0; bj < blocks; ++bj) {
            for(int bk = 0; bk < blocks; ++bk) {
                float tile_a[size*size];
                for(int i = 0; i < size; ++i) {
                    for(int j = 0; j < size; ++j) {
                        tile_a[i*size+j] = a[calc_index(bi, bk, i, j, size, blocks)];
                    }
                }
                float tile_b[size*size];
                for(int i = 0; i < size; ++i) {
                    for(int j = 0; j < size; ++j) {
                        tile_b[i*size+j] = b[calc_index(bk, bj, i, j, size, blocks)];
                    }
                }
                for(int i = 0; i < size; ++i) {
                    for(int j = 0; j < size; ++j) {
                        float s = 0;
                        for(int k = 0; k < size; ++k) {
                            s += tile_a[i*size+k] * tile_b[k*size+j];
                        }
                        q[calc_index(bi, bj, i, j, size, blocks)] += s;
                    }
                }
            }
        }
    }
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

    float *c = new float[size*size];

    matrix_mul(c, a, b, block_size, size/block_size);

    print_matrix("a * b", c, size, size);

    delete[] c;
    delete[] b;
    delete[] a;

    return 0;
}
