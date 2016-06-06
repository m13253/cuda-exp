#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>

static void fill(int a[], int size) {
    for(int n = 0; n < size; ++n) {
        a[n] = n;
    }
}

static void scan(int a[], int size) {
    int i;
    for(i = 1; i < size; i *= 2) {
        for(int n = 0; n < size/2; ++n) {
            int from = (n * 2 + 1) * i - 1;
            int to = from + i;
            if(to < size) {
                a[to] += a[from];
            }
        }
    }
    for(i /= 2; i != 0; i /= 2) {
        for(int n = 0; n < size/2; ++n) {
            int from = (n + 1) * i * 2 - 1;
            int to = from + i;
            if(to < size) {
                a[to] += a[from];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    unsigned device_id = 0;
    if(argc >= 2 && std::sscanf(argv[1], "-d%u", &device_id) == 1) {
        --argc; ++argv;
    }

    unsigned blk_size = 64;
    if(argc >= 2) {
        std::sscanf(argv[1], "%u", &blk_size);
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

    int *a = new int[blk_cnt * blk_size];
    fill(a, blk_cnt * blk_size);

    clock_t start = std::clock();
    for(unsigned i = 0; i < times; ++i) {
        scan(a, blk_cnt * blk_size);
    }
    clock_t end = std::clock();
    clock_t elapsed = end - start;

    if(argc < 4) {
        size_t n = std::min<size_t>(256, blk_size * blk_cnt);
        std::printf("[%d", a[0]);
        for(size_t i = 1; i < n; ++i) {
            std::printf(", %u", a[i]);
        }
        std::puts("]");
    } else {
        std::printf("%.9lg", (double) elapsed / CLOCKS_PER_SEC / times);
    }

    delete[] a;

    return 0;
}
