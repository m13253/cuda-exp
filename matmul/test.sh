#!/bin/bash

# This is the benchmark tool

if [ -n "$1" ]
then
    blk_size="$1"
else
    blk_size=8
fi
echo 'size,matmul_1_gpu_1,matmul_1_gpu_2,matmul_1_gpu_3'
for total in $(seq 8 8 1024)
do
    echo -n "$(($total * $blk_size)),"
    ./matmul_1_gpu "$total" "$blk_size" bench
    echo -n ','
    ./matmul_1_gpu "$total" "$blk_size" bench
    echo -n ','
    ./matmul_1_gpu "$total" "$blk_size" bench
    echo
done
