#!/bin/bash

# This is the benchmark tool

if [ -n "$1" ]
then
    blk_size="$1"
else
    blk_size=512
fi
echo 'blk_size,blk_cnt,scan_0_gpu,scan_0_cpu,scan_1_gpu,scan_1_cpu,scan_2_gpu,scan_2_cpu'
for blk_cnt in $(seq $((4096/$blk_size)) $((4096/$blk_size)) $((131072/$blk_size)))
do
    echo -n "$blk_size,$blk_cnt,"
    ./scan_0_gpu "$blk_size" "$blk_cnt" 32
    echo -n ,
    ./scan_0_cpu "$blk_size" "$blk_cnt" 32
    echo -n ,
    ./scan_1_gpu "$blk_size" "$blk_cnt" 1024
    echo -n ,
    ./scan_1_cpu "$blk_size" "$blk_cnt" 32
    echo -n ,
    ./scan_2_gpu "$blk_size" "$blk_cnt" 1024
    echo -n ,
    ./scan_2_cpu "$blk_size" "$blk_cnt" 32
    echo
done
