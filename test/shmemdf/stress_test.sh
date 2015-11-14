#!/bin/bash
N=10

i=1
while [ $i -le $N ]
do
    echo "Starting round ${i}"

    ./build/oat-exp-server &
    s=$!
    ./build/oat-exp-client foo &
    c1=$!
    ./build/oat-exp-client bar &
    c2=$!

    # Run for a few frames
    sleep 1

    # Kill and give time to clean up
    kill -INT $s $c1 $c2
    sleep 1

    ((i++))
done

