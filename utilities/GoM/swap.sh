#!/bin/bash

for d in *.su ; do
    echo "$d"
    uswapbytes < $d format=0 > ${d: :-2}swapped.su
done
