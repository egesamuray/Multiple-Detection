#!/bin/bash

cd /data/NelsonData/raw-data

for d in *.su ; do
	echo $d
	suoldtonew < $d  > tmp.su
	suwind key=fldr min=1 max=401 < tmp.su | segyhdrs | segywrite tape=${d: :-2}segy
	rm -rf tmp.su
done


# for d in *.su ; do
# 	rm -rf $d
# done