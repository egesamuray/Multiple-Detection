#!/bin/bash

cd /data/GOMdata/data

for d in *.su ; do
	cat $d | suwind key=fldr min=45 max=1045 | segyhdrs | segywrite tape=${d: :-2}segy
done


# for d in *.su ; do
# 	rm -rf $d
# done