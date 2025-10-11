#!/bin/csh -f

# select files that are ready for use in curvelet subtraction.
# shot records consist of 2275 samples, and the first multiples
# does not start before sample nr. 800.
# Therefore select three overlapping time windows each with 512 samples:
# the overlap is 29 samples
# + window 1 : sample  800 : 1311
# + window 2 : sample 1282 : 1793
# + window 3 : sample 1764 : 2275
#
# Each shot consistsing of 180+4 interpolated=184 traces
# Therefore, join each 2 subsequent shot records into panels
# of 2x184=384 traces
# put the shots back-to-back, with the zero offset traces together
#
# The file consists of 1001 shot records, with fldr varying from 45-1045
# Thus, with 2 shots per panel 500 panels are needed.
# Therefore, make 2 overlapping source windows of 2x280 shots:
# + window 1: shot 45+46 : 603+604
# + window 2: shot 486+487 : 1044+1045

set dirin="../data"
set dirout="../data_curvelets"
set base=shots.mult
set base=shots.inter
set base=shots.srme
set nxgath=184

# define NMO curve to flatten events 

set nmopar="tnmo=4,7,10 vnmo=6000,7000,8000 smute=3"

# loop over shot groups

set group=1
while ($group <= 2)

if ($group == 1) set shotmin=45
if ($group == 1) set shotmax=604
if ($group == 2) set shotmin=486
if ($group == 2) set shotmax=1045

# select group of shots from the input file, split odd and even ones

echo " "
echo " Shot records $base group $group with shots $shotmin - $shotmax"
echo " "

set shot2=`expr $shotmin + 1`

suwind < $dirin/$base.su key=fldr min=$shotmin max=$shotmax s=$shotmin j=2 | \
subuffsort key=fldr,offset nxmax=-1 ntrace=200 verbose=1 > tmp1.su
suwind < $dirin/$base.su key=fldr min=$shot2 max=$shotmax s=$shot2 j=2 > tmp2.su

# loop over time windows

set window=1
while ($window <= 3)

if ($window == 1) set sammin=800
if ($window == 1) set sammax=1311
if ($window == 2) set sammin=1282
if ($window == 2) set sammax=1793
if ($window == 3) set sammin=1764
if ($window == 3) set sammax=2275

echo " "
echo " Gathering shot numbers $shotmin - $shotmax, time window nr. $window"
echo " Select time window $window with samples $sammin - $sammax"

file_merge file_in1=tmp1.su file_in2=tmp2.su | \
sunmo $nmopar | \
convert sammin=$sammin sammax=$sammax > $dirout/$base.group$group.window$window.su

# next time window
set window=`expr $window + 1`
end

# next group of shots
set group=`expr $group + 1`
end
