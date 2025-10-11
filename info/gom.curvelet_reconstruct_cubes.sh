#!/bin/csh -f

# restore files that are ready for use in curvelet subtraction
# and reformat them in the normal shot format.
#
# shot records consist of 2275 samples, and the first multiples
# does not start before sample nr. 800.
# Therefore three overlapping time windows were selected each with 512 samples:
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
#
# In this script output from the different shot/time windows are restored
# in the normal shot recrod format in the following manner:
# from shot group 1 select shots  45 - 545
# from shot group 2 select shots 546 - 1045
# use the first 799 samples of the original data (shots.inter.su)
# use sample  1 - 500 from time window 1 (corresponding to samples  800 - 1299)
# use sample 19 - 500 from time window 2 (corresponding to samples 1300 - 1781)
# use sample 19 - 512 from time window 3 (corresponding to samples 1782 - 2275)


set dirin="../data"
set dirout="../data_curvelets"
set base0=shots.inter
set base=shots.T2
set base=shots.T1.5
set baseout=shots.curvsub

# define NMO curve to flatten events 

set nmopar="tnmo=4,7,10 vnmo=6000,7000,8000 smute=3"

if (-f $baseout.su) /bin/rm $baseout.su

# loop over shot groups

set group=1
while ($group <= 2)

if ($group == 1) set shotmin=45
if ($group == 1) set shotmax=545
if ($group == 2) set shotmin=546
if ($group == 2) set shotmax=1045

# select group of shots from the input file, split odd and even ones

echo " "
echo " Shot records $base group $group with shots $shotmin - $shotmax"
echo " "

# loop over time windows

set window=1
while ($window <= 3)

set nsam0=799
if ($window == 1) set sammin=1
if ($window == 1) set sammax=500
if ($window == 2) set sammin=19
if ($window == 2) set sammax=500
if ($window == 3) set sammin=19
if ($window == 3) set sammax=512

echo " "
echo " Gathering shot numbers $shotmin - $shotmax, time window nr. $window"
echo " Select time window $window with samples $sammin - $sammax"

# on first time window, also copy files from the original data
# apply same NMO correction as the cube data
if ($window == 1) then
   cat $dirin/$base0.su | \
   suwind key=fldr min=$shotmin max=$shotmax | \
   sunmo $nmopar | \
   convert sammax=$nsam0 > tmp0.su
endif

# copy data from this window in tmp file
cat $dirout/$base.group$group.window$window.su | \
suwind key=fldr min=$shotmin max=$shotmax | \
subuffsort key=fldr,tracf ntrace=200 nxmax=-1 | \
convert sammin=$sammin sammax=$sammax > tmp$window.su

# next time window
set window=`expr $window + 1`
end

# merge the time windows together
suvcat tmp0.su tmp1.su > tmpa.su
suvcat tmp2.su tmp3.su > tmpb.su
suvcat tmpa.su tmpb.su | \
sunmo $nmopar invert=1 >> $baseout.su

# next group of shots
set group=`expr $group + 1`
end
