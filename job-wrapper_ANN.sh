#!/bin/bash

# # Check provided parameters
# if [ ${#*} -ne 1 ]; then
#     echo "usage: `basename $0` <jobnumber>"
#     exit 1
# fi

#echo "In job_wrapper_ANN.sh with arguments $@" >> /cephfs/user/s6niboei/BAFDEBUG.log

# Store script parameter in a variable with descriptive name
#JOBNUM=$1

# Source shell profile (needed to run setupATLAS)
source /etc/profile

# Set up desired ROOT version (taken from CVMFS)
#setupATLAS


cd /jwd
echo 'Loading anaconda' >> "/cephfs/user/s6niboei/BAFDEBUG.log"
module load anaconda/2020.02-py37
echo 'Activating environment' >> "/cephfs/user/s6niboei/BAFDEBUG.log"
tar xf ${BUDDY}/tf2.1_gpu.tar.gz
source activate /jwd/tf2.1_gpu
python -m pip install guppy3
# debug
mkdir code
mv *.py runANN.sh *.ini *.txt *.root code/
cd code
mkdir out
source runANN.sh $@