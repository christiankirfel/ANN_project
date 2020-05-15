#!/bin/bash

# # Check provided parameters
# if [ ${#*} -ne 1 ]; then
#     echo "usage: `basename $0` <jobnumber>"
#     exit 1
# fi



# Store script parameter in a variable with descriptive name
#JOBNUM=$1

# Source shell profile (needed to run setupATLAS)
source /etc/profile

# Set up desired ROOT version (taken from CVMFS)
setupATLAS


cd /jwd
module load anaconda/2019.10-py37
tar xf ${BUDDY}/tf2_gpu.tar.gz
source activate /jwd/tf2_gpu
mkdir code
mv *.py runANN.sh *.ini *.txt code/
cd code
mkdir out
source runANN.sh ${1}