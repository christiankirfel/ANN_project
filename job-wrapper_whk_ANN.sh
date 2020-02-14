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
export NUMEXPR_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

cd /jwd
module load anaconda/2019.10-py37
tar xf ${BUDDY}/tf2_env.tar.gz
source activate /jwd/tf2_env
tar xf ${BUDDY}/whk_ANN_code.tar.gz
cd whk_ANN_code
python whk_ANN_run.py