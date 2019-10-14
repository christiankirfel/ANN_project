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
module load anaconda/5.3.0-py37
source activate /cephfs/user/s6chkirf/whk_env/

python /cephfs/user/s6chkirf/whk_ANN_defs.py ${1} ${2} ${3} ${4} ${5} ${6} ${7}