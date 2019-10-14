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
#lsetup "root 6.14.04-x86_64-slc6-gcc62-opt" 
lsetup "lcgenv -p LCG_94 x86_64-slc6-gcc7-opt ROOT"
lsetup "lcgenv -p LCG_94 x86_64-slc6-gcc7-opt matplotlib"
lsetup "lcgenv -p LCG_94 x86_64-slc6-gcc7-opt scikitlearn"
lsetup "lcgenv -p LCG_94 x86_64-slc6-gcc7-opt keras"
lsetup "lcgenv -p LCG_94 x86_64-slc6-gcc7-opt ipython"
lsetup "lcgenv -p LCG_94 x86_64-slc6-gcc7-opt cyphon"
lsetup "lcgenv -p LCG_94 x86_64-slc6-gcc7-opt h5py"
lsetup "lcgenv -p LCG_94 x86_64-slc6-gcc7-opt tensorflow"
#export PATH=$PATH:~/.local/bin
export NUMEXPR_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

echo "Job $JOBNUM"

module load anaconda/5.3.0-p37
source activate /cephfs/user/s6chkirf/whk_env/

python /cephfs/user/s6chkirf/whk_ANN_run.py