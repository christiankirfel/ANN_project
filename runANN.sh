#!/bin/bash
#echo "runnANN.sh Arguments are $@" >> /cephfs/user/s6niboei/BAFDEBUG.log
TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_ANN.py "$@"