#!/bin/bash
sleep 180 && while test -d ${_CONDOR_SCRATCH_DIR}/.condor_ssh_to_job_1; do /bin/sleep 3; done