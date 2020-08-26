#!/bin/bash

while true; do
    condor_q -constraint 'JobStatus == 2' -af:hj Cmd ResidentSetSize_RAW RequestMemory DiskUsage_RAW RequestDisk
    sleep 30
done