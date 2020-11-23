# **ANN for the *tW* channel**

## Overview

- All the config is in **config_ANN.ini**
- ANN code is in **ANN_defs.py**
- Plotting is **plot_defs.py**
- Run the ANN with **run_ANN.py**/**runANN.sh**
- Submit to BAF with **JobSubmitter.py**

## Input data

This code expects a single flat n-tuple containing each training set in a single tree with names specificied in the config.

## How to submit jobs to BAF

Submit a single job:

`$ python JobSubmitter.py`

Start grid searches by adding \<option\> \<comma-separated list\>, e.g.

`$ python JobSubmitter.py LambdaValue 0.1,0.25,0.5,1.0`

Result folder will be named accordingly.

Multiple options can be added together to iterate through all possibilities:

`$ python JobSubmitter.py LambdaValue 0.1,0.5 CombinedLearningRate 0.01,0.05`

Be careful: this will try every combination so it can cause a lot of jobs to be started

## Additional options

`-p:` generate submit file but don't actually submit, for debugging 

`-cpu:` submit to CPU instead of GPU nodes

`-cpucores:` specify CPU cores to request (should never be >8 for GPU jobs)

## Separate plotting

Plotting can also be done separately by using the **PlotSubmitter.py** script. It will need some modifications tho to work.

## Other stuff

CheckResource.sh checks current resource usage of jobs, originally to debug a memory leak. Not very useful tho...

condorq.py is a little script to format `condor_q -global` output in a nicer way

ks.py calculates the difference/separation of nom and sys samples

plot_batchsize.py makes a nice plot to look at performance compared to batch size