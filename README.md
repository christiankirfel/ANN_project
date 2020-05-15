change config stuff in **config\_whk\_ANN.ini**

run training using **whk\_ANN\_run.py**

plot using **whk\_ANN\_plot.py**

submit BAF jobs using **JobSubmitter.py**

all the magic happens in **whk\_ANN\_defs.py**

Changelog

**20-05-15**
- Added support for 2nd to last layer as adversarial input
- Added job submitting python script

**20-03-08**
- Moved plotting into a separate file

**19-11-20**
- Added batch size option to network.evaluate

**19-11-08**
- Created new branch for further work
- Started implementing config file

**19-10-14**
- Initial commit with code from Christian's master thesis
