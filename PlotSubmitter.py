import os, sys, glob

# This script breaks with Python 2 so just don't
if sys.version_info[0]<3:
	print('Use Python 3 you muppet')
	sys.exit()
files = []
if len(sys.argv)>1:
	for el in sys.argv[1:]:
		files.append(str(el))
else:
	files = glob.glob('*.tar')

arguments = ''

for f in files:

	f_s = f.replace('ANN_out_','').replace('.tar','')
	if len(sys.argv)==1 and os.path.exists(f_s):
		continue
		#pass
	os.system(f'mkdir -p {f_s}')
	arguments += f'\t{f} {f_s}\n'

script = f"""
#################################################################################################
#                            HTCondor Job Submission File 
# See http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html for further commands
#################################################################################################

# Path to executable
Executable              = plot_wrapper.sh
# Job process number is given as argument to executable
Arguments               = "" 
# Use HTCondor's vanilla universe (see http://research.cs.wisc.edu/htcondor/manual/current/2_4Running_Job.html)
Universe                = vanilla

# Specify files to be transferred (please note that files on CephFS should _NOT_ be transferred!!!)
# Should executable be transferred from the submit node to the job working directory on the worker node?
Transfer_executable     = True
# List of input files to be transferred from the submit node to the job working directory on the worker node
Transfer_input_files    = ../plot_defs.py, PlotOnBAF.py, $(fi)
# List of output files to be transferred from the job working directory on the worker node to the submit node
Transfer_output_files   = $(fi_s)

# Specify job input and output
Error                   = ../logs/err/plot_$(ClusterId).$(Process).err.txt
#Input                   =                                                
Output                  = ../logs/out/plot_$(ClusterId).$(Process).out.txt                          
Log                     = ../logs/log/plot_$(ClusterId).$(Process).log.txt
JobBatchName = "plot_ANN"

# Request resources to the best of your knowledge
# (check log file after job completion to compare requested and used resources)
# Memory in MiB, if no unit is specified!
Request_memory          = 1 GB
Request_cpus            = 1
Request_gpus            = 0
# Disk space in kiB, if no unit is specified!
Request_disk            = 3 GB

# Additional job requirements (note the plus signs)
# Choose OS (options: "SL6", "CentOS7", "Ubuntu1604")
+ContainerOS            = "CentOS7"
+CephFS_IO				= "low"
+MaxRuntimeHours		= 1

queue fi, fi_s from (
{arguments})"""

with open('AutoJobSubmission.jdl','w') as f:
	f.write(script)

#print('Submitting jobs')
os.system('condor_submit AutoJobSubmission.jdl')
os.system('rm -f AutoJobSubmission.jdl')