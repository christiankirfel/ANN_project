'''
Submits jobs for HTCondor GPU.
'''

import configparser as cfg
import os
import sys
import itertools
import logging

if sys.version_info.major < 3:
	print('Use Python 3 you pleblord')
	sys.exit()

argument_string = ''
option = []
varlist = []
o_pseudo = False
o_cpu = False
argv_pos = 1
arguments = ''
req_cpu = 8
req_ram = 16
req_hdd = 20

ANNconfig = cfg.ConfigParser(inline_comment_prefixes="#")
ANNconfig.read('config_ANN.ini')
ANNconfig = ANNconfig['General']

#parse arguments
logging.debug('[JOBSUBMITTER]')
logging.debug('Parsing JobSubmitter arguments')

for el in sys.argv:
	logging.debug(str(el))

try:
	if len(sys.argv) == 1:
		pass
	else:
		# what a mess
		while argv_pos < len(sys.argv):
			logging.debug(f'Parsing \'{str(sys.argv[argv_pos])}\'')
			if str(sys.argv[argv_pos]) == '--pseudo' or str(sys.argv[argv_pos]) == '-p':
				o_pseudo = True
				argv_pos += 1
			elif str(sys.argv[argv_pos]) == '-cpu':
				o_cpu = True
				argv_pos += 1
				logging.debug('Using CPU')
			elif str(sys.argv[argv_pos]) == '-cpucores':
				req_cpu = int(sys.argv[argv_pos+1])
				argv_pos += 2
				logging.debug(f'Requesting {req_cpu} CPU cores')
			else:
				# make sure option is part of the config
				if not str(sys.argv[argv_pos]) in ANNconfig:
					print(f'ERROR: {sys.argv[argv_pos]} is not a valid option.')
					sys.exit(1)
				option.append(str(sys.argv[argv_pos]))
				varlist.append(str(sys.argv[argv_pos+1]).split(','))
				argv_pos += 2
except:
	print('Couldn\'t parse arguments. Format: JobSubmitter.py [options] [comma-separated values]')
	sys.exit(1)

# this looks funky but its just converting the arguments in the proper format
if len(option) != 0:
	for el in option:
		if arguments != '':
			arguments += ' '
		if argument_string != '':
			argument_string += ', '
		arguments = arguments + '$(' + el + ')'
		argument_string = argument_string + el
	argument_string += ' from (\n'
	x = list(itertools.product(*varlist))
	for i, item in enumerate(x):
		argument_string += '\t'
		for j, jtem in enumerate(option):
			if not j==0:
				argument_string += ' '
			argument_string += option[j] + '=' + x[i][j]
		argument_string += '\n'
	argument_string += ')'

import pprint
pprint.pprint(varlist)

shfile = 'runANN.sh'
cfile = 'config_ANN.ini'
pyfile = 'run_ANN.py'
jname = 'ANN_GPU'
wfile = 'job-wrapper_ANN.sh'

if o_cpu:
	req_gpu = ''
else:
	req_gpu = 'Request_gpus			= 1'

script = f"""
#################################################################################################
#                            HTCondor Job Submission File 
# See http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html for further commands
#################################################################################################

# Path to executable
Executable              = {wfile}
# Job process number is given as argument to executable
Arguments               = "{arguments}"
#$(Process) $(Variable) 
# Use HTCondor's vanilla universe (see http://research.cs.wisc.edu/htcondor/manual/current/2_4Running_Job.html)
Universe                = vanilla

# Specify files to be transferred (please note that files on CephFS should _NOT_ be transferred!!!)
# Should executable be transferred from the submit node to the job working directory on the worker node?
Transfer_executable     = True
# List of input files to be transferred from the submit node to the job working directory on the worker node
Transfer_input_files    = {shfile}, variables_{ANNconfig['Region']}.txt, {cfile},  {pyfile},  ANN_defs.py, plot_defs.py
# List of output files to be transferred from the job working directory on the worker node to the submit node
Transfer_output_files   = code/out

# Specify job input and output
Error                   = logs/err/$(ClusterId).$(Process).err.txt
#Input                   =                                                
Output                  = logs/out/$(ClusterId).$(Process).out.txt                          
Log                     = logs/log/$(ClusterId).$(Process).log.txt
JobBatchName = "{jname}"

# Request resources to the best of your knowledge
# (check log file after job completion to compare requested and used resources)
# Memory in MiB, if no unit is specified!
Request_memory          = {req_ram} GB
Request_cpus            = {req_cpu}
{req_gpu}
# Disk space in kiB, if no unit is specified!
Request_disk            = {req_hdd} GB

# Additional job requirements (note the plus signs)
# Choose OS (options: "SL6", "CentOS7", "Ubuntu1604")
+ContainerOS            = "CentOS7"
+CephFS_IO				= "high"
+MaxRuntimeHours		= 3

queue """

logging.debug(f'Argument string is {argument_string}')
script += argument_string
with open('AutoJobSubmission.jdl','w') as f:
	f.write(script)

if not o_pseudo:
	#print('Submitting jobs')
	os.system('condor_submit AutoJobSubmission.jdl')
	os.system('rm -f AutoJobSubmission.jdl')
