'''
Submits jobs for HTCondor GPU.
Format: python JobSubmitter.py [options] [comma-separated values] ...
	e.g. python JobSubmitter.py LambdaValue -0.1,-0.2,-0.3
Additional options:
	-p, --pseudo: Generate the job file but don't submit
'''

import configparser as cfg
import os
import sys
import itertools
import logging

config = cfg.ConfigParser(inline_comment_prefixes="#")
config.read('config_ANN.ini')
config = config['General']
if config['DebugLevel']=='DEBUG':
	logging.basicConfig(level=logging.DEBUG)
else:
	logging.basicConfig(level=logging.WARNING)

if sys.version_info.major < 3:
	print('Use Python 3 you pleblord')
	sys.exit()

argument_string = ''
option = []
varlist = []
o_pseudo = False
argv_pos = 1
arguments = ''

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
		while argv_pos < len(sys.argv):
			logging.debug(f'Parsing \'{str(sys.argv[argv_pos])}\'')
			if str(sys.argv[argv_pos]) == '--pseudo' or str(sys.argv[argv_pos]) == '-p':
				o_pseudo = True
				argv_pos += 1
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

print(arguments)

import pprint
pprint.pprint(varlist)

script = f"""
#################################################################################################
#                            HTCondor Job Submission File 
# See http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html for further commands
#################################################################################################

# Path to executable
Executable              = job-wrapper_ANN.sh
# Job process number is given as argument to executable
Arguments               = "{arguments}"
#$(Process) $(Variable) 
# Use HTCondor's vanilla universe (see http://research.cs.wisc.edu/htcondor/manual/current/2_4Running_Job.html)
Universe                = vanilla

# Specify files to be transferred (please note that files on CephFS should _NOT_ be transferred!!!)
# Should executable be transferred from the submit node to the job working directory on the worker node?
Transfer_executable     = True
# List of input files to be transferred from the submit node to the job working directory on the worker node
Transfer_input_files    = runANN.sh, variables_{config['Region']}.txt, config_ANN.ini,  run_ANN.py,  ANN_defs.py, tW_tt_v29_parton_{config['Region']}.root
# List of output files to be transferred from the job working directory on the worker node to the submit node
#Transfer_output_files   = code/model_prediction.jar, code/sample_validation.jar, code/target_adversarial_validation.jar, code/target_validation.jar, code/history_dict.jar, code/adversary_history_dict.jar, code/l_test.jar, code/l_train.jar, code/tpr.jar, code/fpr.jar, code/auc.jar
Transfer_output_files   = code/out

# Specify job input and output
Error                   = logs/err/$(ClusterId).$(Process).err.txt
#Input                   =                                                
Output                  = logs/out/$(ClusterId).$(Process).out.txt                          
Log                     = logs/log/$(ClusterId).$(Process).log.txt
JobBatchName = "ANN_GPU"

# Request resources to the best of your knowledge
# (check log file after job completion to compare requested and used resources)
# Memory in MiB, if no unit is specified!
Request_memory          = 16 GB
Request_cpus            = 8
Request_gpus            = 1
# Disk space in kiB, if no unit is specified!
Request_disk            = 20 GB

# Additional job requirements (note the plus signs)
# Choose OS (options: "SL6", "CentOS7", "Ubuntu1604")
+ContainerOS            = "CentOS7"
+CephFS_IO				= "high"
+MaxRuntimeHours		= 6

queue """

logging.debug(f'Argument string is {argument_string}')
script += argument_string
with open('AutoJobSubmission.jdl','w') as f:
	f.write(script)

if not o_pseudo:
	print('Submitting jobs')
	os.system('condor_submit AutoJobSubmission.jdl')
	os.system('rm -f AutoJobSubmission.jdl')
