import os, sys

argument_string = ''
option = ''
varlist = []
o_pseudo = False
argv_pos = 1


#parse arguments
try:
	if len(sys.argv) == 1:
		argument_string = ''
	else:
		while argv_pos < len(sys.argv):
			if str(sys.argv[argv_pos]) == '-pseudo':
				o_pseudo = True
				argv_pos += 1
			else:
				argument_string = ' options from (\n'
				option = str(sys.argv[argv_pos])
				print('Option: ' + option)
				varlist = str(sys.argv[argv_pos+1]).split(',')
				for el in varlist:
					print('Value :' + str(el))
					argument_string += ('\t'+option+'='+str(el)+'\n')
				argv_pos+=2
			argument_string+=')'
except:
	print('Couldn\'t parse arguments. Format: JobSubmitter.py [options] [comma-separated values]')
	sys.exit(1)
	

script = """
#################################################################################################
#                            HTCondor Job Submission File 
# See http://research.cs.wisc.edu/htcondor/manual/current/condor_submit.html for further commands
#################################################################################################

# Path to executable
Executable              = job-wrapper_whk_ANN.sh
# Job process number is given as argument to executable
Arguments               = "$(options)"
#$(Process) $(Variable) 
# Use HTCondor's vanilla universe (see http://research.cs.wisc.edu/htcondor/manual/current/2_4Running_Job.html)
Universe                = vanilla

# Specify files to be transferred (please note that files on CephFS should _NOT_ be transferred!!!)
# Should executable be transferred from the submit node to the job working directory on the worker node?
Transfer_executable     = True
# List of input files to be transferred from the submit node to the job working directory on the worker node
Transfer_input_files    = runANN.sh, variables.txt, config_whk_ANN.ini,  whk_ANN_run.py,  whk_ANN_defs.py
# List of output files to be transferred from the job working directory on the worker node to the submit node
#Transfer_output_files   = code/model_prediction.jar, code/sample_validation.jar, code/target_adversarial_validation.jar, code/target_validation.jar, code/history_dict.jar, code/adversary_history_dict.jar, code/l_test.jar, code/l_train.jar, code/tpr.jar, code/fpr.jar, code/auc.jar
Transfer_output_files   = code/out

# Specify job input and output
Error                   = log_whk_ANN/err/$(ClusterId).$(Process).err.txt
#Input                   =                                                
Output                  = log_whk_ANN/out/$(ClusterId).$(Process).out.txt                          
Log                     = log_whk_ANN/log/$(ClusterId).$(Process).log.txt
JobBatchName = "whk_ANN_GPU"

# Request resources to the best of your knowledge
# (check log file after job completion to compare requested and used resources)
# Memory in MiB, if no unit is specified!
Request_memory          = 16 GB
Request_cpus            = 8
Request_gpus            = 1
# Disk space in kiB, if no unit is specified!
Request_disk            = 5 GB

# Additional job requirements (note the plus signs)
# Choose OS (options: "SL6", "CentOS7", "Ubuntu1604")
+ContainerOS            = "CentOS7"

queue"""

script += argument_string
with open('AutoJobSubmission.jdl','w') as f:
	f.write(script)

if not o_pseudo:
	os.system('condor_submit AutoJobSubmission.jdl')
	os.system('rm -f AutoJobSubmission.jdl')

