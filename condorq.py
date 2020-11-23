'''
Make the output of condor_q -global look way more readable
'''

from subprocess import check_output
import logging
import sys

# Set the user here
user = 's6niboei'

# Debug stuff
debug_levels = {'DEBUG': logging.DEBUG,
				'INFO': logging.INFO,
				'WARNING': logging.WARNING,
				'ERROR': logging.ERROR}

if len(sys.argv)==2 and str(sys.argv[1]) in debug_levels.keys():
	logging.basicConfig(level = debug_levels[str(sys.argv[1])])
else:
	logging.basicConfig(level = logging.WARNING)

# check output
file_string = str(check_output(['condor_q','-global']))
# split output by nodes
jobs_list = file_string.split('--')

print_list = [['NODE','JOB START','JOB NAME','JOB STATUS','JOB IDs'],
			['--------','--------','--------','--------','--------']]
TOTAL = 0
HELD = False

# pylint: disable=missing-class-docstring
class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

# fish out the jobs of user and print them in a nice looking way
for job in jobs_list:
	# split lines
	temp_list = job.split('\\n')
	logging.debug(temp_list[0].split(' '))
	if 'Schedd:' in temp_list[0]:
		logging.debug('Found Schedd')
		# get server name
		server_name = temp_list[0].split()[1]
		# get batch name and job numbers
		for j in range(2,6):
			try:
				assert temp_list[j].split()[0] == user
			except:
				continue
			info_list = temp_list[j].split()
			batch_name = info_list[1]

			job_start = f'{info_list[2]} {info_list[3]}'
			job_ids = ' '.join(info_list[9:])
			if logging.root.level == logging.DEBUG:
				logging.debug('INFO_LIST:')
				_ = [logging.debug(f'{k}: {info_list[k]}') for k in range(len(info_list))]
				logging.debug('END_INFO_LIST')

			jobs = {}
			job_states = ['done','run','idle','hold']

			for i, item in enumerate(job_states):
				jobs[item] = int(info_list[i+4]) if info_list[i+4].isdigit() else 0

			jobs_total = sum(jobs.values())

			if jobs_total == 0:
				continue
			else:
				TOTAL += jobs_total
				if jobs['hold'] > 0:
					HELD = True
				columns = [f'{server_name}:',
						f'{job_start}',
						f'{batch_name}',
						f'{jobs["done"]} done, {jobs["run"]} running, {jobs["idle"]} idle, {jobs["hold"]} held.',
						f'{job_ids}']
				print_list.append(columns)

if HELD:
	print(f'{bcolors.FAIL}{bcolors.BOLD}WARNING: JOBS ARE BEING HELD{bcolors.ENDC}')
print(f'TOTAL JOBS: {TOTAL}')

lens = []
for col in zip(*print_list):
	lens.append(max([len(v)+10 for v in col]))
format_ = " ".join(["{:<" + str(l) + "}" for l in lens])
for row in print_list:
	print(format_.format(*row))
	