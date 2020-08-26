import subprocess, time

while True:
	out = subprocess.check_output("condor_q -constraint 'JobStatus == 2' -af:hj Cmd ResidentSetSize_RAW RequestMemory DiskUsage_RAW RequestDisk", shell=True)
	out_list = out.decode('utf-8').split('\n')
	try:
		for i in range(1,len(out_list)-1):
			temp = out_list[i].split()
			ID = temp[0]
			try:
				lel = int(temp[2]) / 1024
				usage = f'{lel} MB'
			except:
				print(f'Can\'t convert {temp[2]}')
				usage = temp[2]
			print(f'{ID}: {usage}')
	except:
		print('No active processes found')
	time.sleep(30)