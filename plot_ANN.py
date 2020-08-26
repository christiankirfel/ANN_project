'''
Script to do all the necessary plotting separate from the main code
'''

import time
time1 = time.time()
import os, sys, glob
import concurrent.futures
from plot_defs import *

# This script breaks with Python 2 so just don't
if sys.version_info[0]<3:
	print('Use Python 3 you muppet')
	sys.exit()

# Check if the code is running on Windows
if sys.platform.startswith('win32'):
	onWindows = True
	print('Windows environment detected')
else:
	onWindows = False

print('Finding pickle jar')


if len(sys.argv) > 2:
	#Process specified files
	files = [str(x) for x in sys.argv[1:]]
else:
	#Process all files in the folder
	files = glob.glob('*.tar')

print(f'Processing {len(files)} files')
import multiprocessing
print(f'{multiprocessing.cpu_count()} cores available')

#If multiple files are given, use multithreading
# Windows is special
if onWindows:
	if __name__ == '__main__':
		with concurrent.futures.ProcessPoolExecutor() as executor:
			executor.map(do_the_things, files)
else:
	with concurrent.futures.ProcessPoolExecutor() as executor:
		executor.map(do_the_things, files)

print(str(time.time() - time1) + ' seconds')