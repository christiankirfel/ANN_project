import glob
from plot_defs import *
import time

time_start = time.time()
f = glob.glob('*.tar')
tarfile = f[0]
print(f'{time.time() - time_start}')

do_the_things(tarfile)
