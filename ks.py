'''
Naively calculates the difference of integrals or separation between nominal and systematic sample network response (tW only)
'''

import tarfile
import pickle
import sys
from scipy import stats
import numpy as np

f = str(sys.argv[1])

with tarfile.open(f) as tar:
	tar.extractall()

path_prefix = ''

sample_validation = pickle.load(open(path_prefix + 'sample_validation.pickle','rb'))
target_validation = pickle.load(open(path_prefix + 'target_validation.pickle','rb'))
model_prediction = pickle.load(open(path_prefix + 'model_prediction.pickle','rb'))
target_adversarial_validation = pickle.load(open(path_prefix + 'target_adversarial_validation.pickle','rb'))

signal_histo = []
signal_sys_histo = []
background_histo = []
background_sys_histo = []
for i in range(len(sample_validation)):
	if target_validation[i] == 1 and target_adversarial_validation[i] == 1:
		signal_histo.append(model_prediction[i])
	if target_validation[i] == 1 and target_adversarial_validation[i] == 0:
		signal_sys_histo.append(model_prediction[i])
	if target_validation[i] == 0 and target_adversarial_validation[i] == 1:
		background_histo.append(model_prediction[i])
	if target_validation[i] == 0 and target_adversarial_validation[i] == 0:
		background_sys_histo.append(model_prediction[i])


sh, she = np.histogram(signal_histo, bins=30, range=(0.,1.), density=True)
ssh, sshe = np.histogram(signal_sys_histo, bins=30, range=(0.,1.), density=True)
if background_sys_histo:
	bh, bhe = np.histogram(background_histo, bins=30, range=(0.,1.), density=True)
	bsh, bshe = np.histogram(background_sys_histo, bins=30, range=(0.,1.), density=True)
	sum_b = 0.
	for i in range(len(bh)):
		temp = bh[i] + bsh[i]
		if temp > 0.:
			sum_b += ((bh[i] - bsh[i])**2/(bh[i] + bsh[i]))
		#sum_b += abs(bh[i]-bsh[i])
	sumb_b = 0.5*sum_b
else:
	sum_b = 0.

sum_s = 0.
for i in range(len(sh)):
	temp = sh[i] + bsh[i]
	if temp > 0.:
		sum_s += ((sh[i] - ssh[i])**2/(sh[i] + bsh[i]))
		#sum_s = abs(sh[i] - ssh[i])
sum_s = 0.5*sum_s

print(f'Signal: {sum_s}, Background: {sum_b}')

