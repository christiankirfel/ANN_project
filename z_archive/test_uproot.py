import tensorflow as tf
import pandas as pd
import uproot as ur


input_path = 'tW_tt_v29_parton_1j1b.root'
signal_sample = 'wt_nominal'
background_sample = 'tt_nominal'
signal_systematics_sample = 'wt_systematic'
background_systematic_sample = 'tt_systematic'

signal_tree = ur.open(input_path)[signal_sample]
background_tree = ur.open(input_path)[background_sample]
signal_systematics_tree = ur.open(input_path)[signal_systematics_sample]
background_systematics_tree = ur.open(input_path)[background_systematic_sample]

list_samples = [signal_tree,signal_systematics_tree,background_tree,background_systematics_tree]
list_names = [signal_sample,signal_systematics_sample,background_sample,background_systematic_sample]


with ur.recreate('testRootFile.root') as f:
	for i in range(len(list_samples)):
		vartype_dict = {}
		var_dict = {}
		vars = [var.decode('utf-8') for var in list_samples[i].iterkeys()]
		for var in vars:
			sample = list_samples[i].pandas.df(var)
			sample_type = np.array(sample).dtype
			if sample_type in ['float32','int32']:
				vartype_dict[var] = sample_type
				var_dict[var] = sample
		#vartype_dict['NN_pred'] = 'float32'
		#var_dict['NN_pred'] = [float(j) for j in self.model_prediction]

		f[list_names[i]] = ur.newtree({var:val for var,val in vartype_dict.items()})
		for var,val in var_dict.items():
			f[list_names[i]][var].newbasket(val)