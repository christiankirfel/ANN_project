'''
Script to do all the necessary plotting separate from the main code
'''
import tarfile, pickle, os, sys, glob
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.metrics import roc_curve, auc

def plot_sep_all():
	'''
	Plot signal background separation, split up into nominal and systematic, with ratio plot
	'''
	print('Eating a pickle')
	plt.title('tW and tt separation')
	axis1 = plt.subplot(211)
	signal_histo = []
	background_histo = []
	signal_sys_histo = []
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
			
	ns1, bins1, patches1 = plt.hist(signal_histo, range=[0., 1.], linewidth = 1, bins=30, histtype="step", density = True, color=color_tW, label = r'$tW_\mathrm{nom}$')
	ns3, bins3, patches3 = plt.hist(background_histo, range=[0., 1.], linewidth = 1, bins=30, histtype="step", density = True, color=color_tt, label = r'$t\bar{t}_\mathrm{nom}$')
	ns2, bins2, patches2 = plt.hist(signal_sys_histo, range=[0., 1.], linewidth = 1, bins=30, histtype="step", density = True, color=color_tW, label = r'$tW_\mathrm{sys}$', linestyle='dashed')
	ns4, bins4, patches4 = plt.hist(background_sys_histo, range=[0., 1.], linewidth = 1, bins=30, histtype="step", density = True, color=color_tt, label = r'$t\bar{t}_\mathrm{sys}$', linestyle='dashed')
	#        plt.hist(model_prediction[target_training.tolist() == 0], range=[0., 1.], linewidth = 2, bins=30, histtype="step", normed=1, color=color_tt)
	#        plt.hist(predicttest__ANN[test_target == 1],   range=[xlo, xhi], linewidth = 2, bins=bins, histtype="step", normed=1, color=color_tW2, label='$Sig_{test}$', linestyle='dashed')
	#        plt.hist(predicttest__ANN[test_target == 0],   range=[xlo, xhi], linewidth = 2, bins=bins, histtype="step", normed=1, color=color_tt2, label='$Bkg_{test}$', linestyle='dashed')
	#        plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
	plt.legend()
	plt.ylabel('Event fraction', fontsize='large')
	plt.legend(frameon=False)

	ratioArray_tW = []
	for iterator in range(len(ns1)):
		if ns1[iterator] > 0:
			ratioArray_tW.append(ns2[iterator]/ns1[iterator])
		else:
			ratioArray_tW.append(1.)

	ratioArray_tt = []
	for iterator in range(len(ns3)):
		if ns3[iterator] > 0:
			ratioArray_tt.append(ns4[iterator]/ns3[iterator])
		else:
			ratioArray_tt.append(1.)
	
	axis2 = plt.subplot(212, sharex = axis1)
	plt.plot(bins1[:-1], ratioArray_tW, color = color_tW, drawstyle = 'steps-mid')
	plt.plot(bins1[:-1], ratioArray_tt, color = color_tt, drawstyle = 'steps-mid')
	plt.hlines(1, xmin = -0.0, xmax = 1.0)
	plt.ylabel('Event ratio', fontsize='large')
	plt.ylim(top = 1.5,bottom=0.75)
	plt.xlabel('Network response', horizontalalignment='left', fontsize='large')
	plt.gcf().savefig('separation_discriminator_2' + path_postfix + '.png')
	#plt.show()
	plt.gcf().clear()

def plot_losses_all():
	'''
	Plot discriminator, adversary and combined loss
	'''
	print('Eating a pickle')
	#lossestest = {"L_f": [], "L_r": [], "L_f - L_r": []}
	#lossestrain = {"L_f": [], "L_r": [], "L_f - L_r": []}
	#lossestest["L_f"].append(l_test[1])
	#lossestest["L_r"].append(-l_test[2])
	#lossestest["L_f - L_r"].append(l_test[0])
	#lossestrain["L_f"].append(l_train[1])
	#lossestrain["L_r"].append(-l_train[2])
	#lossestrain["L_f - L_r"].append(l_train[0])

	ax1 = plt.subplot(311)   
	values_test = np.array(l_test["L_f"])
	values_train = np.array(l_train["L_f"])
	plt.plot(range(len(values_test)), values_test, label=r"$Loss_{net1}^{test}$", color="blue", linestyle='dashed')
	plt.plot(range(len(values_train)), values_train, label=r"$Loss_{net1}^{train}$", color="blue")
	plt.legend(loc="upper right", prop={'size' : 7})
	plt.legend(frameon=False)
	plt.grid()
	ax1.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
	
	ax2 = plt.subplot(312, sharex=ax1) 
	values_test = np.array(l_test["L_r"])
	values_train = np.array(l_train["L_r"])
	plt.plot(range(len(values_test)), values_test, label=r"$\lambda$"+r"$ \cdot Loss_{net2}^{test}$", color="green", linestyle='dashed')
	plt.plot(range(len(values_train)), values_train, label=r"$\lambda$"+r"$ \cdot Loss_{net2}^{train}$", color="green")
	plt.legend(loc="upper right", prop={'size' : 7})
	plt.legend(frameon=False)
	plt.ylabel('Loss')
	plt.grid()
	ax2.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
	
	ax3 = plt.subplot(313, sharex=ax1)
	values_test = np.array(l_test["L_f - L_r"])
	values_train = np.array(l_train["L_f - L_r"])
	plt.plot(range(len(values_test)), values_test, label=r"$Loss_{net1}^{test} - $"+r"$\lambda$"+r"$ \cdot Loss_{net2}^{test}$", color="red", linestyle='dashed')  
	plt.plot(range(len(values_train)), values_train, label=r"$Loss_{net1}^{train} - $"+r"$\lambda$"+r"$ \cdot Loss_{net2}^{train}$", color="red")  
	plt.legend(loc="upper right", prop={'size' : 7})
	plt.grid()
	ax3.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
	plt.xlabel('Epoch')
	
	plt.legend(frameon=False)
	plt.gcf().savefig('losses' + path_postfix + '.png')
	plt.gcf().clear()

def plot_losses_combined():
	ax1 = plt.subplot(311)
	plt.plot(model_loss)
	plt.plot(val_model_loss)

	ax2 = plt.subplot(312, sharex=ax1)
	#print(type(model_1_loss))
	#model_1_loss1 = [-x for x in model_1_loss]
	#val_model_1_loss1 = [-x for x in val_model_1_loss]
	plt.plot(model_2_loss1)
	plt.plot(val_model_2_loss1)

	ax3 = plt.subplot(313, sharex=ax1)
	plt.plot(loss)
	plt.plot(val_loss)

	plt.savefig('combined_losses_' + path_postfix + '.png')
	plt.clf()

def plot_roc():
	'''
	Plot Receiver Operating Curve
	'''
	print('Eating a pickle')
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'g--', label='$AUC_{train}$ = %0.2f'% auc_var)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.,1.])
	plt.ylim([-0.,1.])
	plt.ylabel('True Positive Rate', fontsize='large')
	plt.xlabel('False Positive Rate', fontsize='large')
	plt.legend(frameon=False)
	plt.gcf().savefig('roc' + path_postfix + '.png')
	plt.gcf().clear()

def plot_accuracy():
	'''
	Plot the accuracy. Surprising, isn't it?
	'''
	print('Eating a pickle')
	plt.plot(model_history_array['binary_accuracy'])
	plt.plot(model_history_arra['val_binary_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.show()
	plt.gcf().savefig('acc' + path_postfix + '.png')
	plt.gcf().clear()


print('Finding pickle jar')


color_tW = '#0066ff'
color_tt = '#990000'
color_sys = '#009900'
color_tW2 = '#02590f'
color_tt2 = '#FF6600'

path_prefix = ''
path_postfix = ''

for f in glob.glob('*.tar'):

	folder = f.replace('ANN_out_','').replace('.tar','')

	with tarfile.open(f) as tar:
		tar.extractall(path='')

	print('Opening pickle jar')


	#model_history = pickle.load(open(path_prefix + 'history.pickle', 'rb'))
	model_history_array = pickle.load(open(path_prefix + 'model_history_array.pickle','rb'))
	sample_validation = pickle.load(open(path_prefix + 'sample_validation.pickle','rb'))
	target_validation = pickle.load(open(path_prefix + 'target_validation.pickle','rb'))
	model_prediction = pickle.load(open(path_prefix + 'model_prediction.pickle','rb'))
	adversary_prediction = pickle.load(open(path_prefix + 'adversary_prediction.pickle','rb'))
	target_adversarial_validation = pickle.load(open(path_prefix + 'target_adversarial_validation.pickle','rb'))
	l_test = pickle.load(open(path_prefix + 'losses_test.pickle','rb'))
	l_train = pickle.load(open(path_prefix + 'losses_train.pickle','rb'))

	#print('History:')
	#print(l_test)

	os.system('mkdir ' + folder)
	os.chdir(folder)

	fpr, tpr, threshold = roc_curve(target_validation, model_prediction)
	auc_var = auc(fpr, tpr)
	adversary_fpr, adversary_tpr, adversary_threshold = roc_curve(target_adversarial_validation, adversary_prediction)

	loss = []
	model_2_loss = []
	model_loss = []
	val_loss = []
	val_model_2_loss = []
	val_model_loss = []
	for el in model_history_array:
		loss.append(el['loss'])
		model_2_loss.append(el['model_2_loss'])
		model_loss.append(el['model_loss'])
		val_loss.append(el['val_loss'])
		val_model_2_loss.append(el['val_model_2_loss'])
		val_model_loss.append(el['val_model_loss'])
	for el in model_2_loss:
		model_2_loss1 = [-(x[0]) for x in model_2_loss]
		val_model_2_loss1 = [-(x[0]) for x in model_2_loss]

	#adversary_auc = auc(adversary_fpr, adversary_tpr)
	plot_sep_all()
	plot_roc()
	plot_losses_combined()
	print('All gone')

	os.chdir('..')