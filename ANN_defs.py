#Loading base packages
import os
import sys
import configparser as cfg
import pickle
import logging
from datetime import datetime
import random as rn
import time

#Loading tensorflow
#Setting some options for BAF
import tensorflow as tf
NUM_THREADS=1
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

# pylint: disable=import-error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
from tensorflow.keras.losses import binary_crossentropy

#Loading sklearn for data processing & analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

#Loading the packages for handling the data
import uproot as ur
import numpy as np

#Loading packages needed for plottting
import matplotlib.pyplot as plt

#Defining colours for the plots
#The colours were chosen using the xkcd guice
#color_tW = '#66FFFF'
color_tW = '#0066ff'
#color_tt = '#FF3333'
color_tt = '#990000'
color_sys = '#009900'
color_tW2 = '#02590f'
color_tt2 = '#FF6600'


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Setting up the output directories
output_path = '/cephfs/user/s6niboei/out2/'
array_path = output_path + 'arrays/'
if not os.path.exists(output_path):
	os.makedirs(output_path)
if not os.path.exists(array_path):
	os.makedirs(array_path)


plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#This is the main class for the adversarial neural network setup
class ANN_environment(object):
	'''
	Class implementing an Adversarial Neural Network used for analysis of Wt data
	'''

	def __init__(self,args):
		""" opens files, loads config and initializes variables """

		self.time_start = datetime.now().strftime(r"%y%m%d_%H_%M_%S")
		self.time1 = time.time()
		#logging.debug('Setting up ANN')

		#load the config
		self.load_config()
		self.log_file_dir = self.config['LogFileDir']
		self.setup_logging()
		self.load_custom_config(args)
		self.print_config()

		if self.config['UseTop'] == '1':
			with open(f"/cephfs/user/s6niboei/BAF_top_{self.time_start}",'w') as f:
				pass

		with open(f"variables_{self.config['Region']}.txt","r") as f:
			self.variables = np.asarray([line.strip() for line in f])

		self.root_file_dir = self.config['RootFileDir']

		#The seed is used to make sure that both the events and the labels are shuffeled the same way because they are not inherently connected.
		self.seed = int(self.config['Seed'])
		np.random.seed(self.seed)
		rn.seed(self.seed)
		tf.random.set_seed(self.seed)

		# Check if theres a background systematic
		self.has_background_systematic = False if self.config['BackgroundSystematicsSample'] == 'NONE' else True

		#All information necessary for the input
		#The exact data and targets are set late
		self.input_path = self.config['InputPath'] + '_' + self.config['Region']+'.root'
		self.signal_sample = self.config['SignalSample']
		self.background_sample = self.config['BackgroundSample']
		self.signal_systematics_sample = self.config['SignalSystematicsSample']
		self.background_systematics_sample = self.config['BackgroundSystematicsSample'] if self.has_background_systematic else None

		self.signal_tree = ur.open(self.input_path)[self.signal_sample]
		self.background_tree = ur.open(self.input_path)[self.background_sample]
		self.signal_systematics_tree = ur.open(self.input_path)[self.signal_systematics_sample]
		self.background_systematics_tree = ur.open(self.input_path)[self.background_systematics_sample] if self.has_background_systematic else None

		self.sample_training = None
		self.sample_validation = None
		self.adversarial_training = None
		self.adversarial_validation = None
		self.target_training = None
		self.target_validation = None
		self.target_systematic = None

		#Dimension of the variable input used to define the size of the first layer
		self.input_dimension = self.variables.shape
		#These arrays are used to save loss and accuracy of the two networks
		#That is also important to later be able to use the plotting software desired. matplotlib is not the best tool at all times
		self.discriminator_history_array = []
		self.adversary_history_array = []
		self.model_history_array = []
		self.discriminator_history = None
		#Here are the definitions for the two models
		#All information for the length of the training. Beware that epochs might only come into the pretraining
		#Iterations are used for the adversarial part of the training
		#original: 10 10 1000
		self.discriminator_epochs = int(self.config['DiscriminatorEpochs'])
		self.adversary_epochs = int(self.config['AdversaryEpochs'])
		self.training_iterations = int(self.config['TrainingIterations'])
		#Setup of the networks, nodes and layers
		self.discriminator_layers = int(self.config['DiscriminatorLayers'])
		self.discriminator_nodes = int(self.config['DiscriminatorNodes'])
		self.adversary_layers = int(self.config['AdversaryLayers'])
		self.adversary_nodes = int(self.config['AdversaryNodes'])
		#Setup of the networks, loss and optimisation
		self.discriminator_optimizer = self.get_optimizer('Discriminator')
		self.discriminator_dropout = float(self.config['DiscriminatorDropout'])
		self.discriminator_inputdropout = float(self.config['DiscriminatorInputDropout'])
 		#self.discriminator_loss = binary_crossentropy
		self.adversary_optimizer = self.get_optimizer('Adversary')
		self.adversary_dropout = float(self.config['AdversaryDropout'])
		self.adversary_initializer = self.config['AdversaryInitializer']
 		#self.adversary_loss = binary_crossentropy
		self.combined_optimizer = self.get_optimizer('Combined')
		self.combined_epochs = int(self.config['CombinedEpochs'])

		self.validation_fraction = float(self.config['ValidationFraction'])

		self.batch_size = int(self.config['BatchSize'])
		self.verbosity = int(self.config['Verbosity'])

		#The following set of variables is used to evaluate the result
		#fpr = false positive rate, tpr = true positive rate
		self.tpr = 0.
		self.fpr = 0.
		self.threshold = 0.
		self.auc = 0.

		self.losses_test = {"L_f": [], "L_r": [], "L_f - L_r": []}
		self.losses_train = {"L_f": [], "L_r": [], "L_f - L_r": []}
		self.lambda_value = float(self.config['LambdaValue'])

		#
		self.use_last_layer = self.ini_to_bool(self.config['UseLastLayer'])
		self.train_adversarial = self.ini_to_bool(self.config['TrainAdversarial'])
		self.use_early_stopping = self.ini_to_bool(self.config['UseEarlyStopping'])
		temp = {True: 'Using early stopping', False: 'Not using early stopping'}
		logging.debug(temp[self.use_early_stopping])
		self.early_stopping_mode = self.config['EarlyStoppingMode']
		self.early_stopping_epochs = int(self.config['EarlyStoppingEpochs'])
		if not (self.early_stopping_epochs % 2) == 0:
			logging.error('Early stopping epochs must be even')
			sys.exit(1)
		self.use_early_stopping_discriminator = self.ini_to_bool(self.config['EarlyStoppingDiscriminator'])
		self.losses_es = []
		self.losses_esd = []

		tf.config.optimizer.set_jit(self.ini_to_bool(self.config['UseXLA']))

		# pylint: disable=unused-import
		if self.config['UseGuppy'] == '1':
			try:
				from guppy import hpy
			except:
				logging.warning('guppy not found')
		# pylint: enable=unused-import

		logging.debug('Network initialized')

	def load_config(self):
		"""
		Loads the initial config settings
		"""
		self.custom_config = ''
		self.config_path = "config_ANN.ini"
		self.config = cfg.ConfigParser(inline_comment_prefixes="#")
		self.config.read(self.config_path)
		self.config = self.config['General']

	def setup_logging(self):
		"""
		Sets up the logging system
		"""

		log_file_dir = self.config['LogFileDir']
		log_file_name = f'log_{self.time_start}.txt'
		fname = f'{log_file_dir}{log_file_name}'
		# make sure the file exists
		os.system(f'touch {fname}')

		debug_levels = {'DEBUG': logging.DEBUG,
						'INFO': logging.INFO,
						'WARNING': logging.WARNING,
						'ERROR': logging.ERROR}

		logging.basicConfig(filename=fname, filemode='a', level=debug_levels[self.config['DebugLevel']], format = '%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %H:%M:%S')

	def load_custom_config(self, args):
		"""
		Loads additional config as specified by the command line arguments.
		Arguments should be specified in the format Option=Value
		"""
		logging.debug('Loading custom config')
		logging.debug('Reading arguments:')
		for el in args:
			logging.debug(str(el))

		if len(args) > 1:
			for x in range(1,len(args)):
				try:
					temp = str(args[x]).split('=')
				except:
					logging.error('Additional config should be in the format Option=Value\n')
					sys.exit(1)
				if not temp[0] in self.config:
					logging.error(f'Key {temp[0]} could not be found in config\n')
					sys.exit()
				self.config[temp[0]] = temp[1]
				self.custom_config += '_' + temp[0] + '_' + temp[1]

			#if len(args) % 2 == 1:
			#	for x in range(1,len(args),2):
			#		if self.config.get(str(args[x]))==None:
			#			print('[ERROR] Unknown key ' + str(args[x]))
			#		self.config[str(args[x])] = str(args[x+1])
				logging.info(f'Custom config: {str(temp[0])}={str(temp[1])}')

	def ini_to_bool(self, option):
		'''
		turn option string into a boolean
		'''
		return False if option == '0' else True

	def get_optimizer(self, model):
		'''
		get the optimizer for a model
		'''
		optimizers = {
			'SGD': SGD(lr = float(self.config[f'{model}LearningRate']), momentum = float(self.config[f'{model}Momentum'])),
			'Adagrad': Adagrad(learning_rate = float(self.config[f'{model}LearningRate']), initial_accumulator_value = 0.1, epsilon = 1e-07),
			'Adam': Adam(learning_rate = float(self.config[f'{model}LearningRate']), beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07)
		}
		try:
			return optimizers[self.config[f'{model}Optimizer']]
		except:
			logging.error(f'{self.config[f"{model}Optimizer"]} is not a valid optimizer')
			sys.exit(1)

	def initialize_sample(self):
		"""
		Initializing the data and target samples
		The split function cuts into a training sample and a test sample
		Important note: Have to use the same random seed so that event and target stay in the same order as we shuffle
		"""
		#Signal and background are needed for the classification task, signal and systematic for the adversarial part
		#In this first step the events are retrieved from the tree, using the chosen set of variables

		logging.debug('Initizializing samples')

		#Extract the variables used from the trees into pandas dataframes
		self.events_signal = self.signal_tree.pandas.df(self.variables)
		self.events_background = self.background_tree.pandas.df(self.variables)
		self.events_systematic = self.signal_systematics_tree.pandas.df(self.variables)
		self.events_bkg_systematic = self.background_systematics_tree.pandas.df(self.variables) if self.has_background_systematic else None

		#Setting up the weights. The weights for each tree are stored in 'weight_nominal'
		self.weight_signal = self.signal_tree.pandas.df('weight_nominal')
		self.weight_background = self.background_tree.pandas.df('weight_nominal')
		self.weight_systematic = self.signal_systematics_tree.pandas.df('weight_nominal')
		self.weight_bkg_systematic = self.background_systematics_tree.pandas.df('weight_nominal') if self.has_background_systematic else None

		#Rehsaping the weights
		self.weight_signal = np.reshape(self.weight_signal, (len(self.events_signal), 1))
		self.weight_background = np.reshape(self.weight_background, (len(self.events_background), 1))
		self.weight_background_adversarial = self.weight_background #maybe scale this in the future?
		self.weight_systematic = np.reshape(self.weight_systematic, (len(self.events_systematic), 1))
		self.weight_bkg_systematic = np.reshape(self.weight_bkg_systematic, (len(self.events_bkg_systematic), 1)) if self.has_background_systematic else None

		#Normalisation to the eventcount can be used instead of weights, especially if using data
		#self.norm_signal = np.reshape([1./float(len(self.events_signal)) for x in range(len(self.events_signal))], (len(self.events_signal), 1))
		#self.norm_background = np.reshape([1./float(len(self.events_background)) for x in range(len(self.events_background))], (len(self.events_background), 1))

		#Calculating the weight ratio to scale the signal weight up. This tries to take the high amount of background into account
		if self.has_background_systematic:
			self.weight_ratio = ( self.weight_signal.sum() + self.weight_systematic.sum() )/ (self.weight_background.sum() + self.weight_bkg_systematic.sum())
		else:
			self.weight_ratio = (self.weight_signal.sum() + self.weight_systematic.sum()) / self.weight_background.sum()
		self.weight_signal = self.weight_signal / self.weight_ratio
		self.weight_systematic = self.weight_systematic / self.weight_ratio

		#Setting up the targets
		#target combined is used to make sure the systematics are seen as signal for the first net in the combined training
		#pylint: disable=pointless-string-statement
		'''
			D	A
		Wt	1	1
		tt	0	1
		Wts	1	0
		tts 0	0
		'''
		#pylint: enable=pointless-string-statement
		self.target_signal = np.reshape([1 for x in range(len(self.events_signal))], (len(self.events_signal), 1))
		self.target_background = np.reshape([0 for x in range(len(self.events_background))], (len(self.events_background), 1))
		self.target_systematic = np.reshape([1 for x in range( len( self.events_systematic))], (len(self.events_systematic), 1))
		self.target_bkg_systematic = np.reshape([0 for x in range(len(self.events_bkg_systematic))], (len(self.events_bkg_systematic), 1)) if self.has_background_systematic else None
		self.target_systematic_adversarial = np.reshape([0 for x in range( len( self.events_systematic))], (len(self.events_systematic), 1))
		if self.has_background_systematic:
			self.target_background_adversarial = np.reshape([1 for x in range(len(self.events_background))], (len(self.events_background),1))
		else:
			self.target_background_adversarial = np.reshape( np.random.randint(2, size =len( self.events_background)), (len(self.events_background), 1))
		self.target_bkg_systematic_adversarial = np.reshape([0 for x in range(len(self.events_bkg_systematic))], (len(self.events_bkg_systematic), 1)) if self.has_background_systematic else None

		#The samples and corresponding targets are now split into a sample for training and a sample for testing. Keep in mind that the same random seed should be used for both splits
		sample_split_list = {
			True: (self.events_signal, self.events_background, self.events_systematic, self.events_bkg_systematic),
			False: (self.events_signal, self.events_background, self.events_systematic)
		}
		target_split_list = {
			True: (self.target_signal, self.target_background, self.target_systematic, self.target_bkg_systematic),
			False: (self.target_signal, self.target_background, self.target_systematic)
		}
		target_adversarial_split_list = {
			True: (self.target_signal, self.target_background_adversarial, self.target_systematic_adversarial, self.target_bkg_systematic),
			False: (self.target_signal, self.target_background_adversarial, self.target_systematic_adversarial)
		}
		self.sample_training, self.sample_validation = train_test_split(np.concatenate(sample_split_list[self.has_background_systematic]), test_size = self.validation_fraction, random_state = self.seed)
		self.target_training, self.target_validation = train_test_split(np.concatenate(target_split_list[self.has_background_systematic]), test_size = self.validation_fraction, random_state = self.seed)
		self.target_adversarial, self.target_adversarial_validation = train_test_split(np.concatenate(target_adversarial_split_list[self.has_background_systematic]), test_size = self.validation_fraction, random_state = self.seed)

		#Splitting the weights
		weight_split_list = {
			True: (self.weight_signal, self.weight_systematic, self.weight_background, self.weight_bkg_systematic),
			False: (self.weight_signal, self.weight_systematic, self.weight_background)
		}
		weight_adversarial_split_list = {
			True: (self.weight_signal, self.weight_systematic, self.weight_background_adversarial, self.weight_bkg_systematic),
			False: (self.weight_signal, self.weight_systematic, self.weight_background_adversarial)
		}
		self.weight_training, self.weight_validation = train_test_split(np.concatenate(weight_split_list[self.has_background_systematic]), test_size = self.validation_fraction, random_state = self.seed)
		self.weight_adversarial, self.weight_adversarial_validation = train_test_split(np.concatenate(weight_adversarial_split_list[self.has_background_systematic]), test_size = self.validation_fraction, random_state = self.seed)
		#self.norm_training, self.norm_validation = train_test_split(np.concatenate((self.norm_signal, self.norm_background)), test_size = self.validation_fraction, random_state = self.seed)

		#Setting up a scaler
		#A scaler makes sure that all variables are normalised to 1 and have the same order of magnitude for that reason
		scaler = StandardScaler()
		self.sample_training = scaler.fit_transform(self.sample_training)
		self.sample_validation = scaler.fit_transform(self.sample_validation)

		logging.debug('Samples initialized')

	def build_discriminator(self):
		'''---------------------------------------------------------------------------------------------------------------------------------------
		Here the discriminator is built
		It has an input layer fit to the shape of the variables
		A loop creates the desired amount of deep layers
		It ends in a single sigmoid layer
		Additionally the last layer is saved to be an optional input to the adversary
		'''

		#The discriminator aims to separate signal and background
		#There is an input layer after which the desired amount of hidden layers is added in a loop
		#In the loop normalisation and dropout are added too

		self.network_input = Input( shape = (self.input_dimension) )
		self.layer_discriminator = Dense( self.discriminator_nodes, activation = "elu")(self.network_input)
		self.layer_discriminator = BatchNormalization()(self.layer_discriminator)
		#(experimental) Idea: High dropout in the first layer effectively regularizes variables. Untested.
		self.layer_discriminator = Dropout(self.discriminator_inputdropout)(self.layer_discriminator)
		for _ in range(self.discriminator_layers -1):
			# add hidden layers in a loop with batch normalization and dropout for each
			self.layer_discriminator = Dense(self.discriminator_nodes, activation = "elu")(self.layer_discriminator)
			self.layer_discriminator = BatchNormalization()(self.layer_discriminator)
			self.layer_discriminator = Dropout(self.discriminator_dropout)(self.layer_discriminator)
		self.layer_discriminator = Dense( 1, activation = "sigmoid")(self.layer_discriminator)

		self.model_discriminator = Model(inputs = [self.network_input], outputs = [self.layer_discriminator])
		self.model_discriminator.compile(loss = "binary_crossentropy", weighted_metrics = [metrics.binary_accuracy], optimizer = self.discriminator_optimizer)
		self.model_discriminator.summary()

		self.adversary_input_model = Model(inputs = [self.network_input], outputs = [self.model_discriminator.get_layer(f'dense_{self.discriminator_layers-1}').output])
		#self.adversary_input_model.compile(loss = 'b')
		#self.adversary_input_model.summary()

	def build_adversary(self):
		'''
		Here the adversary is built
		It uses the discriminator output as inputobject has no attribute 'append'

		Optionally the last layer can be used additionally
		In a loop the deep layers are created
		It ends in a single sigmoid layer
		'''
		logging.debug('Building adversary')
		#This is where the adversary is initialized
		#It is just another classifier

		self.adversary_input = Input( shape = (self.input_dimension) )
		# decide whether to use last or second to last layer
		#Idea: If the second to last layer is used as an input to the adversary, that might make it better by giving it more info, instead of just the single node output of the discriminator
		if self.use_last_layer:
			self.layer_adversary = self.model_discriminator(self.network_input)
		else:
			self.layer_adversary = self.adversary_input_model(self.network_input)
		self.layer_adversary = Dense( self.adversary_nodes, activation = 'elu', kernel_initializer = self.adversary_initializer)(self.layer_adversary)
		self.layer_adversary = BatchNormalization()(self.layer_adversary)
		self.layer_adversary = Dropout(self.adversary_dropout)(self.layer_adversary)
		#add hidden layers in a loop with batch normalization and dropout
		for _ in range(self.adversary_layers - 1):
			self.layer_adversary = Dense(self.adversary_nodes, activation = "elu", kernel_initializer = self.adversary_initializer)(self.layer_adversary)
			self.layer_adversary = BatchNormalization()(self.layer_adversary)
			self.layer_adversary = Dropout(self.adversary_dropout)(self.layer_adversary)
		self.layer_adversary = Dense( 1, activation = "sigmoid")(self.layer_adversary)

		self.model_adversary = Model(inputs = [self.network_input], outputs = [self.layer_adversary])
		self.model_adversary.compile(loss = "binary_crossentropy", optimizer = self.adversary_optimizer)
		self.model_adversary.summary()

	def build_combined_training(self):
		'''The discriminator and adversary are added up to a single model running on a combined loss function'''

		logging.debug('Building combined training')
		def make_losses_adversary():
			'''This function creates the loss function used by the adversary'''
			def losses_adversary(y_true, y_pred):
				return self.lambda_value * binary_crossentropy(y_true, y_pred)
			return losses_adversary

		self.model_combined = Model(inputs = self.adversary_input, outputs = [self.model_discriminator(self.adversary_input), self.model_adversary(self.adversary_input)])
		#Compiling a model with multiple loss functions lets Keras use the sum by default
		self.model_combined.compile(loss = ['binary_crossentropy', make_losses_adversary()], optimizer = self.combined_optimizer, metrics = ['accuracy'])

	def run_adversarial_training(self):
		'''
		This function runs the actual adversarial training by alternating between training the discriminator and adversary networks
		'''
		logging.debug('Running adversarial training')

		def make_trainable(network, flag):
			#helper function
			network.trainable = flag
			for l in network.layers:
				l.trainable = flag
			network.compile()

		#run this however many times needed, every iterations is one epoch for each network
		for iteration in range(self.training_iterations):

			logging.info(f'Iteration {iteration+1} of {self.training_iterations}')

			#Only save losses every 5 iterations
			#if iteration%5 == 0 or iteration == (self.training_iterations-1):

			if self.train_adversarial or iteration == 0:
				make_trainable(self.model_discriminator, True)
				make_trainable(self.model_adversary, False)

			logging.debug('Training combined model')
			self.model_history = self.model_combined.fit(self.sample_training, [self.target_training, self.target_adversarial], validation_data = (self.sample_validation, [self.target_validation, self.target_adversarial_validation], [self.weight_validation.ravel(), self.weight_adversarial_validation.ravel()]), epochs = self.combined_epochs, batch_size = self.batch_size, sample_weight = [self.weight_training.ravel(),self.weight_adversarial.ravel()], verbose = self.verbosity)
			self.model_history_array.append(self.model_history.history)

			if self.train_adversarial:
				make_trainable(self.model_discriminator, False)
				make_trainable(self.model_adversary, True)

				logging.debug('Training adversary')
				self.adversary_history = self.model_adversary.fit(self.sample_training, self.target_adversarial, epochs=1, batch_size = self.batch_size, sample_weight = self.weight_training.ravel(), verbose = self.verbosity)
				self.adversary_history_array.append(self.adversary_history.history)

			# Check for early stopping if it is used
			if self.use_early_stopping:
				if self.early_stopping() or self.discriminator_es():
					break

			# Memory debugging
			# pylint: disable=undefined-variable
			if self.config['UseGuppy'] == '1':
				h = hpy()
				hp = h.heap()
				logging.debug(hp)
			# pylint: enable=undefined-variable
			#if self.config['UseTop'] == '1':
		#		#logging.debug(subprocess.call('top -n1', shell=True))
		#		with open(f"/cephfs/user/s6niboei/BAF_top_{self.time_start}",'a') as f:
		#			subprocess.call(['top','-n','1','b'], shell=True, stdout=f)

			#self.save_losses(iteration, self.model_combined)

	def pretrain_adversary(self):
		'''
		Currently unused
		'''

		self.model_adversary.fit(self.sample_training, self.target_adversarial.ravel(), epochs = self.adversary_epochs, batch_size = int(self.config['BatchSize']), sample_weight = self.weight_adversarial.ravel())

	def pretrain_discriminator(self):
		'''
		This function pretrains the discriminator network, this improves the adversarial training speed greatly
		'''

		logging.debug(f'Pretraining discriminator with {str(self.discriminator_epochs)} epochs.')

		#print(self.target_training[12:500])
		#print(self.target_training[-1:-100])

		self.model_discriminator.summary()

		self.discriminator_history = self.model_discriminator.fit(self.sample_training, self.target_training.ravel(), epochs=self.discriminator_epochs, batch_size = int(self.config['BatchSize']), sample_weight = self.weight_training.ravel(), validation_data = (self.sample_validation, self.target_validation, self.weight_validation.ravel()), verbose = self.verbosity)
		self.discriminator_history_array.append(self.discriminator_history)
		print(self.discriminator_history.history.keys())

		#for training_iteration in range(self.training_iterations):
		#    discriminator_history = self.model_discriminator.fit(self.sample_training, self.target_training, epochs=self.discriminator_epochs, validation_data = (self.sample_validation, self.target_validation))
		#    adversary_history = self.model_combined.fit(self.adversarial_training, [self.combined_target, self.adversarial_target], epochs=self.adversary_epochs)

	def early_stopping(self):
		'''
		Stops the training early if the loss stops improving
		'''
		self.losses_es.append(self.model_history.history['val_loss'][0])
		logging.debug('Checking losses for early stopping')
		logging.debug(self.losses_es)
		los_len = len(self.losses_es)
		#collect a list with 10 last losses
		if los_len<=self.early_stopping_epochs:
			return False
		elif los_len==(self.early_stopping_epochs+1):
			self.losses_es.pop(0)
			first = self.losses_es[0]
			last = self.losses_es[self.early_stopping_epochs-1]
			avg = sum(self.losses_es)/self.early_stopping_epochs
			slice_ = int(self.early_stopping_epochs / 2)
			first5 = sum(self.losses_es[:slice_])
			last5 = sum(self.losses_es[-slice_:])
			if self.early_stopping_mode == 'increase':
				if last > avg and first < avg and first5 < last5:
					#stop the training if the loss is going up
					logging.info('Loss is increasing, stopping training')
					return True
				logging.debug('Loss decreasing, keep training')
				return False
			elif self.early_stopping_mode == '0.01':
				if (first5 / last5) < 1.0001:
					logging.info('Loss is barely decreasing, stopping training')
					return True
				logging.debug('Loss is decreasing, keep training')
				return False
			else:
				logging.error('Invalid early stopping mode')
				return True
		else:
			logging.error('Too many items in list in early_stopping')

	def discriminator_es(self):
		'''
		Stops the training if the discriminator stops improving
		'''
		self.losses_esd.append(self.model_history.history['val_model_loss'][0])
		logging.debug('Checking losses for early stopping')
		logging.debug(self.losses_esd)
		los_len = len(self.losses_esd)
		#collect a list with 10 last losses
		if los_len<=self.early_stopping_epochs:
			return False
		elif los_len==(self.early_stopping_epochs+1):
			self.losses_esd.pop(0)
			first = self.losses_esd[0]
			last = self.losses_esd[self.early_stopping_epochs-1]
			avg = sum(self.losses_esd)/self.early_stopping_epochs
			slice_ = int(self.early_stopping_epochs / 2)
			first5 = sum(self.losses_esd[:slice_])
			last5 = sum(self.losses_esd[-slice_:])
			if self.early_stopping_mode == 'increase':
				if last > avg and first < avg and first5 < last5:
					#stop the training if the loss is going up
					logging.info('Loss is increasing, stopping training')
					return True
				logging.debug('Loss decreasing, keep training')
				return False
			elif self.early_stopping_mode == '0.01':
				if (first5 / last5) < 1.0001:
					logging.info('Loss is barely decreasing, stopping training')
					return True
				logging.debug('Loss is decreasing, keep training')
				return False
			else:
				logging.error('Invalid early stopping mode')
				return True
		else:
			logging.error('Too many items in list in early_stopping')

	def predict_model(self):
		'''
		Runs all the necessary prediction on the trained model for evaluating the performance and later use
		'''

		logging.debug('Predicting model')

		self.model_prediction = self.model_discriminator.predict(self.sample_validation, batch_size = self.batch_size).ravel()
		self.fpr, self.tpr, self.threshold = roc_curve(self.target_validation, self.model_prediction)
		self.auc = auc(self.fpr, self.tpr)

		self.adversary_prediction = self.model_adversary.predict(self.sample_validation, batch_size = self.batch_size).ravel()
		self.adversary_fpr, self.adversary_tpr, self.adversary_threshold = roc_curve(self.target_adversarial_validation, self.adversary_prediction)
		self.adversary_auc = auc(self.adversary_fpr, self.adversary_tpr)

		print('Discriminator AUC', self.auc)
		print('Adversary AUC', self.adversary_auc)

	#----------------------------------------------------------------------------------------------Plot structure------------------------------------------------------------------------------------------------------------
	#pylint: disable=unused-argument
	def plot_losses(self, i, l_test, l_train):
		'''
		Old way of plotting the losses of the network, using a separate network.evaluate call
		'''
		#pylint: enable=unused-argument
		print('Printing losses')

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
		plt.plot(range(len(values_test)), values_test, label=str(self.lambda_value)+r"$ \cdot Loss_{net2}^{test}$", color="green", linestyle='dashed')
		plt.plot(range(len(values_train)), values_train, label=str(self.lambda_value)+r"$ \cdot Loss_{net2}^{train}$", color="green")
		plt.legend(loc="upper right", prop={'size' : 7})
		plt.legend(frameon=False)
		plt.ylabel('Loss')
		plt.grid()
		ax2.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

		ax3 = plt.subplot(313, sharex=ax1)
		values_test = np.array(l_test["L_f - L_r"])
		values_train = np.array(l_train["L_f - L_r"])
		plt.plot(range(len(values_test)), values_test, label=r"$Loss_{net1}^{test} - $"+str(float(self.lambda_value))+r"$ \cdot Loss_{net2}^{test}$", color="red", linestyle='dashed')
		plt.plot(range(len(values_train)), values_train, label=r"$Loss_{net1}^{train} - $"+str(float(self.lambda_value))+r"$ \cdot Loss_{net2}^{train}$", color="red")
		plt.legend(loc="upper right", prop={'size' : 7})
		plt.grid()
		ax3.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
		plt.xlabel('Epoch')

		plt.legend(frameon=False)
		plt.gcf().savefig(output_path + 'losses_' + datetime.now().strftime("%H_%M_%S") + '.png')
		plt.gcf().clear()

		#losses_test = {"L_f": [], "L_r": [], "L_f - L_r": []}
		#losses_train = {"L_f": [], "L_r": [], "L_f - L_r": []}

	#This function is inactive for now
	def save_losses(self, i, network):
		'''
		Old function to get the losses of the network from a separate evaluate call instead of during training.
		This gets rid of the train/test offset caused by dropout but makes the network run significantly slower
		'''
		#l_test = network.evaluate(self.sample_training, [self.target_training, self.target_adversarial], sample_weight = [self.weight_training.ravel(),self.weight_adversarial.ravel()], batch_size = int(self.config['BatchSize']))
		#l_train = network.evaluate(self.sample_validation, [self.target_validation, self.target_adversarial_validation], sample_weight = [self.weight_validation.ravel(), self.weight_adversarial_validation.ravel()], batch_size = int(self.config['BatchSize']))
		l_test = network.evaluate(self.sample_training, [self.target_training, self.target_adversarial], batch_size = int(self.config['BatchSize']))
		l_train = network.evaluate(self.sample_validation, [self.target_validation, self.target_adversarial_validation], batch_size = int(self.config['BatchSize']))
		self.losses_test["L_f"].append(l_test[1])
		self.losses_test["L_r"].append(-l_test[2])
		self.losses_test["L_f - L_r"].append(l_test[0])
		self.losses_train["L_f"].append(l_train[1])
		self.losses_train["L_r"].append(-l_train[2])
		self.losses_train["L_f - L_r"].append(l_train[0])
		print('LTEST')
		print(l_test)

		#pickle.dump(lossestest,open('lossestest.jar','wb'))
		#pickle.dump(lossestrain,open('lossestrain.jar','wb'))

		if i == (self.training_iterations-1):
			self.plot_losses(i, self.losses_test, self.losses_train)

	def plot_roc(self):
		'''
		Plots the ROC curve
		'''
		plt.title('Receiver Operating Characteristic')
		plt.plot(self.fpr, self.tpr, 'g--', label='$AUC_{train}$ = %0.2f'% self.auc)
		plt.legend(loc='lower right')
		plt.plot([0,1],[0,1],'r--')
		plt.xlim([-0.,1.])
		plt.ylim([-0.,1.])
		plt.ylabel('True Positive Rate', fontsize='large')
		plt.xlabel('False Positive Rate', fontsize='large')
		plt.legend(frameon=False)
		#plt.show()
		plt.gcf().savefig(output_path + 'roc.png')
		#plt.gcf().savefig(output_path + 'simple_ROC_' + file_extension + '.eps')
		plt.gcf().clear()

	def plot_separation(self):
		'''
		Plots the separation between signal and background events
		'''
		self.signal_histo = []
		self.background_histo = []
		for i in range(len(self.sample_validation)):
			if self.target_validation[i] == 1:
				self.signal_histo.append(self.model_prediction[i])
			if self.target_validation[i] == 0:
				self.background_histo.append(self.model_prediction[i])

		plt.hist(self.signal_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step", density = True, color=color_tW, label = "Signal")
		plt.hist(self.background_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step", density = True, color=color_tt, label = "Background")
		#plt.hist(self.model_prediction[self.target_training.tolist() == 0], range=[0., 1.], linewidth = 2, bins=30, histtype="step", normed=1, color=color_tt)
		#plt.hist(predicttest__ANN[test_target == 1],   range=[xlo, xhi], linewidth = 2, bins=bins, histtype="step", normed=1, color=color_tW2, label='$Sig_{test}$', linestyle='dashed')
	  	#plt.hist(predicttest__ANN[test_target == 0],   range=[xlo, xhi], linewidth = 2, bins=bins, histtype="step", normed=1, color=color_tt2, label='$Bkg_{test}$', linestyle='dashed')
		#plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
		plt.legend()
		plt.xlabel('Network response', horizontalalignment='left', fontsize='large')
		plt.ylabel('Event fraction', fontsize='large')
		plt.legend(frameon=False)
		#plt.title('Normalised')
		#plt.gcf().savefig(output_path + 'ANN_NN_' + file_extension + '.png')
		plt.gcf().savefig(output_path + 'separation_discriminator.png')
		#plt.show()
		plt.gcf().clear()

	def plot_separation_adversary(self):
		'''
		Plot separation of the adversary network
		'''
		#pylint: disable=unused-variable
		plt.title('Adversary Response')
		axis1 = plt.subplot(211)
		self.nominal_histo = []
		self.systematic_histo = []
		self.nominal_adversarial_histo = []
		self.systematic_adversarial_histo = []
		for i in range(len(self.sample_validation)):
			if self.target_adversarial_validation[i] == 1 and self.target_validation[i] == 1:
				self.nominal_histo.append(self.model_prediction[i])
			if self.target_adversarial_validation[i] == 0 and self.target_validation[i] == 1:
				self.systematic_histo.append(self.model_prediction[i])

		ns1, bins1, patches1 = plt.hist(self.nominal_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step", density = True, color=color_tW, label = "Nominal")
		ns2, bins2, patches2 = plt.hist(self.systematic_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step", density = True, color=color_sys, label = "Systematics")
		#plt.hist(self.model_prediction[self.target_training.tolist() == 0], range=[0., 1.], linewidth = 2, bins=30, histtype="step", normed=1, color=color_tt)
		#plt.hist(predicttest__ANN[test_target == 1],   range=[xlo, xhi], linewidth = 2, bins=bins, histtype="step", normed=1, color=color_tW2, label='$Sig_{test}$', linestyle='dashed')
		#plt.hist(predicttest__ANN[test_target == 0],   range=[xlo, xhi], linewidth = 2, bins=bins, histtype="step", normed=1, color=color_tt2, label='$Bkg_{test}$', linestyle='dashed')
		#plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
		plt.legend()
		plt.ylabel('Event fraction', fontsize='large')
		plt.legend(frameon=False)


		ratioArray = []
		for iterator, _ in enumerate(ns1):
			if ns1[iterator] > 0:
				ratioArray.append(ns2[iterator]/ns1[iterator])
			else:
				ratioArray.append(1.)

		axis2 = plt.subplot(212, sharex = axis1)
		#axis2.set_ylim([0., 2.])
		plt.plot(bins1[:-1], ratioArray, color = "blue", drawstyle = 'steps-mid')
		#plt.plot(bins1[:-1], ratioArray, color = "blue", marker = "_", linestyle = 'None', markersize = 12)
		plt.hlines(1, xmin = -0.0, xmax = 1.0)
		plt.xlabel('Network response', horizontalalignment='left', fontsize='large')
		plt.ylabel('Event ratio')
		#plt.title('Normalised')
		#plt.gcf().savefig(output_path + 'ANN_NN_' + file_extension + '.png')
		plt.gcf().savefig(output_path + 'separation_adversary.png')
		#plt.show()
		plt.gcf().clear()
		#pylint: enable=unused-variable

	def plot_all(self):
		'''
		Plot the network response
		'''
		#pylint: disable=unused-variable
		plt.title('Adversary Response')
		axis1 = plt.subplot(211)
		self.nominal_histo = []
		self.systematic_histo = []
		self.nominal_adversarial_histo = []
		self.systematic_adversarial_histo = []
		for i, _ in enumerate(self.sample_validation):
			if self.target_adversarial_validation[i] == 1 and self.target_validation[i] == 1:
				self.nominal_histo.append(self.model_prediction[i])
			if self.target_adversarial_validation[i] == 0 and self.target_validation[i] == 1:
				self.systematic_histo.append(self.model_prediction[i])

		ns1, bins1, patches1 = plt.hist(self.nominal_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step", density = True, color=color_tW, label = "Nominal")
		ns2, bins2, patches2 = plt.hist(self.systematic_histo, range=[0., 1.], linewidth = 2, bins=30, histtype="step", density = True, color=color_sys, label = "Systematics")
		#pylint: enable=unused-variable

	def plot_accuracy(self):
		'''
		Plot the accuracy of the model over time
		'''
		plt.plot(self.model_history.history['binary_accuracy'])
		plt.plot(self.model_history.history['val_binary_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		#plt.show()
		plt.gcf().savefig(output_path + 'acc.png')
		plt.gcf().clear()

	def save_tar(self):
		'''
		Saves some important bits, including predictions and stuff for plotting and tars it for easy transport
		'''

		outfileName = 'ANN_out' + self.custom_config + '_' + self.time_start
		logging.debug(f'Saving tar file: {outfileName}.tar')

		#pickle.dump(self.model_history.history, open('history.pickle', 'wb'))
		pickle.dump(self.sample_validation, open('sample_validation.pickle','wb'))
		pickle.dump(self.target_validation, open('target_validation.pickle','wb'))
		pickle.dump(self.model_prediction, open('model_prediction.pickle','wb'))
		pickle.dump(self.adversary_prediction, open('adversary_prediction.pickle','wb'))
		pickle.dump(self.target_adversarial_validation, open('target_adversarial_validation.pickle','wb'))
		pickle.dump(self.losses_test,open('losses_test.pickle','wb'))
		pickle.dump(self.losses_train,open('losses_train.pickle','wb'))
		pickle.dump(self.model_history_array, open('model_history_array.pickle','wb'))
		pickle.dump(self.adversary_history_array, open('adversary_history_array.pickle','wb'))
		pickle.dump(self.discriminator_history.history, open('discriminator_history.pickle','wb'))

		#writes the options used in this net into a separate text file
		with open('config.txt','w') as f:
			for key in self.config:
				f.write(key + ': ' + self.config[key] + '\n')

		with open('time.txt','w') as f:
			f.write(f'{time.time() - self.time1} seconds')

		import tarfile
		import glob
		tar = tarfile.open('out/' + outfileName + '.tar','w:gz')
		#tar = tarfile.open('/cephfs/user/s6niboei/ANN_out_' + datetime.now().strftime("%H_%M_%S") + '.tar', 'w:gz')
		for f in glob.glob('*.pickle'):
			tar.add(f)
		for f in glob.glob('*.txt'):
			tar.add(f)

		tar.close()

		logging.debug('Tar file written')

	def save_as_root(self):
		'''
		Saves input data as well as model prediction to a single root file for further use
		'''

		outfileName = 'ANN_out' + self.custom_config + '_' + self.time_start
		logging.debug(f'Saving root file: {outfileName}.root')

		# split up the single prediction list
		signal_histo = []
		background_histo = []
		signal_sys_histo = []
		background_sys_histo = []
		for i in range(len(self.sample_validation)):
			if self.target_validation[i] == 1 and self.target_adversarial_validation[i] == 1:
				signal_histo.append(self.model_prediction[i])
			if self.target_validation[i] == 1 and self.target_adversarial_validation[i] == 0:
				signal_sys_histo.append(self.model_prediction[i])
			if self.target_validation[i] == 0 and self.target_adversarial_validation[i] == 1:
				background_histo.append(self.model_prediction[i])
			if self.target_validation[i] == 0 and self.target_adversarial_validation[i] == 0:
				background_sys_histo.append(self.model_prediction[i])


		list_samples = [self.signal_tree, self.signal_systematics_tree, self.background_tree, self.background_systematics_tree]
		list_names = [self.config['SignalSample'],self.config['SignalSystematicsSample'],self.config['BackgroundSample'],self.config['BackgroundSystematicsSample']]
		list_pred = [signal_histo, signal_sys_histo, background_histo, background_sys_histo]

		# read the input data and dump it back into a new root file together with the prediction, until ur.update is implemented
		try:
			with ur.recreate(outfileName + '.root') as f:
				logging.debug('Building root file')

				for i, _ in enumerate(list_samples):
					vartype_dict = {}
					var_dict = {}
					# read all the variables
					vars_ = [var.decode('utf-8') for var in list_samples[i].iterkeys()]
					# for each variable, read the data and its type into dicts, add the NN prediction, then dump it into a new tree
					for var in vars_:
						sample = list_samples[i].pandas.df(var)
						sample_type = np.array(sample).dtype
						if sample_type in ['float32','int32']:
							vartype_dict[var] = sample_type
							var_dict[var] = sample
					vartype_dict['NN_pred'] = 'float32'
					var_dict['NN_pred'] = [float(j) for j in list_pred[i]]

					f[list_names[i]] = ur.newtree({var:val for var,val in vartype_dict.items()})
					for var,val in var_dict.items():
						logging.debug('Filling data')
						f[list_names[i]][var].newbasket(val)
		except:
			logging.error(f'Failed to create root file: {outfileName}')

		os.system(f'mv {outfileName}.root {self.root_file_dir}')
		logging.debug('Root file saved')

	def print_config(self):
		'''
		print the entire config into the log file
		'''
		logging.info('Config used:')
		for key in self.config:
			logging.info(key + ': ' + self.config[key])
