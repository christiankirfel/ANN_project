'''
run script for the ANN network
don't actually use this to run, use runANN.sh instead for additional setup
'''

#To time the code
import time

#timeStart = time.time()

import configparser as cfg
import sys
import ANN_defs

config = cfg.ConfigParser(inline_comment_prefixes="#")
config.read('config_ANN.ini')
config = config['General']
#if config['DebugLevel']=='DEBUG':
#	logging.basicConfig(filename='/cephfs/user/s6niboei/BAFDEBUG.log', filemode='a', level=logging.DEBUG)
#else:
#	logging.basicConfig(level=logging.WARNING)

#logging.debug('run_ANN.py Passing on arguments')
#for el in sys.argv:
#    logging.debug(str(el))

time1 = time.time()
first_training = ANN_defs.ANN_environment(sys.argv)
first_training.initialize_sample()
first_training.build_discriminator()
first_training.build_adversary()
first_training.build_combined_training()
first_training.pretrain_discriminator()
#first_training.predict_model()

first_training.run_adversarial_training()
#with open("/cephfs/user/s6niboei/TF2.2_noXLA.log",'a') as f:
#    print("100 iterations for TF2.2/noXLA took %.3f" % (time1), file=f)
#sys.exit(0)
#print('Time for this batch size')
#print(str(first_training.batch_size, "%.3f" % (time1)))
first_training.predict_model()
time1 = time.time() - time1
print(f'TIME TAKEN: {time1}')
first_training.save_tar()
if config['SaveRootFile'] == '1':
	first_training.save_as_root()

#timeTotal = time.ti
