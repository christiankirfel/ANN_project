'''
run script for the ANN network
don't actually use this to run, use runANN.sh instead for additional setup
'''

#To time the code
import time

#timeStart = time.time()

import sys
import whk_ANN_defs

#In the following options and variables are read in
#This is done to keep the most important features clearly represented

#with open('/cephfs/user/s6chkirf/whk_ANN_variables.txt','r') as varfile:
#with open('whk_ANN_variables.txt','r') as varfile:
#	variableList = varfile.read().splitlines() 

#print(variableList)

#args = sys.argv
#first_training = ANN_environment(variables = variableList)

first_training = whk_ANN_defs.ANN_environment(sys.argv)
first_training.initialize_sample()
first_training.build_discriminator()
first_training.build_adversary()
first_training.build_combined_training()
first_training.pretrain_discriminator()
#first_training.predict_model()

#time1 = time.time()
first_training.run_adversarial_training()
#time1 = time.time() - time1
#with open("/cephfs/user/s6niboei/batchsize_CPU_TF2XLA_" + str(first_training.batch_size) + ".log",'a') as f:
#    print(str(first_training.batch_size), "%.3f" % (time1), file=f)
#print('Time for this batch size')
#print(str(first_training.batch_size, "%.3f" % (time1)))
first_training.predict_model()
first_training.types()
first_training.save_as_root()
#first_training.fill_jar()
#first_training.plot_roc()
#first_training.plot_separation()
#first_training.plot_separation_adversary()
#first_trainings.plot_separation_adversary()
#first_training.plot_losses()

#timeTotal = time.ti