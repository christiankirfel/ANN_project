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



time1 = time.time()

#start the training
print("HI")
first_training = ANN_defs.ANN_environment(sys.argv)
print("WHATS")
first_training.initialize_sample()
print("UP")
first_training.build_discriminator()
print("HOW")
first_training.build_adversary()
print("ARE")
first_training.build_combined_training()
print("YOU")
first_training.pretrain_discriminator()
print("DOING")
first_training.pretrain_adversary()
#first_training.predict_model()
#time_start = time.time()
first_training.run_adversarial_training()
#time_end = time.time()
#time_taken = time_end - time_start
#with open(f"/cephfs/user/s6niboei/benchmarks/{first_training.get_benchmark_filename()}",'a') as f:
#    print(f"{config['TrainingIterations']}" + " iterations took %.3f" % (time_taken), file=f)
#sys.exit(0)
#print('Time for this batch size')
#print(str(first_training.batch_size, "%.3f" % (time1)))
first_training.predict_model()
first_training.plot_results()
time1 = time.time() - time1
print(f'TIME TAKEN: {time1}')
first_training.save_tar()
first_training.save_as_root()

#timeTotal = time.ti
