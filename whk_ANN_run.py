import whk_ANN_defs

first_training = whk_ANN_defs.ANN_environment()
first_training.initialize_sample()
first_training.build_discriminator()
first_training.build_adversary()
first_training.build_combined_training()
first_training.run_adversarial_training()
first_training.predict_model()
first_training.plot_roc()
first_training.plot_separation()
first_training.plot_separation_adversary()
first_training.plot_losses()