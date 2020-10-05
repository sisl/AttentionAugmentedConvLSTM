import os

# Where model weights and config will be saved if you run kitti_train.py
# If you directly download the trained weights, change to appropriate path.

WEIGHTS_DIR = '../../Data/PredNet/model_data/'

if not os.path.exists(WEIGHTS_DIR):
	os.makedirs(WEIGHTS_DIR)

# Where results (prediction plots and evaluation file) will be saved.
RESULTS_SAVE_DIR = '../../Data/PredNet/kitti_results/'

if not os.path.exists(RESULTS_SAVE_DIR):
	os.makedirs(RESULTS_SAVE_DIR)
