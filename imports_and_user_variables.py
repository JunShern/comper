from __future__ import print_function
import os, shutil
import random
import sys
import numpy as np
import pypianoroll
from matplotlib import pyplot as plt
import cPickle as pickle
import IPython
import h5py

from keras.models import load_model
from sklearn.neighbors import NearestNeighbors
import sklearn.externals

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, History
from keras.models import load_model

from scipy.stats import norm
from keras.layers import Lambda, Flatten, Reshape, Dropout
from keras import backend as K
from keras import metrics
from keras import losses
from keras import optimizers

sys.path.append('./snippets')
import pianoroll_utils
import custom_loss

if (len(sys.argv) < 2):
    sys.exit("Imports failed: Missing argument, please specify number of pianorolls to use in dataset.")

# User paths
NUM_FILES = int(sys.argv[1])
LPD5_DIR = "./lpd_5_cleansed"
DATA_DIR = "./pianorolls_{}".format(NUM_FILES)
# Large dataset
UNITS_FILE = './pickle_jar/units_{}_songs_clipped96.h5'.format(NUM_FILES)
SEQ_FILE = './pickle_jar/seq_{}_songs_clipped96.h5'.format(NUM_FILES)
MONO_SEQ_FILE = './pickle_jar/mono_seq_{}_songs_clipped96.h5'.format(NUM_FILES)
# Smaller dataset
PICKLE_FILE = './pickle_jar/units_{}_songs_clipped96.pkl'.format(NUM_FILES)
NORM_PICKLE_FILE = './pickle_jar/norm_units_{}_songs_clipped96.pkl'.format(NUM_FILES)
SEQ_PICKLE_FILE = './pickle_jar/seq_units_{}_songs_clipped96.pkl'.format(NUM_FILES)
SEQ_EMBED_PICKLE_FILE = './pickle_jar/seq_embed_units_{}_songs_clipped96_vaev5.pkl'.format(NUM_FILES)
# Onsets dataset
ONSETS_FILE = './pickle_jar/onsets_{}_songs_clipped96.h5'.format(NUM_FILES)

# Music shape
MIN_PITCH = 13 # 21 # A-1 (MIDI 21)
MAX_PITCH = 108 # C7 (MIDI 108)
BEATS_PER_UNIT = 4
NUM_TRANSPOSITIONS = 3 # Number of transpositions to perform (maximum 12)

# Don't change unless you know what you're doing
BEAT_RESOLUTION = 24 # This is set by the encoding of the lpd-5 dataset, corresponds to number of ticks per beat
PARTITION_NOTE = 60 # Break into left- and right-accompaniments at middle C
NUM_PITCHES = MAX_PITCH - MIN_PITCH + 1
NUM_TICKS = BEATS_PER_UNIT * BEAT_RESOLUTION
MEAN_THRESHOLD = 0.5/127. # Filter out units with mean values less than 0.5
NOTE_DROP_NOISE = 0.3 # Randomly drop 30% of notes when creating input_noisy dataset

print("Imports and variable definitions successful, using {} files.".format(NUM_FILES))