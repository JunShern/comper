import os, shutil
import random
import sys
import numpy as np
import pypianoroll
from matplotlib import pyplot as plt
import cPickle as pickle
import pianoroll_utils

LPD5_DIR = "/media/junshern/s_drive/FYP/MIDI/lpd_5_cleansed_copy"
DATA_DIR = "./pianorolls"
NUM_FILES = 30
PICKLE_FILE = './pickle_jar/units_100_songs.pkl'
# Dataset definitions
NUM_PITCHES = 128
PARTITION_NOTE = 60 # Break into left- and right-accompaniments at middle C
BEAT_RESOLUTION = 24 # This is set by the encoding of the lpd-5 dataset, corresponds to number of ticks per beat
BEATS_PER_UNIT = 4
TICKS_PER_UNIT = BEATS_PER_UNIT * BEAT_RESOLUTION


# Create a new directory for our pianorolls
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
    # Grab a random subset of files from our LPD5 dataset to play with
    files_subset = random.sample(os.listdir(LPD5_DIR), NUM_FILES) # Sampling without replacement
    for filename in files_subset:
        src = os.path.join(LPD5_DIR, filename)
        dest = os.path.join(DATA_DIR, filename)
        shutil.copyfile(src, dest)
    print "Copied", NUM_FILES, "files to", DATA_DIR, "."
else:
    print "Directory already exists."

# Print the first 10 files of our chosen subset
print(os.listdir(DATA_DIR)[:10])

pianorolls = [] # List to hold all our piano rolls
for filename in os.listdir(DATA_DIR):
    # Load pianoroll file as a multitrack object
    multi = pypianoroll.Multitrack(os.path.join(DATA_DIR, filename))
    for track in multi.tracks:
        # Non-empty piano pianoroll
        if track.name == "Piano" and track.pianoroll.shape[0] > 0:
            pianorolls.append(track.pianoroll)
print "Complete. Number of pianorolls collected: ", len(pianorolls)


# This will take a while...
pianorolls_original = pianorolls # Store this somewhere else
print "Transposing", len(pianorolls_original), "pianorolls. This might take a while..."
pianorolls = []
for pianoroll in pianorolls_original:
    print ".",
    # Pick three random transpositions
    for i in [random.choice(range(-5,6)), random.choice(range(-5,6)), random.choice(range(-5,6))]:
        transposed_pianoroll = pianoroll_utils.get_transposed_pianoroll(pianoroll, i)
        pianorolls.append(transposed_pianoroll)
print "Done."
print "New pianorolls has", len(pianorolls), "items."


units = {} # Dictionary to store all data
units["input"] = np.array([]).reshape(0, TICKS_PER_UNIT, NUM_PITCHES)
units["input_next"] = np.array([]).reshape(0, TICKS_PER_UNIT, NUM_PITCHES)
units["comp"] = np.array([]).reshape(0, TICKS_PER_UNIT, NUM_PITCHES)
units["comp_next"] = np.array([]).reshape(0, TICKS_PER_UNIT, NUM_PITCHES)

for pianoroll in pianorolls:
    print ".",
    # Get the units for this pianoroll
    [input_units, input_units_next, comp_units, comp_units_next] = pianoroll_utils.create_units(pianoroll)
    # Append it to the full dataset
    units["input"] = np.concatenate([units["input"], input_units], axis=0)
    units["input_next"] = np.concatenate([units["input_next"], input_units_next], axis=0)
    units["comp"] = np.concatenate([units["comp"], comp_units], axis=0)
    units["comp_next"] = np.concatenate([units["comp_next"], comp_units_next], axis=0)
print("Done extracting units.")

# Print info
print "Collected", units["input"].shape[0], "units from", len(pianorolls), "pianorolls."
print "input_units.shape: ", units["input"].shape
print "input_units_next.shape: ", units["input_next"].shape
print "comp_units.shape: ", units["comp"].shape
print "comp_units_next.shape: ", units["comp_next"].shape

# Save data in a pickle file
with open(PICKLE_FILE, 'wb') as outfile:
    pickle.dump(units, outfile, protocol=pickle.HIGHEST_PROTOCOL)
print "Pickled units to", PICKLE_FILE
