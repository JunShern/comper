import sys
import numpy as np
import pypianoroll
from matplotlib import pyplot as plt

def printInfo(track):
    print "\n#####################"
    print "Name: ", track.name
    print "isDrum: ", track.is_drum
    print "Program: ", track.program
    if len(track.pianoroll):
        print "Active length: ", track.get_active_length()
        print "Active pitch range: ", track.get_active_pitch_range()
    else:
        print "Empty pianoroll."
    print ""

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Load pianoroll matrix
        multi = pypianoroll.Multitrack(sys.argv[1])
        if len(multi.tracks) == 5:
            for track in multi.tracks:
                printInfo(track)
# print("Done loading, now converting")
# pypianoroll.write(loaded, "./lpd_sample.midi")

# Plot the piano-roll
# fig, ax = multi.plot()
# plt.show()