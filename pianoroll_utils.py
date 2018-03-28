"""
Useful functions for plotting and playing pianorolls for Comper
"""
import subprocess
import pypianoroll
from matplotlib import pyplot as plt
import numpy as np
# Dataset definitions
NUM_PITCHES = 128
PARTITION_NOTE = 60 # Break into left- and right-accompaniments at middle C
BEAT_RESOLUTION = 24 # This is set by the encoding of the lpd-5 dataset, corresponds to number of ticks per beat
BEATS_PER_UNIT = 4
TICKS_PER_UNIT = BEATS_PER_UNIT * BEAT_RESOLUTION


def plot_four_units(units, unit_index):
    """
    Given an input dictionary containing "input", "input_next", "comp" and "comp_next",
    plot 2x2 subplots of the four unit pianorolls
    """
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(10, 6, forward=True)
    ax[0,0].set_title('Input')
    ax[0,1].set_title('Input next')
    ax[1,0].set_title('Comp')
    ax[1,1].set_title('Comp next')
    pypianoroll.plot_pianoroll(ax[0,0], units["input"][unit_index], beat_resolution=24)
    pypianoroll.plot_pianoroll(ax[0,1], units["input_next"][unit_index], beat_resolution=24)
    pypianoroll.plot_pianoroll(ax[1,0], units["comp"][unit_index], beat_resolution=24)
    pypianoroll.plot_pianoroll(ax[1,1], units["comp_next"][unit_index], beat_resolution=24)
    fig.tight_layout()
    return

def playPianoroll(pianoroll, bpm=120.0, beat_resolution=24):
    """
    !!----------- Not widely supported ---------------!!
    Given an input pianoroll, creates a MIDI file in /tmp/
    and plays the MIDI file (requires TiMidity++ softsynth)
    [https://wiki.archlinux.org/index.php/timidity]
    
    Returns the exit code of timidity
    """
    FILEPATH = '/tmp/tmp.midi' # For Linux
    track = pypianoroll.Track(pianoroll=pianoroll, program=0, is_drum=False, name='tmp')
    multitrack = pypianoroll.Multitrack(tracks=[track], tempo=bpm, beat_resolution=beat_resolution)
    pypianoroll.write(multitrack, FILEPATH)
    return_code = subprocess.call("timidity " + FILEPATH, shell=True)
    return return_code

def get_transposed_pianoroll(pianoroll, num_semitones):
    """
    Given an input pianoroll matrix of shape [NUM_TICKS, NUM_PITCHES],
    musically-transpose the pianoroll by num_semitones and
    return the new transposed pianoroll.
    """
    num_ticks = pianoroll.shape[0]
    num_pitches = pianoroll.shape[1]
    assert(abs(num_semitones) <= num_pitches)
    
    # Default case, no transposition
    transposed_pianoroll = pianoroll
    # Transpose up
    if (num_semitones > 0):
        transposed_pianoroll = np.hstack([np.zeros(num_semitones*num_ticks).reshape(num_ticks,num_semitones),
                                          pianoroll[:,:num_pitches-num_semitones] ])
    # Transpose down
    elif (num_semitones < 0):
        num_semitones = abs(num_semitones)
        transposed_pianoroll = np.hstack([pianoroll[:,num_semitones:],
                                          np.zeros(num_semitones*num_ticks).reshape(num_ticks,num_semitones) ])
    # Debug assertion
    assert(transposed_pianoroll.shape == (num_ticks, num_pitches))
    return transposed_pianoroll



def chop_to_unit_multiple(pianoroll, ticks_per_unit):
    """
    Given an input pianoroll matrix of shape [NUM_TICKS, NUM_PITCHES],
    truncate the matrix so that it can be evenly divided into M units.
    
    Returns [M, pianoroll_truncated]
    where M is the largest integer such that M*ticks_per_unit <= NUM_TICKS
    and pianoroll_truncated is of shape [M*ticks_per_unit, NUM_PITCHES]
    """
    
    num_ticks = pianoroll.shape[0]
    num_pitches = pianoroll.shape[1]
    
    # Get M
    M = int(num_ticks / ticks_per_unit) # Floor
    # Truncate
    pianoroll_truncated = pianoroll[:M*ticks_per_unit, :]
    
    # Debug assertions
    assert(M*ticks_per_unit <= num_ticks)
    assert(pianoroll_truncated.shape == (M*ticks_per_unit, num_pitches))
    
    return [M, pianoroll_truncated]


def shuffle_left_right(left_units, left_units_next, right_units, right_units_next):
    """
    Given 4 matrices left, left_next, right, and right_next
    return 4 matrices which have left and right randomly exchanged
    while maintaining index order, eg:
    
    [a1,a2,a3,a4]      [a1,b2,b3,a4]
    [a2,a3,a4,a5]  ->  [a2,b3,b4,a5]
    [b1,b2,b3,b4]      [b1,a2,a3,b4]
    [b2,b3,b4,b5]      [b2,a3,a4,b5]
    """
    
    bool_array = np.random.randint(0, 2, left_units.shape[0], dtype=bool) # Random True/False
    
    # Initialize as copies of one side of the accompaniment
    input_units = left_units.copy()
    input_units_next = left_units_next.copy()
    comp_units = right_units.copy()
    comp_units_next = right_units_next.copy()

    # Replace half of array with elements from the other side
    input_units[bool_array, ...] = right_units[bool_array, ...]
    input_units_next[bool_array, ...] = right_units_next[bool_array, ...]
    comp_units[bool_array, ...] = left_units[bool_array, ...]
    comp_units_next[bool_array, ...] = left_units_next[bool_array, ...]
    
    return [input_units, input_units_next, comp_units, comp_units_next]


def create_units(pianoroll, filter_threshold=0):
    """
    Given an input pianoroll matrix of shape [NUM_TICKS, NUM_PITCHES], 
    return input_units, input_units_next, comp_units, comp_units_next_shape
    all of the same shape [M, TICKS PER UNIT, NUM_PITCHES]
    """
    assert(pianoroll.shape[1] == NUM_PITCHES)
    
    # Truncate pianoroll so it can be evenly divided into units
    # Pianoroll is divided into M+1, not M 
    # since we can only get M next-units for M+1 input units
    [M_plus_one, pianoroll] = chop_to_unit_multiple(pianoroll, TICKS_PER_UNIT)
    M = M_plus_one - 1
    
    # Prepare outputs
    input_units = np.zeros([M, TICKS_PER_UNIT, NUM_PITCHES])
    input_units_next = np.zeros([M, TICKS_PER_UNIT, NUM_PITCHES])
    comp_units = np.zeros([M, TICKS_PER_UNIT, NUM_PITCHES])
    comp_units_next = np.zeros([M, TICKS_PER_UNIT, NUM_PITCHES])
    
    # Split pianoroll into left- and right- accompaniments
    left_comp = pianoroll.copy()
    left_comp[:, PARTITION_NOTE:] = 0
    right_comp = pianoroll.copy()
    right_comp[:, :PARTITION_NOTE] = 0
    
    # Get the units by reshaping left_comp and right_comp
    all_left_units = left_comp.reshape(M_plus_one, TICKS_PER_UNIT, NUM_PITCHES)
    all_right_units = right_comp.reshape(M_plus_one, TICKS_PER_UNIT, NUM_PITCHES)
    left_units = all_left_units[:-1,:,:] # All but the last unit
    left_units_next = all_left_units[1:,:,:] # Skip the first unit
    right_units = all_right_units[:-1,:,:] # All but the last unit
    right_units_next = all_right_units[1:,:,:] # Skip the first unit
    
    # Randomly choose between left/right for input/comp units, 
    # so the model learns both sides of the accompaniment
    [input_units, input_units_next, comp_units, comp_units_next] = \
        shuffle_left_right(left_units, left_units_next, right_units, right_units_next)
    
    # Filter out near-empty units
    input_units_means = np.mean(input_units, axis=(1,2)).squeeze()
    filter_array = input_units_means > filter_threshold
    input_units = input_units[filter_array, ...]
    input_units_next = input_units_next[filter_array, ...]
    comp_units = comp_units[filter_array, ...]
    comp_units_next = comp_units_next[filter_array, ...]
    M = np.sum(filter_array) # Recount M after filtering
    
    # Debug assertions
    assert(input_units.shape == (M, TICKS_PER_UNIT, NUM_PITCHES))
    assert(input_units_next.shape == (M, TICKS_PER_UNIT, NUM_PITCHES))
    assert(comp_units.shape == (M, TICKS_PER_UNIT, NUM_PITCHES))
    assert(comp_units_next.shape == (M, TICKS_PER_UNIT, NUM_PITCHES))
    
    return [input_units, input_units_next, comp_units, comp_units_next]
