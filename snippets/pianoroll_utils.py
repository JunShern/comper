"""
Useful functions for plotting and playing pianorolls for Comper
"""
import subprocess
import pypianoroll
from matplotlib import pyplot as plt
import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack
import IPython

def score_note_onsets(pianoroll, min_pitch=0, max_pitch=127, sigma=2, beats_per_unit=4, num_units=1):
    """
    Reward for onsets occuring at 1/2-beat marks (ie. at 24-tick resolution, occuring at 0, 11, 23... )
    Also reward onsets at +/- sigma from those marks, for imperceivable timing inaccuracies
    Impartial to onsets occuring at 1/4-beat marks
    Penalizes onsets occuring at all other ticks
    """
    ticks_per_beat = 24
    assert pianoroll.shape[1] == ticks_per_beat * beats_per_unit * num_units

    # Score mask
    score_mask_row = -np.ones(pianoroll.shape[1])
    for half_beat in range(2*beats_per_unit*num_units):
        hb = half_beat * ticks_per_beat / 2
        next_hb = hb + ticks_per_beat / 2
        # Good ticks
        score_mask_row[hb : hb + sigma + 1] = 1
        score_mask_row[next_hb - sigma : next_hb] = 1
        # Impartial ticks
        score_mask_row[hb + ticks_per_beat / 4] = 0
    score_mask = np.zeros(pianoroll.shape)
    score_mask[:] = score_mask_row # Fill all rows

    # Get note onset matrix
    note_onsets = get_note_onsets(pianoroll, min_pitch, max_pitch)
    
    # Calculate score
    if np.sum(note_onsets) != 0:
        score = np.sum(np.multiply(score_mask, note_onsets)) / np.sum(note_onsets)
        return score
    else:
        return 0

def mask_pitches(pianoroll, pitches, min_pitch=0, max_pitch=127):
    """
    Given a pianoroll matrix and a list of pitch numbers,
    mute all notes in that pianoroll corresponding to those pitches.
    Return the new masked pianoroll.
    """
    assert pianoroll.shape[0] == max_pitch - min_pitch + 1
    pitches = pitches - min_pitch
    masked = pianoroll.copy()
    masked[pitches] = 0
    return masked
    
def get_active_pitches(pianoroll, min_pitch=0, max_pitch=127):
    """
    Given a pianoroll matrix, return a list of all pitches that 
    were played in this pianoroll.
    """
    assert pianoroll.shape[0] == max_pitch - min_pitch + 1
    num_pitches = pianoroll.shape[0]
    
    pitches = np.arange(num_pitches) + min_pitch
    active_pitch_rows = np.any(pianoroll, axis=1) # List of booleans
    active_pitches = pitches[active_pitch_rows]
    return active_pitches

def get_active_pitch_classes(pianoroll, min_pitch=0, max_pitch=127):
    """
    Given a pianoroll matrix, return a list of all pitch classes
    (0-11 from C-B) that were played in this pianoroll.
    """
    active_pitches = get_active_pitches(pianoroll, min_pitch, max_pitch)
    active_pitch_classes = np.unique(active_pitches % 12)
    return active_pitch_classes

def pitch_intersection_over_union(pianoroll_1, pianoroll_2, min_pitch=0, max_pitch=127):
    """
    Given two pianoroll matrices, return the intersection over union
    of their active pitch classes (ignoring octaves)
    """
    assert pianoroll_1.shape[0] == max_pitch - min_pitch + 1
    assert pianoroll_2.shape[0] == max_pitch - min_pitch + 1
    
    notes_1 = set(get_active_pitch_classes(pianoroll_1, min_pitch, max_pitch))
    notes_2 = set(get_active_pitch_classes(pianoroll_2, min_pitch, max_pitch))
    
    intersection = notes_1.intersection(notes_2)
    union = notes_1.union(notes_2)
    if len(union) > 0:
        return float(len(intersection)) / len(union)
    else:
        return 0

def crop_pianoroll(pianoroll, min_pitch, max_pitch):
    """
    Given a pianoroll of shape (128, NUM_TICKS),
    crop the pitch axis to range from min_pitch:max_pitch+1
    (inclusive of min_pitch and max_pitch)
    """
    assert pianoroll.shape[0] == 128
    output = pianoroll[min_pitch:max_pitch+1, :] # Crop pitch range of pianoroll
    assert output.shape[0] == max_pitch - min_pitch + 1
    return output

def pad_pianoroll(pianoroll, min_pitch, max_pitch):
    """
    Given a pianoroll of shape (NUM_PITCHES, NUM_TICKS),
    return a zero-padded matrix (128, NUM_TICKS)
    """
    assert pianoroll.shape[0] == max_pitch - min_pitch + 1
    ticks = pianoroll.shape[1]
    front = np.zeros((min_pitch-1, ticks))
    back = np.zeros((128-max_pitch, ticks))
    output = np.vstack((front, pianoroll, back))
    assert output.shape[0] == 128
    return output

def plot_velocities(ax, pianoroll, beat_resolution=24):
    """
    Given a pianoroll, plot the sum of all velocities occurring
    at each time tick.
    """
    assert pianoroll.shape[1] == 96
    num_ticks = pianoroll.shape[1]
    num_beat = 4
    
    ax.step(range(num_ticks), np.sum(pianoroll, axis=0))
    ax.set_ylabel('total velocity')
    ax.set_xlabel('ticks')
    ax.set_xlim([0, num_ticks-1])
    ax.set_ylim([0, 5]) # May exceed but it's okay
    
    # Beat lines
    if beat_resolution is not None:
        num_beat = num_ticks//beat_resolution
        xticks_major = beat_resolution * np.arange(0, num_beat)
        xticks_minor = beat_resolution * (0.5 + np.arange(0, num_beat))
        xtick_labels = np.arange(1, 1 + num_beat)
        ax.set_xticks(xticks_major)
        ax.set_xticklabels('')
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xtick_labels, minor=True)
        ax.tick_params(axis='x', which='minor', width=0)
        ax.set_xlabel('beats')
    ax.grid(axis='both', color='k', linestyle=':', linewidth=.5)
    return

    
def plot_onsets(ax, onsets, beat_resolution=24):
    """
    Plots a pitch class histogram, given an onsets vector
    of shape (96).
    """
    assert len(onsets) == 96
    num_ticks = len(onsets)
    num_beat = 4
    
    ax.step(range(num_ticks), onsets)
    ax.set_ylabel('total onsets velocity')
    ax.set_xlabel('ticks')
    ax.set_xlim([0, num_ticks-1])
    ax.set_ylim([0, 5]) # May exceed but it's okay
    
    # Beat lines
    if beat_resolution is not None:
        num_beat = num_ticks//beat_resolution
        xticks_major = beat_resolution * np.arange(0, num_beat)
        xticks_minor = beat_resolution * (0.5 + np.arange(0, num_beat))
        xtick_labels = np.arange(1, 1 + num_beat)
        ax.set_xticks(xticks_major)
        ax.set_xticklabels('')
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xtick_labels, minor=True)
        ax.tick_params(axis='x', which='minor', width=0)
        ax.set_xlabel('beats')
    ax.grid(axis='both', color='k', linestyle=':', linewidth=.5)
    return

def plot_pitch_class_histogram(ax, hist):
    """
    Plots a pitch class histogram, given a histogram vector
    of shape (12).
    """
    assert len(hist) == 12
    x_pos = np.arange(len(hist))
    ax.bar(x_pos, hist, align='center', alpha=0.5)
    ax.set_xticks(x_pos, minor=False)
    ax.set_ylabel('normalized velocity sum')
    ax.set_xlabel('pitch class')
    ax.set_ylim([0,1])
    return 

def plot_pianoroll(ax, pianoroll, min_pitch=0, max_pitch=127, beat_resolution=None, cmap='Blues'):
    """
    Plots a pianoroll matrix of shape (NUM_PITCHES, NUM_TICKS)
    Code adapted from 
    https://salu133445.github.io/pypianoroll/_modules/pypianoroll/plot.html#plot_pianoroll
    """
    assert pianoroll.shape[0] == max_pitch - min_pitch + 1
    num_ticks = pianoroll.shape[1]
    
    ax.imshow(pianoroll.astype('float32'), cmap=cmap, aspect='auto', 
              vmin=0, vmax=1, origin='lower', interpolation='none')
    ax.set_ylabel('pitch')
    lowest_octave = ((min_pitch - 1) // 12 + 1) - 2
    highest_octave = max_pitch // 12 - 2
    ax.set_yticks(np.arange((lowest_octave + 2) * 12, max_pitch+1, 12) - min_pitch)
    ax.set_yticklabels(['C{}'.format(i) for i in range(lowest_octave, highest_octave + 1)])
    
    ax.set_xlabel('ticks')
    # Beat lines
    if beat_resolution is not None:
        num_beat = num_ticks//beat_resolution
        xticks_major = beat_resolution * np.arange(0, num_beat)
        xticks_minor = beat_resolution * (0.5 + np.arange(0, num_beat))
        xtick_labels = np.arange(1, 1 + num_beat)
        ax.set_xticks(xticks_major)
        ax.set_xticklabels('')
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xtick_labels, minor=True)
        ax.tick_params(axis='x', which='minor', width=0)
        ax.set_xlabel('beats')
    ax.grid(axis='both', color='k', linestyle=':', linewidth=.5)
    return

# DEPRECATED
# def plot_four_units(units, unit_index, min_pitch, max_pitch):
#     """
#     Given an input dictionary containing "input", "input_next", "comp" and "comp_next",
#     plot 2x2 subplots of the four unit pianorolls
#     """
#     fig, ax = plt.subplots(2,2)
#     fig.set_size_inches(10, 6, forward=True)
#     ax[0,0].set_title('Input')
#     ax[0,1].set_title('Input next')
#     ax[1,0].set_title('Comp')
#     ax[1,1].set_title('Comp next')
#     plot_pianoroll(ax[0,0], units["input"][unit_index], min_pitch, max_pitch, beat_resolution=24)
#     plot_pianoroll(ax[0,1], units["input_next"][unit_index], min_pitch, max_pitch, beat_resolution=24)
#     plot_pianoroll(ax[1,0], units["comp"][unit_index], min_pitch, max_pitch, beat_resolution=24)
#     plot_pianoroll(ax[1,1], units["comp_next"][unit_index], min_pitch, max_pitch, beat_resolution=24)
#     fig.tight_layout()
#     return

# DEPRECATED
# def play_pianoroll(pianoroll, min_pitch=0, max_pitch=127, bpm=120.0, beat_resolution=24):
#     """
#     !!----------- Not widely supported ---------------!!
#     Given an input pianoroll, creates a MIDI file in /tmp/
#     and plays the MIDI file (requires TiMidity++ softsynth)
#     [https://wiki.archlinux.org/index.php/timidity]
    
#     Returns the exit code of timidity
#     """
#     FILEPATH = '/tmp/tmp.midi' # For Linux
#     if min_pitch != 0 or max_pitch != 127:
#         print(min_pitch, max_pitch)
#         pianoroll = pad_pianoroll(pianoroll, min_pitch, max_pitch) # Pad to full 128 pitches
#     track = pypianoroll.Track(pianoroll=pianoroll, program=0, is_drum=False, name='tmp')
#     multitrack = pypianoroll.Multitrack(tracks=[track], tempo=bpm, beat_resolution=beat_resolution)
#     pypianoroll.write(multitrack, FILEPATH)
#     return_code = subprocess.call("timidity " + FILEPATH, shell=True)
#     return return_code

def play_pianoroll(pianoroll, min_pitch=0, max_pitch=127, filelabel='0', process=True):
    if process:
        pianoroll = pianoroll_preprocess(pianoroll, min_pitch, max_pitch) # Get proper notes from probability matrix
    filepath = play_midi_events(pianoroll_2_events(pianoroll, min_pitch, max_pitch), filelabel)
    IPython.display.display(IPython.display.Audio(filepath))
    return

def play_midi_events(events, filelabel=0):
    COMP_CHANNEL = 5
    beats_per_bar = 4
    ticks_per_beat = 24

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for tick_event in events:
        for msg in tick_event:
            track.append(msg.copy(channel=COMP_CHANNEL, time=0))
        # This effectively acts as a time.sleep for 1 tick
        track.append(Message('note_off', note=0, velocity=0, time=16))
    FILEPATH = '/tmp/tmp_'+filelabel
    MIDIPATH = FILEPATH + '.mid'
    WAVPATH = FILEPATH + '.wav'
    mid.save(MIDIPATH)
    return_code = subprocess.call("timidity {} -Ow -o {}".format(MIDIPATH, WAVPATH), shell=True)
    if return_code == 0:
        return WAVPATH
    else:
        return return_code

def pianoroll_preprocess(pianoroll, min_pitch=0, max_pitch=127, empty_threshold=0.1, max_threshold=0.20):
    """
    Takes an input matrix of pianoroll note probabilities and
    extracts a clean pianoroll from the probabilities.
    Returns a pianoroll matrix of the same shape.
    """
    assert pianoroll.shape[0] == max_pitch - min_pitch + 1
    num_pitches = pianoroll.shape[0]
    num_ticks = pianoroll.shape[1]
    
    assert np.max(pianoroll) <= 1 # Pianorolls must be normalized between 0 and 1
    pianoroll_ = pianoroll.copy() * 127

    if np.max(pianoroll_) < 127 * empty_threshold:
    # If all notes are ghost notes
        pianoroll_[:] = 0
    else: 
        pianoroll_[pianoroll_ < np.max(pianoroll_) * max_threshold] = 0

    # events = [[] for _ in range(num_ticks)] # Each tick gets a list to store events
    clipped = pianoroll_.astype(int)

    binarized = clipped.astype(bool)
    padded = np.pad(binarized, ((0, 0), (1, 1)), 'constant')
    diff = np.diff(padded.astype(int), axis=1)

    for p in range(num_pitches):
        note_ons = np.nonzero(diff[p,:] > 0)[0]
        note_offs = np.nonzero(diff[p,:] < 0)[0]
        for idx, note_on in enumerate(note_ons):
            velocity = np.mean(clipped[p, note_on:note_offs[idx]])
            clipped[p, note_on:note_offs[idx]] = velocity
    return clipped / 127.

def get_note_onsets(pianoroll, min_pitch=0, max_pitch=127):
    """
    Takes an input pianoroll of shape (NUM_PITCHES, NUM_TICKS)
    and returns a list of tick-indices for all note-on events 
    for each pitch, as a list of lists.
    """
    assert pianoroll.shape[0] == max_pitch - min_pitch + 1
    num_pitches = pianoroll.shape[0]

    binarized = pianoroll.astype(bool)
    padded = np.pad(binarized, ((0, 0), (1, 1)), 'constant')
    diff = np.diff(padded.astype(int), axis=1)
    note_ons = np.array(diff > 0)[:,:-1] # Discard last column

    assert note_ons.shape == pianoroll.shape
    return note_ons

def pianoroll_2_events(pianoroll, min_pitch=0, max_pitch=127):
    """
    Takes an input pianoroll of shape (NUM_PITCHES, NUM_TICKS) 
    and returns a list of quantized events
    "Adjacent nonzero values of the same pitch will be considered a 
    single note with their mean as its velocity.", as per pypianoroll.
    https://github.com/salu133445/pypianoroll/blob/master/pypianoroll/multitrack.py#L1171
    """
    assert pianoroll.shape[0] == max_pitch - min_pitch + 1
    num_pitches = pianoroll.shape[0]
    num_ticks = pianoroll.shape[1]
    
    assert np.max(pianoroll) <= 1 # Pianorolls must be normalized between 0 and 1
    pianoroll = pianoroll * 127

    events = [[] for _ in range(num_ticks)] # Each tick gets a list to store events
    clipped = pianoroll.astype(int)
    binarized = clipped.astype(bool)
    padded = np.pad(binarized, ((0, 0), (1, 1)), 'constant')
    diff = np.diff(padded.astype(int), axis=1)

    for p in range(num_pitches):
        pitch = min_pitch + p
        note_ons = np.nonzero(diff[p,:] > 0)[0]
        note_offs = np.nonzero(diff[p,:] < 0)[0]
        for idx, note_on in enumerate(note_ons):
            velocity = np.mean(clipped[p, note_on:note_offs[idx]])
            # Create message events
            on_msg = mido.Message('note_on', note=pitch, velocity=int(velocity), time=0)
            events[note_ons[idx]].append(on_msg)
            if note_offs[idx] < num_ticks:
                off_msg = mido.Message('note_on', note=pitch, velocity=0, time=0)
                events[note_offs[idx]].append(off_msg)
    return events

def get_transposed_pianoroll(pianoroll, num_semitones):
    """
    Given an input pianoroll matrix of shape [NUM_PITCHES, NUM_TICKS],
    musically-transpose the pianoroll by num_semitones and
    return the new transposed pianoroll.
    """
    num_pitches = pianoroll.shape[0]
    num_ticks = pianoroll.shape[1]
    assert(abs(num_semitones) <= num_pitches)
    
    # Default case, no transposition
    transposed_pianoroll = pianoroll.copy()
    # Transpose up
    if (num_semitones > 0):
        transposed_pianoroll = np.vstack([np.zeros(num_semitones*num_ticks).reshape(num_semitones, num_ticks),
                                          pianoroll[:num_pitches-num_semitones, :] ])
    # Transpose down
    elif (num_semitones < 0):
        num_semitones = abs(num_semitones)
        transposed_pianoroll = np.vstack([pianoroll[num_semitones:, :],
                                          np.zeros(num_semitones*num_ticks).reshape(num_semitones, num_ticks) ])
    # Debug assertion
    assert(transposed_pianoroll.shape == (num_pitches, num_ticks))
    return transposed_pianoroll



def chop_to_unit_multiple(pianoroll, ticks_per_unit):
    """
    Given an input pianoroll matrix of shape [NUM_PITCHES, NUM_TICKS],
    truncate the matrix so that it can be evenly divided into M units.
    
    Returns [M, pianoroll_truncated]
    where M is the largest integer such that M*ticks_per_unit <= NUM_TICKS
    and pianoroll_truncated is of shape [NUM_PITCHES, M*ticks_per_unit]
    """
    
    num_pitches = pianoroll.shape[0]
    num_ticks = pianoroll.shape[1]
    
    # Get M
    M = int(num_ticks / ticks_per_unit) # Floor
    # Truncate
    pianoroll_truncated = pianoroll[:, :M*ticks_per_unit]
    
    # Debug assertions
    assert(M*ticks_per_unit <= num_ticks)
    assert(pianoroll_truncated.shape == (num_pitches, M*ticks_per_unit))
    
    return [M, pianoroll_truncated]


def shuffle_left_right(left_units, right_units):
    """
    Given 2 matrices of left and right pianorolls units,
    return 2 matrices which have left and right randomly exchanged
    while maintaining index order, eg:
    
    [a1,a2,a3,a4]  ->  [a1,b2,b3,a4]
    [b1,b2,b3,b4]      [b1,a2,a3,b4]
    """
    
    bool_array = np.random.randint(0, 2, left_units.shape[0], dtype=bool) # Random True/False
    
    # Initialize as copies of one side of the accompaniment
    input_units = left_units.copy()
    comp_units = right_units.copy()

    # Replace half of array with elements from the other side
    input_units[bool_array, ...] = right_units[bool_array, ...]
    comp_units[bool_array, ...] = left_units[bool_array, ...]
    
    return [input_units, comp_units]


def create_units(pianoroll, num_pitches, ticks_per_unit, partition_note,
    min_pitch=0, filter_threshold=0, shuffle=True, return_full_units=False):
    """
    Given an input pianoroll matrix of shape [NUM_PITCHES, ticks_per_unit], 
    return input_units and comp_units of shape [M, NUM_PITCHES, ticks_per_unit]
    """
    assert(pianoroll.shape[0] == num_pitches)
    
    # Truncate pianoroll so it can be evenly divided into units
    [M, pianoroll] = chop_to_unit_multiple(pianoroll, ticks_per_unit)
    
    # Split pianoroll into left- and right- accompaniments
    partition_note = partition_note - min_pitch
    left_comp = pianoroll.copy()
    left_comp[partition_note:, :] = 0
    right_comp = pianoroll.copy()
    right_comp[:partition_note, :] = 0
    
    # Get the units by reshaping left_comp and right_comp
    left_units = left_comp.T.reshape(M, ticks_per_unit, num_pitches).swapaxes(1,2)
    right_units = right_comp.T.reshape(M, ticks_per_unit, num_pitches).swapaxes(1,2)
    full_units = pianoroll.T.reshape(M, ticks_per_unit, num_pitches).swapaxes(1,2)
    
    # Randomly choose between left/right for input/comp units, 
    # so the model learns both sides of the accompaniment
    if shuffle:
        [input_units, comp_units] = shuffle_left_right(left_units, right_units)
    else:
        [input_units, comp_units] = [left_units, right_units]

    # Filter out both units if near-empty input units
    input_units_means = np.mean(input_units, axis=(1,2)).squeeze()
    filter_array = input_units_means >= filter_threshold
    input_units = input_units[filter_array, ...]
    comp_units = comp_units[filter_array, ...]
    full_units = full_units[filter_array, ...]
    M = np.sum(filter_array) # Recount M after filtering
    
    # Debug assertions
    assert(input_units.shape == (M, num_pitches, ticks_per_unit))
    assert(comp_units.shape == (M, num_pitches, ticks_per_unit))
    assert(full_units.shape == (M, num_pitches, ticks_per_unit))
    
    if return_full_units:
        return [input_units, comp_units, full_units]
    else:
        return [input_units, comp_units]

def one_hot_to_pianoroll(one_hot_matrix):
    """
    Given a one-hot matrix [NUM_VECTORS=NUM_TICKS, NUM_TOKENS=NUM_PITCHES],
    return a pianoroll matrix of shape [NUM_PITCHES, NUM_TICKS] but 
    with special tokens converted according to:
    - Start token [...,0,0,0,1] -> Copy next token
    - Empty token [1,0,0,0,...] -> [0,0,0,...,0]
    """
    assert np.all(np.sum(one_hot_matrix, axis=1) == 1)
    num_tokens = one_hot_matrix.shape[1]
    proll = one_hot_matrix.copy()
    # Define special tokens
    empty_token = np.zeros(num_tokens)
    empty_token[0] = 1
    start_token = np.zeros(num_tokens)
    start_token[-1] = 1
    # Start tokens
    start_token_rows = np.all(proll == start_token, axis=1) # Boolean indexing array
    next_token_rows = np.concatenate([[False], start_token_rows[:-1]]) # Boolean array shifted
    proll[start_token_rows] = proll[next_token_rows] # Replace with next token
    # End tokens
    empty_token_rows = np.all(proll == empty_token, axis=1)
    proll[empty_token_rows] = np.zeros(num_tokens) # Replace with zeros
    # Change to (NUM_PITCHES, NUM_TICKS)
    proll = proll.swapaxes(0,1)
    return proll
    
def create_bass_units(pianoroll, num_pitches=128, ticks_per_unit=96, filter_threshold=0):
    """
    Given an input pianoroll matrix of shape [NUM_PITCHES, ticks_per_unit], 
    return input_units and bass_units of shape [M, NUM_PITCHES, ticks_per_unit]
    where bass_units contain the lowest-pitched notes of each tick and 
    input_units contain the remaining notes.
    """
    assert pianoroll.shape[0] == num_pitches
    
    # Truncate pianoroll so it can be evenly divided into units
    [M, pianoroll] = chop_to_unit_multiple(pianoroll, ticks_per_unit)
    
    # Split pianoroll into left- and right- accompaniments
    binary_pianoroll = pianoroll.astype(bool)
    # Using argmax to find lowest pitches, all-zero columns will select the first pitch. 
    # So we use this [1,0,0,...] vector as the OHE for empty token
    pitch_indices = np.argmax(binary_pianoroll, axis=0)
    input_pianoroll = pianoroll.copy()
    input_pianoroll[pitch_indices, np.arange(input_pianoroll.shape[1])] = 0
    bass_pianoroll = np.zeros(pianoroll.shape)
    bass_pianoroll[pitch_indices, np.arange(bass_pianoroll.shape[1])] = 1
    
    # Get the units by reshaping left_comp and right_comp
    input_units = input_pianoroll.T.reshape(M, ticks_per_unit, num_pitches).swapaxes(1,2)
    bass_units = bass_pianoroll.T.reshape(M, ticks_per_unit, num_pitches).swapaxes(1,2)
    
    # Filter out both units if near-empty input units
    input_units_means = np.mean(input_units, axis=(1,2)).squeeze()
    filter_array = input_units_means >= filter_threshold
    input_units = input_units[filter_array, ...]
    bass_units = bass_units[filter_array, ...]
    M = np.sum(filter_array) # Recount M after filtering
    
    # Replace first column of all bass units with start-of-sequence token [...,0,0,0,1]
    start_token = np.zeros((num_pitches))
    start_token[-1] = 1
    bass_units[:, :, 0] = start_token
    
    # Debug assertions
    assert(input_units.shape == (M, num_pitches, ticks_per_unit))
    assert(bass_units.shape == (M, num_pitches, ticks_per_unit))
    
    return [input_units, bass_units]

def get_pianoroll(filepath, min_pitch=0, max_pitch=127):
    """
    Given a path to a pypianoroll .npz file from LPD5-clean dataset, 
    extract the Piano track into a 0-1 normalized pianoroll matrix of
    shape (NUM_PITCHES, ?) where NUM_PITCHES = max_pitch - min_pitch + 1
    """
    # Load pianoroll file as a multitrack object
    multi = pypianoroll.Multitrack(filepath)
    for track in multi.tracks:
        # Non-empty piano pianoroll
        if track.name == "Piano":
            if track.pianoroll.shape[0] > 0:
                proll = track.pianoroll.T
                # Clip and normalize velocities between 0 and 1
                proll = proll.clip(0, 127) / 127.
                # Crop pitch range of pianoroll
                proll = crop_pianoroll(proll, min_pitch, max_pitch)
                return proll
            else:
                return np.array([]) # No pianoroll
    # Error
    print("Unexpected condition: No Piano track in file", filepath)
    return np.array([]) # No pianoroll
