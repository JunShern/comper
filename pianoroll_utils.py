"""
Useful functions for plotting and playing pianorolls for Comper
"""
import subprocess
import pypianoroll
from matplotlib import pyplot as plt

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
