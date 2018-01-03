import time
import random
import collections
import mido

MEMORY_LENGTH = 50
COMP_CHANNEL = 2 # MIDI channel number (1-16)

class Composer(object):
    """
    Base class for possible composer types, implements:
    - causal memory states of music
    - callback function for new notes
    - empty comping function
    """
    def __init__(self):
        self.player_notes = collections.deque(maxlen=MEMORY_LENGTH)
        self.own_notes = collections.deque(maxlen=MEMORY_LENGTH)
        self.active_notes = []

    def register_player_note(self, msg):
        """
        Keep track of all messages, and which notes are active
        """
        if msg.type == "note_on":
            if not msg.note in self.active_notes:
                self.active_notes.append(msg.note)
            self.player_notes.append(msg)
        elif msg.type == "note_off":
            if msg.note in self.active_notes:
                self.active_notes.remove(msg.note)
            self.player_notes.append(msg)

    def add_to_own_memory(self, msg):
        self.own_notes.append(msg)

    def generate_comp(self, _):
        time.sleep(.2)
        return

class RandomMemory(Composer):
    """
    Simply plays back notes at random from memory of player_notes
    """
    def __init__(self):
        Composer.__init__(self)

    def generate_comp(self, outport):
        if len(self.player_notes):
            msg = self.player_notes[random.randint(0, len(self.player_notes)-1)]
            msg = msg.copy(channel=COMP_CHANNEL-1) # Mido channels from 0-15, MIDI 1-16
            outport.send(msg)
        time.sleep(.2)

class Arpeggiator(Composer):
    """
    Arpeggiate notes from a currently active notes (eg. held chord)
    """
    def __init__(self):
        Composer.__init__(self)
        self.active_notes = []

    def generate_comp(self, outport):
        for n in self.active_notes:
            msg = mido.Message('note_on', note=n, velocity=100, channel=COMP_CHANNEL-1)
            outport.send(msg)
            time.sleep(0.2)
            msg = mido.Message('note_off', note=n, velocity=100, channel=COMP_CHANNEL-1)
            outport.send(msg)