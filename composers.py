import random
import collections

MEMORY_LENGTH = 50
COMP_CHANNEL = 2 # MIDI channel number (1-16)

class Composer:
    """
    Base class for possible composer types, implements:
    - causal memory states of music
    - callback function for new notes
    - empty comping function
    """
    def __init__(self):
        self.player_notes = collections.deque(maxlen=MEMORY_LENGTH)
        self.own_notes = collections.deque(maxlen=MEMORY_LENGTH)

    def add_to_player_memory(self, msg):
        if msg.type == "note_on" or msg.type == "note_off":
            self.player_notes.append(msg)

    def add_to_own_memory(self, msg):
        self.own_notes.append(msg)

    def generate_comp(self, _):
        return

class RandomMemory(Composer):
    """
    Simply plays back notes at random from memory of player_notes
    """
    def __init__(self):
        Composer.__init__(self)

    def generate_comp(self, outport):
        if len(self.player_notes):
            newmsg = self.player_notes[random.randint(0, len(self.player_notes)-1)]
            newmsg = newmsg.copy(channel=COMP_CHANNEL-1) # Mido channels from 0-15, MIDI 1-16
            print(newmsg)
            outport.send(newmsg)
