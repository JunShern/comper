import mido
import random
import collections
import time

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

    def message_callback(self, msg):
        print(msg)
        outport.send(msg) # Direct bypass
        if msg.type == "note_on" or msg.type == "note_off":
            self.add_to_player_memory(msg)

    def add_to_player_memory(self, msg):
        self.player_notes.append(msg)

    def add_to_own_memory(self, msg):
        self.own_notes.append(msg)

    def generate_comp(self):
        return

class RandomMemory(Composer):
    """
    Simply plays back notes at random from memory of player_notes
    """
    def __init__(self):
        Composer.__init__(self)

    def generate_comp(self):
        if len(self.player_notes):
            newmsg = self.player_notes[random.randint(0, len(self.player_notes)-1)]
            newmsg = newmsg.copy(channel=COMP_CHANNEL-1) # Mido channels from 0-15, MIDI 1-16
            print(newmsg)
            outport.send(newmsg)

if __name__ == "__main__":
    comp = RandomMemory()
    with mido.open_output('mido_out', virtual=True, autoreset=True) as outport, \
        mido.open_input('mido_in', virtual=True, callback=comp.message_callback) as inport:
        # Prep the composer
        # inst_change = mido.Message('program_change', channel=COMP_CHANNEL, program=random.randint(1, 127))
        # outport.send(inst_change)
        while 1:
            time.sleep(.2)
            comp.generate_comp()
