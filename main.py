import mido
import random
import collections
import time

MEMORY_LENGTH = 50
COMP_CHANNEL = 1

class Composer:
    """Composer class stores causal state of music, and generates new notes."""

    def __init__(self):
        self.player_notes = collections.deque(maxlen=MEMORY_LENGTH)
        self.own_notes = collections.deque(maxlen=MEMORY_LENGTH)
        # Prep the composer
        inst_change = mido.Message('program_change', channel=COMP_CHANNEL, program=random.randint(1, 127))
        outport.send(inst_change)

    def message_callback(self, msg):
        outport.send(msg) # Direct bypass
        self.add_to_player_memory(msg)

    def add_to_player_memory(self, msg):
        self.player_notes.append(msg)

    def add_to_own_memory(self, msg):
        self.own_notes.append(msg)

    def generate_comp(self):
        return
        # if len(self.player_notes):
        #     newmsg = self.player_notes[random.randint(0, len(self.player_notes)-1)]
        #     outport.send(newmsg)
        # newmsg = mido.Message(msg.type, channel=COMP_CHANNEL, note=msg.note+4, velocity=msg.velocity)
        # return newmsg.copy(channel=COMP_CHANNEL)

if __name__ == "__main__":
    comp = Composer()
    with mido.open_output('mido_out', virtual=True, autoreset=True) as outport, \
        mido.open_input('mido_in', virtual=True, callback=comp.message_callback) as inport:
        while 1:
            time.sleep(25)
            comp.generate_comp()
