import time
import random
import collections
import mido

MEMORY_LENGTH = 500
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
        self.previous_event_time = 0

    def register_player_note(self, msg):
        """
        Keep track of all messages, and which notes are active
        """
        # Delta time gives the time since the previous MIDI event
        deltatime = 0.5 # Arbitrarily set time for first note
        timenow = time.time()
        if self.previous_event_time:
            deltatime = timenow - self.previous_event_time
        self.previous_event_time = timenow
        msg.time = deltatime

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
            time.sleep(msg.time)
            outport.send(msg)
        else:
            time.sleep(0.2)

class Arpeggiator(Composer):
    """
    Arpeggiate notes from a currently active notes (eg. held chord)
    """
    def __init__(self):
        Composer.__init__(self)

    def generate_comp(self, outport):
        for n in self.active_notes:
            msg = mido.Message('note_on', note=n, velocity=100, channel=COMP_CHANNEL-1)
            outport.send(msg)
            time.sleep(0.2)
            msg = mido.Message('note_off', note=n, velocity=100, channel=COMP_CHANNEL-1)
            outport.send(msg)

class MarkovMonophonicDurationless(Composer):
    """
    Markov generator based only on note frequencies,
    no regard to note durations or other characteristics.
    """
    def __init__(self):
        Composer.__init__(self)
        self.markov_chain = {} # This will be a dictionary of state:[nextstates]

    def register_player_note(self, msg):
        """
        Keep track of all messages, and which notes are active.
        Also, update Markov Chain.
        """
        # Delta time gives the time since the previous MIDI event
        deltatime = 0
        timenow = time.clock()
        if self.previous_event_time:
            deltatime = timenow - self.previous_event_time
        self.previous_event_time = timenow
        msg.time = deltatime

        if msg.type == "note_on":
            if not msg.note in self.active_notes:
                self.active_notes.append(msg.note)
                if self.player_notes:
                    # Add to Markov Chain
                    if self.player_notes[-1].note in self.markov_chain:
                        self.markov_chain[self.player_notes[-1].note].append(msg.note)
                    else: # First initialization
                        self.markov_chain[self.player_notes[-1].note] = [msg.note]
            self.player_notes.append(msg)
        elif msg.type == "note_off":
            if msg.note in self.active_notes:
                self.active_notes.remove(msg.note)
            self.player_notes.append(msg)

    def generate_comp(self, outport):
        note_ = 0
        if self.markov_chain:
            # Generate new note from Markov Chain (if the state has been registered in the chain)
            if self.own_notes and (self.own_notes[-1].note in self.markov_chain):
                note_ = random.choice(self.markov_chain[self.own_notes[-1].note])
            # Runs only the first time, before generator has produced anything
            elif self.player_notes and (self.player_notes[-1].note in self.markov_chain):
                note_ = random.choice(self.markov_chain[self.player_notes[-1].note])

        if note_:
            msg = mido.Message('note_on', note=note_, velocity=100, channel=COMP_CHANNEL-1)
            self.add_to_own_memory(msg)
            outport.send(msg)
            time.sleep(0.2)
            msg = mido.Message('note_off', note=note_, velocity=100, channel=COMP_CHANNEL-1)
            self.add_to_own_memory(msg)
            outport.send(msg)
        else:
            time.sleep(0.2)
